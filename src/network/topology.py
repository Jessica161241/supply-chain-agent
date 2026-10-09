# -*- coding: utf-8 -*-
"""B 同学：构建三层供应链网络拓扑（region -> city -> store）

职责
----
1. 读取 config/network_config.yaml；
2. 构造 SupplyChainGraph（nodes: {名: EchelonConfig} + edges: {(上游,下游): EdgeConfig}）；
3. 返回 NetworkSupplyChain 对象（默认 OrderUpToPolicy 基线）。

接口
----
build_network_from_config(config_path) -> NetworkSupplyChain
run_baseline_policy(network, demand, forecasts_mean, forecasts_std) -> SimulationResult

框架事实（读本地 deepbullwhip 0.4.1 源码实测）
----
* ``SupplyChainGraph.__init__(nodes: dict[str, EchelonConfig],
  edges: dict[tuple[str, str], EdgeConfig])``——边的连接关系由 **dict 的 key
  (上游名, 下游名)** 表达；``EdgeConfig`` 只承载 lead_time / capacity /
  transport_cost，不含 source/target（对齐清单 §1.7：EdgeConfig.lead_time 不生效，
  补货提前期配在节点的 EchelonConfig.lead_time 上）。
* ``NetworkSupplyChain.__init__(graph, policies=None, cost_fns=None)``——
  不接受 echelons/edges/policy 关键字；``policies`` 是 **{节点名: OrderingPolicy}**
  的字典，**每个节点一个独立实例**（策略对象持有状态、跨时间步常驻）。
* 若某节点未在 policies 中指定，框架会用默认策略。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from deepbullwhip import (
    EchelonConfig,
    EdgeConfig,
    NetworkSupplyChain,
    OrderUpToPolicy,
    SupplyChainGraph,
)

__all__ = [
    "build_network_from_config",
    "build_coordinated_network_from_config",
    "run_baseline_policy",
]


def build_network_from_config(config_path: str | Path) -> NetworkSupplyChain:
    """从 YAML 配置构建三层供应链网络（默认 OrderUpTo 基线策略）。

    Parameters
    ----------
    config_path : str or Path
        ``config/network_config.yaml`` 的路径。

    Returns
    -------
    NetworkSupplyChain
        可直接调用 ``.simulate(demand, forecasts_mean, forecasts_std)`` 的对象。

    Raises
    ------
    FileNotFoundError
        配置文件不存在时。
    KeyError
        YAML 缺少必要字段时。
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    net = cfg["network"]

    # ---- 1. 构造节点：{节点名: EchelonConfig} ----
    nodes: dict[str, EchelonConfig] = {}

    # 区域仓
    rw = net["regional_warehouse"]
    nodes[rw["name"]] = EchelonConfig(
        name=rw["name"],
        lead_time=rw["lead_time"],
        holding_cost=rw["holding_cost"],
        backorder_cost=rw["backorder_cost"],
        depreciation_rate=rw["depreciation_rate"],
        service_level=rw["service_level"],
        initial_inventory=rw["initial_inventory"],
    )

    # 城市仓 + 终端门店
    store_defaults = net["store_defaults"]
    for cw in net["city_warehouses"]:
        nodes[cw["name"]] = EchelonConfig(
            name=cw["name"],
            lead_time=cw["lead_time"],
            holding_cost=cw["holding_cost"],
            backorder_cost=cw["backorder_cost"],
            depreciation_rate=cw["depreciation_rate"],
            service_level=cw["service_level"],
            initial_inventory=cw["initial_inventory"],
        )
        for store_name in cw["stores"]:
            nodes[store_name] = EchelonConfig(
                name=store_name,
                lead_time=store_defaults["lead_time"],
                holding_cost=store_defaults["holding_cost"],
                backorder_cost=store_defaults["backorder_cost"],
                depreciation_rate=store_defaults["depreciation_rate"],
                service_level=store_defaults["service_level"],
                initial_inventory=store_defaults["initial_inventory"],
            )

    # ---- 2. 构造边：{(上游, 下游): EdgeConfig} ----
    # 注：对齐清单 §1.7 实测 EdgeConfig 的 lead_time 不生效，
    # 这里只用它占位（capacity=inf, transport_cost=0.0 均为默认）。
    edges: dict[tuple[str, str], EdgeConfig] = {}

    for cw in net["city_warehouses"]:
        # 区域仓 -> 城市仓
        edges[(rw["name"], cw["name"])] = EdgeConfig()
        # 城市仓 -> 门店
        for store_name in cw["stores"]:
            edges[(cw["name"], store_name)] = EdgeConfig()

    # ---- 3. 构造 SupplyChainGraph ----
    graph = SupplyChainGraph(nodes=nodes, edges=edges)

    # ---- 4. 每个节点一个独立的 OrderUpToPolicy 实例 ----
    # 独立实例很重要：策略对象在仿真期间常驻、持有库存状态，不可共用。
    policies = {name: OrderUpToPolicy(lead_time=nodes[name].lead_time) for name in nodes}

    # ---- 5. 构造 NetworkSupplyChain ----
    return NetworkSupplyChain(graph=graph, policies=policies)


def run_baseline_policy(network: NetworkSupplyChain,
                        demand: dict,
                        forecasts_mean: dict,
                        forecasts_std: dict):
    """用默认 OrderUpTo 策略跑一遍仿真。

    Parameters
    ----------
    network : NetworkSupplyChain
        :func:`build_network_from_config` 的返回值。
    demand : dict
        ``{节点名: 需求序列 (T,)}``，只需为**叶子门店**提供（中间节点的需求
        由框架按下游订单之和推出）。
    forecasts_mean / forecasts_std : dict
        ``{节点名: 预测均值/标准差 (T,)}``，由 A 同学的
        ``build_forecast_bundle`` 提供，只需覆盖叶子门店
        （对齐清单 §1.4）。

    Returns
    -------
    NetworkSimulationResult
        含 node_results / edge_flows / cumulative_bullwhip / total_cost。
    """
    return network.simulate(demand, forecasts_mean, forecasts_std)

def build_coordinated_network_from_config(
    config_path: str | Path, alpha: float = 0.3
) -> tuple[NetworkSupplyChain, SignalBus]:
    """构建带协同策略的网络（OrderUpTo 基线 + 下游聚合缺口信号）。

    Parameters
    ----------
    config_path : str or Path
        ``config/network_config.yaml`` 的路径。
    alpha : float, default 0.3
        协同强度系数；``alpha=0`` 时等价于纯 OrderUpTo。

    Returns
    -------
    tuple[NetworkSupplyChain, SignalBus]
        ``(网络对象, 共享信号总线)``。网络里每个节点的策略均为
        :class:`CoordinatedOrderingPolicy`，共享同一个 ``SignalBus``；
        总线一并返回是为了让调用方能在仿真后读取信号快照
        （网络对象本身不暴露策略字典）。
    """
    from src.coordination.custom_policy import CoordinatedOrderingPolicy
    from src.coordination.signal_protocol import SignalBus

    # 复用 build_network_from_config 的图构造逻辑
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    net_cfg = cfg["network"]

    # --- 与 build_network_from_config 相同的节点/边构造（此处直接内联，避免抽象泄漏）---
    nodes: dict[str, EchelonConfig] = {}
    rw = net_cfg["regional_warehouse"]
    nodes[rw["name"]] = EchelonConfig(
        name=rw["name"], lead_time=rw["lead_time"],
        holding_cost=rw["holding_cost"], backorder_cost=rw["backorder_cost"],
        depreciation_rate=rw["depreciation_rate"],
        service_level=rw["service_level"],
        initial_inventory=rw["initial_inventory"],
    )
    store_defaults = net_cfg["store_defaults"]
    for cw in net_cfg["city_warehouses"]:
        nodes[cw["name"]] = EchelonConfig(
            name=cw["name"], lead_time=cw["lead_time"],
            holding_cost=cw["holding_cost"], backorder_cost=cw["backorder_cost"],
            depreciation_rate=cw["depreciation_rate"],
            service_level=cw["service_level"],
            initial_inventory=cw["initial_inventory"],
        )
        for store_name in cw["stores"]:
            nodes[store_name] = EchelonConfig(
                name=store_name, lead_time=store_defaults["lead_time"],
                holding_cost=store_defaults["holding_cost"],
                backorder_cost=store_defaults["backorder_cost"],
                depreciation_rate=store_defaults["depreciation_rate"],
                service_level=store_defaults["service_level"],
                initial_inventory=store_defaults["initial_inventory"],
            )
    edges: dict[tuple[str, str], EdgeConfig] = {}
    for cw in net_cfg["city_warehouses"]:
        edges[(rw["name"], cw["name"])] = EdgeConfig()
        for store_name in cw["stores"]:
            edges[(cw["name"], store_name)] = EdgeConfig()
    graph = SupplyChainGraph(nodes=nodes, edges=edges)

    # --- 共享总线 + 每个节点一个协同策略 ---
    bus = SignalBus()
    policies: dict[str, CoordinatedOrderingPolicy] = {}
    for name in nodes:
        downstream = graph.downstream_neighbors(name)
        policies[name] = CoordinatedOrderingPolicy(
            lead_time=nodes[name].lead_time,
            service_level=nodes[name].service_level,
            signal_bus=bus,
            node_name=name,
            downstream_names=downstream,
            alpha=alpha,
        )

    return NetworkSupplyChain(graph=graph, policies=policies), bus
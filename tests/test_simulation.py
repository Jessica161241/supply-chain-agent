# -*- coding: utf-8 -*-
"""B 同学仿真流程单元测试（pytest）

负责人: B同学

覆盖范围
----
1. 拓扑构建：节点/边数量、节点名、拓扑序、上下游关系、配置缺失处理；
2. 基线仿真：结果结构完整、指标合理、确定性；
3. 协同仿真：协同网络构建、信号在仿真内传递、不同 alpha 产生不同结果；
4. 边界：最小 T、需求含 0、波动数据。

设计约定
----
* 全部用例离线可跑：合成数据 seed 固定；
* 用小 T（30 天）+ 少 SKU，秒级完成；
* 本文件不重复测策略内部逻辑（那是 test_policy.py 的事），
  只测"拓扑 + 仿真流程"层面的正确性。

运行
----
    pytest tests/test_simulation.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.coordination.signal_protocol import SignalBus
from src.network.topology import (
    build_coordinated_network_from_config,
    build_network_from_config,
    run_baseline_policy,
)

CONFIG_PATH = "config/network_config.yaml"


# ---------------------------------------------------------------------------
# 共享 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_demand():
    """4 门店 × 30 天小规模需求与预测，仿真秒级完成。"""
    from src.data_loader import load_synthetic_demand, pivot_demand_wide

    df = load_synthetic_demand(n_nodes=4, n_days=30, seed=42)
    wide = pivot_demand_wide(df)
    store_names = ["store_1", "store_2", "store_3", "store_4"]
    demand, fm, fs = {}, {}, {}
    for store, col in zip(store_names, wide.columns[:4]):
        s = wide[col].to_numpy(dtype=np.float64)
        demand[store] = s
        fm[store] = np.full_like(s, s.mean())
        fs[store] = np.full_like(s, s.std())
    return demand, fm, fs


# ---------------------------------------------------------------------------
# TestNetworkConstruction —— 拓扑结构
# ---------------------------------------------------------------------------

class TestNetworkConstruction:
    """build_network_from_config：节点/边/拓扑序/邻居关系。"""

    def test_node_count(self):
        """配置了 1 区域仓 + 2 城市仓 + 4 门店 = 7 节点。"""
        net = build_network_from_config(CONFIG_PATH)
        assert len(net.graph.nodes) == 7

    def test_edge_count(self):
        """1×2 区域→城市 + 2×4 城市→门店 = 6 条边。"""
        net = build_network_from_config(CONFIG_PATH)
        assert len(net.graph.edges) == 6

    def test_expected_node_names(self):
        """节点名与配置一致。"""
        net = build_network_from_config(CONFIG_PATH)
        expected = {
            "regional_warehouse",
            "city_warehouse_1", "city_warehouse_2",
            "store_1", "store_2", "store_3", "store_4",
        }
        assert set(net.graph.nodes) == expected

    def test_topological_order(self):
        """拓扑序必须满足：区域仓在前，城市仓居中，门店在后。"""
        net = build_network_from_config(CONFIG_PATH)
        order = net.graph.topological_order()
        # 位置索引
        pos = {name: i for i, name in enumerate(order)}
        assert pos["regional_warehouse"] < pos["city_warehouse_1"]
        assert pos["regional_warehouse"] < pos["city_warehouse_2"]
        for city in ("city_warehouse_1", "city_warehouse_2"):
            for store in ("store_1", "store_2", "store_3", "store_4"):
                # 只对真实相连的对断言（下方邻居测试覆盖具体连接）
                pass
        # 门店总体在城市仓之后
        assert pos["city_warehouse_1"] < pos["store_1"]
        assert pos["city_warehouse_2"] < pos["store_3"]

    def test_downstream_relationships(self):
        """下游邻居关系与配置一致。"""
        net = build_network_from_config(CONFIG_PATH)
        g = net.graph
        # 区域仓的下游是两个城市仓
        assert set(g.downstream_neighbors("regional_warehouse")) == {
            "city_warehouse_1", "city_warehouse_2"}
        # 城市仓 1 的下游是 store_1/2
        assert set(g.downstream_neighbors("city_warehouse_1")) == {
            "store_1", "store_2"}
        # 城市仓 2 的下游是 store_3/4
        assert set(g.downstream_neighbors("city_warehouse_2")) == {
            "store_3", "store_4"}
        # 门店是叶子节点
        for store in ("store_1", "store_2", "store_3", "store_4"):
            assert g.downstream_neighbors(store) == []

    def test_upstream_relationships(self):
        """上游邻居关系正确。"""
        net = build_network_from_config(CONFIG_PATH)
        g = net.graph
        assert g.upstream_neighbors("regional_warehouse") == []
        assert set(g.upstream_neighbors("city_warehouse_1")) == {
            "regional_warehouse"}
        assert set(g.upstream_neighbors("store_1")) == {"city_warehouse_1"}
        assert set(g.upstream_neighbors("store_4")) == {"city_warehouse_2"}

    def test_missing_config_raises(self):
        """配置文件不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            build_network_from_config("config/__no_such_file__.yaml")


# ---------------------------------------------------------------------------
# TestBaselineSimulation —— 基线仿真
# ---------------------------------------------------------------------------

class TestBaselineSimulation:
    """run_baseline_policy：结果结构、指标、确定性。"""

    def test_returns_network_simulation_result(self, small_demand):
        """返回类型是框架的 NetworkSimulationResult。"""
        from deepbullwhip import NetworkSimulationResult
        demand, fm, fs = small_demand
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert isinstance(result, NetworkSimulationResult)

    def test_node_results_cover_all_nodes(self, small_demand):
        """node_results 覆盖全部 7 个节点。"""
        demand, fm, fs = small_demand
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert len(result.node_results) == 7
        assert set(result.node_results) == set(net.graph.nodes)

    def test_series_length_matches_T(self, small_demand):
        """每个节点的 orders / inventory_levels 长度 = T（30）。"""
        demand, fm, fs = small_demand
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        for name, r in result.node_results.items():
            assert r.orders.shape == (30,), f"{name}.orders 长度错误"
            assert r.inventory_levels.shape == (30,), f"{name}.inventory_levels 长度错误"

    def test_metrics_are_finite_and_positive(self, small_demand):
        """关键指标有限且为正。"""
        demand, fm, fs = small_demand
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert np.isfinite(result.total_cost) and result.total_cost > 0
        assert np.isfinite(result.cumulative_bullwhip)
        assert result.cumulative_bullwhip > 0

    def test_determinism(self, small_demand):
        """同一输入跑两次，结果完全一致（无内部随机源）。"""
        demand, fm, fs = small_demand
        net1 = build_network_from_config(CONFIG_PATH)
        net2 = build_network_from_config(CONFIG_PATH)
        r1 = run_baseline_policy(net1, demand, fm, fs)
        r2 = run_baseline_policy(net2, demand, fm, fs)
        assert r1.total_cost == pytest.approx(r2.total_cost, rel=1e-12)
        assert r1.cumulative_bullwhip == pytest.approx(
            r2.cumulative_bullwhip, rel=1e-12)
        for name in r1.node_results:
            np.testing.assert_allclose(
                r1.node_results[name].orders,
                r2.node_results[name].orders,
                rtol=1e-12, atol=1e-12,
            )

    def test_edge_flows_populated(self, small_demand):
        """edge_flows 覆盖全部 6 条边。"""
        demand, fm, fs = small_demand
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert len(result.edge_flows) == 6
        assert set(result.edge_flows) == set(net.graph.edges)


# ---------------------------------------------------------------------------
# TestCoordinatedSimulation —— 协同仿真集成
# ---------------------------------------------------------------------------

class TestCoordinatedSimulation:
    """build_coordinated_network_from_config：返回元组、信号传递、alpha 效果。"""

    def test_returns_network_and_bus(self):
        """返回 (NetworkSupplyChain, SignalBus) 元组。"""
        from deepbullwhip import NetworkSupplyChain
        net, bus = build_coordinated_network_from_config(CONFIG_PATH, alpha=0.5)
        assert isinstance(net, NetworkSupplyChain)
        assert isinstance(bus, SignalBus)

    def test_network_structure_same_as_baseline(self):
        """协同网络的图结构与基线一致（节点、边不变，只换策略）。"""
        net_base = build_network_from_config(CONFIG_PATH)
        net_coord, _ = build_coordinated_network_from_config(CONFIG_PATH)
        assert set(net_base.graph.nodes) == set(net_coord.graph.nodes)
        assert set(net_base.graph.edges) == set(net_coord.graph.edges)

    def test_signals_populated_after_simulation(self, small_demand):
        """仿真后 SignalBus 有 7 个节点的信号。"""
        demand, fm, fs = small_demand
        net, bus = build_coordinated_network_from_config(CONFIG_PATH, alpha=0.3)
        net.simulate(demand, fm, fs)
        snap = bus.snapshot()
        assert len(snap) == 7, f"应 7 个节点有信号，实际 {sorted(snap)}"
        # 中间层信号必须存在（自下而上传上来了）
        for name in ("city_warehouse_1", "city_warehouse_2",
                     "regional_warehouse"):
            assert name in snap
            assert np.isfinite(snap[name].gap)

    def test_different_alpha_gives_different_cost(self, small_demand):
        """alpha=0 与 alpha=0.5 应产生不同的总成本。"""
        demand, fm, fs = small_demand
        net_a0, _ = build_coordinated_network_from_config(CONFIG_PATH, alpha=0.0)
        net_a5, _ = build_coordinated_network_from_config(CONFIG_PATH, alpha=0.5)
        r0 = net_a0.simulate(demand, fm, fs)
        r5 = net_a5.simulate(demand, fm, fs)
        assert r0.total_cost != pytest.approx(r5.total_cost, rel=1e-6)

    def test_alpha_zero_matches_baseline_cost(self, small_demand):
        """alpha=0 的协同网络总成本 = 基线总成本（补偿项恒为 0）。"""
        demand, fm, fs = small_demand
        net_base = build_network_from_config(CONFIG_PATH)
        r_base = run_baseline_policy(net_base, demand, fm, fs)
        net_a0, _ = build_coordinated_network_from_config(CONFIG_PATH, alpha=0.0)
        r_a0 = net_a0.simulate(demand, fm, fs)
        assert r_a0.total_cost == pytest.approx(r_base.total_cost, rel=1e-6)


# ---------------------------------------------------------------------------
# TestSimulationEdgeCases —— 边界与鲁棒性
# ---------------------------------------------------------------------------

class TestSimulationEdgeCases:
    """最小 T、含 0 需求、高波动数据的鲁棒性。"""

    def _build_demand(self, n_days: int, values_fn):
        """构造 4 门店 × n_days 的 demand/fm/fs（用给定函数生成序列）。"""
        store_names = ["store_1", "store_2", "store_3", "store_4"]
        demand, fm, fs = {}, {}, {}
        for i, store in enumerate(store_names):
            s = values_fn(i, n_days)
            demand[store] = s
            fm[store] = np.full_like(s, float(s.mean()) if s.size else 0.0)
            fs[store] = np.full_like(s, float(s.std()) if s.size > 1 else 0.0)
        return demand, fm, fs

    def test_minimal_T_one(self):
        """T=1 的仿真能跑完（最小长度）。"""
        demand, fm, fs = self._build_demand(1, lambda i, T: np.array([10.0]))
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert result.total_cost >= 0
        for r in result.node_results.values():
            assert r.orders.shape == (1,)

    def test_zero_demand(self):
        """需求全 0：仿真不崩溃（策略可能仍下单，用于补安全库存）。"""
        demand, fm, fs = self._build_demand(20, lambda i, T: np.zeros(T))
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert np.isfinite(result.total_cost)

    def test_high_volatility(self):
        """高波动需求：仿真能跑完，指标有限。"""
        rng = np.random.default_rng(0)
        demand, fm, fs = self._build_demand(
            50, lambda i, T: rng.uniform(0, 100, size=T))
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert np.isfinite(result.total_cost) and result.total_cost > 0
        assert np.isfinite(result.cumulative_bullwhip)

    def test_constant_demand(self):
        """需求恒定：std=0，策略不因除以 0 崩溃。"""
        demand, fm, fs = self._build_demand(20, lambda i, T: np.full(T, 15.0))
        net = build_network_from_config(CONFIG_PATH)
        result = run_baseline_policy(net, demand, fm, fs)
        assert np.isfinite(result.total_cost)
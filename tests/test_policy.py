# -*- coding: utf-8 -*-
"""B 同学协同模块单元测试（pytest）

负责人: B同学（src/coordination/* 为 B 全权）

覆盖范围
----
1. ``src/coordination/signal_protocol.py`` —— AggregatedSignal 数据类、
   SignalBus 的 write / read / read_downstream / reset / snapshot；
2. ``src/coordination/custom_policy.py``   —— CoordinatedOrderingPolicy 的
   alpha=0 退化、信号补偿、信号上报、reset；
3. ``src/network/topology.py``             —— build_coordinated_network_from_config
   的集成验证（小规模仿真，验证信号跨节点传递）。

设计约定
----
* 全部用例离线可跑：数据一律用合成兜底（seed 固定可复现）；
* 不修改 A 的模块；不依赖 M5 真实数据；
* 集成测试用小 T（30 天）+ 少 SKU，保证秒级完成。

运行
----
    pytest tests/test_policy.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.coordination.custom_policy import CoordinatedOrderingPolicy
from src.coordination.signal_protocol import AggregatedSignal, SignalBus
from src.network.topology import (
    build_coordinated_network_from_config,
    build_network_from_config,
)


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------

def _make_signal(name: str, gap: float, gap_std: float = 0.0, t: int = 0,
                 n_sources: int = 1) -> AggregatedSignal:
    """构造一个测试用信号，减少重复。"""
    return AggregatedSignal(node_name=name, timestep=t, gap=gap,
                            gap_std=gap_std, n_sources=n_sources)


def _expected_order_up_to(lead_time: int, service_level: float,
                          ip: float, fm: float, fs: float) -> float:
    """独立重算 OrderUpTo 期望订单量，用于对照。"""
    z = stats.norm.ppf(service_level)
    S = (lead_time + 1) * fm + z * fs * np.sqrt(lead_time + 1)
    return max(0.0, S - ip)


# ---------------------------------------------------------------------------
# TestAggregatedSignal
# ---------------------------------------------------------------------------

class TestAggregatedSignal:
    """聚合信号数据类：字段、默认值。"""

    def test_required_fields_only(self):
        """只传必需字段也能构造，默认 gap_std=0.0、n_sources=1。"""
        s = AggregatedSignal(node_name="store_1", timestep=0, gap=10.0)
        assert s.node_name == "store_1"
        assert s.timestep == 0
        assert s.gap == 10.0
        assert s.gap_std == 0.0
        assert s.n_sources == 1

    def test_all_fields(self):
        """全字段构造保留所有值。"""
        s = _make_signal("city_1", gap=12.5, gap_std=3.3, t=7, n_sources=2)
        assert s.node_name == "city_1"
        assert s.gap == 12.5
        assert s.gap_std == 3.3
        assert s.timestep == 7
        assert s.n_sources == 2


# ---------------------------------------------------------------------------
# TestSignalBus
# ---------------------------------------------------------------------------

class TestSignalBus:
    """共享信号总线：写入/读取/聚合/清空/快照。"""

    def test_write_and_read(self):
        """写入后能读到同一对象。"""
        bus = SignalBus()
        s = _make_signal("store_1", gap=5.0)
        bus.write(s)
        assert bus.read("store_1") is s

    def test_read_missing_returns_none(self):
        """未写入的节点读取返回 None。"""
        bus = SignalBus()
        assert bus.read("nobody") is None

    def test_write_overwrites(self):
        """同一节点再次写入覆盖旧值（总线只保留最新）。"""
        bus = SignalBus()
        bus.write(_make_signal("store_1", gap=5.0, t=0))
        bus.write(_make_signal("store_1", gap=8.0, t=1))
        latest = bus.read("store_1")
        assert latest.gap == 8.0 and latest.timestep == 1

    def test_reset_clears_all(self):
        """reset() 清空所有信号。"""
        bus = SignalBus()
        bus.write(_make_signal("a", gap=1.0))
        bus.write(_make_signal("b", gap=2.0))
        bus.reset()
        assert bus.read("a") is None and bus.read("b") is None
        assert bus.snapshot() == {}

    def test_snapshot_is_copy(self):
        """snapshot 返回浅拷贝——修改返回值不影响总线。"""
        bus = SignalBus()
        bus.write(_make_signal("store_1", gap=5.0))
        snap = bus.snapshot()
        snap["hacked"] = _make_signal("x", gap=999.0)
        assert bus.read("x") is None

    # ---- read_downstream 三种情况 ----

    def test_read_downstream_none_when_empty(self):
        """下游一个信号都没有 -> 返回 None。"""
        bus = SignalBus()
        assert bus.read_downstream(["store_1", "store_2"]) is None

    def test_read_downstream_single_returns_as_is(self):
        """只有 1 个下游有信号 -> 直接返回该信号（不新建对象）。"""
        bus = SignalBus()
        s = _make_signal("store_1", gap=5.0, gap_std=1.5, t=3)
        bus.write(s)
        got = bus.read_downstream(["store_1", "store_2"])
        assert got is s

    def test_read_downstream_aggregates(self):
        """多个下游 -> gap 求和、gap_std 按方差合成、n_sources 累加、timestep 取最大。"""
        bus = SignalBus()
        bus.write(_make_signal("store_1", gap=3.0, gap_std=1.0, t=2, n_sources=1))
        bus.write(_make_signal("store_2", gap=4.0, gap_std=2.0, t=5, n_sources=1))
        agg = bus.read_downstream(["store_1", "store_2"])
        assert agg is not None
        assert agg.gap == pytest.approx(7.0)
        assert agg.gap_std == pytest.approx(np.sqrt(1.0 ** 2 + 2.0 ** 2))
        assert agg.n_sources == 2
        assert agg.timestep == 5
        assert agg.node_name == "AGG(2)"

    def test_read_downstream_skips_missing(self):
        """下游列表里混入未上报的节点 -> 只聚合有的，不报错。"""
        bus = SignalBus()
        bus.write(_make_signal("store_1", gap=3.0))
        agg = bus.read_downstream(["store_1", "store_2", "store_3"])
        # 只有 store_1 有信号 -> 等价于单信号路径
        assert agg is not None and agg.gap == 3.0


# ---------------------------------------------------------------------------
# TestCoordinatedOrderingPolicy
# ---------------------------------------------------------------------------

class TestCoordinatedOrderingPolicy:
    """协同订货策略：退化、补偿、上报、reset。"""

    def test_alpha_zero_matches_order_up_to(self):
        """alpha=0 且无下游 -> compute_order 与手算 OrderUpTo 完全一致。"""
        policy = CoordinatedOrderingPolicy(lead_time=2, service_level=0.95,
                                           alpha=0.0)
        order = policy.compute_order(inventory_position=30.0,
                                     forecast_mean=10.0,
                                     forecast_std=3.0)
        expected = _expected_order_up_to(2, 0.95, 30.0, 10.0, 3.0)
        assert order == pytest.approx(expected)

    def test_negative_inventory_position_increases_order(self):
        """库存头寸为负时，订单应更大（S - IP 中 IP 为负 -> 加）。"""
        policy = CoordinatedOrderingPolicy(lead_time=1, alpha=0.0)
        order_pos = policy.compute_order(20.0, 10.0, 2.0)
        order_neg = policy.compute_order(-20.0, 10.0, 2.0)
        assert order_neg > order_pos
        assert order_neg - order_pos == pytest.approx(40.0)

    def test_no_bus_behaves_like_order_up_to(self):
        """无 signal_bus 时不报错，等价于纯 OrderUpTo。"""
        policy = CoordinatedOrderingPolicy(lead_time=3, alpha=0.5,
                                           signal_bus=None)
        order = policy.compute_order(10.0, 8.0, 2.0)
        expected = _expected_order_up_to(3, 0.95, 10.0, 8.0, 2.0)
        assert order == pytest.approx(expected)

    def test_coordination_adds_alpha_times_downstream_gap(self):
        """有下游信号时，订单 = 基线下单 + alpha * 下游 gap。"""
        bus = SignalBus()
        # 下游上报 20 的缺口
        bus.write(_make_signal("store_1", gap=20.0))
        policy = CoordinatedOrderingPolicy(
            lead_time=2, alpha=0.5,
            signal_bus=bus, node_name="city_1",
            downstream_names=["store_1"],
        )
        order = policy.compute_order(10.0, 10.0, 2.0)
        baseline = _expected_order_up_to(2, 0.95, 10.0, 10.0, 2.0)
        assert order == pytest.approx(baseline + 0.5 * 20.0)

    def test_leaf_node_ignores_downstream_signal(self):
        """叶子节点（downstream_names=[]）即使总线上有信号也不读。"""
        bus = SignalBus()
        bus.write(_make_signal("nobody", gap=999.0))
        policy = CoordinatedOrderingPolicy(
            lead_time=1, alpha=1.0,
            signal_bus=bus, node_name="store_1",
            downstream_names=[],
        )
        order = policy.compute_order(5.0, 10.0, 2.0)
        baseline = _expected_order_up_to(1, 0.95, 5.0, 10.0, 2.0)
        assert order == pytest.approx(baseline)

    def test_writes_own_signal_to_bus(self):
        """compute_order 后，本节点信号写入总线，gap = 基线下单量。"""
        bus = SignalBus()
        policy = CoordinatedOrderingPolicy(
            lead_time=2, alpha=0.3,
            signal_bus=bus, node_name="store_1",
            downstream_names=[],
        )
        policy.compute_order(10.0, 10.0, 2.0)
        sig = bus.read("store_1")
        assert sig is not None
        assert sig.node_name == "store_1"
        assert sig.gap == pytest.approx(
            _expected_order_up_to(2, 0.95, 10.0, 10.0, 2.0))
        assert sig.gap_std == pytest.approx(2.0)
        assert sig.n_sources == 1
        assert sig.timestep == 0

    def test_timestep_increments(self):
        """每次 compute_order 后 timestep 递增。"""
        bus = SignalBus()
        policy = CoordinatedOrderingPolicy(
            lead_time=1, signal_bus=bus, node_name="store_1")
        for _ in range(3):
            policy.compute_order(5.0, 5.0, 1.0)
        assert bus.read("store_1").timestep == 2

    def test_reset_clears_timestep(self):
        """reset 后 timestep 归零。"""
        bus = SignalBus()
        policy = CoordinatedOrderingPolicy(
            lead_time=1, signal_bus=bus, node_name="store_1")
        policy.compute_order(5.0, 5.0, 1.0)
        policy.compute_order(5.0, 5.0, 1.0)
        policy.reset()
        policy.compute_order(5.0, 5.0, 1.0)
        assert bus.read("store_1").timestep == 0

    def test_higher_alpha_increases_order(self):
        """同一场景下 alpha 越大，订单越多（有下游信号时）。"""
        orders = []
        for a in (0.0, 0.5, 1.0):
            bus = SignalBus()
            bus.write(_make_signal("store_1", gap=10.0))
            policy = CoordinatedOrderingPolicy(
                lead_time=1, alpha=a, signal_bus=bus,
                node_name="city_1", downstream_names=["store_1"])
            orders.append(policy.compute_order(20.0, 10.0, 2.0))
        assert orders[0] < orders[1] < orders[2]


# ---------------------------------------------------------------------------
# TestCoordinationIntegration
# ---------------------------------------------------------------------------

class TestCoordinationIntegration:
    """集成：用真实拓扑跑小规模仿真，验证信号跨节点传递。"""

    @pytest.fixture
    def small_demand(self):
        """小规模合成数据：4 门店 × 30 天。"""
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

    def test_coordinated_network_builds(self):
        """build_coordinated_network_from_config 返回 (NetworkSupplyChain, SignalBus)。"""
        net, bus = build_coordinated_network_from_config(
            "config/network_config.yaml", alpha=0.5)
        assert net is not None
        assert isinstance(bus, SignalBus)
        assert len(net.graph.nodes) == 7
        assert len(net.graph.edges) == 6

    def test_signals_reach_upstream(self, small_demand):
        """仿真后，中间层与区域仓都能在 SignalBus 里读到信号。"""
        demand, fm, fs = small_demand
        net, bus = build_coordinated_network_from_config(
            "config/network_config.yaml", alpha=0.3)
        net.simulate(demand, fm, fs)

        snap = bus.snapshot()
        # 至少应有 7 个节点的信号（4 门店 + 2 城市 + 1 区域）
        assert len(snap) == 7, f"应 7 个节点有信号，实际 {list(snap)}"
        # 中间节点的信号必须存在（证明自下而上传上来了）
        for name in ("city_warehouse_1", "city_warehouse_2",
                     "regional_warehouse"):
            assert name in snap, f"缺少 {name} 的信号"

    def test_alpha_zero_matches_baseline(self, small_demand):
        """alpha=0 的协同仿真总成本应与纯 OrderUpTo 基线一致。"""
        demand, fm, fs = small_demand

        net_base = build_network_from_config("config/network_config.yaml")
        from src.network.topology import run_baseline_policy
        res_base = run_baseline_policy(net_base, demand, fm, fs)

        net_coord, _ = build_coordinated_network_from_config(
            "config/network_config.yaml", alpha=0.0)
        res_coord = net_coord.simulate(demand, fm, fs)

        # alpha=0 时协同补偿项恒为 0 -> 与基线完全一致（数值允许极小浮点误差）
        assert res_coord.total_cost == pytest.approx(res_base.total_cost, rel=1e-6)

    def test_coordination_does_not_break_simulation(self, small_demand):
        """alpha=0.5 的仿真能正常跑完，且结果结构完整。"""
        demand, fm, fs = small_demand
        net, _ = build_coordinated_network_from_config(
            "config/network_config.yaml", alpha=0.5)
        result = net.simulate(demand, fm, fs)

        assert result.total_cost > 0
        assert len(result.node_results) == 7
        for name, r in result.node_results.items():
            assert r.orders.shape == (30,)
            assert r.inventory_levels.shape == (30,)
# -*- coding: utf-8 -*-
"""仿真对比脚本：baseline vs 协同策略

职责
----
1. 跑一遍基线（纯 OrderUpTo）；
2. 对每个 alpha 跑一遍协同策略；
3. 汇总三项指标：Bullwhip Ratio / True Fill Rate / Total Cost；
4. 导出 CSV 到 outputs/results/，供 dashboard.py 与 05 demo 消费。

关键口径
----
框架自带 ``fill_rate`` 是"库存非负期数占比"，不是真正的订单满足率。
本模块自行计算 **True Fill Rate**：

    true_fill_rate = 1 - 累计缺货量 / 累计需求量

* 需求：叶子节点用外部 demand；中间节点用"下游订单之和"（框架的口径）；
* 缺货量：节点每期 ``max(0, -inventory_levels)`` 之和。

本模块输出**两个口径**的 true fill rate：
* ``true_fill_rate``：全节点（含区域仓）；
* ``true_fill_rate_excl_top``：排除区域仓。
  区域仓 fill rate 恒低是 deepbullwhip 不建模顶层补货导致的框架限制
  （见 docs/risk_log.md R4），协同策略实际作用的层级是城市仓与门店，
  因此这个口径更能反映协同的真实收益。

对外接口
----
    run_comparison(config_path, demand, forecasts_mean, forecasts_std,
                   alphas, save_dir) -> BenchmarkResult

    BenchmarkResult.summary_df       # 每个策略一行
    BenchmarkResult.node_metrics_df  # 每个 (策略, 节点) 一行
    BenchmarkResult.raw_results      # {策略名: NetworkSimulationResult}
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录（src/simulation/run_benchmark.py 的上两级）——
# 脚本直跑时把项目根加入 sys.path，才能 import src.* 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.network.topology import (
    build_coordinated_network_from_config,
    build_network_from_config,
)

__all__ = ["BenchmarkResult", "run_comparison"]

DEFAULT_SAVE_DIR = PROJECT_ROOT / "outputs" / "results"

# 顶层节点名——算"可优化层 fill rate"时排除它
_TOP_NODE = "regional_warehouse"


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """一次 benchmark 跑批的全部结果。

    Attributes
    ----------
    summary_df : pd.DataFrame
        每个策略一行，列：``policy`` / ``alpha`` / ``total_cost`` /
        ``cumulative_bullwhip`` / ``framework_fill_rate`` / ``true_fill_rate`` /
        ``true_fill_rate_excl_top``。

        * ``true_fill_rate``：全节点口径（含区域仓）；
        * ``true_fill_rate_excl_top``：排除区域仓（协同策略实际作用层）。
          区域仓 fill rate 恒低是 deepbullwhip 不建模顶层补货导致的框架限制，
          见 docs/risk_log.md R4。
    node_metrics_df : pd.DataFrame
        每个 (策略, 节点) 一行，列：``policy`` / ``node`` / ``orders_mean`` /
        ``inv_mean`` / ``bullwhip`` / ``cost`` / ``true_fill_rate``。
    raw_results : dict[str, NetworkSimulationResult]
        ``{策略名: 原始仿真结果}``，供 dashboard.py 画图用。
    """

    summary_df: pd.DataFrame
    node_metrics_df: pd.DataFrame
    raw_results: dict


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _cumulative_shortage(inv) -> float:
    inv = np.asarray(inv, dtype=np.float64)
    backlog = np.maximum(0.0, -inv)
    backlog_prev = np.concatenate([[0.0], backlog[:-1]])
    return float(np.maximum(0.0, backlog - backlog_prev).sum())


def _true_fill_rate(
    result,
    graph,
    demand: dict,
    node_name: str,
) -> float:
    """计算单节点的真 fill rate（1 - 缺货量 / 需求量）。

    Parameters
    ----------
    result : NetworkSimulationResult
    graph : SupplyChainGraph
    demand : dict
        叶子节点的外部需求（中间节点由下游订单之和推导）。
    node_name : str

    Returns
    -------
    float
        [0, 1] 之间的比例；需求全 0 时返回 1.0（无缺货可言）。
    """
    inv = np.asarray(result.node_results[node_name].inventory_levels,
                     dtype=np.float64)
    shortage = _cumulative_shortage(inv)

    downstream = graph.downstream_neighbors(node_name)
    if not downstream:
        # 叶子节点：需求来自外部输入
        node_demand = np.asarray(demand[node_name], dtype=np.float64)
    else:
        # 中间节点：需求 = 下游订单之和（与框架 simulate 内部一致）
        T = len(inv)
        node_demand = np.zeros(T, dtype=np.float64)
        for dn in downstream:
            node_demand += np.asarray(
                result.node_results[dn].orders, dtype=np.float64)

    total_demand = float(node_demand.sum())
    if total_demand <= 0:
        return 1.0
    return max(0.0, 1.0 - shortage / total_demand)


def _evaluate(
    policy_name: str,
    alpha: float | None,
    result,
    graph,
    demand: dict,
) -> tuple[dict, pd.DataFrame, "NetworkSimulationResult"]:
    """从一次仿真结果中提取汇总指标与节点级指标。"""
    # ---- 节点级指标 ----
    rows = []
    framework_frs = []
    for name, r in result.node_results.items():
        true_fr = _true_fill_rate(result, graph, demand, name)
        framework_frs.append(float(r.fill_rate))
        rows.append({
            "policy": policy_name,
            "node": name,
            "orders_mean": float(np.mean(r.orders)),
            "inv_mean": float(np.mean(r.inventory_levels)),
            "bullwhip": float(r.bullwhip_ratio),
            "cost": float(r.total_cost),
            "true_fill_rate": true_fr,
        })
    node_df = pd.DataFrame(rows)

    # ---- 系统级 true fill rate：两个口径 ----
    #   全节点口径：所有节点加总（含 region，受框架顶层限制，恒低）
    #   可优化层口径：排除 regional_warehouse（协同策略实际的作用层）
    total_shortage = 0.0
    total_demand = 0.0
    total_shortage_excl = 0.0
    total_demand_excl = 0.0

    for name in result.node_results:
        inv = np.asarray(result.node_results[name].inventory_levels,
                         dtype=np.float64)
        shortage = _cumulative_shortage(inv)
        total_shortage += shortage

        downstream = graph.downstream_neighbors(name)
        if not downstream:
            node_demand = np.asarray(demand[name], dtype=np.float64)
        else:
            T = len(inv)
            node_demand = np.zeros(T, dtype=np.float64)
            for dn in downstream:
                node_demand += np.asarray(
                    result.node_results[dn].orders, dtype=np.float64)
        node_demand_sum = float(node_demand.sum())
        total_demand += node_demand_sum

        if name != _TOP_NODE:
            total_shortage_excl += shortage
            total_demand_excl += node_demand_sum

    sys_true_fr = (1.0 - total_shortage / total_demand
                   if total_demand > 0 else 1.0)
    sys_true_fr_excl = (1.0 - total_shortage_excl / total_demand_excl
                        if total_demand_excl > 0 else 1.0)

    summary = {
        "policy": policy_name,
        "alpha": alpha if alpha is not None else np.nan,
        "total_cost": float(result.total_cost),
        "cumulative_bullwhip": float(result.cumulative_bullwhip),
        "framework_fill_rate": float(np.mean(framework_frs)),
        "true_fill_rate": max(0.0, sys_true_fr),
        "true_fill_rate_excl_top": max(0.0, sys_true_fr_excl),
    }
    return summary, node_df, result


# ---------------------------------------------------------------------------
# 对外主函数
# ---------------------------------------------------------------------------

def run_comparison(
    config_path: str | Path,
    demand: dict,
    forecasts_mean: dict,
    forecasts_std: dict,
    alphas: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0),
    save_dir: str | Path | None = None,
) -> BenchmarkResult:
    """跑基线 + 多个 alpha 的协同策略，返回汇总与节点级对比。

    Parameters
    ----------
    config_path : str or Path
        ``config/network_config.yaml`` 路径。
    demand, forecasts_mean, forecasts_std : dict
        叶子门店的需求与预测（见 ``build_forecast_bundle``）。
    alphas : tuple[float, ...], default (0.0, 0.5, 0.8, 1.0)
        要扫描的协同强度。``0.0`` 会走协同网络，但因补偿项为 0，
        总成本应与基线一致（可作一致性校验）。
    save_dir : str or Path or None, default None
        CSV 保存目录；None 时用 ``outputs/results/``。
        传 ``False`` 可跳过落盘（仅返回结果，用于测试）。

    Returns
    -------
    BenchmarkResult
        summary_df / node_metrics_df / raw_results。
    """
    summaries = []
    node_dfs = []
    raw = {}

    # ---- 1. 基线 ----
    net_base = build_network_from_config(config_path)
    res_base = net_base.simulate(demand, forecasts_mean, forecasts_std)
    s, nd, r = _evaluate("baseline", None, res_base, net_base.graph, demand)
    summaries.append(s); node_dfs.append(nd)
    raw["baseline"] = r

    # ---- 2. 各 alpha 协同 ----
    for a in alphas:
        name = f"coord(a={a})"
        net_a, _bus = build_coordinated_network_from_config(config_path, alpha=a)
        res_a = net_a.simulate(demand, forecasts_mean, forecasts_std)
        s, nd, r = _evaluate(name, a, res_a, net_a.graph, demand)
        summaries.append(s); node_dfs.append(nd)
        raw[name] = r

    summary_df = pd.DataFrame(summaries)
    node_metrics_df = pd.concat(node_dfs, ignore_index=True)

    # ---- 3. 落盘 ----
    if save_dir is not False:
        out_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / "benchmark_summary.csv",
                          index=False, float_format="%.6f",
                          encoding="utf-8-sig")
        node_metrics_df.to_csv(out_dir / "benchmark_node_metrics.csv",
                               index=False, float_format="%.6f",
                               encoding="utf-8-sig")

    return BenchmarkResult(
        summary_df=summary_df,
        node_metrics_df=node_metrics_df,
        raw_results=raw,
    )


# ---------------------------------------------------------------------------
# 内部自测（python src/simulation/run_benchmark.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """自测：小规模数据跑一遍完整对比，验证结构与一致性。"""
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    print("=" * 68)
    print("  src/simulation/run_benchmark.py 自测")
    print("=" * 68)
    print(f"解释器: {sys.executable}")

    failures = []

    def report(name, ok, detail=""):
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"       {line}")
        if not ok:
            failures.append(name)

    # ---- 步骤 1：准备小规模需求 ----
    try:
        from src.data_loader import load_synthetic_demand, pivot_demand_wide
        from src.forecasting.custom_forecaster import WeightedMAForecaster

        df = load_synthetic_demand(n_nodes=4, n_days=30, seed=42)
        wide = pivot_demand_wide(df)
        store_names = ["store_1", "store_2", "store_3", "store_4"]
        demand, fm, fs = {}, {}, {}
        wma = WeightedMAForecaster(window=5)
        for store, col in zip(store_names, wide.columns[:4]):
            s = wide[col].to_numpy(dtype=np.float64)
            m, sd = wma.generate_forecasts(s)
            demand[store] = s; fm[store] = m; fs[store] = sd
        report("准备小规模数据（4 门店 × 30 天）", True,
               f"demand 覆盖 {len(demand)} 门店，T={len(demand['store_1'])}")
    except Exception:  # noqa: BLE001
        import traceback
        report("准备小规模数据", False, traceback.format_exc())
        return 1

    # ---- 步骤 2：跑 run_comparison（不落盘）----
    try:
        result = run_comparison(
            "config/network_config.yaml",
            demand, fm, fs,
            alphas=(0.0, 0.5),
            save_dir=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.summary_df) == 3          # baseline + 0.0 + 0.5
        assert len(result.node_metrics_df) == 3 * 7  # 3 策略 × 7 节点
        assert "true_fill_rate_excl_top" in result.summary_df.columns
        report("run_comparison 结构", True,
               result.summary_df.round(4).to_string(index=False))
    except Exception:  # noqa: BLE001
        import traceback
        report("run_comparison 结构", False, traceback.format_exc())
        return 1

    # ---- 步骤 3：alpha=0 与基线一致性 ----
    try:
        s = result.summary_df.set_index("policy")
        base_cost = s.loc["baseline", "total_cost"]
        a0_cost = s.loc["coord(a=0.0)", "total_cost"]
        assert abs(base_cost - a0_cost) < 1e-6, (
            f"alpha=0 应等同基线：{base_cost} vs {a0_cost}")
        report("alpha=0 与基线一致性", True,
               f"baseline={base_cost:,.2f}  coord(a=0.0)={a0_cost:,.2f}")
    except Exception:  # noqa: BLE001
        import traceback
        report("alpha=0 与基线一致性", False, traceback.format_exc())

    # ---- 步骤 4：两个 fill rate 口径都在 [0, 1] ----
    try:
        for col in ("true_fill_rate", "true_fill_rate_excl_top"):
            fr = result.summary_df[col].to_numpy()
            assert ((fr >= 0) & (fr <= 1)).all(), f"{col} 越界: {fr}"
        # excl_top 应 >= 全节点口径（去掉了恒低的区域仓）
        assert (result.summary_df["true_fill_rate_excl_top"]
                >= result.summary_df["true_fill_rate"] - 1e-9).all()
        report("fill rate 两口径在 [0, 1] 且 excl_top >= 全节点", True,
               f"全节点: {result.summary_df['true_fill_rate'].round(4).tolist()}\n"
               f"排顶层: {result.summary_df['true_fill_rate_excl_top'].round(4).tolist()}")
    except Exception:  # noqa: BLE001
        import traceback
        report("fill rate 两口径", False, traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：run_comparison 结构与指标口径均无报错。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
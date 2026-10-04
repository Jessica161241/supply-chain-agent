# -*- coding: utf-8 -*-
"""agent/agent_wrapper.py —— 全流程 Agent 封装（A+B 共同文件）

当前状态（2026-10-01）
----
* **A 侧已完成**：需求准备（M5→合成三级降级，封装 src/data_loader.py）+
  预测产出（逐节点整段 (T,) 的 (forecast_mean, forecast_std)，封装
  src/forecasting/custom_forecaster.py，可插拔任意 Forecaster）。
* **B 侧待交付**：src/network/topology.py（3 层 NetworkSupplyChain 构造）与
  src/coordination/*（协同策略）——仿真执行环节以 TODO(B) 占位，
  调用即抛 NotImplementedError，**A 侧不伪造 B 的仿真行为**。

A↔B 接口约定（源自 docs/AB_interface_checklist.md §3.1-3.3）
----
* A → B 需求数据：长表 DataFrame（date/store_id/item_id/demand，团队注意事项 §8
  口径）+ ``ReplayDemandGenerator``（整表聚合 / 逐节点两种构造）。
* A → B 预测交接物：``dict[节点名, (fm, fs)]``——fm/fs 均为 shape (T,) 的
  float64 数组；**fs 允许为 0**（历史不足时的启动瞬态），B 侧不得假设 fs > 0。
* B → A 结果（待 B 交付后对接）：``NetworkSimulationResult`` 的
  ``node_results`` / ``edge_flows`` / ``cumulative_bullwhip`` / ``total_cost``。

对外接口
----
::

    prepare_demand(use_m5=False, ...)               -> DemandBundle
    build_forecast_bundle(demand_df, forecaster=None) -> dict[str, (fm, fs)]
    run_network_simulation(bundle, forecast_map=None) -> NotImplementedError（TODO(B)）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from deepbullwhip.demand.replay import ReplayDemandGenerator
from deepbullwhip.forecast import Forecaster

__all__ = [
    "DemandBundle",
    "prepare_demand",
    "build_forecast_bundle",
    "run_network_simulation",
]

# 项目根目录（agent/agent_wrapper.py 的上一级），与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认需求口径：
#   M5 路径 = CA_1 × 4 个 dept（对齐清单 §1.2 选型）；
#   合成兜底 = 5 节点 × 365 天（对齐 config/network_config.yaml：sku_count=5、
#   simulation_days=365；demand.mean=10 / demand.std=3 由 data_loader 内置实现）
_DEFAULT_STORE = "CA_1"
_DEFAULT_DEPTS = ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
_DEFAULT_N_NODES = 5
_DEFAULT_N_DAYS = 365


# ---------------------------------------------------------------------------
# 数据侧封装（A 侧已完成）
# ---------------------------------------------------------------------------

@dataclass
class DemandBundle:
    """一次仿真所需的数据侧交接物（A 侧产出）。

    Attributes
    ----------
    demand_df : pd.DataFrame
        长表需求（date/store_id/item_id/demand，已清洗：
        NaN/Inf 记 0、负值截断、逐序列日期连续）。
    source : str
        数据来源标签（"M5 ..." / "synthetic AR(1) ..."）。
    replay_gen : ReplayDemandGenerator
        整表聚合（多列按行求和）的回放生成器。
    node_generators : dict[str, ReplayDemandGenerator]
        逐节点回放生成器（每列一个），B 拓扑中给每个终端门店各配一个。
    """

    demand_df: pd.DataFrame
    source: str
    replay_gen: ReplayDemandGenerator
    node_generators: dict[str, ReplayDemandGenerator]

    def summary(self) -> str:
        """返回一行人读摘要（形状、来源、节点数）。"""
        return (f"DemandBundle(source={self.source!r}, "
                f"shape={self.demand_df.shape}, "
                f"nodes={len(self.node_generators)})")


def prepare_demand(use_m5: bool = False,
                   store_ids: str | list[str] = _DEFAULT_STORE,
                   item_ids: list[str] = _DEFAULT_DEPTS,
                   n_nodes: int = _DEFAULT_N_NODES,
                   n_days: int = _DEFAULT_N_DAYS,
                   seed: int = 42) -> DemandBundle:
    """需求准备：M5 真实数据（use_m5=True）→ 合成数据兜底，三级降级封装。

    Parameters
    ----------
    use_m5 : bool, default False
        True 时尝试加载 M5 本地 CSV（data/raw/m5/sales_train_evaluation.csv）；
        失败时打印原因并自动降级为合成数据，不阻塞流程。
    store_ids / item_ids : M5 门店与 dept 级序列 ID（仅 use_m5=True 时生效）。
    n_nodes / n_days / seed : 合成数据参数（仅降级路径生效）。
        默认 n_nodes=5 / n_days=365 对齐 config 基准（sku_count / simulation_days）。

    Returns
    -------
    DemandBundle
        宽表 + 整表/逐节点 ReplayDemandGenerator。
    """
    # A 全权模块：确保项目根在 sys.path（脚本直跑时 CWD 可能不在项目根）
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_loader import (
        build_replay_generator, load_m5_demand, load_synthetic_demand,
    )

    if use_m5:
        try:
            demand_df = load_m5_demand(store_ids, item_ids)
            source = f"M5 {store_ids} x{len(item_ids)}"
            print(f"[agent] [M5] 加载成功：{len(item_ids)} 条 dept 级序列")
        except (FileNotFoundError, ImportError, ValueError) as exc:
            print(f"[agent] [M5] 不可用（{type(exc).__name__}），降级合成："
                  f"{str(exc).splitlines()[0]}")
            demand_df = load_synthetic_demand(n_nodes=n_nodes, n_days=n_days,
                                              seed=seed)
            source = "synthetic AR(1) (M5 fallback)"
    else:
        demand_df = load_synthetic_demand(n_nodes=n_nodes, n_days=n_days, seed=seed)
        source = "synthetic AR(1)"
        print("[agent] [合成] 使用框架 AR(1) 合成需求（seed 可复现）")

    replay_gen = build_replay_generator(demand_df)
    node_generators: dict[str, ReplayDemandGenerator] = {}
    for (store, item), g in demand_df.groupby(["store_id", "item_id"], sort=True):
        # 逐节点：按 (store_id, item_id) 过滤长表后各自构造回放生成器
        node_generators[f"{store}_{item}"] = build_replay_generator(g)
    return DemandBundle(demand_df=demand_df, source=source,
                        replay_gen=replay_gen, node_generators=node_generators)


# ---------------------------------------------------------------------------
# 预测侧封装（A 侧已完成）
# ---------------------------------------------------------------------------

def build_forecast_bundle(demand_df: pd.DataFrame,
                          forecaster: Forecaster | None = None,
                          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """生成 A→B 的预测交接物：逐节点整段 (T,) 的 (forecast_mean, forecast_std)。

    Parameters
    ----------
    demand_df : pd.DataFrame
        宽表需求（来自 :func:`prepare_demand` 或 src.data_loader）。
    forecaster : Forecaster or None, default None
        预测器实例；None 时用 ``WeightedMAForecaster()``。
        可插拔：也可传框架自带基线（Naive/MovingAverage/ExponentialSmoothing）
        或任何 ``Forecaster`` 子类实例。

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        ``{节点名: (fm, fs)}``——两个 shape (T,) 的 float64 数组；
        fs 允许为 0（启动瞬态），B 侧不得假设 fs > 0。

    Raises
    ------
    ValueError
        框架返回形状与输入不一致，或输入含非数值列/整列 NaN 时。
    """
    # A 全权模块：确保项目根在 sys.path（脚本直跑时 CWD 可能不在项目根）
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.baseline_forecasters import _coerce_demand_columns
    from src.forecasting.custom_forecaster import WeightedMAForecaster

    if forecaster is None:
        forecaster = WeightedMAForecaster()

    columns = _coerce_demand_columns(demand_df)
    bundle: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col, series in columns.items():
        fm, fs = forecaster.generate_forecasts(series)
        fm = np.asarray(fm, dtype=np.float64)
        fs = np.asarray(fs, dtype=np.float64)
        if fm.shape != series.shape or fs.shape != series.shape:
            raise ValueError(
                f"预测器返回形状 {fm.shape}/{fs.shape} 与输入 {series.shape} 不一致"
            )
        bundle[col] = (fm, fs)
    return bundle


# ---------------------------------------------------------------------------
# 仿真执行（TODO(B)：等待 src/network/topology.py 交付后接入）
# ---------------------------------------------------------------------------

def run_network_simulation(bundle: DemandBundle,
                           forecast_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                           ):
    """TODO(B)：执行 3 层网络仿真——等待 B 交付后接入，当前故意占位。

    预期接入方式（对齐清单 §3.2/3.3；先按"路线甲"跑通：
    外部预测只注入叶子门店，中间层用框架自推的下游订单均值）::

        from src.network.topology import build_network_from_config   # TODO(B)
        chain = build_network_from_config("config/network_config.yaml")
        #                                        ^ 函数名取自《项目代码结构指南》§2.5
        result = chain.simulate(
            demand={node: bundle.demand_df[node] ...},          # 逐节点需求序列
            forecasts_mean={node: fm},                          # A 的 forecast_map
            forecasts_std={node: fs},
        )
        return result   # NetworkSimulationResult（§3.3 交接给 A 做指标校验）

    Parameters
    ----------
    bundle : DemandBundle
        :func:`prepare_demand` 的产出（数据侧交接物）。
    forecast_map : dict[str, (fm, fs)] or None, default None
        :func:`build_forecast_bundle` 的产出（预测侧交接物）；
        None 时自动生成（默认 WeightedMAForecaster）。

    Raises
    ------
    NotImplementedError
        B 的拓扑/协同模块尚未交付——A 侧不伪造仿真行为。
    """
    if forecast_map is None:
        forecast_map = build_forecast_bundle(bundle.demand_df)
    raise NotImplementedError(
        "run_network_simulation 等待 B 同学交付 src/network/topology.py 与 "
        "src/coordination/* 后接入（预期接口见本方法 docstring 与 "
        "docs/AB_interface_checklist.md §3.3）。"
        "当前 A 侧交接物已就绪：DemandBundle + forecast_map"
        + (f"（{len(forecast_map)} 个节点）" if forecast_map else ""))


# ---------------------------------------------------------------------------
# 内部自测（python agent/agent_wrapper.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """自测 A 侧封装：需求准备 -> 预测交接物 -> B 占位的 NotImplementedError。"""
    # Windows 中文控制台默认 GBK，防 UnicodeEncodeError 导致自测中断
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    print("=" * 68)
    print("  agent/agent_wrapper.py 自测")
    print("=" * 68)
    print(f"解释器: {sys.executable}")

    failures: list[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"       {line}")
        if not ok:
            failures.append(name)

    # ---------- 步骤 1：prepare_demand ----------
    try:
        bundle = prepare_demand()  # 合成数据路径（离线）
        assert isinstance(bundle, DemandBundle)
        assert bundle.demand_df.shape == (_DEFAULT_N_NODES * _DEFAULT_N_DAYS, 4)
        assert list(bundle.demand_df.columns) == ["date", "store_id", "item_id",
                                                  "demand"]
        assert len(bundle.node_generators) == _DEFAULT_N_NODES
        report("prepare_demand()（合成兜底路径）", True,
               bundle.summary() + f"\n{bundle.demand_df.head(2).round(2).to_string()}")
    except Exception:  # noqa: BLE001
        import traceback
        report("prepare_demand()（合成兜底路径）", False, traceback.format_exc())

    # ---------- 步骤 2：build_forecast_bundle（默认 + 可插拔基线） ----------
    try:
        forecast_map = build_forecast_bundle(bundle.demand_df)
        assert set(forecast_map) == set(bundle.node_generators)
        T = bundle.demand_df["date"].nunique()  # 仿真期数（config simulation_days）
        for node, (fm, fs) in forecast_map.items():
            assert fm.shape == (T,) and fs.shape == (T,)
            assert np.isfinite(fm).all() and np.isfinite(fs).all()
            assert (fs >= 0).all()

        # 可插拔：传入框架自带基线预测器
        from deepbullwhip.forecast import NaiveForecaster
        alt = build_forecast_bundle(bundle.demand_df, forecaster=NaiveForecaster())
        assert set(alt) == set(forecast_map)
        report("build_forecast_bundle(demand_df)", True,
               f"{len(forecast_map)} 节点 × (T,)={(T,)}，fm/fs 有限且 fs>=0；"
               f"插拔 NaiveForecaster 亦正常（{len(alt)} 节点）")
    except Exception:  # noqa: BLE001
        import traceback
        report("build_forecast_bundle(demand_df)", False, traceback.format_exc())

    # ---------- 步骤 3：run_network_simulation 占位（TODO(B)） ----------
    try:
        run_network_simulation(bundle, forecast_map)
    except NotImplementedError as exc:
        report("run_network_simulation 占位（TODO(B)）", True,
               f"按设计抛 NotImplementedError（不伪造 B 行为）：\n  {exc}")
    except Exception:  # noqa: BLE001
        import traceback
        report("run_network_simulation 占位（TODO(B)）", False,
               traceback.format_exc())
    else:
        report("run_network_simulation 占位（TODO(B)）", False,
               "应抛 NotImplementedError，却正常返回了——占位失效！")

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：A 侧封装就绪，B 侧接入点已按 TODO 占位。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

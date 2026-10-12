# -*- coding: utf-8 -*-
"""agent/agent_wrapper.py —— 全流程 Agent 封装

职责
----
把"需求准备 → 预测 → 网络仿真"串成三个可独立调用的函数，外部无需了解内部细节：

* :func:`prepare_demand`：需求准备（M5 → 合成三级降级，封装 src/data_loader.py）；
* :func:`build_forecast_bundle`：预测产出（逐节点整段 (T,) 的
  ``(forecast_mean, forecast_std)``，封装 src/forecasting/custom_forecaster.py，
  可插拔任意 ``Forecaster``）；
* :func:`run_network_simulation`：网络仿真执行，封装 src/network/topology.py
  与 src/coordination/*，支持纯 OrderUpTo 基线与协同策略
  （OrderUpTo + 下游聚合缺口信号）两种模式。

模块间交接约定（详见 docs/AB_interface_checklist.md §3.1-3.3）
----
* 需求数据：长表 DataFrame（date/store_id/item_id/demand）
  + ``ReplayDemandGenerator``（整表聚合 / 逐节点两种构造）。
* 预测交接物：``dict[节点名, (fm, fs)]``——fm/fs 均为 shape (T,) 的
  float64 数组；**fs 允许为 0**（历史不足时的启动瞬态），消费侧不得假设 fs > 0。
* 仿真结果：``NetworkSimulationResult`` 的 ``node_results`` / ``edge_flows``
  / ``cumulative_bullwhip`` / ``total_cost``。

对外接口
----
::

    prepare_demand(use_m5=False, ...)                  -> DemandBundle
    build_forecast_bundle(demand_df, forecaster=None)  -> dict[str, (fm, fs)]
    run_network_simulation(bundle, forecast_map=None,
                           alpha=0.8, coordinated=True) -> NetworkSimulationResult
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
#   合成兜底 = 4 节点 × 365 天（对齐 config/network_config.yaml：
#   sku_count=4、simulation_days=365，正好一一映射到 store_1..store_4；
#   demand.mean=10 / demand.std=3 由 data_loader 内置实现）
_DEFAULT_STORE = "CA_1"
_DEFAULT_DEPTS = ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
_DEFAULT_N_NODES = 4
_DEFAULT_N_DAYS = 365

# 拓扑叶子门店名（与 config/network_config.yaml 一致）
_STORE_NAMES = ["store_1", "store_2", "store_3", "store_4"]

# 协同强度默认值——实测 alpha 扫描结论：
#   相变临界落在 alpha = 1.0 ~ 1.05 之间（1.0 时总成本 16,040；
#   1.05 骤升至 35,558）。取 0.8 距临界保留约 20~24% 安全边际，
#   成本改善已接近最优。
_DEFAULT_ALPHA = 0.8


# ---------------------------------------------------------------------------
# 数据侧封装
# ---------------------------------------------------------------------------

@dataclass
class DemandBundle:
    """一次仿真所需的数据侧交接物。

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
        默认 n_nodes=4 / n_days=365 对齐 config 基准（sku_count / simulation_days），
        4 条序列一一映射到 store_1..store_4。

    Returns
    -------
    DemandBundle
        长表 + 整表/逐节点 ReplayDemandGenerator。
    """
    # 确保项目根在 sys.path（脚本直跑时 CWD 可能不在项目根）
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
# 预测侧封装
# ---------------------------------------------------------------------------

def build_forecast_bundle(demand_df: pd.DataFrame,
                          forecaster: Forecaster | None = None,
                          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """生成供仿真使用的预测交接物：逐节点整段 (T,) 的 (forecast_mean, forecast_std)。

    Parameters
    ----------
    demand_df : pd.DataFrame
        长表需求（来自 :func:`prepare_demand` 或 src.data_loader）。
    forecaster : Forecaster or None, default None
        预测器实例；None 时用 ``WeightedMAForecaster()``。
        可插拔：也可传框架自带基线（Naive/MovingAverage/ExponentialSmoothing）
        或任何 ``Forecaster`` 子类实例。

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        ``{节点名: (fm, fs)}``——两个 shape (T,) 的 float64 数组；
        fs 允许为 0（启动瞬态），消费侧不得假设 fs > 0。

    Raises
    ------
    ValueError
        框架返回形状与输入不一致，或输入含非数值列/整列 NaN 时。
    """
    # 确保项目根在 sys.path（脚本直跑时 CWD 可能不在项目根）
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
# 仿真执行
# ---------------------------------------------------------------------------

def run_network_simulation(
    bundle: DemandBundle,
    forecast_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    alpha: float = _DEFAULT_ALPHA,
    coordinated: bool = True,
    config_path: str | Path | None = None,
):
    """执行完整网络仿真，返回 ``NetworkSimulationResult``。

    Parameters
    ----------
    bundle : DemandBundle
        :func:`prepare_demand` 的产出（数据侧交接物）。
    forecast_map : dict[str, (fm, fs)] or None, default None
        :func:`build_forecast_bundle` 的产出（预测侧交接物）；
        None 时自动生成（默认 WeightedMAForecaster）。
        key 既支持 A 宽表列名（如 "SYN_SKU_1"），也支持拓扑节点名（如 "store_1"）。
    alpha : float, default 0.8
        协同强度系数。默认值来自实测 alpha 扫描：
        相变临界在 1.0~1.05 之间，取 0.8 保留约 20~24% 安全边际。
    coordinated : bool, default True
        True  → 协同网络（OrderUpTo + 下游聚合缺口信号）；
        False → 纯 OrderUpTo 基线（对照实验用）。
    config_path : str or Path or None, default None
        拓扑配置文件路径；None 时用
        ``<project_root>/config/network_config.yaml``。

    Returns
    -------
    NetworkSimulationResult
        含 ``node_results`` / ``edge_flows`` / ``cumulative_bullwhip`` / ``total_cost``。

    Raises
    ------
    KeyError
        ``forecast_map`` 缺少必需的序列列或节点名。
    FileNotFoundError
        配置文件不存在。
    """
    # 确保项目根在 sys.path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_loader import pivot_demand_wide
    from src.network.topology import (
        build_coordinated_network_from_config,
        build_network_from_config,
    )

    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "network_config.yaml"
    config_path = Path(config_path)

    # ---- 1. 从长表提取叶子门店需求：前 4 条序列依次映射到 store_1..4 ----
    wide = pivot_demand_wide(bundle.demand_df)
    series_cols = list(wide.columns)[:len(_STORE_NAMES)]
    col_to_store = dict(zip(series_cols, _STORE_NAMES))

    demand = {
        store: wide[col].to_numpy(dtype=np.float64)
        for col, store in col_to_store.items()
    }

    # ---- 2. 处理预测：支持 A 宽表列名 或 拓扑节点名 两种 key ----
    if forecast_map is None:
        from src.forecasting.custom_forecaster import WeightedMAForecaster
        benchmark_forecaster = WeightedMAForecaster(
            window=8, weights=[1, 1, 1, 1, 2, 2, 3, 4])
        forecast_map = build_forecast_bundle(
            bundle.demand_df, forecaster=benchmark_forecaster)

    fm: dict[str, np.ndarray] = {}
    fs: dict[str, np.ndarray] = {}
    for col, store in col_to_store.items():
        if col in forecast_map:
            fm[store], fs[store] = forecast_map[col]
        elif store in forecast_map:
            fm[store], fs[store] = forecast_map[store]
        else:
            raise KeyError(
                f"forecast_map 缺少序列列 {col!r} 或拓扑节点 {store!r}；"
                f"现有 key: {list(forecast_map)}"
            )

    # ---- 3. 构建网络（协同 or 基线）----
    if coordinated:
        net, _bus = build_coordinated_network_from_config(
            config_path, alpha=alpha)
    else:
        net = build_network_from_config(config_path)

    # ---- 4. 执行仿真 ----
    return net.simulate(demand, fm, fs)


# ---------------------------------------------------------------------------
# 内部自测（python agent/agent_wrapper.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """自测 A+B 封装：需求准备 → 预测交接物 → 真实仿真（协同 + 基线）。"""
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
        preview = bundle.demand_df.head(2).copy()
        preview["demand"] = preview["demand"].round(2)
        report("prepare_demand()（合成兜底路径）", True,
               bundle.summary() + f"\n{preview.to_string()}")
    except Exception:  # noqa: BLE001
        import traceback
        report("prepare_demand()（合成兜底路径）", False, traceback.format_exc())

    # ---------- 步骤 2：build_forecast_bundle（默认 + 可插拔基线） ----------
    try:
        # 与 05_final_demo / benchmark 保持同一预测口径（window=8 加权），
        # 避免同一方案在不同入口下数字不一致
        from src.forecasting.custom_forecaster import WeightedMAForecaster
        benchmark_forecaster = WeightedMAForecaster(
            window=8, weights=[1, 1, 1, 1, 2, 2, 3, 4])
        forecast_map = build_forecast_bundle(
            bundle.demand_df, forecaster=benchmark_forecaster)
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

    # ---------- 步骤 3：run_network_simulation（协同 + 基线两模式） ----------
    try:
        result_coord = run_network_simulation(bundle, forecast_map,
                                              alpha=_DEFAULT_ALPHA,
                                              coordinated=True)
        assert result_coord is not None
        assert result_coord.total_cost > 0
        assert np.isfinite(result_coord.cumulative_bullwhip)
        assert len(result_coord.node_results) == 7
        report(f"run_network_simulation（协同 alpha={_DEFAULT_ALPHA}）", True,
               f"total_cost={result_coord.total_cost:,.2f}  "
               f"bullwhip={result_coord.cumulative_bullwhip:.4f}  "
               f"nodes={len(result_coord.node_results)}")
    except Exception:  # noqa: BLE001
        import traceback
        report("run_network_simulation（协同）", False, traceback.format_exc())

    try:
        result_base = run_network_simulation(bundle, forecast_map,
                                             coordinated=False)
        assert result_base is not None
        assert result_base.total_cost > 0
        report("run_network_simulation（基线 OrderUpTo）", True,
               f"total_cost={result_base.total_cost:,.2f}  "
               f"bullwhip={result_base.cumulative_bullwhip:.4f}")
    except Exception:  # noqa: BLE001
        import traceback
        report("run_network_simulation（基线）", False, traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：封装链路全部就绪。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
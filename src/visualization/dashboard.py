# -*- coding: utf-8 -*-
"""可视化绘图函数

职责
----
* **预测层**：需求序列总览、真实 vs 各预测器均值、预测 std 不确定性区间；
* **仿真层**：benchmark 对比仪表盘、牛鞭放大图、节点成本对比。
* 设计为**可复用库函数**：notebook、``run_benchmark``、``agent_wrapper``
  均可直接调用；函数只绘制并返回 Figure，**不调用 plt.show()**
  （展示由调用方决定）；``save_path`` 非 None 时自动创建父目录并落盘。

对外接口
----
::

    # 预测层
    plot_demand_overview(demand_df, data_source="", save_path=None) -> Figure
    plot_forecast_mean(demand_df, col, forecast_map, data_source="", save_path=None) -> Figure
    plot_uncertainty_band(demand_df, col, forecast_mean, forecast_std,
                          std_curves=None, sigmas=(1, 2),
                          data_source="", save_path=None) -> Figure

    # 仿真对比层
    plot_summary_dashboard(benchmark_result, data_source="", save_path=None) -> Figure
    plot_bullwhip_amplification(benchmark_result, data_source="", save_path=None) -> Figure
    plot_node_cost_comparison(benchmark_result, data_source="", save_path=None) -> Figure

绘图约定
----
* 图内文字一律英文（matplotlib 默认字体无中文字形，避免缺字形告警）；
  中文说明放在调用方的 print / 文档里。
* 预测层 ``forecast_map`` / ``std_curves`` 为"名称 -> 数组"的有序映射，
  遍历顺序即图例顺序；键名 ``"WeightedMA"`` 的曲线以红色高亮。
* 仿真对比层接受 ``BenchmarkResult``（``run_benchmark.run_comparison`` 的返回值），
  画 baseline vs 各 alpha 协同策略的对比。
* 默认落盘目录 ``FIGURES_DIR = <项目根>/outputs/figures/``。

依赖
----
仅 numpy / pandas / matplotlib。仿真对比函数额外需要 ``run_benchmark.BenchmarkResult``
作为输入（仅类型契约，不 import，保持运行时零耦合）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "FIGURES_DIR",
    # 预测层
    "plot_demand_overview",
    "plot_forecast_mean",
    "plot_uncertainty_band",
    # 仿真对比层
    "plot_summary_dashboard",
    "plot_bullwhip_amplification",
    "plot_node_cost_comparison",
]

# 项目根目录（src/visualization/dashboard.py 的上两级），与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认图片目录（与 notebook 02、任务约定一致）
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_path: str | Path | None) -> Path | None:
    """按需落盘：save_path 非 None 时创建父目录并保存，返回实际路径。"""
    if save_path is None:
        return None
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _title_with_source(base: str, data_source: str) -> str:
    """拼装标题：data_source 非空时以 [tag] 形式追加（标签用英文安全写法）。"""
    return f"{base} [{data_source}]" if data_source else base


def _ensure_wide(demand_df: pd.DataFrame) -> pd.DataFrame:
    """长表输入（date/store_id/item_id/demand）自动转宽表；宽表原样返回。"""
    cols = set(demand_df.columns)
    if {"date", "demand"} <= cols and ({"store_id", "item_id"} & cols):
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.data_loader import pivot_demand_wide
        return pivot_demand_wide(demand_df)
    return demand_df


# ---------------------------------------------------------------------------
# 预测相关绘图
# ---------------------------------------------------------------------------

def plot_demand_overview(demand_df: pd.DataFrame,
                         data_source: str = "",
                         save_path: str | Path | None = None) -> plt.Figure:
    """绘制多条需求序列的总览曲线（一列一条线）。

    Parameters
    ----------
    demand_df : pd.DataFrame
        长表（date/store_id/item_id/demand，自动 pivot）或宽表需求。
    data_source : str, default ""
        数据来源英文标签，用于标题。
    save_path : str or Path or None, default None
        保存路径；None 只绘制不落盘。

    Returns
    -------
    matplotlib.figure.Figure
    """
    demand_df = _ensure_wide(demand_df)
    fig, ax = plt.subplots(figsize=(11, 4))
    for col in demand_df.columns:
        ax.plot(demand_df.index, demand_df[col], lw=1.0, label=str(col))
    ax.set_xlabel("period")
    ax.set_ylabel("demand")
    ax.set_title(_title_with_source("Demand series overview", data_source))
    ax.legend(fontsize=8, ncol=min(len(demand_df.columns), 4))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_forecast_mean(demand_df: pd.DataFrame,
                       col: str,
                       forecast_map: dict[str, tuple[np.ndarray, np.ndarray]],
                       data_source: str = "",
                       save_path: str | Path | None = None) -> plt.Figure:
    """绘制真实需求 vs 各预测器均值曲线。

    Parameters
    ----------
    demand_df : pd.DataFrame
        长表（自动 pivot）或宽表需求。
    col : str
        要绘制的序列列名（宽表列）。
    forecast_map : dict[str, tuple[np.ndarray, np.ndarray]]
        ``{预测器名称: (fm, fs)}``；键名 ``"WeightedMA"`` 红色高亮。
    data_source : str, default ""
    save_path : str or Path or None, default None

    Returns
    -------
    matplotlib.figure.Figure
    """
    demand_df = _ensure_wide(demand_df)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(demand_df.index, demand_df[col], color="black", lw=1.2, label="Actual")
    for name, (fm, _fs) in forecast_map.items():
        if name == "WeightedMA":
            ax.plot(demand_df.index, fm, color="red", lw=1.5, label="WeightedMA mean")
        else:
            ax.plot(demand_df.index, fm, lw=0.9, alpha=0.85, label=f"{name} mean")
    ax.set_xlabel("period")
    ax.set_ylabel("demand")
    ax.set_title(_title_with_source(f"Actual vs forecast mean - {col}", data_source))
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_uncertainty_band(demand_df: pd.DataFrame,
                          col: str,
                          forecast_mean: np.ndarray,
                          forecast_std: np.ndarray,
                          std_curves: dict[str, np.ndarray] | None = None,
                          sigmas: tuple[float, ...] = (1, 2),
                          data_source: str = "",
                          save_path: str | Path | None = None) -> plt.Figure:
    """绘制预测不确定性：均值 ±kσ 阴影带（上图）+ std(t) 曲线对比（下图）。

    Parameters
    ----------
    demand_df : pd.DataFrame
    col : str
    forecast_mean, forecast_std : np.ndarray
        shape (T,) 的预测均值 / 标准差。
    std_curves : dict[str, np.ndarray] or None
        额外对比的 std 曲线（``{名称: fs 数组}``）。
    sigmas : tuple[float, ...], default (1, 2)
    data_source : str, default ""
    save_path : str or Path or None, default None

    Returns
    -------
    matplotlib.figure.Figure
    """
    demand_df = _ensure_wide(demand_df)
    std_curves = std_curves or {}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   height_ratios=[2, 1])
    ax1.plot(demand_df.index, demand_df[col], color="black", lw=1.0, label="Actual")
    ax1.plot(demand_df.index, forecast_mean, color="red", lw=1.4,
             label="Forecast mean")
    for k in sigmas:
        alpha = 0.25 / k if k else 0.25
        ax1.fill_between(demand_df.index, forecast_mean - k * forecast_std,
                         forecast_mean + k * forecast_std, color="red",
                         alpha=alpha, label=f"±{k:g}σ")
    ax1.set_ylabel("demand")
    ax1.set_title(_title_with_source(f"Forecast uncertainty band - {col}", data_source))
    ax1.legend(ncol=3, fontsize=8)

    ax2.plot(demand_df.index, forecast_std, color="red", lw=1.2,
             label="forecast std")
    for name, fs in std_curves.items():
        ax2.plot(demand_df.index, fs, lw=1.0, alpha=0.8, label=f"{name} std")
    ax2.set_xlabel("period")
    ax2.set_ylabel("forecast std")
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 仿真对比绘图
# ---------------------------------------------------------------------------

def plot_summary_dashboard(benchmark_result,
                           data_source: str = "",
                           save_path: str | Path | None = None) -> plt.Figure:
    """绘制 benchmark 对比仪表盘：三子图（Cost / Bullwhip / Fill Rate）。

    Parameters
    ----------
    benchmark_result : BenchmarkResult
        ``run_benchmark.run_comparison()`` 的返回值，含 ``summary_df``。
    data_source : str, default ""
    save_path : str or Path or None, default None

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = benchmark_result.summary_df
    policies = df["policy"].tolist()
    colors = ["#7f7f7f"] + ["#1f77b4", "#ff7f0e", "#2ca02c",
                            "#d62728", "#9467bd"][:len(policies) - 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---- 子图 1：Total Cost ----
    ax = axes[0]
    ax.bar(policies, df["total_cost"], color=colors)
    ax.set_ylabel("Total cost")
    ax.set_title("Total Cost (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(df["total_cost"]):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    # ---- 子图 2：Cumulative Bullwhip ----
    ax = axes[1]
    ax.bar(policies, df["cumulative_bullwhip"], color=colors)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=1)
    ax.set_ylabel("Cumulative bullwhip")
    ax.set_title("Bullwhip Ratio (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(df["cumulative_bullwhip"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # ---- 子图 3：True Fill Rate（排除顶层）----
    ax = axes[2]
    fr = df["true_fill_rate_excl_top"] * 100.0
    ax.bar(policies, fr, color=colors)
    ax.set_ylabel("Fill rate (%)")
    ax.set_title("True Fill Rate (excl. top, higher is better)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(fr):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle(_title_with_source("Benchmark Summary", data_source))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_bullwhip_amplification(benchmark_result,
                                data_source: str = "",
                                save_path: str | Path | None = None) -> plt.Figure:
    """绘制各节点的牛鞭系数对比（baseline vs 各 alpha 协同策略）。

    横轴为节点，纵轴为牛鞭系数，不同策略分组柱状图。
    红色虚线 y=1 为"无放大"参考线。

    Parameters
    ----------
    benchmark_result : BenchmarkResult
    data_source : str, default ""
    save_path : str or Path or None, default None

    Returns
    -------
    matplotlib.figure.Figure
    """
    node_df = benchmark_result.node_metrics_df
    policies = node_df["policy"].unique().tolist()
    nodes = node_df["node"].unique().tolist()

    n_pol = len(policies)
    n_node = len(nodes)
    width = 0.8 / n_pol
    x = np.arange(n_node)

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, policy in enumerate(policies):
        sub = node_df[node_df["policy"] == policy].set_index("node")
        vals = [float(sub.loc[n, "bullwhip"]) if n in sub.index else 0.0
                for n in nodes]
        ax.bar(x + i * width, vals, width, label=policy,
               color=colors[i % len(colors)])

    ax.axhline(y=1.0, color="red", linestyle="--", lw=1,
               label="No amplification (y=1)")
    ax.set_xticks(x + width * (n_pol - 1) / 2)
    ax.set_xticklabels(nodes, rotation=20)
    ax.set_ylabel("Bullwhip ratio")
    ax.set_title(_title_with_source("Bullwhip amplification by node", data_source))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_node_cost_comparison(benchmark_result,
                              data_source: str = "",
                              save_path: str | Path | None = None) -> plt.Figure:
    """绘制各节点的成本对比（baseline vs 各 alpha 协同策略）。

    Parameters
    ----------
    benchmark_result : BenchmarkResult
    data_source : str, default ""
    save_path : str or Path or None, default None

    Returns
    -------
    matplotlib.figure.Figure
    """
    node_df = benchmark_result.node_metrics_df
    policies = node_df["policy"].unique().tolist()
    nodes = node_df["node"].unique().tolist()

    n_pol = len(policies)
    width = 0.8 / n_pol
    x = np.arange(len(nodes))

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, policy in enumerate(policies):
        sub = node_df[node_df["policy"] == policy].set_index("node")
        vals = [float(sub.loc[n, "cost"]) if n in sub.index else 0.0
                for n in nodes]
        ax.bar(x + i * width, vals, width, label=policy,
               color=colors[i % len(colors)])

    ax.set_xticks(x + width * (n_pol - 1) / 2)
    ax.set_xticklabels(nodes, rotation=20)
    ax.set_ylabel("Total cost")
    ax.set_title(_title_with_source("Cost breakdown by node", data_source))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 内部自测（python src/visualization/dashboard.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """自测六个绘图函数：返回 Figure、按路径落盘、临时文件测后清理。"""
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    print("=" * 68)
    print("  src/visualization/dashboard.py 自测")
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

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data_loader import load_synthetic_demand, pivot_demand_wide
        from src.forecasting.baseline_forecasters import generate_baseline_forecasts
        from src.forecasting.custom_forecaster import WeightedMAForecaster

        demand_df = load_synthetic_demand()
        wide_df = pivot_demand_wide(demand_df)
        col = wide_df.columns[0]
        wma = WeightedMAForecaster(window=8, weights=[1, 1, 1, 1, 2, 2, 3, 4])
        baseline_fc = generate_baseline_forecasts(wide_df[[col]])
        custom_fm, custom_fs = wma.generate_forecasts(
            wide_df[col].to_numpy(dtype=np.float64))

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # ---- 预测层自测 ----
        paths = [FIGURES_DIR / f"test_dashboard_{n}.png"
                 for n in ("overview", "mean", "band")]
        try:
            fig1 = plot_demand_overview(demand_df, data_source="selftest",
                                        save_path=paths[0])
            fmap = dict(baseline_fc[col])
            fmap["WeightedMA"] = (custom_fm, custom_fs)
            fig2 = plot_forecast_mean(demand_df, col, fmap,
                                      data_source="selftest", save_path=paths[1])
            fig3 = plot_uncertainty_band(
                demand_df, col, custom_fm, custom_fs,
                std_curves={"ExponentialSmoothing":
                            baseline_fc[col]["ExponentialSmoothing"][1]},
                data_source="selftest", save_path=paths[2])
            assert fig1 is not None and fig2 is not None and fig3 is not None
            for p in paths:
                assert p.exists() and p.stat().st_size > 0, f"未落盘: {p}"
            report("预测层三个绘图函数（overview / mean / band）", True,
                   "均返回 Figure 并按 save_path 落盘：\n  "
                   + "\n  ".join(str(p) for p in paths))
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)
            plt.close("all")

        # ---- 仿真对比层自测（需要跑一次 benchmark）----
        from src.simulation.run_benchmark import run_comparison
        store_names = ["store_1", "store_2", "store_3", "store_4"]
        dem, fm, fs = {}, {}, {}
        for store, c in zip(store_names, wide_df.columns[:4]):
            s = wide_df[c].to_numpy(dtype=np.float64)
            dem[store] = s
            fm[store] = np.full_like(s, s.mean())
            fs[store] = np.full_like(s, s.std())
        bench = run_comparison("config/network_config.yaml",
                               dem, fm, fs,
                               alphas=(0.0, 0.8),
                               save_dir=False)
        b_paths = [FIGURES_DIR / f"test_b_{n}.png"
                   for n in ("summary", "bullwhip", "cost")]
        try:
            fb1 = plot_summary_dashboard(bench, data_source="selftest",
                                          save_path=b_paths[0])
            fb2 = plot_bullwhip_amplification(bench, data_source="selftest",
                                              save_path=b_paths[1])
            fb3 = plot_node_cost_comparison(bench, data_source="selftest",
                                            save_path=b_paths[2])
            assert fb1 is not None and fb2 is not None and fb3 is not None
            for p in b_paths:
                assert p.exists() and p.stat().st_size > 0, f"未落盘: {p}"
            report("仿真对比层三个绘图函数（summary / bullwhip / cost）", True,
                   "均返回 Figure 并按 save_path 落盘：\n  "
                   + "\n  ".join(str(p) for p in b_paths))
        finally:
            for p in b_paths:
                Path(p).unlink(missing_ok=True)
            plt.close("all")

    except Exception:  # noqa: BLE001
        import traceback
        report("dashboard 自测", False, traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：六个绘图函数返回 Figure、落盘均无问题。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
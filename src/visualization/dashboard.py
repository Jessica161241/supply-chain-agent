# -*- coding: utf-8 -*-
"""A 同学预测相关可视化绘图函数（src/visualization/dashboard.py 的预测层）

职责边界
----
* 只做**预测相关**图表：需求序列总览、真实 vs 各预测器均值、预测 std 不确定性区间；
  拓扑图 / 成本曲线 / 协同对比等 B 侧图表不在此实现
  （分工硬边界：A 不写 src/network、src/coordination 相关代码）。
* 设计为**可复用库函数**：notebook（02）、src/simulation/run_benchmark、
  agent/agent_wrapper 均可直接调用；函数只绘制并返回 Figure，**不调用 plt.show()**
  （展示由调用方决定）；``save_path`` 非 None 时自动创建父目录并落盘。

对外接口
----
::

    plot_demand_overview(demand_df, data_source="", save_path=None) -> Figure
    plot_forecast_mean(demand_df, col, forecast_map, data_source="", save_path=None) -> Figure
    plot_uncertainty_band(demand_df, col, forecast_mean, forecast_std,
                          std_curves=None, sigmas=(1, 2),
                          data_source="", save_path=None) -> Figure

绘图约定
----
* 图内文字一律英文（matplotlib 默认字体无中文字形，避免缺字形告警）；
  中文说明放在调用方的 print / 文档里。
* ``forecast_map`` / ``std_curves`` 为"名称 -> 数组"的有序映射，遍历顺序即图例顺序；
  键名 ``"WeightedMA"`` 的曲线以红色高亮（A 自定义预测器）。
* 默认落盘目录 ``FIGURES_DIR = <项目根>/outputs/figures/``，与 notebook 02 一致。

依赖
----
仅 numpy / pandas / matplotlib，零 B 模块（src/network、src/coordination）依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "FIGURES_DIR",
    "plot_demand_overview",
    "plot_forecast_mean",
    "plot_uncertainty_band",
]

# 项目根目录（src/visualization/dashboard.py 的上两级），与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认图片目录（与 notebook 02、任务约定一致）
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_path: str | Path | None) -> Path | None:
    """按需落盘：save_path 非 None 时创建父目录并保存，返回实际路径。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        待保存的图。
    save_path : str or Path or None
        保存路径；None 表示不落盘。

    Returns
    -------
    Path or None
        实际写入的路径；未落盘时为 None。
    """
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
    """长表输入（date/store_id/item_id/demand）自动转宽表；宽表原样返回。

    Parameters
    ----------
    demand_df : pd.DataFrame
        长表或宽表需求。

    Returns
    -------
    pd.DataFrame
        宽表（index = date，columns = "{store_id}_{item_id}"）。
    """
    cols = set(demand_df.columns)
    if {"date", "demand"} <= cols and ({"store_id", "item_id"} & cols):
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.data_loader import pivot_demand_wide
        return pivot_demand_wide(demand_df)
    return demand_df


# ---------------------------------------------------------------------------
# 对外绘图函数
# ---------------------------------------------------------------------------

def plot_demand_overview(demand_df: pd.DataFrame,
                         data_source: str = "",
                         save_path: str | Path | None = None) -> plt.Figure:
    """绘制多条需求序列的总览曲线（一列一条线）。

    Parameters
    ----------
    demand_df : pd.DataFrame
        长表（date/store_id/item_id/demand，自动 pivot）或宽表需求，
        来自 src.data_loader。
    data_source : str, default ""
        数据来源英文标签（如 "M5 CA_1 x4" / "synthetic AR(1)"），用于标题。
    save_path : str or Path or None, default None
        保存路径；None 只绘制不落盘。

    Returns
    -------
    matplotlib.figure.Figure
        绘制好的图（不关闭，由调用方决定 show / close）。
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
        要绘制的序列列名（宽表列，形如 "CA_1_FOODS_1" / "SYN_SKU_1"）。
    forecast_map : dict[str, tuple[np.ndarray, np.ndarray]]
        ``{预测器名称: (fm, fs)}``——fm 为 shape (T,) 的预测均值数组
        （来自 ``generate_forecasts``）；fs 仅占位不绘制。
        键名 "WeightedMA" 红色高亮，其余为半透明细线。
    data_source : str, default ""
        数据来源英文标签，用于标题。
    save_path : str or Path or None, default None
        保存路径；None 只绘制不落盘。

    Returns
    -------
    matplotlib.figure.Figure
        绘制好的图。
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
        长表（自动 pivot）或宽表需求（用于取真实值与 date 轴）。
    col : str
        序列列名（宽表列）。
    forecast_mean : np.ndarray
        shape (T,) 的预测均值（如 WeightedMA 的 ``generate_forecasts`` 输出）。
    forecast_std : np.ndarray
        shape (T,) 的预测标准差（A↔B 契约：残差口径，允许为 0）。
    std_curves : dict[str, np.ndarray] or None, default None
        额外对比的 std 曲线（``{名称: fs 数组}``，如
        ``{"ExponentialSmoothing": fs}``）。
    sigmas : tuple[float, ...], default (1, 2)
        阴影带倍数，从内到外透明度递减。
    data_source : str, default ""
        数据来源英文标签，用于标题。
    save_path : str or Path or None, default None
        保存路径；None 只绘制不落盘。

    Returns
    -------
    matplotlib.figure.Figure
        绘制好的双子图。
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
# 内部自测（python src/visualization/dashboard.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """自测三个绘图函数：返回 Figure、按路径落盘、临时文件测后清理。

    使用合成数据（data_loader 兜底路径），M5 缺失不影响本自测。
    返回进程退出码：0 全部通过，1 有失败项。
    """
    # Windows 中文控制台默认 GBK，防 UnicodeEncodeError 导致自测中断
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

        demand_df = load_synthetic_demand()      # 长表（config 对齐默认 5×365）
        wide_df = pivot_demand_wide(demand_df)   # 逐序列断言用的宽表
        col = wide_df.columns[0]
        wma = WeightedMAForecaster(window=8, weights=[1, 1, 1, 1, 2, 2, 3, 4])
        baseline_fc = generate_baseline_forecasts(wide_df[[col]])
        custom_fm, custom_fs = wma.generate_forecasts(
            wide_df[col].to_numpy(dtype=np.float64))

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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
            report("三个绘图函数（overview / mean / band）", True,
                   f"均返回 Figure 并按 save_path 落盘：\n  "
                   + "\n  ".join(str(p) for p in paths))
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)
            plt.close("all")

        # 边界：import 依赖不含 B 的模块（AST 扫描，注释/文档提及不算依赖）
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(sys.modules[__name__]))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        bad = [m for m in imported
               if m.startswith("src.network") or m.startswith("src.coordination")]
        assert not bad, f"dashboard 依赖了 B 的模块: {bad}"
        report("边界检查：零 B 模块依赖", True,
               f"AST 扫描全部 import（共 {len(imported)} 个模块），"
               "无 src.network / src.coordination")
    except Exception:  # noqa: BLE001
        import traceback
        report("dashboard 自测", False, traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：三个绘图函数返回 Figure、落盘、边界均无问题。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

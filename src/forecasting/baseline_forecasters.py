# -*- coding: utf-8 -*-
"""基线预测器（Naive / MovingAverage / ExponentialSmoothing）滚动预测误差对比

负责人: A同学（src/forecasting/* 为 A 全权，B 只读）

职责
----
1. 实例化 DeepBullwhip 内置的三种基线预测器，在需求时序上执行滚动预测；
2. 对每个预测器得到整段 (T,) 的 (forecast_mean, forecast_std) 输出；
3. 计算预测误差指标 MAE / RMSE / MAPE（MAPE 对真实值为 0 的期做掩码，防除零）；
4. 对外提供 ``run_baseline_comparison(demand_data) -> pd.DataFrame``，
   行 = 预测器名称，列 = MAE / RMSE / MAPE；
5. 结果可保存到 ``outputs/results/baseline_metrics.csv``，
   作为 custom_forecaster.py 自定义预测器的对比基准。

对外接口
----
::

    generate_baseline_forecasts(demand_data)
        -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    compute_forecast_metrics(y_true, y_pred) -> dict[str, float]
    run_baseline_comparison(demand_data)      -> pd.DataFrame
    save_baseline_metrics(metrics_df, filename="baseline_metrics.csv") -> Path

框架事实（读本地 deepbullwhip 0.4.1 源码实测，非文档转述）
----
* ``Forecaster.forecast(demand_history, steps_ahead=1)`` 的真实签名返回
  **两个标量** ``(mean, std)``（deepbullwhip/forecast/base.py，见
  docs/AB_interface_checklist.md §1.1）；产出整段 (T,) 数组的是**继承来的
  具体方法** ``generate_forecasts(demand) -> (fm, fs)``，两者形状都是 (T,)。
  本模块因此只调用 ``generate_forecasts`` 拿整段数组交付——满足
  "预测输出必须整段 mean/std 数组、不允许单点" 的要求，
  且不自定义 Forecaster 子类（那是 custom_forecaster.py 的工作）。
* 滚动窗口语义：``generate_forecasts`` 内部 ``history = demand[: t + 1]``
  **包含当期观测**（框架 simulate() 即按此口径消费 fm/fs），因此
  第 0 期三个基线的 fm[0] 恒等于 y[0]、fs[0]=0（启动瞬态，属框架行为）。
  误差指标按同一口径直接对齐 y[t] vs fm[t]，与框架消费方式一致。
* 三个基线构造参数（均为框架默认值，未引入自定义参数）：
  ``NaiveForecaster()``（历史均值）、
  ``MovingAverageForecaster(window=10)``、
  ``ExponentialSmoothingForecaster(alpha=0.3)``。
* 框架自带 metrics 只覆盖牛鞭/库存/成本口径，**没有** MAE/RMSE/MAPE，
  误差指标由本模块按标准公式计算。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from deepbullwhip.forecast import (
    ExponentialSmoothingForecaster,
    Forecaster,
    MovingAverageForecaster,
    NaiveForecaster,
)

__all__ = [
    "BASELINE_FORECASTERS",
    "generate_baseline_forecasts",
    "compute_forecast_metrics",
    "run_baseline_comparison",
    "save_baseline_metrics",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 项目根目录（src/forecasting/baseline_forecasters.py 的上两级），与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 结果输出目录（任务约定：outputs/results/baseline_metrics.csv）
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# 三种官方基线预测器：名称 -> 类。行序固定按此排列。
# 全部使用框架默认参数（window=10 / alpha=0.3），保证与框架自带行为一致。
BASELINE_FORECASTERS: dict[str, type[Forecaster]] = {
    "Naive": NaiveForecaster,
    "MovingAverage": MovingAverageForecaster,
    "ExponentialSmoothing": ExponentialSmoothingForecaster,
}

METRIC_COLUMNS = ["MAE", "RMSE", "MAPE"]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _clean_values(values: np.ndarray | pd.Series) -> np.ndarray:
    """需求序列基础清洗：float64 一维，NaN/Inf 记 0，负值截断为 0。

    与 src/data_loader.py::_clean_series 同一口径（对齐清单 §2.4 第 22 条：
    M5 无销量日即 0，不插值），保持两个模块对"干净需求"的定义一致。

    Parameters
    ----------
    values : array-like
        原始需求序列。

    Returns
    -------
    np.ndarray
        shape (T,) 的 float64 数组，全部有限且 >= 0。
    """
    arr = np.array(values, dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = 0.0
    np.clip(arr, 0.0, None, out=arr)
    return arr


def _coerce_demand_columns(demand_data: pd.DataFrame | pd.Series | np.ndarray
                           ) -> dict[str, np.ndarray]:
    """把输入统一成 {列名: 清洗后一维数组}。

    支持两种 DataFrame 口径：
    * **长表**（团队注意事项 §8：date/store_id/item_id/demand 四列，
      data_loader 两个加载函数的返回格式）→ 自动 pivot 成宽表再拆列，
      列名为 ``"{store_id}_{item_id}"``；
    * **宽表**（index = 期数，columns = 序列）→ 直接按列拆。

    Parameters
    ----------
    demand_data : pd.DataFrame or pd.Series or np.ndarray
        长表 / 宽表需求；也接受单条 pd.Series 或 1-D np.ndarray
        （列名记为 "demand"）。

    Returns
    -------
    dict[str, np.ndarray]
        每列一条清洗后的需求序列。

    Raises
    ------
    TypeError
        输入类型不受支持时。
    ValueError
        表为空、含非数值列、或整列全 NaN 时。
    """
    if isinstance(demand_data, pd.Series):
        frame = demand_data.to_frame(name=demand_data.name or "demand")
    elif isinstance(demand_data, pd.DataFrame):
        frame = demand_data
        # 长表（date/store_id/item_id/demand，团队注意事项 §8 口径）→ 自动转宽表
        if {"date", "demand"} <= set(frame.columns) and (
                {"store_id", "item_id"} & set(frame.columns)):
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from src.data_loader import pivot_demand_wide
            frame = pivot_demand_wide(frame)
    elif isinstance(demand_data, np.ndarray):
        if demand_data.ndim != 1:
            raise ValueError(f"np.ndarray 输入必须是一维，收到 shape {demand_data.shape}")
        return {"demand": _clean_values(demand_data)}
    else:
        raise TypeError(
            "demand_data 应为 pd.DataFrame / pd.Series / np.ndarray，"
            f"收到 {type(demand_data).__name__}"
        )

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("demand_data 为空（0 行或 0 列），无法执行滚动预测")

    all_nan_cols = [str(c) for c in frame.columns if frame[c].isna().all()]
    if all_nan_cols:
        raise ValueError(f"以下列整列均为 NaN，无有效需求数据: {all_nan_cols}")

    non_numeric = [str(c) for c in frame.columns
                   if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"以下列不是数值类型，无法作为需求序列: {non_numeric}")

    return {str(col): _clean_values(frame[col]) for col in frame.columns}


def _new_baseline_instances() -> list[tuple[str, Forecaster]]:
    """按固定顺序实例化三种基线预测器（每次全新实例，无跨调用状态）。

    Returns
    -------
    list[tuple[str, Forecaster]]
        [(名称, 实例), ...]，顺序与 ``BASELINE_FORECASTERS`` 一致。
    """
    return [(name, cls()) for name, cls in BASELINE_FORECASTERS.items()]


# ---------------------------------------------------------------------------
# 对外函数
# ---------------------------------------------------------------------------

def generate_baseline_forecasts(
    demand_data: pd.DataFrame | pd.Series | np.ndarray,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """对每条需求序列执行三种基线预测器的滚动预测。

    内部调用各预测器**继承来的** ``generate_forecasts(demand)``（框架真实
    API），返回整段 (T,) 的 (forecast_mean, forecast_std) 数组——不是单点。

    Parameters
    ----------
    demand_data : pd.DataFrame or pd.Series or np.ndarray
        data_loader 处理后的需求宽表（一行一期，一列一条序列），
        也接受单条 Series / 1-D 数组。

    Returns
    -------
    dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
        外层 key = 序列列名（Series / ndarray 输入时为 "demand"）；
        内层 key = 预测器名称（"Naive" / "MovingAverage" / "ExponentialSmoothing"）；
        值 = ``(fm, fs)``，两个 shape (T,) 的 float64 数组。
        notebook 画"真实值 vs 预测 mean"直接取 ``forecasts[col][name][0]``。

    Raises
    ------
    TypeError
        输入类型不受支持时。
    ValueError
        输入内容不合法，或框架返回形状与输入不一致时。
    """
    columns = _coerce_demand_columns(demand_data)
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for col, series in columns.items():
        per_forecaster: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, forecaster in _new_baseline_instances():
            fm, fs = forecaster.generate_forecasts(series)
            fm = np.asarray(fm, dtype=np.float64)
            fs = np.asarray(fs, dtype=np.float64)
            if fm.shape != series.shape or fs.shape != series.shape:
                raise ValueError(
                    f"预测器 {name} 返回形状 {fm.shape}/{fs.shape} 与输入 "
                    f"{series.shape} 不一致，框架行为异常"
                )
            per_forecaster[name] = (fm, fs)
        results[col] = per_forecaster
    return results


def compute_forecast_metrics(y_true: np.ndarray | pd.Series,
                             y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """计算三种标准预测误差指标：MAE / RMSE / MAPE（%）。

    MAPE 除零处理：真实值为 0 的期**不参与** MAPE（掩码 y != 0，标准做法），
    MAE / RMSE 仍用全部期数；若所有真实值均为 0，MAPE 记 NaN 而非崩溃。

    Parameters
    ----------
    y_true : array-like
        真实需求，shape (T,)。
    y_pred : array-like
        预测均值，shape (T,)，与 ``y_true`` 等长对齐。

    Returns
    -------
    dict[str, float]
        ``{"MAE": float, "RMSE": float, "MAPE": float}``，
        MAE/RMSE 与需求同量纲，MAPE 为百分数（如 12.5 表示 12.5%）。

    Raises
    ------
    ValueError
        两序列为空或长度不一致时。
    """
    true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true_arr.size == 0:
        raise ValueError("y_true 为空，无法计算误差指标")
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"y_true 与 y_pred 长度不一致: {true_arr.shape} vs {pred_arr.shape}"
        )

    err = true_arr - pred_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    nonzero_mask = true_arr != 0.0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs(err[nonzero_mask] / true_arr[nonzero_mask])) * 100.0
        )
    else:
        mape = float("nan")  # 全为 0 的需求无 MAPE 意义，显式 NaN 而非除零崩溃

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_baseline_comparison(demand_data: pd.DataFrame | pd.Series | np.ndarray
                            ) -> pd.DataFrame:
    """运行三种基线预测器的滚动预测并输出误差对比总表。

    每个预测器经 ``generate_forecasts`` 得到整段 (T,) 的 mean/std 数组后，
    与真实需求逐期对齐计算 MAE / RMSE / MAPE。多列输入时按**全部列拼接**
    汇总成一个误差值（data_loader 产出的各列等长，等价于逐列平均）；
    单列输入即为该列的误差。

    Parameters
    ----------
    demand_data : pd.DataFrame or pd.Series or np.ndarray
        data_loader 处理后的需求宽表（index = 期数，columns = 序列）。

    Returns
    -------
    pd.DataFrame
        index = 预测器名称（"Naive" / "MovingAverage" / "ExponentialSmoothing"，
        index 名为 ``forecaster``，顺序固定）；
        columns = ``["MAE", "RMSE", "MAPE"]``（float64）。

    Raises
    ------
    TypeError
        输入类型不受支持时。
    ValueError
        输入内容不合法时（见 ``_coerce_demand_columns``）。
    """
    forecasts = generate_baseline_forecasts(demand_data)
    columns = _coerce_demand_columns(demand_data)

    rows: dict[str, dict[str, float]] = {}
    for name, _forecaster in _new_baseline_instances():
        # 多列拼接成一个大样本池计算总误差（各列等长时 = 逐列平均）
        y_true = np.concatenate([columns[col] for col in columns])
        y_pred = np.concatenate([forecasts[col][name][0] for col in columns])
        rows[name] = compute_forecast_metrics(y_true, y_pred)

    metrics_df = pd.DataFrame.from_dict(rows, orient="index",
                                        columns=METRIC_COLUMNS, dtype=np.float64)
    metrics_df.index.name = "forecaster"
    return metrics_df


def save_baseline_metrics(metrics_df: pd.DataFrame,
                          filename: str = "baseline_metrics.csv") -> Path:
    """把基线误差对比表保存到 outputs/results/ 目录。

    目录不存在时自动创建。CSV 以 UTF-8-SIG 编码写入（Excel 打开不乱码），
    浮点保留 6 位小数。

    Parameters
    ----------
    metrics_df : pd.DataFrame
        ``run_baseline_comparison`` 的返回值。
    filename : str, default "baseline_metrics.csv"
        文件名（不含目录），保存到 ``outputs/results/`` 下。

    Returns
    -------
    Path
        实际写入的 CSV 绝对路径。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    metrics_df.to_csv(out_path, float_format="%.6f", encoding="utf-8-sig")
    return out_path


# ---------------------------------------------------------------------------
# 内部自测（python src/forecasting/baseline_forecasters.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """依次自测：数据加载 → 框架滚动语义对齐 → 基线对比 → CSV 落盘。

    M5 数据缺失（FileNotFoundError / ImportError）视为 SKIP 并自动降级到
    合成兜底数据；其余任何异常记 FAIL。
    返回进程退出码：0 全部通过，1 有失败项。
    """
    # Windows 中文控制台默认 GBK，防 UnicodeEncodeError 导致自测中断
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    import deepbullwhip

    print("=" * 68)
    print("  src/forecasting/baseline_forecasters.py 自测")
    print("=" * 68)
    print(f"解释器: {sys.executable}")
    print(f"deepbullwhip {getattr(deepbullwhip, '__version__', '?')} | "
          f"pandas {pd.__version__} | numpy {np.__version__}")

    failures: list[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"       {line}")
        if not ok:
            failures.append(name)

    # ---------- 步骤 1：导入 data_loader 并加载数据（M5 → 合成兜底） ----------
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_loader import load_m5_demand, load_synthetic_demand

    demand_data: pd.DataFrame | None = None
    source = ""
    try:
        demand_data = load_m5_demand(
            "CA_1", ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
        )
        source = "M5 本地CSV"
    except (FileNotFoundError, ImportError) as exc:
        print(f"\n[SKIP] load_m5_demand —— {str(exc).splitlines()[0]}")
        print("       data/raw 暂无 M5 CSV 属预期情况，改用合成兜底数据完成自测。")
    except Exception:  # noqa: BLE001
        report("load_m5_demand(...)", False, traceback.format_exc())

    if demand_data is None:
        try:
            demand_data = load_synthetic_demand(n_nodes=4, n_days=156, seed=42)
            source = "合成 AR(1) 兜底"
        except Exception:  # noqa: BLE001
            report("load_synthetic_demand(4, 156, 42)", False, traceback.format_exc())

    if demand_data is not None:
        report("数据加载（data_loader）", True,
               f"来源: {source}；形状 {demand_data.shape}；"
               f"列名 {list(demand_data.columns)}")

    # ---------- 步骤 2：框架滚动语义对齐（手算校验，证明未编造 API） ----------
    try:
        toy = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
        toy_fc = generate_baseline_forecasts(toy)
        fm_naive, fs_naive = toy_fc["y"]["Naive"]
        fm_es, fs_es = toy_fc["y"]["ExponentialSmoothing"]
        # Naive = 历史均值（含当期）: [1, 1.5, 2, 2.5]
        np.testing.assert_allclose(fm_naive, [1.0, 1.5, 2.0, 2.5])
        # ES(alpha=0.3): level 逐步 1 -> 1.3 -> 1.81 -> 2.467（0.3*4 + 0.7*1.81）
        np.testing.assert_allclose(fm_es, [1.0, 1.3, 1.81, 2.467], rtol=1e-9)
        assert fs_naive[0] == 0.0 and fs_es[0] == 0.0  # 第 0 期启动瞬态
        report("滚动预测数组 (fm, fs) 语义对齐", True,
               "Naive fm = [1, 1.5, 2, 2.5]；ES fm = [1, 1.3, 1.81, 2.467]\n"
               "与 generate_forecasts 的 history=demand[:t+1] 口径逐值吻合")
    except Exception:  # noqa: BLE001
        report("滚动预测数组 (fm, fs) 语义对齐", False, traceback.format_exc())

    # ---------- 步骤 3：run_baseline_comparison 结构与指标 ----------
    if demand_data is None:
        report("run_baseline_comparison(demand_data)", False, "依赖步骤 1 的数据")
    else:
        try:
            metrics_df = run_baseline_comparison(demand_data)
            assert list(metrics_df.columns) == METRIC_COLUMNS, (
                f"列错误: {list(metrics_df.columns)}"
            )
            assert list(metrics_df.index) == list(BASELINE_FORECASTERS), (
                f"行错误: {list(metrics_df.index)}"
            )
            assert metrics_df.index.name == "forecaster"
            assert metrics_df[["MAE", "RMSE"]].notna().all().all(), "MAE/RMSE 有 NaN"
            assert (metrics_df[["MAE", "RMSE"]] >= 0).all().all(), "MAE/RMSE 出现负值"

            # 单条序列输入（长表按 (store_id, item_id) 过滤）可独立成表（逐列口径）
            _, first_series_df = next(iter(
                demand_data.groupby(["store_id", "item_id"], sort=True)))
            first_key = (f"{first_series_df['store_id'].iloc[0]}"
                         f"_{first_series_df['item_id'].iloc[0]}")
            single = run_baseline_comparison(first_series_df)

            # MAPE 除零鲁棒性：真实值含 0 / 全 0 时不得崩溃
            # y=[0,2,4], pred=[1,2,3] -> 掩码后 |0/2|,|(-1)/4| 均值 *100 = 12.5
            m_with_zero = compute_forecast_metrics(np.array([0.0, 2.0, 4.0]),
                                                   np.array([1.0, 2.0, 3.0]))
            assert np.isclose(m_with_zero["MAPE"], 12.5), m_with_zero
            m_all_zero = compute_forecast_metrics(np.zeros(3), np.ones(3))
            assert np.isnan(m_all_zero["MAPE"])
            # 手算校验 MAE/RMSE: err=[1,0,1] -> MAE=2/3, RMSE=sqrt(2/3)
            assert np.isclose(m_with_zero["MAE"], 2.0 / 3.0)
            assert np.isclose(m_with_zero["RMSE"], np.sqrt(2.0 / 3.0))

            report("run_baseline_comparison(demand_data)", True,
                   f"输入形状 {demand_data.shape}\n误差对比总表:\n"
                   + metrics_df.round(4).to_string()
                   + f"\n单序列输入（{first_key}）MAE 对比: "
                   f"{single['MAE'].round(4).to_dict()}\n"
                   f"MAE/RMSE 手算校验与 MAPE 除零掩码（含 0 / 全 0 -> NaN）: OK")
        except Exception:  # noqa: BLE001
            report("run_baseline_comparison(demand_data)", False,
                   traceback.format_exc())

    # ---------- 步骤 4：CSV 落盘 ----------
    try:
        metrics_df = run_baseline_comparison(
            demand_data if demand_data is not None
            else load_synthetic_demand(n_nodes=4, n_days=156, seed=42)
        )
        out_path = save_baseline_metrics(metrics_df)
        assert out_path.exists() and out_path.stat().st_size > 0
        reloaded = pd.read_csv(out_path, index_col="forecaster")
        assert list(reloaded.columns) == METRIC_COLUMNS
        report(f"save_baseline_metrics -> {out_path.name}", True,
               f"已写入: {out_path}\n回读校验: 形状 {reloaded.shape}，"
               f"列 {list(reloaded.columns)} OK")
    except Exception:  # noqa: BLE001
        report("save_baseline_metrics", False, traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：滚动预测、误差对比、CSV 落盘均无报错"
          "（M5 缺失时自动走合成兜底）。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

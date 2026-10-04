# -*- coding: utf-8 -*-
"""自定义预测器 WeightedMAForecaster（加权移动平均，含残差标准差）

负责人: A同学（src/forecasting/* 为 A 全权，B 只读）

职责
----
继承 DeepBullwhip 抽象基类 ``Forecaster``，实现加权移动平均预测器
``WeightedMAForecaster`` 并用框架 ``@register`` 装饰器注册，
可接入 ``baseline_forecasters.py`` 的对比流程参与误差指标计算。

对外接口
----
::

    WeightedMAForecaster(window=10, weights=None)   # @register("forecaster", "weighted_ma")
    run_comparison_with_custom(demand_data, forecaster=None, name=None)
        -> pd.DataFrame   # 3 个官方基线 + 自定义预测器的误差对比总表

返回契约（本任务强制）
----
``forecast(demand_history, steps_ahead)`` 返回
``(forecast_mean: np.ndarray, forecast_std: np.ndarray)``，
**两个数组、长度都等于 steps_ahead**——均值与标准差同时输出，
std 代表预测不确定性，供下游补货策略使用，禁止只返回单点。

框架事实与兼容性设计（读本地 deepbullwhip 0.4.1 源码实测）
----
* 基类抽象方法 ``Forecaster.forecast`` 的原始签名返回**两个标量**
  ``tuple[float, float]``（deepbullwhip/forecast/base.py，见
  docs/AB_interface_checklist.md §1.1）；全框架**唯一**调用
  ``.forecast()`` 的位置是 ``generate_forecasts`` 内部的
  ``fm[t], fs[t] = self.forecast(history)``（base.py:58）。
* 本类按任务要求让 ``forecast()`` 返回长度=steps_ahead 的数组；
  为不破坏框架的滚动流程，**重写了具体方法 ``generate_forecasts``**：
  滚动语义与基类逐点一致（``history = demand[: t + 1]``，取 1 步预测
  结果的第 0 个元素填入 (T,) 数组），仅做"数组 -> 标量槽位"的类型适配。
  在框架内重写该方法有官方先例——自带的 ``DeepARForecaster`` 就重写了
  ``generate_forecasts``（deepbullwhip/forecast/deepar.py）。
  ``BenchmarkRunner`` / ``simulate()`` 只消费 ``generate_forecasts``
  的 (T,) 数组，因此本类与框架全链路兼容。
* 注册：``@register("forecaster", "weighted_ma")``，之后
  ``deepbullwhip.registry.get("forecaster", "weighted_ma")`` 即可实例化。
* std 口径：对全部可用历史做"1 步加权预测 vs 真实值"的滚动残差，
  取残差样本标准差（ddof=1）。历史太短（可预测残差不足 2 个）时按
  框架自带预测器同款约定回退 0.0（Naive/MA/ES 在 len(history)<=1 时
  也返回 fs=0.0）——这是数据驱动的启动瞬态，不是把 std 写死为常数：
  只要历史 >= window+2 期，std 随数据变化且非零（除非需求完全恒定）。
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from deepbullwhip._types import TimeSeries
from deepbullwhip.forecast import Forecaster
from deepbullwhip.registry import get, get_class, register

__all__ = ["WeightedMAForecaster", "run_comparison_with_custom"]

# 项目根目录（src/forecasting/custom_forecaster.py 的上两级），与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------

@register("forecaster", "weighted_ma")
class WeightedMAForecaster(Forecaster):
    """加权移动平均预测器：mean = 加权均值，std = 历史滚动预测残差的标准差。

    Parameters
    ----------
    window : int, default 10
        加权窗口长度（参与加权平均的最近期数），>= 1。
    weights : Sequence[float] or None, default None
        与窗口切片**同序**的权重：weights[0] 乘窗口内最旧一期，
        weights[-1] 乘最新一期（np.average 的对齐约定）。
        None 时退化为等权（即普通移动平均）。构造时会归一化使权重和为 1；
        长度必须等于 window，且不允许负权重、权重和必须为正。

    Notes
    -----
    * 多步预测（steps_ahead > 1）为平坦外推：各步 mean 相同（移动平均类
      方法的标准做法）；各步 std 取同一残差标准差——在 i.i.d. 残差假设下
      平坦外推的多步误差方差不含步长 h 项，故不人为放大。
    * 短历史（len(history) < window）时用最近 len(history) 期 +
      对应尾部权重重归一化计算 mean；残差不足 2 个时 std 回退 0.0（见模块
      docstring 的 std 口径说明）。
    """

    def __init__(self, window: int = 10, weights: Sequence[float] | None = None) -> None:
        """初始化窗口与归一化权重。参数校验失败抛 ValueError。"""
        window = int(window)
        if window < 1:
            raise ValueError(f"window 至少为 1，收到 {window}")
        self.window = window
        self.weights = self._normalize_weights(window, weights)

    @staticmethod
    def _normalize_weights(window: int,
                           weights: Sequence[float] | None) -> np.ndarray:
        """校验并归一化权重（和为 1）。

        Parameters
        ----------
        window : int
            窗口长度，权重个数必须与之相等。
        weights : Sequence[float] or None
            原始权重；None 表示等权。

        Returns
        -------
        np.ndarray
            shape (window,) 的归一化权重。

        Raises
        ------
        ValueError
            长度与 window 不符、含非有限值、存在负权重、或权重和 <= 0 时。
        """
        if weights is None:
            return np.full(window, 1.0 / window, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != window:
            raise ValueError(
                f"权重个数 ({w.size}) 必须等于 window ({window})"
            )
        if not np.all(np.isfinite(w)):
            raise ValueError("权重必须全部为有限值")
        if (w < 0).any():
            raise ValueError(f"权重不允许为负: {w}")
        total = float(w.sum())
        if total <= 0:
            raise ValueError(f"权重之和必须为正，收到 {total}")
        return w / total

    def forecast(self, demand_history: TimeSeries,
                 steps_ahead: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """加权移动平均预测，同时返回均值与标准差数组。

        mean：最近 ``window`` 期（不足则取全部可用期）的加权平均，
        权重按"最旧 -> 最新"与切片同序对齐。
        std：全部可用历史上"1 步加权预测 vs 真实值"的滚动残差样本
        标准差（ddof=1），即用历史预测误差的经验分布估计预测不确定性。

        Parameters
        ----------
        demand_history : TimeSeries
            观测到的需求历史（一维，含当期；与框架 forecast 约定一致）。
        steps_ahead : int, default 1
            预测步数，>= 1。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(forecast_mean, forecast_std)``，两个 shape (steps_ahead,)
            的 float64 数组；std 非负、随历史数据变化（非写死常数）。

        Raises
        ------
        ValueError
            ``steps_ahead < 1`` 或历史为空时。
        """
        steps = int(steps_ahead)
        if steps < 1:
            raise ValueError(f"steps_ahead 至少为 1，收到 {steps_ahead}")

        history = _clean_demand_history(demand_history)
        if history.size == 0:
            raise ValueError("demand_history 为空，无法预测")

        # ---- mean：加权平均（短历史时截取尾部权重并重归一化） ----
        m = history.size
        if m >= self.window:
            mean_value = float(np.average(history[-self.window:],
                                          weights=self.weights))
        else:
            w_tail = self.weights[-m:]
            mean_value = float(np.average(history[-m:],
                                          weights=w_tail / w_tail.sum()))

        # ---- std：全历史上 1 步加权预测的滚动残差标准差 ----
        residuals = [
            history[t] - float(np.average(history[t - self.window:t],
                                          weights=self.weights))
            for t in range(self.window, m)
        ]
        if len(residuals) >= 2:
            std_value = float(np.std(residuals, ddof=1))
        else:
            # 启动瞬态：可预测残差不足 2 个，无离散度可言。
            # 与框架自带预测器 len(history)<=1 时 fs=0.0 的约定一致（非写死常数）。
            std_value = 0.0

        return np.full(steps, mean_value, dtype=np.float64), \
            np.full(steps, std_value, dtype=np.float64)

    def generate_forecasts(self, demand: TimeSeries) -> tuple[TimeSeries, TimeSeries]:
        """滚动预测整段需求序列，返回 (T,) 的 (fm, fs) 数组。

        重写基类同名的具体方法以适配本类 ``forecast()`` 的数组返回值：
        基类实现把返回值直接赋给 float 数组的单个槽位（base.py:58
        ``fm[t], fs[t] = self.forecast(history)``），遇到数组返回会抛
        "setting an array element with a sequence"。本重写保持滚动语义
        **逐点一致**——每个 t 用 ``demand[:t+1]`` 做 1 步预测、取第 0 个
        元素——仅改变类型适配（框架内 DeepARForecaster 有同样的重写先例）。

        Parameters
        ----------
        demand : TimeSeries
            完整需求序列，shape (T,)。

        Returns
        -------
        tuple[TimeSeries, TimeSeries]
            ``(fm, fs)``，两个 shape (T,) 的 float64 数组，与
            ``simulate()`` / ``BenchmarkRunner`` 的消费口径一致。
        """
        series = _clean_demand_history(demand)
        T = series.size
        fm = np.zeros(T, dtype=np.float64)
        fs = np.zeros(T, dtype=np.float64)
        for t in range(T):
            mean_arr, std_arr = self.forecast(series[:t + 1], steps_ahead=1)
            fm[t] = float(np.asarray(mean_arr).reshape(-1)[0])
            fs[t] = float(np.asarray(std_arr).reshape(-1)[0])
        return fm, fs


def _clean_demand_history(values: np.ndarray | pd.Series | Sequence[float]) -> np.ndarray:
    """预测输入清洗：float64 一维，NaN/Inf 记 0，负值截断为 0。

    与 src/data_loader.py / baseline_forecasters.py 同一口径
    （对齐清单 §2.4 第 22 条：无销量日即 0，不插值）。

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


# ---------------------------------------------------------------------------
# 接入基线对比流程
# ---------------------------------------------------------------------------

def run_comparison_with_custom(demand_data: pd.DataFrame | pd.Series | np.ndarray,
                               forecaster: Forecaster | None = None,
                               name: str | None = None) -> pd.DataFrame:
    """把自定义预测器加入基线误差对比，输出 4 行对比总表。

    前 3 行来自 ``run_baseline_comparison``（Naive / MovingAverage /
    ExponentialSmoothing），最后一行为 ``forecaster``（默认
    ``WeightedMAForecaster()``）在相同数据、相同口径（``generate_forecasts``
    滚动输出 vs 真实需求逐期对齐，多列拼接汇总）下的误差指标。

    Parameters
    ----------
    demand_data : pd.DataFrame or pd.Series or np.ndarray
        data_loader 处理后的需求宽表（一行一期，一列一条序列）。
    forecaster : Forecaster or None, default None
        参与对比的自定义预测器实例；None 时用 ``WeightedMAForecaster()``。
        也可以传任意 ``Forecaster`` 子类实例（含框架自带预测器）。
    name : str or None, default None
        对比表中的行名；None 时默认 "WeightedMA"，
        传入自定义 forecaster 时用其类名。

    Returns
    -------
    pd.DataFrame
        index = 预测器名称（index 名 ``forecaster``，基线 3 行 + 自定义 1 行）；
        columns = ``["MAE", "RMSE", "MAPE"]``。

    Raises
    ------
    TypeError / ValueError
        输入不合法时（沿用 baseline_forecasters 的校验）。
    """
    # 同包（A 全权模块）内部复用；直接以脚本运行本文件时退化为同目录导入
    try:
        from src.forecasting.baseline_forecasters import (
            _coerce_demand_columns, compute_forecast_metrics,
            run_baseline_comparison,
        )
    except ImportError:  # noqa: F401 —— python src/forecasting/custom_forecaster.py 直跑
        from baseline_forecasters import (  # type: ignore[no-redef]
            _coerce_demand_columns, compute_forecast_metrics,
            run_baseline_comparison,
        )

    if forecaster is None:
        forecaster = WeightedMAForecaster()
    if name is None:
        name = type(forecaster).__name__ if not isinstance(
            forecaster, WeightedMAForecaster) else "WeightedMA"

    columns = _coerce_demand_columns(demand_data)
    y_true = np.concatenate([columns[col] for col in columns])
    y_pred = np.concatenate([
        np.asarray(forecaster.generate_forecasts(columns[col])[0],
                   dtype=np.float64)
        for col in columns
    ])
    custom_row = pd.DataFrame(
        [compute_forecast_metrics(y_true, y_pred)],
        index=[name], columns=["MAE", "RMSE", "MAPE"], dtype=np.float64,
    )

    combined = pd.concat([run_baseline_comparison(demand_data), custom_row])
    combined.index.name = "forecaster"
    return combined


# ---------------------------------------------------------------------------
# 内部自测（python src/forecasting/custom_forecaster.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """依次自测：数据加载 -> 构造与权重归一化 -> forecast 数组契约（手算
    校验 mean/std）-> 框架注册 -> generate_forecasts 兼容性 -> 接入对比流程。

    M5 数据缺失（FileNotFoundError / ImportError）视为 SKIP 并自动降级到
    合成兜底数据。返回进程退出码：0 全部通过，1 有失败项。
    """
    # Windows 中文控制台默认 GBK，防 UnicodeEncodeError 导致自测中断
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    import deepbullwhip

    print("=" * 68)
    print("  src/forecasting/custom_forecaster.py 自测")
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

    # ---------- 步骤 1：导入 data_loader 并加载数据（M5 -> 合成兜底） ----------
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
               f"来源: {source}；形状 {demand_data.shape}")

    # ---------- 步骤 2：构造与权重归一化 ----------
    try:
        wma_uniform = WeightedMAForecaster(window=4)
        np.testing.assert_allclose(wma_uniform.weights, [0.25, 0.25, 0.25, 0.25])

        wma = WeightedMAForecaster(window=3, weights=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(wma.weights, [1 / 6, 2 / 6, 3 / 6])
        assert abs(float(wma.weights.sum()) - 1.0) < 1e-12

        for bad_kwargs, why in [
            (dict(window=3, weights=[1, 2]), "权重个数 != window"),
            (dict(window=3, weights=[1, -2, 3]), "负权重"),
            (dict(window=3, weights=[0, 0, 0]), "权重和为 0"),
            (dict(window=0), "window < 1"),
        ]:
            try:
                WeightedMAForecaster(**bad_kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"应报错但通过了: {why} ({bad_kwargs})")

        report("构造与权重归一化", True,
               f"等权默认: window=4 -> {[0.25] * 4}\n"
               f"自定义: [1,2,3] -> 归一化 {[1/6, 1/3, 1/2]}\n"
               f"非法输入（长度不符/负权重/零和/window<1）全部拦截: OK")
    except Exception:  # noqa: BLE001
        report("构造与权重归一化", False, traceback.format_exc())

    # ---------- 步骤 3：forecast 数组契约 + 手算校验（证明非编造口径） ----------
    try:
        toy = [2.0, 1.0, 4.0, 3.0, 6.0, 4.0]  # window=3, weights=[1,2,3]
        mean_arr, std_arr = wma.forecast(toy)
        assert isinstance(mean_arr, np.ndarray) and isinstance(std_arr, np.ndarray), (
            "forecast 必须返回 np.ndarray 对"
        )
        assert mean_arr.shape == (1,) and std_arr.shape == (1,), (
            f"steps_ahead=1 时形状应为 (1,): {mean_arr.shape}, {std_arr.shape}"
        )
        # 手算: mean = WMA([3,6,4]) = (3*1 + 6*2 + 4*3)/6 = 4.5
        np.testing.assert_allclose(mean_arr[0], 4.5, rtol=1e-12)
        # 手算: 残差 = y_t - WMA(前3期) = [1/3, 3, -2/3]
        indep_residuals = []
        for t in range(3, len(toy)):
            indep_residuals.append(
                toy[t] - sum(w * toy[t - 3 + i] for i, w in
                             enumerate([1 / 6, 2 / 6, 3 / 6]))
            )
        np.testing.assert_allclose(std_arr[0], np.std(indep_residuals, ddof=1),
                                   rtol=1e-9)
        assert std_arr[0] > 0, "有波动的数据 std 必须为正"

        # 多步: 数组长度 == steps_ahead，平坦外推
        mean4, std4 = wma.forecast(toy, steps_ahead=4)
        assert mean4.shape == (4,) and std4.shape == (4,)
        np.testing.assert_allclose(mean4, [4.5] * 4)
        np.testing.assert_allclose(std4, std_arr[0])

        # std 非写死：恒定数据 -> 0（数据驱动），不同噪声数据 -> 不同正数
        std_const = wma.forecast([5.0] * 6)[1][0]
        std_a = wma.forecast([2.0, 1.0, 4.0, 3.0, 6.0, 4.0])[1][0]
        std_b = wma.forecast([10.0, 12.0, 9.0, 14.0, 11.0, 15.0])[1][0]
        assert std_const == 0.0 and std_a > 0 and std_b > 0 and std_a != std_b

        # 短历史（< window）：尾部权重重归一化，std 回退 0（启动瞬态）
        mean_s, std_s = wma.forecast([7.0, 9.0])
        np.testing.assert_allclose(mean_s[0], 8.2, rtol=1e-12)  # 7*0.4 + 9*0.6
        assert std_s[0] == 0.0

        report("forecast 返回契约与手算校验", True,
               f"mean/std 均为 np.ndarray，len == steps_ahead（1 步 (1,)，4 步 (4,)）\n"
               f"手算: mean=4.5, std={std_arr[0]:.6f}（独立重算残差逐值吻合）\n"
               f"std 随数据变化: 恒定数据={std_const}, 噪声A={std_a:.4f}, "
               f"噪声B={std_b:.4f}\n"
               f"短历史 [7,9]: mean=8.2（尾部权重重归一化）, std=0（启动瞬态）")
    except Exception:  # noqa: BLE001
        report("forecast 返回契约与手算校验", False, traceback.format_exc())

    # ---------- 步骤 4：框架注册校验 ----------
    try:
        assert get_class("forecaster", "weighted_ma") is WeightedMAForecaster, (
            "get_class 未取到本类"
        )
        inst = get("forecaster", "weighted_ma")
        assert isinstance(inst, WeightedMAForecaster), "get 实例化失败"
        assert getattr(WeightedMAForecaster, "_registry_key", None) == (
            "forecaster/weighted_ma"
        )
        assert isinstance(inst, Forecaster), "未继承框架 Forecaster 基类"
        report("框架注册（@register）", True,
               "register('forecaster', 'weighted_ma') 生效\n"
               "get_class / get 均可取到；isinstance(Forecaster) = True")
    except Exception:  # noqa: BLE001
        report("框架注册（@register）", False, traceback.format_exc())

    # ---------- 步骤 5：generate_forecasts 兼容性（框架消费口径） ----------
    try:
        series = demand_data[demand_data.columns[0]].to_numpy(
            dtype=np.float64) if demand_data is not None else np.array(
            [2.0, 1.0, 4.0, 3.0, 6.0, 4.0])
        fm, fs = wma.generate_forecasts(series)
        assert isinstance(fm, np.ndarray) and isinstance(fs, np.ndarray)
        assert fm.shape == (series.size,) and fs.shape == (series.size,)
        assert np.isfinite(fm).all() and np.isfinite(fs).all() and (fs >= 0).all()

        # 与"逐期 forecast(demand[:t+1], 1) 取第 0 元素"的手工滚动逐值一致
        fm_manual = np.array([wma.forecast(series[:t + 1])[0][0]
                              for t in range(series.size)])
        fs_manual = np.array([wma.forecast(series[:t + 1])[1][0]
                              for t in range(series.size)])
        np.testing.assert_allclose(fm, fm_manual, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fs, fs_manual, rtol=1e-12, atol=1e-12)
        assert WeightedMAForecaster.generate_forecasts is not (
            Forecaster.generate_forecasts), "缺少数组适配重写"
        report("generate_forecasts 兼容性", True,
               f"输入 (T,)={series.shape} -> fm/fs 均为 (T,)，有限且 fs>=0\n"
               f"与基类滚动语义（history=demand[:t+1]）逐值一致: OK")
    except Exception:  # noqa: BLE001
        report("generate_forecasts 兼容性", False, traceback.format_exc())

    # ---------- 步骤 6：接入基线对比流程 ----------
    if demand_data is None:
        report("run_comparison_with_custom(demand_data)", False, "依赖步骤 1 的数据")
    else:
        try:
            combined = run_comparison_with_custom(demand_data)
            assert list(combined.columns) == ["MAE", "RMSE", "MAPE"]
            assert list(combined.index) == [
                "Naive", "MovingAverage", "ExponentialSmoothing", "WeightedMA"
            ], f"行错误: {list(combined.index)}"
            assert combined.loc["WeightedMA"].notna().all()
            report("run_comparison_with_custom(demand_data)", True,
                   f"输入形状 {demand_data.shape}\n含自定义预测器的误差对比总表:\n"
                   + combined.round(4).to_string())
        except Exception:  # noqa: BLE001
            report("run_comparison_with_custom(demand_data)", False,
                   traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：WeightedMAForecaster 数组契约、注册、框架兼容、"
          "对比流程均无报错。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

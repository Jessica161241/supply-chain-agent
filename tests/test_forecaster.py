# -*- coding: utf-8 -*-
"""预测与数据加载模块单元测试（pytest）

覆盖范围
----
1. ``src/data_loader.py``            —— 长表契约（date/store_id/item_id/demand）、
                                        config 参数对齐、pivot_demand_wide、
                                        build_replay_generator（长表/宽表/Series 三态）、
                                        load_m5_demand（仅本地 CSV + 参数校验，不依赖外网）
2. ``src/forecasting/baseline_forecasters.py`` —— 长表输入、输出格式、指标口径
3. ``src/forecasting/custom_forecaster.py``    —— WeightedMAForecaster 数组契约与 mean/std 合法性
4. ``src/visualization/dashboard.py``          —— 长表输入自动 pivot、落盘、边界

设计约定
----
* 全部用例离线可跑：数据一律用合成兜底（seed 固定可复现）；M5 真实数据用例
  在 ``data/raw/m5/sales_train_evaluation.csv`` 缺失时 ``pytest.skip``，
  保证 CI 不因网络/数据缺失而失败。
* 合成数据默认参数对齐 config/network_config.yaml（4 SKU × 365 天、
  demand.mean=10、demand.std=3），见 data_loader 模块 docstring。
* 不触碰 ``config/network_config.yaml``；不含 policy / simulation / 拓扑类测试。

运行
----
    pytest tests/test_forecaster.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from deepbullwhip.demand.replay import ReplayDemandGenerator
from deepbullwhip.forecast import Forecaster
from deepbullwhip.registry import get_class

from src.data_loader import (
    LONG_COLUMNS,
    M5_DATA_ROOT,
    build_replay_generator,
    load_m5_demand,
    load_synthetic_demand,
    pivot_demand_wide,
)
from src.forecasting.baseline_forecasters import (
    BASELINE_FORECASTERS,
    METRIC_COLUMNS,
    compute_forecast_metrics,
    generate_baseline_forecasts,
    run_baseline_comparison,
    save_baseline_metrics,
)
from src.forecasting.custom_forecaster import (
    WeightedMAForecaster,
    run_comparison_with_custom,
)

# 绘图断言走无头后端（须在 pyplot 被导入前设置）
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.visualization import dashboard as dashboard_module
from src.visualization.dashboard import (
    FIGURES_DIR,
    plot_demand_overview,
    plot_forecast_mean,
    plot_uncertainty_band,
)

# M5 本地数据根目录直接复用 src.data_loader.M5_DATA_ROOT（单一事实来源）

# ---------------------------------------------------------------------------
# 共享 fixture
# ---------------------------------------------------------------------------

# 10 期有波动的短历史（window=3 时滚动残差充足，用于手算校验 std>0）
TOY_HISTORY = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 4.0, 5.0, 3.0, 7.0, 5.0])
TOY_WEIGHTS = [1.0, 2.0, 3.0]  # 归一化后 [1/6, 2/6, 3/6]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """长表合成需求（默认 4 SKU × 365 天，config 对齐；seed=42 可复现）。"""
    return load_synthetic_demand()


@pytest.fixture
def wide_df(synthetic_df) -> pd.DataFrame:
    """长表转宽表（index=date，columns=SYN_SKU_1..4），供逐序列断言。"""
    return pivot_demand_wide(synthetic_df)


@pytest.fixture
def toy_history() -> np.ndarray:
    """10 期确定性短历史，供 forecast 手算校验。"""
    return TOY_HISTORY.copy()


@pytest.fixture
def weighted_ma() -> WeightedMAForecaster:
    """window=3、权重 [1,2,3] 的加权移动平均预测器实例。"""
    return WeightedMAForecaster(window=3, weights=TOY_WEIGHTS)


# ---------------------------------------------------------------------------
# src/data_loader.py
# ---------------------------------------------------------------------------

class TestDataLoader:
    """数据加载模块：长表契约、config 对齐、交接对象、M5 参数校验。"""

    def test_load_synthetic_demand_long_contract(self, synthetic_df):
        """长表契约：四列/行数/日期数/SKU 数/排序/无 NaN/非负。"""
        assert isinstance(synthetic_df, pd.DataFrame)
        assert list(synthetic_df.columns) == LONG_COLUMNS
        assert synthetic_df.shape == (4 * 365, 4)         # n_nodes × n_days
        assert synthetic_df["date"].nunique() == 365      # config simulation_days
        assert sorted(synthetic_df["item_id"].unique()) == [
            f"SKU_{i}" for i in range(1, 5)]              # config sku_count=4
        assert synthetic_df["date"].is_monotonic_increasing
        assert not synthetic_df.isna().any().any()
        assert (synthetic_df["demand"] >= 0).all()

    def test_load_synthetic_demand_config_aligned(self, synthetic_df):
        """对齐 config 基准：demand.mean=10、demand.std=3（AR(1) 平稳 std）。"""
        g = synthetic_df.groupby("item_id")["demand"]
        assert (g.mean() - 10.0).abs().max() < 0.6, "均值偏离 demand.mean=10"
        assert (g.std() - 3.0).abs().max() < 0.4, "std 偏离 demand.std=3"

    def test_load_synthetic_demand_reproducible_and_clean(self):
        """同 seed 结果逐位一致、不同 seed 不同。"""
        df_a = load_synthetic_demand(n_nodes=2, n_days=50, seed=42)
        df_b = load_synthetic_demand(n_nodes=2, n_days=50, seed=42)
        df_c = load_synthetic_demand(n_nodes=2, n_days=50, seed=7)
        assert df_a.equals(df_b), "同 seed 必须可复现"
        assert not df_a.equals(df_c), "不同 seed 必须产生不同数据"
        assert not df_a.isna().any().any()
        assert (df_a["demand"] >= 0).all()

    @pytest.mark.parametrize("kwargs", [
        {"n_nodes": 0, "n_days": 30},
        {"n_nodes": 2, "n_days": 1},
    ])
    def test_load_synthetic_demand_rejects_invalid_args(self, kwargs):
        """非法 n_nodes / n_days 必须抛 ValueError。"""
        with pytest.raises(ValueError):
            load_synthetic_demand(**kwargs)

    def test_pivot_demand_wide(self, synthetic_df):
        """长表 → 宽表：形状/列名/index 名/无 NaN/数值与长表一致。"""
        wide = pivot_demand_wide(synthetic_df)
        assert wide.shape == (365, 4)
        assert list(wide.columns) == [f"SYN_SKU_{i}" for i in range(1, 5)]
        assert wide.index.name == "date"
        assert not wide.isna().any().any()
        sku1 = synthetic_df[synthetic_df["item_id"] == "SKU_1"]
        np.testing.assert_allclose(wide["SYN_SKU_1"].to_numpy(),
                                   sku1["demand"].to_numpy())

    def test_pivot_demand_wide_rejects_invalid(self):
        """缺列 / 空表必须抛 ValueError。"""
        with pytest.raises(ValueError):
            pivot_demand_wide(pd.DataFrame({"a": [1.0]}))
        with pytest.raises(ValueError):
            pivot_demand_wide(pd.DataFrame(columns=LONG_COLUMNS))

    def test_build_replay_generator_long_aggregates_by_date(
            self, synthetic_df, wide_df):
        """长表输入按 date 聚合求和成单序列（整表口径）。"""
        gen = build_replay_generator(synthetic_df)
        assert isinstance(gen, ReplayDemandGenerator)
        assert gen.data.shape == (365,)
        np.testing.assert_allclose(gen.data, wide_df.sum(axis=1).to_numpy())

    def test_build_replay_generator_wide_backcompat(self):
        """宽表输入（向后兼容）：多列按行求和。"""
        wide = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        gen = build_replay_generator(wide)
        np.testing.assert_allclose(gen.data, [4.0, 6.0])

    def test_build_replay_generator_series(self, wide_df):
        """pd.Series 输入直接取该序列。"""
        gen = build_replay_generator(wide_df["SYN_SKU_1"])
        assert isinstance(gen, ReplayDemandGenerator)
        np.testing.assert_allclose(gen.data, wide_df["SYN_SKU_1"].to_numpy())

    def test_build_replay_generator_per_series_filter(self, synthetic_df):
        """逐节点构造：按 item_id 过滤长表后各自构造（B 拓扑推荐用法）。"""
        for item, g in synthetic_df.groupby("item_id"):
            gen = build_replay_generator(g)
            assert isinstance(gen, ReplayDemandGenerator)
            assert gen.data.shape == (365,)
            np.testing.assert_allclose(
                gen.data, g.sort_values("date")["demand"].to_numpy())

    def test_build_replay_generator_rejects_invalid_input(self):
        """空表 / 整列 NaN / 全零序列 / 长表全零必须报 ValueError。"""
        with pytest.raises(ValueError):
            build_replay_generator(pd.DataFrame())
        with pytest.raises(ValueError):
            build_replay_generator(pd.DataFrame({"a": [np.nan, np.nan]}))
        with pytest.raises(ValueError):
            build_replay_generator(pd.DataFrame({"a": [0.0, 0.0, 0.0]}))
        with pytest.raises(ValueError):                 # 长表全零
            build_replay_generator(pd.DataFrame(
                {"date": pd.date_range("2024-01-01", periods=3),
                 "store_id": "SYN", "item_id": "SKU_1",
                 "demand": [0.0, 0.0, 0.0]}))

    def test_load_m5_demand_validates_ids_without_network(self):
        """M5 参数校验在文件访问之前完成——无需数据/网络即可测。"""
        with pytest.raises(ValueError):
            load_m5_demand("CA_9", ["FOODS_1"])          # 非法门店
        with pytest.raises(ValueError):
            load_m5_demand("CA_1", ["FOODS_9"])          # 非法 dept
        with pytest.raises(ValueError):
            load_m5_demand("CA_1", ["FOODS_1_001"])      # item 级 ID（框架无此维度）

    def test_load_m5_demand_local_csv_or_skip(self):
        """M5 真实数据用例：仅读本地 CSV，缺数据时 skip，保证 CI 不碰外网。"""
        csv_path = M5_DATA_ROOT / "m5" / "sales_train_evaluation.csv"
        if not csv_path.exists():
            pytest.skip(f"本地无 M5 CSV（{csv_path}），跳过真实数据用例")
        df = load_m5_demand("CA_1", ["FOODS_1", "HOUSEHOLD_1"])
        assert list(df.columns) == LONG_COLUMNS
        assert df["item_id"].nunique() == 2 and df["store_id"].nunique() == 1
        n_days = df["date"].nunique()
        assert n_days == 365                    # 对齐 config simulation_days（截断）
        assert df.shape == (n_days * 2, 4)
        assert not df.isna().any().any()
        assert (df["demand"] >= 0).all()
        # 每条序列日期连续（daily，相邻日相差 1 天），起点为 M5 锚点 2011-01-29
        for _, g in df.groupby("item_id"):
            d = pd.to_datetime(g.sort_values("date")["date"])
            assert (d.diff().dropna() == pd.Timedelta(days=1)).all()
            assert d.iloc[0] == pd.Timestamp("2011-01-29")


# ---------------------------------------------------------------------------
# src/forecasting/baseline_forecasters.py
# ---------------------------------------------------------------------------

class TestBaselineForecasters:
    """基线预测对比：长表输入、输出格式、指标口径、滚动预测数组形状。"""

    def test_run_baseline_comparison_output_format(self, synthetic_df):
        """长表输入 -> 返回 pd.DataFrame：行=3 个基线、列=MAE/RMSE/MAPE、无 NaN。"""
        metrics = run_baseline_comparison(synthetic_df)
        assert isinstance(metrics, pd.DataFrame)
        assert list(metrics.columns) == METRIC_COLUMNS == ["MAE", "RMSE", "MAPE"]
        assert list(metrics.index) == list(BASELINE_FORECASTERS)
        assert metrics.index.name == "forecaster"
        assert not metrics.isna().any().any(), "误差表不允许出现 NaN"
        assert (metrics[["MAE", "RMSE"]] >= 0).all().all()

    def test_run_baseline_comparison_synthetic_values(self, synthetic_df):
        """合成数据上指标为正数（AR(1) 无零值 -> MAPE 也有效）。"""
        metrics = run_baseline_comparison(synthetic_df)
        assert (metrics > 0).all().all()

    def test_generate_baseline_forecasts_shapes(self, synthetic_df):
        """长表输入 -> 每个 (序列, 预测器) 产出 (T,) 的 (fm, fs)，有限且 fs>=0。"""
        forecasts = generate_baseline_forecasts(synthetic_df)
        T = synthetic_df["date"].nunique()
        assert set(forecasts) == {f"SYN_SKU_{i}" for i in range(1, 5)}
        for col, per_fc in forecasts.items():
            assert set(per_fc) == set(BASELINE_FORECASTERS)
            for name, (fm, fs) in per_fc.items():
                assert isinstance(fm, np.ndarray) and isinstance(fs, np.ndarray)
                assert fm.shape == (T,) and fs.shape == (T,)
                assert np.isfinite(fm).all() and np.isfinite(fs).all()
                assert (fs >= 0).all(), f"{col}/{name} 出现负 std"

    def test_generate_baseline_forecasts_accepts_wide(self):
        """宽表输入（向后兼容）同样产出逐列 (fm, fs)。"""
        wide = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                             "b": [4.0, 3.0, 2.0, 1.0]})
        forecasts = generate_baseline_forecasts(wide)
        assert set(forecasts) == {"a", "b"}
        for per_fc in forecasts.values():
            for fm, fs in per_fc.values():
                assert fm.shape == (4,) and fs.shape == (4,)

    def test_compute_forecast_metrics_hand_values(self):
        """指标公式手算校验 + MAPE 除零掩码（含 0）与全零 NaN 路径。"""
        # y=[0,2,4], pred=[1,2,3]: MAE=2/3, RMSE=sqrt(2/3), MAPE=|0/2|、|1/4| 均值*100=12.5
        m = compute_forecast_metrics(np.array([0.0, 2.0, 4.0]),
                                     np.array([1.0, 2.0, 3.0]))
        assert m["MAE"] == pytest.approx(2.0 / 3.0)
        assert m["RMSE"] == pytest.approx(np.sqrt(2.0 / 3.0))
        assert m["MAPE"] == pytest.approx(12.5)
        # 真实值全为 0：MAPE 显式 NaN，不除零崩溃
        m0 = compute_forecast_metrics(np.zeros(3), np.ones(3))
        assert np.isnan(m0["MAPE"])

    def test_run_baseline_comparison_rejects_empty(self):
        """空输入必须抛 ValueError。"""
        with pytest.raises(ValueError):
            run_baseline_comparison(pd.DataFrame())


# ---------------------------------------------------------------------------
# src/forecasting/custom_forecaster.py
# ---------------------------------------------------------------------------

class TestCustomForecaster:
    """WeightedMAForecaster：数组契约、mean/std 合法性、注册与框架兼容。"""

    def test_forecast_returns_two_arrays_len_steps_ahead(self, weighted_ma, toy_history):
        """steps_ahead=7 -> 返回两个 np.ndarray，mean/std 形状都是 (7,)。"""
        mean_arr, std_arr = weighted_ma.forecast(toy_history, steps_ahead=7)
        assert isinstance(mean_arr, np.ndarray)
        assert isinstance(std_arr, np.ndarray)
        assert mean_arr.shape == (7,)
        assert std_arr.shape == (7,)

    def test_forecast_default_steps_ahead_is_one(self, weighted_ma, toy_history):
        """默认 steps_ahead=1 -> 形状 (1,)。"""
        mean_arr, std_arr = weighted_ma.forecast(toy_history)
        assert mean_arr.shape == (1,) and std_arr.shape == (1,)

    def test_forecast_mean_hand_value(self, weighted_ma, toy_history):
        """mean = 加权平均（权重同序对齐）：WMA([3,7,5]) = 32/6。"""
        mean_arr, _ = weighted_ma.forecast(toy_history, steps_ahead=3)
        np.testing.assert_allclose(mean_arr, [32.0 / 6.0] * 3)

    def test_forecast_std_not_hardcoded_zero(self, weighted_ma, toy_history):
        """std 来自历史滚动残差：有波动数据 std>0（不允许全 0 / 写死常数）。"""
        _, std_arr = weighted_ma.forecast(toy_history, steps_ahead=7)
        assert not np.all(std_arr == 0), "std 不允许硬编码为 0"
        # 与独立重算的滚动残差标准差逐值一致
        w = np.array(TOY_WEIGHTS) / sum(TOY_WEIGHTS)
        residuals = np.array([
            toy_history[t] - float(np.average(toy_history[t - 3:t], weights=w))
            for t in range(3, len(toy_history))
        ])
        np.testing.assert_allclose(std_arr, np.std(residuals, ddof=1), rtol=1e-9)
        # 不同数据 -> 不同 std（证明非常数）
        std_other = weighted_ma.forecast(
            [10.0, 12.0, 9.0, 14.0, 11.0, 15.0, 13.0, 16.0])[1][0]
        assert std_other > 0 and std_other != pytest.approx(std_arr[0])
        # 恒定数据的 0 是数据驱动结果，非写死
        assert weighted_ma.forecast([5.0] * 6)[1][0] == 0.0

    def test_forecast_values_finite_nonnegative_std(self, weighted_ma, toy_history):
        """mean/std 无 NaN、无 Inf；std 非负。"""
        mean_arr, std_arr = weighted_ma.forecast(toy_history, steps_ahead=7)
        assert np.isfinite(mean_arr).all() and np.isfinite(std_arr).all()
        assert not np.isnan(mean_arr).any() and not np.isnan(std_arr).any()
        assert (std_arr >= 0).all()

    def test_forecast_validates_steps_ahead(self, weighted_ma, toy_history):
        """steps_ahead < 1 必须抛 ValueError。"""
        with pytest.raises(ValueError):
            weighted_ma.forecast(toy_history, steps_ahead=0)

    def test_constructor_normalizes_and_validates_weights(self):
        """权重归一化（和为 1）与非法输入拦截。"""
        wma = WeightedMAForecaster(window=3, weights=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(wma.weights, [1 / 6, 2 / 6, 3 / 6])
        np.testing.assert_allclose(
            WeightedMAForecaster(window=4).weights, [0.25] * 4)  # 默认等权
        for kwargs in [dict(window=3, weights=[1, 2]),       # 长度不符
                       dict(window=3, weights=[1, -2, 3]),   # 负权重
                       dict(window=3, weights=[0, 0, 0]),    # 零和
                       dict(window=0)]:                      # window < 1
            with pytest.raises(ValueError):
                WeightedMAForecaster(**kwargs)

    def test_registered_in_framework_registry(self):
        """@register('forecaster','weighted_ma') 生效，且继承框架 Forecaster。"""
        assert get_class("forecaster", "weighted_ma") is WeightedMAForecaster
        assert getattr(WeightedMAForecaster, "_registry_key", None) == (
            "forecaster/weighted_ma")
        assert isinstance(WeightedMAForecaster(), Forecaster)

    def test_generate_forecasts_framework_compatible(self, weighted_ma, toy_history):
        """重写的 generate_forecasts 输出 (T,) 且与逐期 forecast 手工滚动一致。"""
        fm, fs = weighted_ma.generate_forecasts(toy_history)
        assert isinstance(fm, np.ndarray) and isinstance(fs, np.ndarray)
        assert fm.shape == (10,) and fs.shape == (10,)
        assert np.isfinite(fm).all() and np.isfinite(fs).all() and (fs >= 0).all()
        for t in range(len(toy_history)):
            m1, s1 = weighted_ma.forecast(toy_history[:t + 1])
            assert fm[t] == pytest.approx(m1[0])
            assert fs[t] == pytest.approx(s1[0])

    def test_run_comparison_with_custom_adds_row(self, synthetic_df, weighted_ma):
        """自定义预测器接入对比表：基线 3 行 + WeightedMA 共 4 行，无 NaN。"""
        combined = run_comparison_with_custom(synthetic_df, forecaster=weighted_ma)
        assert isinstance(combined, pd.DataFrame)
        assert list(combined.columns) == ["MAE", "RMSE", "MAPE"]
        assert list(combined.index) == [
            "Naive", "MovingAverage", "ExponentialSmoothing", "WeightedMA"]
        assert not combined.isna().any().any()
        assert (combined[["MAE", "RMSE"]] >= 0).all().all()


# ---------------------------------------------------------------------------
# 补充覆盖：保存 / Series 输入 / dashboard 预测绘图
# ---------------------------------------------------------------------------

def test_save_baseline_metrics_roundtrip(synthetic_df):
    """save_baseline_metrics 落盘 + 回读一致（临时文件名，测后清理）。"""
    metrics = run_baseline_comparison(synthetic_df)
    out = save_baseline_metrics(metrics, filename="test_metrics_roundtrip_tmp.csv")
    try:
        assert out.exists() and out.stat().st_size > 0
        reloaded = pd.read_csv(out, index_col="forecaster")
        assert list(reloaded.columns) == METRIC_COLUMNS
        np.testing.assert_allclose(reloaded["MAE"].to_numpy(),
                                   metrics["MAE"].to_numpy(), rtol=1e-5)
    finally:
        out.unlink(missing_ok=True)


def test_compute_forecast_metrics_accepts_series():
    """compute_forecast_metrics 接受 pd.Series 输入（与 ndarray 同结果）。"""
    m = compute_forecast_metrics(pd.Series([0.0, 2.0, 4.0]),
                                 pd.Series([1.0, 2.0, 3.0]))
    assert m["MAE"] == pytest.approx(2.0 / 3.0)
    assert m["MAPE"] == pytest.approx(12.5)


class TestDashboard:
    """预测绘图函数：长表输入自动 pivot、返回 Figure、不依赖仿真模块。"""

    def test_dashboard_plots_return_figure_and_save(self, synthetic_df):
        """三个绘图函数（长表输入）返回 Figure 并按 save_path 落盘（测后清理）。"""
        col = pivot_demand_wide(synthetic_df).columns[0]   # "SYN_SKU_1"
        wma = WeightedMAForecaster(window=3, weights=TOY_WEIGHTS)
        baseline_fc = generate_baseline_forecasts(
            synthetic_df[synthetic_df["item_id"] == "SKU_1"])
        custom_fm, custom_fs = wma.generate_forecasts(
            synthetic_df[synthetic_df["item_id"] == "SKU_1"]
            .sort_values("date")["demand"].to_numpy())

        paths = [FIGURES_DIR / f"test_dashboard_{n}.png"
                 for n in ("overview", "mean", "band")]
        try:
            fig1 = plot_demand_overview(synthetic_df, data_source="test",
                                        save_path=paths[0])
            forecast_map = dict(baseline_fc["SYN_SKU_1"])
            forecast_map["WeightedMA"] = (custom_fm, custom_fs)
            fig2 = plot_forecast_mean(synthetic_df, col, forecast_map,
                                      data_source="test", save_path=paths[1])
            fig3 = plot_uncertainty_band(
                synthetic_df, col, custom_fm, custom_fs,
                std_curves={"ExponentialSmoothing":
                            baseline_fc["SYN_SKU_1"]["ExponentialSmoothing"][1]},
                data_source="test", save_path=paths[2])
            import matplotlib.figure
            for fig in (fig1, fig2, fig3):
                assert isinstance(fig, matplotlib.figure.Figure)
            for p in paths:
                assert p.exists() and p.stat().st_size > 0
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
            plt.close("all")

    def test_dashboard_has_no_b_module_dependency(self):
        """dashboard 的 import 依赖不含 src.network / src.coordination（分工边界）。"""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(dashboard_module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        bad = [m for m in imported
               if m.startswith("src.network") or m.startswith("src.coordination")]
        assert not bad, f"dashboard 依赖了 B 的模块: {bad}"

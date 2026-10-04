# supply-chain-agent 架构文档

> 赛题二·方向二「多级库存智能协同」
> 状态：A 同学模块已实现并通过单元测试；B 同学模块（拓扑/协同/仿真执行）待交付。
> A/B 参数与契约的完整核对记录见 [AB_interface_checklist.md](AB_interface_checklist.md)。

---

## 1. 项目定位

在「区域仓 → 城市仓 → 终端门店」三层供应链网络上，用聚合缺口信号实现跨节点补货
协同，对比基线策略与自研协同策略的三项指标：`Bullwhip Ratio`、`Fill Rate`、
`Total Cost`。仿真框架为 `deepbullwhip 0.4.1`（pip 安装）。

## 2. 总体数据流

```
M5 本地CSV（data/raw/m5/）─┐
                           ├─> src/data_loader.py ──> 长表 DataFrame（date/store_id/item_id/demand）
合成 AR(1)（seed 兜底）────┘         │
                                     ├─> ReplayDemandGenerator（A→B 需求交接物）
                                     │
                                     └─> src/forecasting/ ──> dict[节点, (fm, fs)]（A→B 预测交接物，
                                                                  fm/fs 均为 (T,) float64）
                                                 │
                                                 v
                       src/network/topology.py（B，待交付）──> NetworkSupplyChain.simulate
                       src/coordination/*（B，待交付）          │
                                                                 v
                                       NetworkSimulationResult（B→A 结果交接物）
                                                                 │
                                                                 v
                       src/visualization/dashboard.py ──> outputs/figures/ + outputs/results/
```

## 3. 模块清单与职责

| 模块 | 负责人 | 状态 | 说明 |
|---|---|---|---|
| `src/data_loader.py` | A | ✅ 已实现 | M5 / 合成数据加载、清洗、ReplayDemandGenerator 交接 |
| `src/forecasting/baseline_forecasters.py` | A | ✅ 已实现 | 三种官方基线滚动预测 + MAE/RMSE/MAPE 对比 |
| `src/forecasting/custom_forecaster.py` | A | ✅ 已实现 | 自定义 WeightedMAForecaster（mean + 残差 std） |
| `src/visualization/dashboard.py` | A | ✅ 已实现 | 预测相关绘图三函数（需求总览/均值对比/不确定带） |
| `tests/test_forecaster.py` | A | ✅ 已实现 | pytest 单元测试（24 用例） |
| `agent/agent_wrapper.py` | A+B | 🟡 A 侧完成 | 数据/预测封装已完成；仿真执行 TODO(B) 占位 |
| `src/network/topology.py` | B | ⬜ 待交付 | 3 层 NetworkSupplyChain 构造（指南预期签名：`build_network_from_config(config_path)`） |
| `src/coordination/*` | B | ⬜ 待交付 | 聚合缺口信号 + 协同补货策略 |
| `config/network_config.yaml` | B | ✅ 在库 | 拓扑参数（B 维护，A 不改） |
| `src/simulation/run_benchmark.py` | A+B | ⬜ 待联调 | 基线 vs 协同的指标对比跑批 |

## 4. A 同学模块详细设计

### 4.1 数据层 `src/data_loader.py`

对外接口（前三个为冻结契约函数，第四个为长表→宽表工具）：

```python
load_m5_demand(store_ids, item_ids) -> pd.DataFrame                  # 长表
load_synthetic_demand(n_nodes=5, n_days=365, seed=42) -> pd.DataFrame  # 长表
build_replay_generator(demand_df) -> ReplayDemandGenerator
pivot_demand_wide(demand_long_df) -> pd.DataFrame                    # 长表 -> 宽表
```

* **返回格式（团队注意事项 §8 口径）**：长表，固定四列
  `[date, store_id, item_id, demand]`——`date` 为 datetime64（M5 daily 口径逐日日期，
  锚点 2011-01-29 起、截取前 365 天；合成数据为 2024-01-01 起的逐日日期）；
  `item_id` 承载 dept 级序列 ID（框架无 item 维度，对齐清单 §1.2）。
  供内部流程/节点映射使用的宽表用 `pivot_demand_wide` 转换
  （index=date，columns="{store_id}_{item_id}"）。
* **M5 加载**：内部调用框架 `load_dataset("m5", store=..., dept=...,
  freq="daily", data_root=...)`，本地 CSV（`data/raw/m5/sales_train_evaluation.csv`）优先，
  缺失且有 Kaggle 凭证时走框架 `load_m5` 自动下载（缓存 `~/.deepbullwhip/data/m5/`），
  再失败则抛带三类修复提示的 `FileNotFoundError`；daily 序列截取前 365 天
  （`M5_MAX_PERIODS`，对齐 config simulation_days）。
* **维度口径**：非法 ID 抛 `ValueError`（框架本身会静默返回全 0，必须拦截）。
* **清洗口径**：NaN/Inf 记 0、负值截断为 0、不做插值（对齐清单 §2.4 第 22 条）；
  每条序列日期连续无空洞。
* **合成数据默认参数对齐 config**（团队注意事项 §8）：5 SKU × 365 天、
  demand.mean=10、demand.std=3（AR(1) rho=0.6、创新项 sigma=2.4 → 序列平稳 std≈3）；
  每节点 seed+i 差异化，同 seed 逐位可复现。
* **交接对象**：`build_replay_generator` 支持三种输入——长表（按 date 聚合求和 =
  整表口径）、宽表（按行求和，向后兼容）、`pd.Series`（直接取该列）；
  逐节点构造用 `build_replay_generator(demand_df[demand_df.item_id == xxx])` 过滤后传入。
* 环境变量 `M5_DATA_ROOT` 可覆盖数据目录（不影响函数签名）。

### 4.2 预测层 `src/forecasting/`

**基线对比 `baseline_forecasters.py`**（全部使用框架真实 API 与默认参数）：

```python
generate_baseline_forecasts(demand_data) -> dict[列名, dict[预测器, (fm, fs)]]
compute_forecast_metrics(y_true, y_pred) -> {"MAE", "RMSE", "MAPE"}
run_baseline_comparison(demand_data) -> pd.DataFrame   # 行=预测器，列=MAE/RMSE/MAPE
save_baseline_metrics(metrics_df, filename="baseline_metrics.csv") -> Path
```

* 三种官方基线：`NaiveForecaster`（历史均值，非"上期值复制"）、
  `MovingAverageForecaster(window=10)`、`ExponentialSmoothingForecaster(alpha=0.3)`；
  滚动预测用框架继承方法 `generate_forecasts(demand) -> (fm, fs)`（均 `(T,)`）。
* `generate_forecasts` 的 history 含当期（`demand[:t+1]`），第 0 期 fm 恒等于 y[0]、
  fs=0（启动瞬态，框架口径）。
* MAPE 对真实值为 0 的期做掩码；全零序列 MAPE 记 NaN（不除零崩溃）。

**自定义预测器 `custom_forecaster.py`**：

```python
@register("forecaster", "weighted_ma")
class WeightedMAForecaster(Forecaster):
    def __init__(self, window: int = 10, weights: Sequence[float] | None = None) -> None
    def forecast(self, demand_history, steps_ahead: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]          # 长度 == steps_ahead
    def generate_forecasts(self, demand) -> tuple[TimeSeries, TimeSeries]   # (T,) 重写适配
```

* **mean**：最近 `window` 期加权平均，权重与切片同序（`weights[-1]` 乘最近一期），
  构造时归一化；短历史截取尾部权重重归一化。
* **std**：全部可用历史上"1 步加权预测 vs 真实值"的滚动残差样本标准差（ddof=1）
  ——预测不确定性来自历史预测误差的经验分布，非写死常数；可预测残差不足 2 个时
  按框架自带预测器同款约定回退 0.0。
* **多步**：平坦外推（MA 类标准做法），std 各步相同（i.i.d. 残差假设下平坦外推的
  多步误差方差不含步长项）。
* **框架兼容**：框架唯一调用 `forecast()` 的位置在基类 `generate_forecasts` 内部
  （把返回值赋给 float 数组单个槽位）；本类重写 `generate_forecasts` 做数组→标量
  适配，滚动语义与基类逐点一致（框架 DeepAR 预测器有同样的重写先例），
  `BenchmarkRunner` / `simulate()` 消费口径不受影响。
* 注册名 `weighted_ma`：`deepbullwhip.registry.get("forecaster", "weighted_ma")` 可直接实例化。

**对比接入**：`run_comparison_with_custom(demand_data, forecaster=None, name=None)`
把自定义预测器追加进基线对比表（3 基线 + 1 自定义，共 4 行）。

### 4.3 可视化层 `src/visualization/dashboard.py`

三个纯函数（返回 `Figure`，`save_path` 给定时落盘 `outputs/figures/`，
不调 `plt.show()`；零 B 模块依赖）：

```python
plot_demand_overview(demand_df, data_source="", save_path=None)
plot_forecast_mean(demand_df, col, forecast_map, data_source="", save_path=None)
plot_uncertainty_band(demand_df, col, forecast_mean, forecast_std,
                      std_curves=None, sigmas=(1, 2), data_source="", save_path=None)
```

`forecast_map` 为 `{预测器名: (fm, fs)}`，键 `WeightedMA` 红色高亮；
`plot_uncertainty_band` 绘制 ±1σ/±2σ 阴影带与 std(t) 对比子图。

### 4.4 Agent 封装 `agent/agent_wrapper.py`（A 侧部分）

```python
prepare_demand(use_m5=False, ...)        -> DemandBundle   # 数据准备（M5→合成三级降级）
build_forecast_bundle(demand_df, forecaster=None)
                                         -> dict[节点名, (fm, fs)]   # 预测交接物
run_network_simulation(bundle, forecast_map=None)          # TODO(B) 占位
```

`run_network_simulation` 当前抛 `NotImplementedError` 并在 docstring 中写明预期接入
方式（`src/network.topology.build_network() -> NetworkSupplyChain` →
`chain.simulate(demand, forecasts_mean, forecasts_std)`）——B 交付前 A 侧不伪造仿真行为。

## 5. A↔B 接口契约（摘要，全文见对齐清单 §3）

1. **需求数据**：长表 DataFrame（date/store_id/item_id/demand，团队注意事项 §8 口径）
   + `pivot_demand_wide` 长表→宽表工具 + `ReplayDemandGenerator`（整表聚合 / 逐节点）。
2. **预测矩阵**：`dict[节点名, (fm, fs)]`，`(T,)` float64；
   **fs 允许为 0，B 侧不得假设 fs > 0**；外部预测只对叶子门店生效
   （中间层框架用下游订单均值，见对齐清单 §1.4）。
3. **仿真结果**：`NetworkSimulationResult`（node_results / edge_flows /
   cumulative_bullwhip / total_cost）；框架 `fill_rate` 是"库存非负期数占比"口径，
   非真正订单满足率（对齐清单 §1.6）。

## 6. 产物路径

| 产物 | 路径 | 生成方 |
|---|---|---|
| 基线误差表 | `outputs/results/baseline_metrics.csv` | `save_baseline_metrics` |
| 含自定义预测器的 4 行对比表 | `outputs/results/baseline_metrics_with_custom.csv` | 同上 |
| 均值对比图 / 不确定带图 | `outputs/figures/fig1_baseline_vs_custom_mean.png`、`fig2_forecast_uncertainty_band.png` | dashboard + notebook 02 |
| 需求总览图 | `outputs/figures/fig_quickstart_demand_overview.png` | dashboard |

## 7. 变更记录

| 日期 | 变更 | 来源 |
|---|---|---|
| 2026-10-01 | 初版：按 A 模块真实实现撰写（数据层/预测层/可视化/agent A 侧）；B 模块标注待交付 | A 同学 |
| 2026-10-01 | 与《方向二项目代码结构指南.docx》核对：指南未定义需求 DataFrame 列结构（长表/宽表均未提及），config 具体数值（sku_count/simulation_days）以 B 的 network_config.yaml 为准；B 拓扑预期函数名对齐为 build_network_from_config；dashboard 指南另期望 plot_summary_dashboard / plot_bullwhip_amplification（依赖 B 仿真 results，待其交付后补充） | A 同学 |
| 2026-10-04 | 数据层切换长表契约（团队注意事项 §8）：load_m5_demand/load_synthetic_demand 返回 [date, store_id, item_id, demand]，新增 pivot_demand_wide；合成默认参数对齐 config（5 SKU×365 天、mean=10、std=3）；forecasting/dashboard/agent 输入适配长表（自动 pivot） | A 同学 |
| 2026-10-04 | 仿真时间口径拍板：M5 加载切 daily 并截取前 365 天（对齐 config simulation_days=365），156 周 weekly 口径作废；M5 选型仍 4 条序列，是否扩到 sku_count=5 待 B 拍板 | A 同学 |

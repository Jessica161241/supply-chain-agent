# supply-chain-agent 架构文档

> 赛题二 · 方向二「多级库存智能协同」
> 状态（2026-10-11）：**A 侧与 B 侧模块均已实现并通过单元测试**，联调完成。
> A/B 参数与契约的完整核对记录见 [`AB_interface_checklist.md`](AB_interface_checklist.md)。
> （对外方案说明 `solution.md` 与最终演示 `notebooks/05_final_demo.ipynb` 目前为本地初稿，尚未提交。）

---

## 1. 项目定位

在「区域仓 → 城市仓 → 终端门店」三层供应链网络上，用聚合缺口信号实现跨节点补货协同，
对比基线策略与自研协同策略的三项指标：`Bullwhip Ratio`、`Fill Rate`、`Total Cost`。
仿真框架为 `deepbullwhip 0.4.1`（pip 安装，GitHub 仓库 404，源码用 `scripts/fetch_source.sh` 拉到 `vendor/`）。

---

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
                       src/network/topology.py ──> NetworkSupplyChain（7 节点 / 6 边）
                       src/coordination/*      ──> CoordinatedOrderingPolicy + SignalBus（共享信号总线）
                                                 │
                                                 v
                                       NetworkSimulationResult（B→A 结果交接物）
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        v                        v                        v
              src/simulation/            agent/agent_wrapper.py    src/visualization/
              run_benchmark.py           （一键全链路入口）          dashboard.py
                        │                                              │
                        └──────────────> outputs/results/ <─────────────┘
                                         outputs/figures/
```

---

## 3. 模块清单与职责

| 模块 | 负责人 | 状态 | 说明 |
|---|---|---|---|
| `src/data_loader.py` | A | ✅ 已实现 | M5 / 合成数据加载、清洗、长表 ↔ 宽表、ReplayDemandGenerator 交接 |
| `src/forecasting/baseline_forecasters.py` | A | ✅ 已实现 | 三种官方基线滚动预测 + MAE/RMSE/MAPE 对比 |
| `src/forecasting/custom_forecaster.py` | A | ✅ 已实现 | 自研 WeightedMAForecaster（加权 mean + 残差 std） |
| `src/visualization/dashboard.py` | A + B | ✅ 已实现 | A 侧预测绘图 3 函数 + B 侧仿真对比绘图 3 函数 |
| `tests/test_forecaster.py` | A | ✅ 已实现 | pytest 单元测试（34 用例） |
| `agent/agent_wrapper.py` | A + B | ✅ 已实现 | A 侧数据/预测封装 + B 侧仿真接入，三函数跑通全链路 |
| `config/network_config.yaml` | B | ✅ 在库 | 拓扑参数（A 不改） |
| `src/network/topology.py` | B | ✅ 已实现 | 3 层 NetworkSupplyChain 构造（基线 / 协同两种） |
| `src/coordination/signal_protocol.py` | B | ✅ 已实现 | 聚合缺口信号 `AggregatedSignal` + 共享总线 `SignalBus` |
| `src/coordination/custom_policy.py` | B | ✅ 已实现 | `CoordinatedOrderingPolicy` = OrderUpTo + alpha × 下游缺口 |
| `src/simulation/run_benchmark.py` | A + B | ✅ 已实现 | 基线 vs 多 alpha 协同的指标对比跑批 |
| `tests/test_policy.py` / `tests/test_simulation.py` | B | ✅ 已实现 | 拓扑/策略/仿真单元测试（24 + 22 用例） |
| `notebooks/05_final_demo.ipynb` | A（整合 B 素材） | ⏳ 初稿（本地，暂未提交） | 最终演示，已执行、输出内嵌 |
| `docs/solution.md` | A | ⏳ 初稿（本地，暂未提交） | 对外方案说明 |
| `scripts/*` | A | ✅ 已实现 | 环境自检 / 源码查阅 / 骨架重建 |

测试总量：**80 collected → 79 passed, 1 skipped**（skip 项为依赖真实 M5 数据的用例）。

---

## 4. A 同学模块详细设计

### 4.1 数据层 `src/data_loader.py`

对外接口（前三个为冻结契约函数，第四个为长表→宽表工具）：

```python
load_m5_demand(store_ids, item_ids) -> pd.DataFrame                  # 长表
load_synthetic_demand(n_nodes=4, n_days=365, seed=42) -> pd.DataFrame  # 长表
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
* **合成数据默认参数对齐 config**：4 SKU × 365 天、
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
run_baseline_comparison(demand_data) -> pd.DataFrame   # index=预测器名，列=MAE/RMSE/MAPE
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

run_comparison_with_custom(demand_data, forecaster=None, name=None) -> pd.DataFrame
```

* **mean**：最近 `window` 期加权平均，权重与切片同序（`weights[-1]` 乘最近一期），
  构造时归一化；短历史截取尾部权重重归一化。
* **std**：全部可用历史上"1 步加权预测 vs 真实值"的滚动残差样本标准差（ddof=1）
  ——预测不确定性来自历史预测误差的经验分布，非写死常数；可预测残差不足 2 个时
  按框架自带预测器同款约定回退 0.0。
* **多步**：平坦外推（MA 类标准做法），std 各步相同（i.i.d. 残差假设下平坦外推的
  多步误差方差不含步长项）。
* **框架兼容（重要）**：本类 `forecast()` 返回**数组**（长度 = `steps_ahead`），
  与框架抽象签名声明的 `tuple[float, float]` 不同；因此**重写**了基类的具体方法
  `generate_forecasts` 做数组→标量适配，滚动语义与基类**逐点一致**。
  已实测确认安全：框架全代码库只在 `benchmark/runner.py:160` 调用
  `generate_forecasts()`，`forecast()` 仅被基类 `generate_forecasts` 内部调用，
  而该方法已被本类重写；框架内 `DeepARForecaster` 有同样的重写先例。
  该偏差记录于 [`AB_interface_checklist.md`](AB_interface_checklist.md) §3.2，待 A/B 正式签字确认。
* 注册名 `weighted_ma`：`deepbullwhip.get("forecaster", "weighted_ma")` 可直接实例化。

**对比接入**：`run_comparison_with_custom(...)` 把自定义预测器追加进基线对比表
（3 基线 + 1 自定义，共 4 行；index 名为 `forecaster`）。

### 4.3 可视化层 `src/visualization/dashboard.py`

六个纯函数（返回 `Figure`，`save_path` 给定时落盘 `outputs/figures/`，
不调 `plt.show()`）：

```python
# A 侧（预测）—— 零 B 模块依赖
plot_demand_overview(demand_df, data_source="", save_path=None)
plot_forecast_mean(demand_df, col, forecast_map, data_source="", save_path=None)
plot_uncertainty_band(demand_df, col, forecast_mean, forecast_std,
                      std_curves=None, sigmas=(1, 2), data_source="", save_path=None)

# B 侧（仿真对比）—— 只吃 BenchmarkResult 的类型契约，不 import，运行时零耦合
plot_summary_dashboard(benchmark_result, data_source="", save_path=None)
plot_bullwhip_amplification(benchmark_result, data_source="", save_path=None)
plot_node_cost_comparison(benchmark_result, data_source="", save_path=None)
```

`forecast_map` 为 `{预测器名: (fm, fs)}`，键 `WeightedMA` 红色高亮；
`plot_uncertainty_band` 绘制 ±1σ/±2σ 阴影带与 std(t) 对比子图。

### 4.4 Agent 封装 `agent/agent_wrapper.py`

```python
prepare_demand(use_m5=False, n_nodes=4, n_days=365, seed=42) -> DemandBundle
build_forecast_bundle(demand_df, forecaster=None) -> dict[节点名, (fm, fs)]
run_network_simulation(bundle, forecast_map=None, alpha=0.8, coordinated=True)
        -> NetworkSimulationResult
```

* `DemandBundle`（数据类）：`demand_df` / `source` / `replay_gen` / `node_generators`。
* `prepare_demand` 封装三级降级（M5 本地 CSV → Kaggle → 合成 AR(1)），
  并同时构造整表聚合与逐节点两种 `ReplayDemandGenerator`。
* `build_forecast_bundle` 逐节点调 `generate_forecasts`，产出 `(T,)` 的 `(fm, fs)`；
  预测器**可插拔**（默认 `WeightedMAForecaster`，也可传任意框架 `Forecaster` 子类）。
* `run_network_simulation` 已接入 B 侧拓扑：`coordinated=True` 走协同网络
  （`build_coordinated_network_from_config`），`False` 走纯 OrderUpTo 基线
  （`build_network_from_config`）；`forecast_map` 的 key 同时支持 A 宽表列名与拓扑节点名。
* 自测：`python agent/agent_wrapper.py` → 四步全 PASS（含协同与基线两次真实仿真）。

### 4.5 B 侧模块（接口摘要，实现归属 B）

> 详细设计以 B 侧代码与其 docstring 为准，此处只记录 A 需要知道的契约面。

```python
# src/network/topology.py
build_network_from_config(config_path) -> NetworkSupplyChain
build_coordinated_network_from_config(config_path, alpha=0.3)
        -> tuple[NetworkSupplyChain, SignalBus]
run_baseline_policy(network, demand, forecasts_mean, forecasts_std) -> NetworkSimulationResult

# src/coordination/signal_protocol.py
AggregatedSignal(node_name, timestep, gap, gap_std, n_sources)
SignalBus()  # .write() / .read() / .read_downstream() / .reset() / .snapshot()

# src/coordination/custom_policy.py
CoordinatedOrderingPolicy(lead_time, service_level, signal_bus, node_name,
                          downstream_names, alpha)
    # compute_order = max(0, S - IP) + alpha * 下游聚合 gap

# src/simulation/run_benchmark.py
run_comparison(config_path, demand, forecasts_mean, forecasts_std,
               alphas=(0.0,0.5,0.8,1.0), save_dir=None) -> BenchmarkResult
    # BenchmarkResult: summary_df / node_metrics_df / raw_results
```

---

## 5. A↔B 接口契约（摘要，全文见对齐清单 §3）

1. **需求数据**：长表 DataFrame（date/store_id/item_id/demand，团队注意事项 §8 口径）
   + `pivot_demand_wide` 长表→宽表工具 + `ReplayDemandGenerator`（整表聚合 / 逐节点）。
2. **预测矩阵**：`dict[节点名, (fm, fs)]`，`(T,)` float64；
   **fs 允许为 0，B 侧不得假设 fs > 0**；外部预测只对叶子门店生效
   （中间层框架用下游订单均值，见对齐清单 §1.4）。
3. **仿真结果**：`NetworkSimulationResult`（node_results / edge_flows /
   cumulative_bullwhip / total_cost）；框架 `fill_rate` 是"库存非负期数占比"口径，
   非真正订单满足率（对齐清单 §1.6），对外主口径用自研的
   `true_fill_rate_excl_top`（`1 - 缺货量/需求量`，且排除顶层区域仓 ——
   顶层无上游补货来源，其 fill rate 恒为 0 属框架限制）。

---

## 6. 产物路径

| 产物 | 路径 | 生成方 |
|---|---|---|
| 基线误差表 | `outputs/results/baseline_metrics.csv` | `save_baseline_metrics`（A） |
| 含自定义预测器的 4 行对比表 | `outputs/results/baseline_metrics_with_custom.csv` | 同上（A） |
| 系统级 benchmark 对比 | `outputs/results/benchmark_summary.csv` | `run_comparison`（B） |
| 节点级 benchmark 对比 | `outputs/results/benchmark_node_metrics.csv` | `run_comparison`（B） |
| 环境自检留档 | `outputs/results/env_check.json` | `scripts/quickstart_verify.py`（A） |
| 需求总览图 | `outputs/figures/fig_demo_demand_overview.png`、`fig_quickstart_demand_overview.png` | `plot_demand_overview`（A） |
| 均值对比图 | `outputs/figures/fig1_baseline_vs_custom_mean.png` | `plot_forecast_mean`（A） |
| 不确定带图 | `outputs/figures/fig2_forecast_uncertainty_band.png` | `plot_uncertainty_band`（A） |
| 对比总览 / 牛鞭放大 / 节点成本 | `outputs/figures/fig_benchmark_*.png`、`fig_node_cost_comparison.png` | B 侧绘图函数 |

---

## 7. 变更记录

| 日期 | 变更 | 来源 |
|---|---|---|
| 2026-10-01 | 初版：按 A 模块真实实现撰写（数据层/预测层/可视化/agent A 侧）；B 模块标注待交付 | A 同学 |
| 2026-10-01 | 与《方向二项目代码结构指南.docx》核对：指南未定义需求 DataFrame 列结构（长表/宽表均未提及），config 具体数值（sku_count/simulation_days）以 B 的 network_config.yaml 为准；B 拓扑预期函数名对齐为 build_network_from_config；dashboard 指南另期望 plot_summary_dashboard / plot_bullwhip_amplification（依赖 B 仿真 results，待其交付后补充） | A 同学 |
| 2026-10-04 | 数据层切换长表契约（团队注意事项 §8）：load_m5_demand/load_synthetic_demand 返回 [date, store_id, item_id, demand]，新增 pivot_demand_wide；合成默认参数对齐 config；forecasting/dashboard/agent 输入适配长表（自动 pivot） | A 同学 |
| 2026-10-04 | 仿真时间口径拍板：M5 加载切 daily 并截取前 365 天（对齐 config simulation_days=365），156 周 weekly 口径作废 | A 同学 |
| 2026-10-11 | 全面对齐现状：B 侧模块（topology / coordination / run_benchmark）状态由"待交付"改为"已实现"，新增 §4.5 B 侧接口摘要；`run_network_simulation` 由"TODO(B) 占位"改为已接入；dashboard 由 3 函数更新为 6 函数（A 3 + B 3）；测试用例数由 24 修正为 34（文件）/ 80（全量）；新增预测器数组返回契约的框架兼容说明；产物路径表补全 benchmark 与环境自检产物 | A 同学 |

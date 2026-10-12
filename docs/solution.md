# 多级库存智能协同：方案说明

> 赛题二 · 方向二
> 网络结构：区域仓 → 城市仓 → 终端门店（三层）
> 仿真框架：DeepBullwhip 0.4.1
>
> 相关文档：[架构文档](architecture.md) · [接口与口径核对清单](AB_interface_checklist.md) · [风险排查日志](risk_log.md) · [最终演示](../notebooks/05_final_demo.ipynb)

---

## 1. 问题定义

三级供应链网络中，各节点依据本地库存状态独立决策补货量。在需求自下游向上游传递的过程中，
订单波动逐级放大，形成牛鞭效应（bullwhip effect），表现为上游订单方差显著高于终端需求方差、
各层级库存与需求错配、系统总成本抬升。

本方案的目标是：在保持各节点独立决策、不交换经营明细数据的前提下，建立自下而上的信息通路，
抑制订单波动放大并降低系统总成本。

评价采用三项指标：

| 指标 | 定义 | 优化方向 |
|---|---|---|
| Bullwhip Ratio | 订单方差与需求方差之比，度量波动放大程度 | 降低 |
| Fill Rate | 已满足订单占订单总量的比例 | 提高 |
| Total Cost | 持有成本与缺货成本之和在仿真期内的累计值 | 降低 |

---

## 2. 方案总览

### 2.1 方法概述

各级节点在每个决策周期向上游上报一个聚合缺口标量，该标量由缺口总量与不确定性两部分组成、
不含任何订单明细。上游节点在原有 OrderUpTo 订货量的基础上叠加该项信号的线性补偿：

```
order = max(0, S − IP) + alpha × gap_downstream
```

其中 `S` 为目标库存水平，`IP` 为库存头寸，`alpha` 为协同强度系数。信息方向自下而上，
且不暴露任何节点的原始经营数据。

### 2.2 系统架构

```
┌──────────────────────── 需求与预测层 ────────────────────────────┐
│                                                                  │
│  合成 AR(1)（seed=42，本次实验数据来源）                          │
│  M5 本地 CSV / Kaggle（加载路径已实现，本次未启用）               │
│                │                                                 │
│                v                                                 │
│            data_loader ──> 长表 [date, store_id, item_id, demand]│
│                │                                                 │
│                ├──> ReplayDemandGenerator                        │
│                │                                                 │
│                └──> forecasting/ ──> (mean, std) 矩阵            │
│                      基线 3 种 + 自研 WeightedMA                  │
└────────────────────────────────┬─────────────────────────────────┘
                                 │  (mean, std)  仅对叶子门店生效
                                 v
┌──────────────────────── 拓扑与协同层 ────────────────────────────┐
│                                                                  │
│   config/network_config.yaml                                     │
│            │                                                     │
│            v                                                     │
│   topology.py ──> NetworkSupplyChain（7 节点 / 6 边）             │
│            │                │                                    │
│            │                └── 每节点一个策略对象（仿真期常驻）    │
│            │                          │                          │
│            │        ┌─────────────────┴─────────────────┐        │
│            │        │  CoordinatedOrderingPolicy        │        │
│            │        │   order = OrderUpTo               │        │
│            │        │         + alpha × 下游聚合缺口    │        │
│            │        └────────────┬──────────────────────┘        │
│            │                     │ 写 / 读                       │
│            │              ┌──────v──────┐                        │
│            │              │  SignalBus  │  共享信号总线           │
│            │              └─────────────┘                        │
│            v                                                     │
│   NetworkSupplyChain.simulate() ──> NetworkSimulationResult      │
│                                          │                       │
│   run_benchmark.py ──> Bullwhip / Fill Rate / Total Cost         │
└──────────────────────────────────────────┬───────────────────────┘
                                           v
                        dashboard.py ──> outputs/figures/ + outputs/results/
```

### 2.3 数据流

需求序列 → 清洗为长表 → 终端门店预测 `(mean, std)` → 注入叶子节点 →
仿真过程中信号自下而上传递 → 输出基线与协同策略的三项指标对比。

---

## 3. 核心方法：聚合缺口信号

### 3.1 信号定义

`AggregatedSignal`（`src/coordination/signal_protocol.py`）字段定义如下：

| 字段 | 含义 |
|---|---|
| `node_name` | 上报节点名；多源聚合时标记为 `AGG(n)` |
| `timestep` | 信号产生的时间步 |
| `gap` | 聚合缺口总量，非负，不含订单明细 |
| `gap_std` | 缺口的不确定性，非负，沿用预测标准差口径 |
| `n_sources` | 参与聚合的来源节点数 |

多下游聚合规则为：`gap` 求和，`gap_std` 按方差可加性合并（`√Σσᵢ²`），`n_sources` 累加。

### 3.2 隐私保护边界

信号仅承载"下游合计缺口"这一聚合标量，不包含以下任何信息：

* 单笔订单的 SKU 与数量；
* 各节点的库存明细；
* 需求序列本身。

上游节点接收到的是一个标量缺口值，无法据此反推任一节点的原始经营数据。

### 3.3 跨节点信号传递机制

框架的订货接口 `OrderingPolicy.compute_order(inventory_position, forecast_mean,
forecast_std)` 不提供跨节点参数，这是本方案落地时的主要技术约束。经阅读框架源码并实测验证，
存在无需修改框架的解法，依据以下三点：

1. 策略对象在 `NetworkSupplyChain.__init__` 中一次性构造，并在整个仿真期内常驻；
2. `compute_order()` 在每个时间步被调用一次；
3. `simulate()` 内部按反向拓扑序处理节点，即每步顺序为 store → city → regional。

因此，在构造策略时注入一个共享可变对象（`SignalBus`），各节点策略即可通过该对象互相读写；
又因处理顺序为反向拓扑序，下游节点在本步写入的信号在同一时间步内即可被上游读取，
与自下而上的信息流方向一致。

实测结果（3 层拓扑，T = 40）：store → city 与 city → regional 两级分别有 37/40 期非零读数，
前 3 期为零对应库存尚未消耗的启动瞬态。据此，原计划的兜底方案 `manual_stepper.py`
（手写仿真循环）未予启用。完整排查过程见[风险排查日志](risk_log.md) R1。

### 3.4 协同订货规则

```python
# 1. OrderUpTo 基线
S = (lead_time + 1) * forecast_mean + z_α * forecast_std * sqrt(lead_time + 1)
base_order = max(0, S - inventory_position)

# 2. 协同补偿：读取下游聚合信号
coordination = alpha * agg.gap          # agg 为 None 时取 0

# 3. 下单量
order = base_order + coordination

# 4. 上报本节点缺口
bus.write(AggregatedSignal(gap=base_order, gap_std=forecast_std, ...))
```

参数行为：`alpha = 0` 时策略退化为纯 OrderUpTo；`alpha` 增大时协同补偿增强、总成本下降；
`alpha` 超过临界值后系统进入振荡发散区间（见 §6.2）。

---

## 4. 需求与预测模块

### 4.1 数据来源与清洗

`src/data_loader.py` 提供四个对外函数：

```python
load_m5_demand(store_ids, item_ids)                    -> pd.DataFrame   # 长表
load_synthetic_demand(n_nodes=4, n_days=365, seed=42)  -> pd.DataFrame   # 长表
build_replay_generator(demand_df)                      -> ReplayDemandGenerator
pivot_demand_wide(demand_long_df)                      -> pd.DataFrame   # 长表 → 宽表
```

本方案实验结果基于合成 AR(1) 需求数据，生成参数如下：

| 项 | 取值 |
|---|---|
| 序列数 | 4（对应拓扑的 4 个终端门店） |
| 期数 | 365（daily，对齐 `simulation_days`） |
| 均值 / 标准差 | `mean ≈ 10`、平稳 `std ≈ 3` |
| 自相关结构 | AR(1)，`rho = 0.6`，创新项 `sigma = 2.4` |
| 随机种子 | 42（同种子下逐位可复现） |

M5 真实数据的加载路径已实现并保留，采用本地 CSV → Kaggle 自动下载 → 合成数据的三级降级策略，
本次实验未启用；切换至真实数据可通过 `prepare_demand(use_m5=True)` 完成。

**长表口径**为固定四列 `[date, store_id, item_id, demand]`。框架的 M5 加载接口仅提供
`store` 与 `dept` 两个筛选维度，不含 item 维度，因此 `item_id` 字段承载 dept 级序列标识。

**清洗规则**：NaN 与 Inf 记为 0，负值截断为 0，不做插值，各序列日期连续无空洞。
切换至 M5 数据后，daily 序列截取前 365 期。

### 4.2 预测输出契约

预测器输出强制为二元组，且标准差不得为固定常数。

| 项 | 约定 |
|---|---|
| 单点接口 | `Forecaster.forecast(history, steps_ahead=1) -> tuple[float, float]`，返回两个标量 |
| 整段接口 | `generate_forecasts(demand) -> (fm, fs)`，两个数组形状均为 `(T,)`，类型 float64 |
| 标准差来源 | 全部可用历史上"单步加权预测值与真实值之差"的滚动残差样本标准差（ddof = 1） |
| 边界条件 | `fs` 允许为 0（历史长度不足 2 期时的启动瞬态），消费方不应假设 `fs > 0` |

框架抽象签名声明 `forecast()` 返回标量，而产出整段 `(T,)` 数组的是继承方法
`generate_forecasts()`。二者的差异与实测依据见[核对清单](AB_interface_checklist.md) §3.2。

### 4.3 预测器与误差对比

自研预测器 `WeightedMAForecaster` 的均值为最近 `window` 期加权平均（权重递增，近期权重较高），
标准差为滚动残差标准差，通过 `@register("forecaster", "weighted_ma")` 注册至框架注册表。

在 4 条合成序列（365 期）上的滚动预测误差如下：

| 预测器 | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Naive | 2.2608 | 2.8561 | 28.27 |
| MovingAverage | 2.0898 | 2.6436 | 25.74 |
| ExponentialSmoothing | 1.4825 | 1.8647 | 17.94 |
| WeightedMA（自研） | 1.5882 | 1.9940 | 19.24 |

在本次合成数据上，自研预测器的点估计精度低于框架自带的指数平滑法
（MAPE 19.24% 对 17.94%）。其选型依据在于以下三点，而非点估计精度：

1. 提供完整的不确定性度量 `std(t)`，且该值直接来源于残差经验分布，被下游补货策略的
   安全库存项 `z_α · σ · √(L+1)` 直接消费；
2. 权重可配置，为后续接入外部信号预留了可插拔接口；
3. 完全自持实现，不依赖框架内部行为，便于审计与替换。

---

## 5. 网络与仿真设置

### 5.1 拓扑与参数

拓扑定义于 `config/network_config.yaml`：

```
regional_warehouse ──┬──> city_warehouse_1 ──┬──> store_1
                     │                      └──> store_2
                     └──> city_warehouse_2 ──┬──> store_3
                                            └──> store_4
```

共 7 节点、6 条边。节点级参数如下：

| 层级 | `lead_time` | `holding_cost` | `backorder_cost` | `service_level` | `initial_inventory` |
|---|---|---|---|---|---|
| 区域仓 | 4 | 0.08 | 0.40 | 0.95 | 50 |
| 城市仓 | 2 | 0.15 | 0.50 | 0.95 | 50 |
| 终端门店 | 1 | 0.20 | 0.60 | 0.95 | 50 |

缺货成本随层级下移递增，与框架自带 `beer_game_config()` 的参数结构一致。

补货提前期需配置在节点上（`EchelonConfig.lead_time`）。框架 `simulate()` 不读取
`EdgeConfig` 中的 `lead_time`、`capacity` 与 `transport_cost`，该字段仅参与拓扑关系表达
与图形渲染。

### 5.2 仿真与评测设置

* 仿真周期 T = 365（daily），随机种子固定为 42，全程可复现；
* 需求为合成 AR(1) 序列（`mean = 10`、`std = 3`），4 条序列映射至 4 个终端门店；
* 对比基线为框架默认的 `OrderUpToPolicy`；
* 外部预测输入（`mean` / `std`）仅对叶子门店生效，中间层节点的预测由框架以
  "最近 8 期下游订单均值"替代。

### 5.3 Fill Rate 口径定义

框架自带的 `fill_rate` 采用以下定义：

```python
fill_rate = float(np.mean(inv >= 0))     # 库存非负的期数占比
```

该指标度量的是库存非负期数占仿真总期数的比例，而非订单满足率，与本赛题对 Fill Rate 的
要求不一致。因此本方案另行计算订单满足率，并在结果中并列给出三种口径：

| 口径 | 定义 | 用途 |
|---|---|---|
| `framework_fill_rate` | 框架自带：库存非负期数占比 | 与框架内部指标对照 |
| `true_fill_rate` | `1 − 缺货量 / 需求量`，含区域仓 | 完整性参考 |
| `true_fill_rate_excl_top` | 同 `true_fill_rate`，排除区域仓 | 对外主口径 |

其中缺货量取各期 `max(0, −inventory_levels)` 之和，需求量在叶子节点取外部输入、
在中间节点取下游订单之和（与框架 `simulate()` 内部口径一致）。

排除区域仓的原因是：`simulate()` 不建模顶层节点的补货来源，区域仓库存无上游补充，
其满足率恒为 0，属框架建模限制。协同策略作用于城市仓层级，故采用排除顶层的口径
方能反映策略的实际效果。

---

## 6. 实验结果

### 6.1 主结果

| 指标 | 基线（OrderUpTo） | 协同（alpha = 0.8） | 变化 |
|---|---|---|---|
| 总成本 | 22,563.92 | 16,576.44 | −26.5% |
| 累计牛鞭系数 | 4.4828 | 5.0290 | +0.55 |
| 可优化层 fill rate（排除区域仓） | 92.79% | 94.86% | +2.07 pp |
| 框架口径 fill rate | 55.19% | 61.02% | +5.8 pp |

节点级结果（alpha = 0.8）：

| 节点 | 成本（基线 → 协同） | 订单满足率（基线 → 协同） |
|---|---|---|
| regional_warehouse | 12,079 → 11,310 | 0.00 → 0.00 |
| city_warehouse_1 | 4,192 → 1,636 | 0.00 → 0.585 |
| city_warehouse_2 | 4,360 → 1,698 | 0.00 → 0.573 |
| store_1 … store_4 | 477 / 474 / 476 / 505（不变） | 0.98 ~ 0.99（不变） |

协同策略的收益主要来源于中间层。城市仓由持续缺货状态（满足率为 0，库存均值为 −22）
转为基本满足状态（满足率约 0.58）；终端门店在两种策略下服务水平接近，未受策略变更影响。

### 6.2 alpha 扫描与相变临界

| alpha | 总成本 | 累计牛鞭 | 可优化层 fill rate |
|---|---|---|---|
| 0.0（= 基线） | 22,563.92 | 4.4828 | 92.79% |
| 0.3 | 19,447.46 | 4.6489 | 93.43% |
| 0.5 | 17,914.11 | 4.8406 | 93.87% |
| 0.8 | 16,576.44 | 5.0290 | 94.86% |
| 1.0 | 16,040.37 | 4.4626 | 98.97% |
| 1.05 | 35,557.74 | 4.4404 | 99.59% |
| 1.1 | 55,512.90 | 4.4299 | 99.59% |

`alpha = 0` 与基线的总成本逐位一致（差值 0.00e+00），可作为实现正确性的一致性校验。
总成本随 `alpha` 增大而单调下降，直至相变临界；临界位于 `alpha = 1.0` 与 `1.05` 之间：
`alpha = 1.0` 时总成本为 16,040，`alpha = 1.05` 时升至 35,558，`alpha = 1.1` 时为 55,513
（约为 `alpha = 1.0` 处的 3.5 倍），此后持续恶化，呈振荡发散特征。

综合成本表现与稳健性，主结果取 `alpha = 0.8`，距相变临界保留约 20 ~ 24% 的参数裕度。

相变后订单满足率升至 0.994，其机制是以过量订货压低缺货水平，代价是持有成本随订单量
同步攀升。这表明单一指标不足以评价策略优劣，需三项指标联立判断。

### 6.3 牛鞭系数与总成本的解耦

协同策略下累计牛鞭系数由 4.4828 升至 5.0290，而总成本下降 26.5%。其原因是协同补偿将
区域仓的消耗显著压低（12,079 → 11,310），缺货成本的下降幅度超过牛鞭系数上升带来的影响。
若仅依据牛鞭系数评价，将得出协同策略劣于基线的相反结论。

该现象在协同规则设计中具有普遍性：上游节点牛鞭系数下降而总成本上升的反直觉结果，
源于补偿项与订货策略参数之间的耦合。协同效果取决于参数设计，参数选取需以三项指标的
联合表现为依据。

---

## 7. 工程实现与可复现性

### 7.1 运行环境

* Python 3.10 或 3.11（实测环境为 3.10.4）；
* 依赖版本锁定于 `requirements.txt`：numpy 2.2.6 / pandas 2.3.3 / scipy 1.15.3 /
  matplotlib 3.10.9 / deepbullwhip 0.4.1 / pyyaml；
* DeepBullwhip 通过 PyPI 安装；其 GitHub 仓库不可访问，框架源码可通过
  `scripts/fetch_source.sh` 拉取至 `vendor/` 备查。

### 7.2 复现步骤

```bash
python scripts/quickstart_verify.py            # 环境自检，8 项全部通过
python -m pytest -q                            # 79 passed, 1 skipped
python agent/agent_wrapper.py                  # 全链路自测，4/4 通过
jupyter nbconvert --to notebook --execute --inplace notebooks/05_final_demo.ipynb
```

### 7.3 测试覆盖

| 测试文件 | 覆盖范围 |
|---|---|
| `tests/test_forecaster.py` | 数据加载、基线预测器、自研预测器、`(mean, std)` 契约与边界条件 |
| `tests/test_policy.py` | 拓扑构建、协同策略、信号聚合 |
| `tests/test_simulation.py` | 仿真循环、指标计算、基准跑批 |

累计 80 项用例，其中 79 项通过、1 项跳过（依赖真实 M5 数据的用例）。

### 7.4 模块接口

各模块通过 [`AB_interface_checklist.md`](AB_interface_checklist.md) §3 定义的冻结契约交互，
不依赖彼此的内部实现：

| 交接物 | 形态 |
|---|---|
| 需求数据 | 长表 DataFrame（`date` / `store_id` / `item_id` / `demand`）与 `ReplayDemandGenerator` |
| 预测 | `dict[节点名, (fm, fs)]`，两个 `(T,)` float64 数组；`fs` 允许为 0 |
| 仿真结果 | `NetworkSimulationResult`（`node_results` / `edge_flows` / `cumulative_bullwhip` / `total_cost`） |

---

## 8. 局限与后续工作

| # | 局限 | 影响 | 后续方向 |
|---|---|---|---|
| 1 | 顶层节点补货未建模 | 区域仓满足率恒为 0，需采用排除顶层的口径 | 在框架外补充供应商模型，或自研仿真循环 |
| 2 | 中间层无法直接使用外部预测 | 框架对中间节点强制采用近期下游订单均值 | 由中间节点策略对象自行持有预测序列（接口已预留） |
| 3 | 边级容量与运输成本不参与计算 | 方案未包含产能约束与运输成本建模 | 在策略层扩展约束 |
| 4 | 自研预测器点估计精度低于指数平滑 | 见 §4.3 | 引入季节性与促销特征，或改用学习型预测器 |
| 5 | 实验基于合成数据 | 结论尚未在真实数据上验证 | 接入 `data/raw/m5/sales_train_evaluation.csv` 后重跑全流程 |
| 6 | 相变临界仅做粗粒度扫描 | 最优 `alpha` 未精细定位 | 细化参数网格，并以多随机种子重复实验给出置信区间 |

---

## 9. 附录：接口定义

### 9.1 需求与预测输出

```python
# 需求：长表 [date, store_id, item_id, demand]
demand_df = load_synthetic_demand(n_nodes=4, n_days=365, seed=42)

# 预测：(mean, std)，两个 (T,) float64 数组；fs 允许为 0
forecast_map = build_forecast_bundle(demand_df)     # dict[节点名, (fm, fs)]
```

### 9.2 仿真结果

```python
NetworkSimulationResult:
    node_results: dict[str, EchelonResult]      # key = 节点名
    edge_flows:   dict[tuple[str, str], TimeSeries]
    cumulative_bullwhip: float
    total_cost: float

EchelonResult:
    name, orders, inventory_levels, costs      # 均为 (T,)
    bullwhip_ratio, fill_rate, total_cost      # 标量（fill_rate 口径见 §5.3）
```

### 9.3 全流程入口

```python
from agent.agent_wrapper import (
    prepare_demand, build_forecast_bundle, run_network_simulation,
)

bundle = prepare_demand(use_m5=False)                  # 需求（合成 AR(1)，seed=42）
fc     = build_forecast_bundle(bundle.demand_df)       # 预测 (mean, std)
base   = run_network_simulation(bundle, fc, coordinated=False)   # 基线
coord  = run_network_simulation(bundle, fc, coordinated=True)    # 协同
```

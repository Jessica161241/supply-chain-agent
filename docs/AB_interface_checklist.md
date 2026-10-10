# 拓扑 & 需求口径 核对清单（A ↔ B 对齐用）

> 赛题二·方向二「多级库存智能协同」
> 用途：A、B 双方在动手写代码前，逐条确认参数与接口口径，避免联调时才发现对不上。
> 标记：**主导** = 该项由谁拍板；**确认** = 谁需要签字认可。
> 状态：`[ ]` 未确认 / `[x]` 已确认（确认后在括号里写上名字和日期）

---

## 0. 怎么用这份清单

1. **先读第 1 节**。里面每一条都是读框架源码实测出来的，其中 3 条直接推翻了我们原文档里的写法，不先对齐会白写代码。
2. 第 2 节按模块分组，逐条过。**主导方**负责给出具体数值，**确认方**负责检查自己那侧的代码能吃下这个值。
3. 第 3 节是接口契约，**冻结后任何一方要改动，必须提前通知对方**。
4. 第 4 节是遗留待办。

---

## 1. ⚠️ 先对齐：7 条实测框架事实（3 条推翻了原文档）

以下结论全部来自实际读 `deepbullwhip 0.4.1` 源码 + 本机跑通验证，不是猜测。
源码可用 `bash scripts/fetch_source.sh --wheel` 拉到 `vendor/` 下自行复核。

### 1.1 ❗`Forecaster.forecast()` 返回的是**标量**，不是数组

`task.md` 第 128 行写的是：

```python
def forecast(self, demand_history, steps_ahead) -> Tuple[np.ndarray, np.ndarray]
```

**真实的抽象方法签名是**（`deepbullwhip/forecast/base.py`）：

```python
@abstractmethod
def forecast(
    self, demand_history: TimeSeries, steps_ahead: int = 1
) -> tuple[float, float]:
    """Return (forecast_mean, forecast_std) given history."""
```

差异：

| | 原文档写法 | 真实签名 |
|---|---|---|
| 返回类型 | `Tuple[np.ndarray, np.ndarray]` | `tuple[float, float]`（**两个标量**） |
| 语义 | 一次返回 N 天预测 | **只返回当期那一个点的** (mean, std) |

产出**整个 (T,) 数组**的是另一个方法 —— `generate_forecasts()`，它是 `Forecaster` 上的**具体方法**（不是抽象方法，不用重写）：

```python
def generate_forecasts(self, demand: TimeSeries) -> tuple[TimeSeries, TimeSeries]:
    """Run rolling forecast over a full demand series. 返回 (fm, fs)，形状都是 (T,)"""
```

> **对 A 同学的实际影响**：写 `custom_forecaster.py` 时，
> 只需要重写 `forecast(self, demand_history, steps_ahead=1)` 并返回两个 `float`；
> 要得到喂给仿真的 `(T,)` 数组，直接调用继承来的 `generate_forecasts(demand)`。
> **不要**按原文档写成返回数组，那样会与框架的调用方式对不上。
>
> 另注：`steps_ahead` 参数虽然存在，但框架自带的三个预测器
> （Naive / MovingAverage / ExponentialSmoothing）**都没有真正使用它**，
> 一律按 1 步预测处理。我们如果要做多步预测，属于超出框架默认行为，需自己实现。

- 主导：**A**（预测模块） 确认：**B**（要接住这个输出）
- [ ] 已对齐

### 1.2 ❗M5 加载只有 `store` + `dept` 两个维度，**没有 `item` 维度**

`task.md` 第 131 行写的接口是 `load_m5_demand(store_ids, item_ids)`。

框架实际提供的两个 M5 入口，签名分别是：

```python
# deepbullwhip/datasets/loader.py —— 读本地 CSV
load_dataset(name, freq="weekly", data_root=None, **kwargs)
    -> 对 m5 而言，kwargs 是 store="CA_1", dept="FOODS_1"

# deepbullwhip/datasets/m5.py —— 走 Kaggle 下载
load_m5(store="CA_1", dept="FOODS_1", freq="weekly", cache_dir=...)
```

内部实现是 `df["store_id"].eq(store) & df["dept_id"].eq(dept)`，**只按 store + dept 聚合**，没有 item 维度。

M5 的合法取值：

- `store`：`CA_1..CA_4`, `TX_1..TX_3`, `WI_1..WI_3`（共 10 个）
- `dept`：`FOODS_1..FOODS_3`, `HOBBIES_1..HOBBIES_2`, `HOUSEHOLD_1..HOUSEHOLD_2`（共 7 个）
- `freq`：`daily` / `weekly`（weekly 会把 daily 按 7 天求和；注意源码只实现了 daily/weekly/monthly 中的 weekly 和 monthly 聚合分支）

> **对 A 同学的实际影响**：`load_m5_demand(store_ids, item_ids)` 这个签名要改成
> **`dept_ids` 而不是 `item_ids`**。一次调用返回**一条**序列；
> 想要"一个终端门店的需求"，就把多条 (store, dept) 序列**求和**。
>
> **本阶段建议选型**（4 条序列 → 4 个终端门店，便于映射到 2 个城市仓）：
>
> | 终端节点 | 对应 M5 序列 | 归属 |
> |---|---|---|
> | store_1 | `CA_1` × `FOODS_1` | city_warehouse_1 |
> | store_2 | `CA_1` × `FOODS_2` | city_warehouse_1 |
> | store_3 | `CA_1` × `HOUSEHOLD_1` | city_warehouse_2 |
> | store_4 | `CA_1` × `HOUSEHOLD_2` | city_warehouse_2 |
>
> 好处：同一个 store（`CA_1`）内部不同品类，需求尺度接近，便于对比；
> 两个城市仓各带 2 个终端，便于演示"聚合信号上报"的效果。

- 主导：**A** 确认：**B**
- [ ] 已对齐

### 1.3 ❗`load_dataset()` **不负责下载**，且默认路径在 pip 安装下不存在

PyPI README 的示例 `load_dataset("m5", store="CA_1", dept="FOODS_1", freq="weekly")`
在 pip 安装环境里会直接抛 `FileNotFoundError`。原因：

```python
_DATA_ROOT = Path(__file__).parents[2] / "data" / "raw"
```

装到 `site-packages` 后，这个相对路径指向的根本不是我们的项目目录。

> **解决**：必须显式传 `data_root`：
> ```python
> load_dataset("m5", store="CA_1", dept="FOODS_1", freq="weekly",
>              data_root="data/raw")   # 读 data/raw/m5/sales_train_evaluation.csv
> ```
> 或者改用 `load_m5()`（它自带 Kaggle 下载 + 缓存逻辑，缓存到 `~/.deepbullwhip/data/m5/`）。
> 我们的 `scripts/quickstart_verify.py` 已经按"本地 CSV → Kaggle → 合成数据"三级降级写好了。

- 主导：**A** 确认：**B**
- [ ] 已对齐

### 1.4 🔴 `forecasts_mean / forecasts_std` **只对"面向需求的叶子节点"生效**

**这条最影响 A/B 接口，务必仔细看。**

`NetworkSupplyChain.simulate(demand, forecasts_mean, forecasts_std)` 内部逻辑
（`chain/network_sim.py`）：

```python
downstream = self.graph.downstream_neighbors(node_name)

if not downstream:
    # 面向需求的叶子节点（终端门店）：用外部传入的 demand / forecasts
    d  = float(demand[node_name][t])
    fm = float(forecasts_mean[node_name][t])
    fs = float(forecasts_std[node_name][t])
else:
    # 中间节点（城市仓、区域仓）：需求 = 下游订单之和，
    # 预测值由框架自己从"最近 8 期下游订单"算出来，跟外部传的 forecasts 无关！
    d  = sum(period_orders[dn] for dn in downstream)
    recent = all_downstream_orders[-min(8, len(all_downstream_orders)):]
    fm = float(np.mean(recent))
    fs = float(np.std(recent)) if len(recent) > 1 else 1.0
```

也就是说，在 3 层拓扑 `regional_warehouse → city_warehouse → store` 中：

| 节点 | 需求来源 | 预测来源 |
|---|---|---|
| `store`（叶子） | 外部传入的 M5 需求 | **A 同学的 mean/std** ✅ |
| `city_warehouse` | 下游 store 的订单之和 | ❌ 框架用"最近 8 期下游订单均值"顶替 |
| `regional_warehouse` | 下游 city 的订单之和 | ❌ 同上 |

**`task.md` 第 96 行写的**「将标准 mean/std 预测矩阵输出交付 B 同学，B 将预测注入
`NetworkSupplyChain` 各节点补货接口」—— **只有叶子节点成立**。

> **两条可选路线，必须二选一，这是本节最需要拍的板：**
>
> **路线甲（省事，接受框架默认）**：A 的预测矩阵只用于终端门店；
> 城市仓 / 区域仓用框架自己推出的"下游订单均值"。
> 代价：中间层没有真正的需求预测，"智能协同"的成色打折。
>
> **路线乙（推荐，但要 B 多写代码）**：B 在 `custom_policy.py` 里，让中间节点
> 的策略对象**自己持有** A 提供的预测序列（构造时注入），
> `compute_order()` 里用注入的预测而非 `forecast_mean` 参数。
> 这样中间层也能吃上 A 的多步预测。**这正好和第 1.5 节的"共享状态"机制是同一套写法。**
>
> 建议：**按路线乙设计接口，按路线甲先跑通**——即接口预留 `forecasts` 的注入位，
> 第一版先不用，跑通后再切。

- 主导：**A**（提供预测） + **B**（决定如何注入） 确认：双方
- [ ] 已选定路线（甲 / 乙）

### 1.5 ✅ 跨节点信号**可以传**，不必一上来就写 `manual_stepper.py`

这是本项目最大的技术风险点（`task.md` 第 20 行）。**已实测验证：框架支持。**

**原理**：`OrderingPolicy.compute_order(inventory_position, forecast_mean, forecast_std)`
的签名里**确实没有**跨节点参数的钩子。**但是**——

- 策略对象是在 `NetworkSupplyChain.__init__` 里**一次性构造**、整个仿真期间**常驻**的；
- `compute_order()` 每个时间步都会被调用一次。

所以只要在**构造策略时传入一个共享的可变对象**（下面叫 `SignalBus`），
各节点的策略就能通过这个共享对象互相读写，**完全不需要改框架**。

更关键的是**节点处理顺序天然正确**。`simulate()` 内部：

```python
sim_order = list(reversed(self._topo_order))   # 反向拓扑序 = 面向需求的节点排在最前
```

即每个时间步内：**先处理 store → 再处理 city_warehouse → 最后 regional_warehouse**。
所以 store 在本步上报的信号，city 在同一步就能读到 —— **正是我们要的"自下而上"信息流方向**。

**实测证据**（3 层拓扑，`regional → city → store`，T=40）

复现命令：`python docs/poc_cross_node_signal.py`

```
HOP 1  store -> city     : 非零读数 37 / 40 期
HOP 2  city  -> regional : 非零读数 37 / 40 期
```

（前 3 期读数为 0 是库存尚未消耗的启动瞬态，属正常。）

**注意**：上面这个实验只验证了**信号传递机制可行**，
它顺带暴露了一个值得警惕的现象 —— 朴素的协同规则**不一定改善总成本**：

```
node                  BW基线   BW协同     成本基线   成本协同
store                   3.29     3.29          66         66
city_warehouse          6.61     5.14  ↓       69         78  ↑
regional_warehouse      1.80     1.30  ↓       69         61  ↓
TOTAL                                        203        206  ↑
```

即：两个上游节点的牛鞭系数都降了，但**总成本反而升了**（203 → 206）。
**这说明协同策略的收益完全取决于规则设计，B 同学需要做参数调优
（如 `alpha` 系数、信号定义），不能默认"加了协同就一定更好"。**
这个对照实验也正好是路演里"我们做了 A/B 对比"的素材来源。

> 完整可运行实验已留档：`docs/poc_cross_node_signal.py`（含溯源注释）。
> 该文件是 A 同学的**可行性排查证据**，不是 B 的业务实现 ——
> 信号语义与隐私保护设计仍需 B 重新设计。

> **结论**：`manual_stepper.py` 从"大概率要写的兜底方案"降级为"**低优先级备选**"。
> 建议 B 同学**先按共享状态机制实现 `custom_policy.py`**，把兜底方案的时间省下来做策略调优。

- 主导：**B**（实现） 确认：**A**（把结论同步给 B）
- [ ] 已同步给 B 同学

### 1.6 ⚠️ 框架返回的 `fill_rate` **不是真正的订单满足率**

`NetworkSimulationResult.node_results[name].fill_rate` 的计算是：

```python
fill_rate = float(np.mean(inv >= 0))     # 库存非负的期数占比
```

`metrics/inventory.py` 里注册的 `FILL_RATE` 指标也是同一个口径。

**这不是**"已满足订单 / 总订单"，而是"有多少期库存没断过"。
用它来交赛题的 `Fill Rate` 指标，说服力不足，评委如果追问会站不住。

> **建议**：B 同学在 `run_benchmark.py` 里自行计算真正的 fill rate，
> 数据源是够的 —— `EchelonResult` 里有 `orders` / `inventory_levels`，
> `NetworkSimulationResult.edge_flows` 里有每条边的物流量。
> 可以在协同策略里顺带统计累计缺货量，用 `1 - 缺货量/总需求` 得到真值。
>
> **这条要写进提交材料的口径说明**，避免自研指标和框架指标混用被质疑。

- 主导：**B** 确认：**A**（指标对齐性校验是 A 的职责）
- [ ] 已对齐，且明确最终报告用哪个口径

### 1.7 ⚠️ `EdgeConfig.lead_time` **对仿真结果没有影响**

`NetworkSupplyChain.__init__` 构造各节点时用的是**节点自己的** `lead_time`：

```python
SupplyChainEchelon(name=name, lead_time=cfg.lead_time, ...)   # cfg 是 EchelonConfig
```

`EdgeConfig`（含 `lead_time` / `capacity` / `transport_cost`）在整个 `simulate()` 流程里
**从未被读取**，只用于拓扑关系（谁是谁的上下游）和画图。

> **对 B 同学的实际影响**：写 `network_config.yaml` 时，**补货提前期要配在节点上
> （`EchelonConfig.lead_time`），配在边上不生效**。
> 边上的 `capacity`（产能上限）和 `transport_cost`（运输成本）同样不参与计算，
> 如果方案里承诺了"运输成本"或"产能约束"，需要额外说明是自研扩展。

- 主导：**B** 确认：**A**
- [ ] 已对齐

---

## 2. 参数核对清单

> 3 层拓扑：`regional_warehouse` → `city_warehouse` → `terminal store`
> 下面所有 `EchelonConfig` / `EdgeConfig` 字段名均来自框架源码，可直接使用。

### 2.1 拓扑结构参数（B 主导，A 确认）

| # | 参数 | 建议值 | 说明 | 主导 | 确认 | 状态 |
|---|---|---|---|---|---|---|
| 1 | 区域仓数量 | 1 | 顶层单一供应源 | B | A | [ ] |
| 2 | 城市仓数量 | 2 | 让"聚合上报"有对照组 | B | A | [ ] |
| 3 | 终端门店数量 | 4 | 每城市仓带 2 个 | B | A | [ ] |
| 4 | 节点命名 | `regional_warehouse` / `city_warehouse_1,2` / `store_1..4` | 建议全小写下划线 | B | A | [ ] |
| 5 | 连接关系 | 1→2 全连，2→4 各带 2 | 分叉树 | B | A | [ ] |
| 6 | 边是否带容量约束 | 否 | 框架不支持，见 1.7 | B | A | [ ] |

### 2.2 节点级参数（B 主导，A 确认）

对应 `EchelonConfig(name, lead_time, holding_cost, backorder_cost, depreciation_rate, service_level, initial_inventory)`

| # | 参数 | 区域仓 | 城市仓 | 终端门店 | 主导 | 确认 | 状态 |
|---|---|---|---|---|---|---|---|
| 7 | `lead_time`（补货提前期，**配在节点上**） | 4 | 2 | 1 | B | A | [ ] |
| 8 | `holding_cost`（单位持有成本） | 0.08 | 0.15 | 0.20 | B | A | [ ] |
| 9 | `backorder_cost`（单位缺货成本） | 0.40 | 0.50 | 0.60 | B | A | [ ] |
| 10 | `service_level`（目标服务水平） | 0.95 | 0.95 | 0.95 | B | A | [ ] |
| 11 | `initial_inventory`（初始库存） | 50 | 50 | 50 | B | A | [ ] |
| 12 | `depreciation_rate`（折损率） | 0.0 | 0.0 | 0.0 | B | A | [ ] |

> 第 8/9 条的取值逻辑：下游离消费者越近，缺货代价越高（`backorder_cost` 递增），
> 持有成本同理。这与框架自带的 `beer_game_config()` / `consumer_2tier_config()` 的设定一致。
> 第 12 条若不做易腐品场景就保持 0；响应赛题"生鲜"叙事的话再考虑，**需 A/B 共同决定**。

- [ ] 第 7~12 条全部确认

### 2.3 仿真运行参数（A 主导，B 确认）

| # | 参数 | 建议值 | 说明 | 主导 | 确认 | 状态 |
|---|---|---|---|---|---|---|
| 13 | 仿真周期数 T | 365 天（daily） | 2026-10-04 拍板：对齐 config simulation_days，原 156 周口径作废 | A | B | [x] |
| 14 | 随机种子 `seed` | 42 | 全局固定，保证可复现 | A | B | [ ] |
| 15 | 频率 | daily | 对齐 config simulation_days=365（原 weekly 作废） | A | B | [x] |
| 16 | 评估指标 | BWR / Fill Rate / Total Cost | 赛题指定三项 | B | A | [ ] |
| 17 | 对比基线 | `order_up_to`（OUT） | 框架默认策略 | B | A | [ ] |

### 2.4 需求口径（A 主导，B 确认）

| # | 参数 | 值 | 说明 | 主导 | 确认 | 状态 |
|---|---|---|---|---|---|---|
| 18 | 数据集 | M5（兜底：合成 AR(1)） | | A | B | [ ] |
| 19 | store 维度 | `CA_1` | 见 1.2 选型表 | A | B | [ ] |
| 20 | dept 维度 | `FOODS_1` / `FOODS_2` / `HOUSEHOLD_1` / `HOUSEHOLD_2` | 共 4 条 | A | B | [ ] |
| 21 | 需求聚合方式 | 单条序列直接使用；多品类求和成"门店需求" | | A | B | [ ] |
| 22 | 缺失值 / 零值处理 | 保留 0（M5 无销量日即 0），不做插值 | **需 A/B 共同确认** | A | B | [ ] |
| 23 | 是否截断 | 取前 365 天，超出部分丢弃（对齐 config simulation_days=365） | 2026-10-04 A 按队长注意事项拍板 | A | B | [x] |
| 24 | 需求归一化 | 不归一化，保留原始销量单位 | 保证总成本可解释 | A | B | [ ] |
| 25 | 各终端需求映射 | store_1..4 分别对应第 20 条的 4 条序列 | | A | B | [ ] |

---

## 3. 接口契约（冻结项）

改动以下任一条，必须提前通知对方。

### 3.1 A → B：需求数据交接

```python
# src/data_loader.py —— 2026-10-04 按团队注意事项 §8 更新：返回长表
load_m5_demand(store_ids, item_ids) -> pd.DataFrame
    # 长表 [date, store_id, item_id, demand]；item_id 承载 dept 级序列 ID（框架无 item 维度）

load_synthetic_demand(n_nodes=5, n_days=365, seed=42) -> pd.DataFrame
    # 兜底长表；默认参数对齐 config 基准（sku_count=5 / simulation_days=365 /
    # demand.mean=10 / demand.std=3）

build_replay_generator(demand_df) -> ReplayDemandGenerator
    # 只接受 1-D 序列（data: np.ndarray | list[float]）；长表输入按 date 聚合求和，
    # 逐节点用 demand_df[item_id == xxx] 过滤后传入

pivot_demand_wide(demand_long_df) -> pd.DataFrame
    # 长表 -> 宽表（index=date，columns="{store_id}_{item_id}"），供按列映射拓扑节点
```

- 主导：**A** 确认：**B** — [ ]

### 3.2 A → B：预测矩阵交接

```python
# src/forecasting/custom_forecaster.py
class WeightedMAForecaster(Forecaster):
    def forecast(self, demand_history: TimeSeries, steps_ahead: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]:   # ← 数组，长度 == steps_ahead（见下方说明）
        ...

    def generate_forecasts(self, demand) -> tuple[TimeSeries, TimeSeries]:  # ← 重写基类，做数组→标量适配
        ...

# 得到整段 (T,) 预测数组（供仿真用）—— 只调这个
fm, fs = forecaster.generate_forecasts(demand)   # 两个都是 (T,)
```

约定：

- `fm` / `fs` 形状均为 `(T,)`，`float64`
- `fs` 允许为 0（历史长度不足 2 时框架就是这么返回的），**B 侧不能假设 `fs > 0`**
- 若走**路线乙**（1.4 节），A 额外提供 `dict[str, tuple[np.ndarray, np.ndarray]]`，
  key 是节点名，覆盖到中间层 —— 具体容器格式待 A/B 再定

#### ⚠️ `forecast()` 返回值口径差异（2026-10-11 实测复核，待双方签字）

| | 框架抽象签名 | 本类实现 |
|---|---|---|
| 声明/返回 | `tuple[float, float]`（两个标量） | `tuple[np.ndarray, np.ndarray]`，各长 `steps_ahead` |

本类采用**数组契约**以支持多步预测，代价是必须**重写基类 `generate_forecasts`**
（基类实现做 `fm[t], fs[t] = self.forecast(history)`，遇到数组会抛
"setting an array element with a sequence"）。

**2026-10-11 全代码库实测复核结论：该差异对 B 侧完全不可见，安全。** 依据：

1. 框架内**唯一**的预测调用点是 `benchmark/runner.py:160`
   → `fm, fs = forecaster.generate_forecasts(demand)`；
2. `forecast()` 仅被基类 `generate_forecasts` 内部调用（`forecast/base.py:58`），
   而该方法在本类中已被重写；
3. 框架自带的 `DeepARForecaster` 有**同样的重写先例**（`forecast/deepar.py:176,181`）。

因此 B 侧只需按 `generate_forecasts()` 的 `(T,)` 口径消费，无需感知此差异。
**本条仍待 B 同学签字确认**，确认后把"待确认"改为"已确认"。

- 主导：**A** 确认：**B** — [ ]

### 3.3 B → A：仿真结果交接

```python
NetworkSimulationResult:
    node_results: dict[str, EchelonResult]      # key = 节点名
    edge_flows:   dict[tuple[str,str], TimeSeries]
    cumulative_bullwhip: float
    total_cost: float

EchelonResult:
    name, orders, inventory_levels, costs,      # 均为 (T,)
    bullwhip_ratio, fill_rate, total_cost       # 标量
```

- A 用 `node_results[name].orders` / `.inventory_levels` 做预测误差与指标对齐性校验
- ⚠️ `fill_rate` 口径问题见 1.6
- 主导：**B** 确认：**A** — [ ]

### 3.4 文件归属（硬边界）

| 目录 / 文件 | 归属 | 另一方 |
|---|---|---|
| `src/data_loader.py` | A | B 只读 |
| `src/forecasting/*` | **A 全权** | B 只读 |
| `config/network_config.yaml` | B | A 只读、只提参数意见 |
| `src/network/*` | **B 全权** | A 只读 |
| `src/coordination/*` | **B 全权** | A 只参与"上报哪些聚合指标"的口径讨论，**不写实现代码** |
| `src/simulation/run_benchmark.py` | A+B 联调 | 共同 |
| `src/visualization/dashboard.py` | A+B | 共同，A 负责预测相关图表 |
| `agent/agent_wrapper.py` | A+B | 共同 |
| `tests/test_forecaster.py` | A | |
| `tests/test_policy.py` / `test_simulation.py` | B | |
| `docs/solution.md` / `architecture.md` / `setup_guide.md` / `acceptance_checklist.md` | A 主笔 | B 可补充自己章节 |
| `docs/AB_interface_checklist.md` | A 主导维护 | B 确认 |
| `docs/risk_log.md` | B | |
| `scripts/*` | A | |
| `README.md` / `README_A.md` | A | `README_A.md` 为 A 侧工作日志 |
| `requirements.txt` / `.gitignore` / `.gitattributes` | A | B 只读 |

---

## 4. 遗留待办

| # | 事项 | 负责人 | 状态 |
|---|---|---|---|
| 1 | **拍板 1.4 节：中间层预测走路线甲还是路线乙** | A + B | [ ] 目前按路线甲跑通 |
| 2 | 确认 2.4 第 22 条：M5 的 0 值是否保留 | A + B | [x] 保留 0、不插值（已实现） |
| 3 | 确认 1.6：最终提交的 Fill Rate 用自研口径还是框架口径 | B + A | [x] 双口径输出，对外主口径取 `true_fill_rate_excl_top`（`1 - 缺货量/需求量` 且排除顶层区域仓） |
| 4 | 确认是否做易腐品 / 折损场景（影响 2.2 第 12 条） | A + B | [x] 不做，`depreciation_rate=0` |
| 5 | 拿到真实 M5 数据（`data/raw/m5/sales_train_evaluation.csv`） | A | [ ] 未拿到，当前走合成兜底；拿到后 `--require-m5` 重跑 |
| 6 | B 交付 `network_config.yaml` + `topology.py` 后重跑 `quickstart_verify.py` 验证 | A | [x] 已重跑，8/8 通过（步骤 8 已由 SKIP 变为实跑） |
| 7 | `task.md` 第 82 行的 `docs/risk_shturl` 疑为 `risk_log.md` 的笔误 | A + B | [x] 确认为笔误，实际文件名 `docs/risk_log.md` |
| 8 | 签字确认 3.2 节 `forecast()` 数组返回口径差异 | B | [ ] 已实测复核安全，待 B 签字 |

---

## 5. 变更记录

| 日期 | 变更 | 来源 |
|---|---|---|
| 2026-09-28 | 初版；修正 `task.md` 中 Forecaster 签名、M5 item 维度两处错误；新增 forecasts 生效范围、跨节点信号可行性、fill_rate 口径、EdgeConfig 无效四条实测结论 | A 同学读框架源码 + 本机实测 |
| 2026-10-01 | 本文件曾在 git 拉取中丢失，由 A 同学按原文恢复；§3.2 的 forecast 返回类型与新任务要求（返回长度=steps_ahead 的数组）存在口径差异，恢复版保留原记录，**以 custom_forecaster.py 实际实现为准（数组契约），待 A/B 确认后更新本条** | A 同学 |
| 2026-10-04 | 按团队注意事项 §8（数据长表口径 + config 基准）更新 §3.1：data_loader 返回长表 [date, store_id, item_id, demand]，新增 pivot_demand_wide 长表→宽表工具；load_synthetic_demand 默认参数对齐 config（5 SKU×365 天、mean=10、std=3）；§2.3 的 156 周 weekly 口径与 config 365 天 daily 口径的差异仍待 A/B 拍板 | A 同学（按队长注意事项执行） |
| 2026-10-04 | 仿真时间口径拍板（A 依队长注意事项选 config 基准，task.md/旧清单的 156 周口径作废）：§2.3 第 13/15 条与 §2.4 第 23 条改为 **365 天 daily**——M5 加载切 daily 并截取前 365 天（M5_MAX_PERIODS=365，400 天 CSV 实测截断正确）；M5 选型仍为 4 条 dept 序列，是否扩到 sku_count=5 待 B 拍板 | A 同学 |
| 2026-10-11 | 代码冻结日复核：§3.2 补记 `forecast()` 数组返回口径差异的**全代码库调用路径实测结论**（框架只调 `generate_forecasts()`，本类已重写该方法，差异对 B 不可见；DeepAR 有同样先例），状态从"待 A/B 确认后更新"改为"待 B 签字"；§3.4 文件归属表补全 docs/scripts/README/仓库配置行；§4 遗留待办逐条结项（#2/#3/#4/#6/#7 已完成，#1/#5/#8 待办），新增第 8 条（数组契约签字） | A 同学 |

> 本文所有"实测"结论对应的源码位置：
> `deepbullwhip/forecast/base.py`、`deepbullwhip/policy/base.py`、
> `deepbullwhip/chain/network_sim.py`、`deepbullwhip/chain/echelon.py`、
> `deepbullwhip/chain/graph.py`、`deepbullwhip/datasets/{loader,m5}.py`、
> `deepbullwhip/metrics/inventory.py`、`deepbullwhip/_types.py`
>
> 复核方式：`bash scripts/fetch_source.sh --wheel` 之后到 `vendor/` 下比对。

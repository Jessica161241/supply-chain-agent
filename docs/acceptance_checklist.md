# 本阶段验收清单

> 赛题二 · 方向二「多级库存智能协同」
> 用途：交付前逐项自检。**每一行都有可执行的验证方式**，不靠"我觉得做完了"。
> 标记：`[x]` 已验证通过 / `[ ]` 待验证 / `[—]` 不适用
>
> 基线日期：2026-10-11（代码冻结）。复验命令统一在项目根执行，
> 先激活 venv 并清理失效代理（见 [`setup_guide.md`](setup_guide.md) §2）。

---

## A 组 · 环境与依赖（5 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| A1 | Python 版本合规 | `python -V` | `3.10.x` 或 `3.11.x` | [x] |
| A2 | 依赖按锁定版本装齐 | `python -m pip install -r requirements.txt` | 无报错；`pip check` 通过 | [x] |
| A3 | 框架可导入 | `python -c "import deepbullwhip as d; print(d.__version__)"` | `0.4.1` | [x] |
| A4 | 环境自检全过 | `python scripts/quickstart_verify.py` | `共 8 项，通过 8 项，失败 0 项`，退出码 0 | [x] |
| A5 | 框架源码可查阅 | `bash scripts/fetch_source.sh --wheel` | `vendor/deepbullwhip-wheel/deepbullwhip/` 存在 | [x] |

---

## B 组 · 数据加载（6 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| B1 | 长表四列契约 | `load_synthetic_demand().columns` | `[date, store_id, item_id, demand]` | [x] |
| B2 | 合成数据默认参数对齐 config | `load_synthetic_demand()` 形状 | `4 × 365 = 1460` 行，`n_nodes=4` / `n_days=365` | [x] |
| B3 | `seed` 逐位可复现 | 同 seed 调用两次比对 | 两次数值完全相同 | [x] |
| B4 | M5 三级降级 | 无 `data/raw/m5/` 时调 `load_m5_demand()` | 抛带修复提示的 `FileNotFoundError`，不崩溃 | [x] |
| B5 | 非法 M5 ID 被拦截 | `load_m5_demand("XX_9", ["FOODS_1"])` | 抛 `ValueError`（框架本身会静默返回全 0） | [x] |
| B6 | 长表 → 宽表 | `pivot_demand_wide(df).shape` | `(365, 4)`，列名 `SYN_SKU_1..4` | [x] |

---

## C 组 · 预测模块（8 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| C1 | 单点接口返回 `(steps_ahead,)` 数组 | `WeightedMAForecaster().forecast(hist, 1)` | 返回两个 shape `(1,)` 的 `float64` 数组（**已知偏差**，见文末） | [x] |
| C2 | **整段接口返回 `(T,)`** | `generate_forecasts(series)` | 两个数组形状均 `(T,)`、`float64` | [x] |
| C3 | **std 来自历史残差，不是写死的 0** | 检查 `fs` 是否随历史变化 | 历史 ≥ 2 期后 `fs > 0` 且逐期变化 | [x] |
| C4 | 短历史边界不崩 | 空序列 / 1 期序列 | 正常返回，`fs` 按框架约定回退 0.0 | [x] |
| C5 | 三种官方基线可跑 | `run_baseline_comparison(wide)` | 返回 Naive / MovingAverage / ExponentialSmoothing 三行 | [x] |
| C6 | 自研预测器入表 | `run_comparison_with_custom(wide)` | 返回 **4 行**（3 基线 + 1 自研），index 名为 `forecaster` | [x] |
| C7 | 误差表落盘 | `save_baseline_metrics(df)` | `outputs/results/baseline_metrics_with_custom.csv` 存在且可回读 | [x] |
| C8 | 注册表可实例化 | `deepbullwhip.get("forecaster", "weighted_ma")` | 返回 `WeightedMAForecaster` 类 | [x] |

---

## D 组 · 网络与协同（5 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| D1 | 拓扑节点/边数 | `build_network_from_config(cfg)` | 7 节点 / 6 边 | [x] |
| D2 | 连接关系正确 | 打印 `graph.edges` | 区域仓→2 城市仓，每城市仓→2 门店 | [x] |
| D3 | 拓扑序正确 | `graph.topological_order()` | 区域仓在前，门店在后 | [x] |
| D4 | **跨节点信号可传** | 跑协同仿真，读 `SignalBus.snapshot()` | 各节点均有非零信号快照 | [x] |
| D5 | **隐私保护成立** | 检查 `AggregatedSignal` 字段 | 只有 `gap` / `gap_std` / `n_sources`，无订单明细、无库存明细、无需求序列 | [x] |

---

## E 组 · 仿真与评测（A+B 共同，7 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| E1 | 基线仿真可跑 | `run_baseline_policy(net, demand, fm, fs)` | 返回 `NetworkSimulationResult`，`total_cost > 0` | [x] |
| E2 | **alpha=0 与基线逐位一致** | 比对 `total_cost` | 差值 `0.00e+00`（一致性自检） | [x] |
| E3 | alpha 扫描完整 | `run_comparison(alphas=(0.0,0.3,0.5,0.8,1.0))` | `summary_df` 6 行（baseline + 5 个 alpha） | [x] |
| E4 | 节点级指标完整 | `node_metrics_df` | 每策略 7 行 × 6 策略 = 42 行 | [x] |
| E5 | **fill rate 双口径都在** | 检查 `summary_df` 列 | 同时含 `framework_fill_rate` / `true_fill_rate` / `true_fill_rate_excl_top` | [x] |
| E6 | **主结果复现** | `coord(a=0.8).total_cost` | `16,576.44`（vs 基线 `22,563.92`，−26.5%） | [x] |
| E7 | 结果 CSV 落盘 | `ls outputs/results/benchmark_*.csv` | `benchmark_summary.csv` + `benchmark_node_metrics.csv` | [x] |

---

## F 组 · 单元测试与代码质量（4 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| F1 | 全量测试通过 | `python -m pytest -q` | `80 passed, 1 skipped`，无 failed | [x] |
| F2 | 预测模块测试覆盖 | `pytest tests/test_forecaster.py -q` | 覆盖数据加载 / 基线 / 自研 / 边界 | [x] |
| F3 | 各模块自测可跑 | `python src/data_loader.py`、`python agent/agent_wrapper.py` 等 | 末尾输出 `全部就绪` / 无 FAIL | [x] |
| F4 | A 模块有 docstring | 抽查 `src/data_loader.py`、`src/forecasting/*` | 每个公开函数都有 Parameters / Returns | [x] |

---

## G 组 · 文档、可视化与交付完整性（8 项）

| # | 验收项 | 验证方式 | 期望 | 状态 |
|---|---|---|---|---|
| G1 | 方案说明齐备 | `ls docs/solution.md` | 存在，含问题定义 / 方法 / 实验设置 / 结果 / 局限 | [x] |
| G2 | 架构文档与现状一致 | 通读 `docs/architecture.md` | 不再出现"B 待交付""TODO(B)"等过期表述 | [x] |
| G3 | 接口对齐清单在库 | `ls docs/AB_interface_checklist.md` | 存在，7 条实测事实完整 | [x] |
| G4 | 环境指南在库 | `ls docs/setup_guide.md` | 存在，含代理坑 / 三级降级 / FAQ | [x] |
| G5 | 最终 Demo notebook 可执行 | `jupyter nbconvert --execute notebooks/05_final_demo.ipynb` | 0 个 error 输出，图表内嵌 | [x] |

| G6 | 结果图表齐备 | `ls outputs/figures/*.png` | ≥ 6 张：需求总览 / 均值对比 / 不确定带 / 对比总览 / 牛鞭放大 / 节点成本 | [x] |
| G7 | 环境自检结果留档 | `ls outputs/results/env_check.json` | 存在，`n_failed = 0` | [x] |
| G8 | 骨架可重建 | `bash scripts/init_project_skeleton.sh --dry-run` | 幂等，不覆盖已有文件 | [x] |

---

## 复验一键脚本

```bash
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy=""   # 清失效代理
source .venv/Scripts/activate

python -m pytest -q
python scripts/quickstart_verify.py --json-out outputs/results/env_check.json
python agent/agent_wrapper.py
jupyter nbconvert --to notebook --execute --inplace notebooks/05_final_demo.ipynb
```

---

## 已知偏差（如实记录，不隐藏）

| # | 事项 | 现状 | 处理 |
|---|---|---|---|
| 1 | 顶层区域仓 fill rate 恒为 0 | `simulate()` 不建模顶层补货来源 | 对外主口径改用 `true_fill_rate_excl_top`，并在 `solution.md` §5.3 说明 |
| 2 | 框架自带 `fill_rate` 不是真满足率 | 口径为"库存非负期数占比" | 自研 `1 − 缺货量/需求量`，两个口径同时给出 |
| 3 | 自研预测器点估计精度略逊于指数平滑 | MAPE 19.24% vs 17.94% | 已在 `solution.md` §4.3 明确写出，并说明自研价值在 std 与可插拔性 |
| 4 | 实验数据为合成 AR(1)，非真实 M5 | 仓库不含 M5 原始 CSV（gitignore） | 数据口径已冻结，拿到 CSV 后 `--require-m5` 重跑即可 |
| 5 | 边上的容量 / 运输成本不生效 | 框架 `EdgeConfig` 未被 `simulate()` 读取 | 方案未承诺这两项；如需，做自研扩展 |
| 6 | notebooks 01/02 为"已清空输出"提交 | 便于 diff review | 需要展示时本地重新执行 |
| 7 | `WeightedMAForecaster.forecast()` 返回**数组**而非框架抽象签名声明的标量 | 框架抽象签名写 `tuple[float, float]`；本类返回 `(steps_ahead,)` 数组以支持多步预测，并**重写了 `generate_forecasts`** 做适配 | 已实测确认安全：框架全代码库只在 `benchmark/runner.py` 调 `generate_forecasts()`，`forecast()` 仅被基类 `generate_forecasts` 内部调用，而该方法已被重写；框架内 DeepAR 预测器有同样的重写先例。偏差记录于 `AB_interface_checklist.md` §3.2，**待 A/B 正式签字确认** |

---

## 变更记录

| 日期 | 变更 | 来源 |
|---|---|---|
| 2026-10-01 | 初版（在 git 拉取事故中丢失） | — |
| 2026-10-11 | 按当前实现重建：A~G 七组共 43 项，逐项附可执行验证方式；新增"已知偏差"一节 | — |

# README_A —— A 同学工作记录与交接

> 本文件是 A 侧的**工作日志与交接说明**，不是项目入口。
> 项目说明、快速开始、目录结构、文档索引请见 [README.md](README.md)。

**最近更新：2026-10-11（代码冻结日）**

---

## 1. 排期对照（对照《精确日程表与 AB 协同分工指南》）

| 日期 | 排期要求 | 状态 | 产物 |
|---|---|---|---|
| 9.21-9.22 | 环境搭建、跑通 Quick Start、锁定 3 层拓扑与需求口径 | ✅ | `.venv`、`requirements.txt`、`docs/AB_interface_checklist.md` |
| 9.23 | 精读教程 01+02，理解预测的 mean/std 输出要求 | ✅ | 对齐结论见 checklist §1.1 |
| 9.24-9.25 | 交付 `src/data_loader.py` | ✅ | 长表契约 + 三级降级 + Replay 生成 |
| 9.26-9.27 | 交付 `src/forecasting/baseline_forecasters.py` | ✅ | 3 基线 + MAE/RMSE/MAPE 对比 |
| 9.28-9.30 | 交付 `src/forecasting/custom_forecaster.py` | ✅ | `WeightedMAForecaster`（mean + 残差 std） |
| 10.1-10.3 | 单测 `tests/test_forecaster.py` + 代码注释 | ✅ | 34 用例 |
| 10.4-10.7 | 协助 B 对齐接口数据格式 | ✅ | 长表口径冻结；`pivot_demand_wide` 提供 |
| 10.8-10.9 | 【A+B 联调】`run_benchmark.py`，跑通完整仿真 | ✅ | B 主导，A 侧预测接入 |
| 10.10-10.11 | 【Demo 封版】`dashboard.py` + `agent_wrapper.py` + `05_final_demo.ipynb` | ✅ | 本文件末尾清单 |
| 10.12-10.13 | 共同完善 `architecture.md` 与打包 | ⏳ | 架构文档已更新；打包待做 |
| 10.14 | 直接提交，当天不写新代码 | ⏳ | — |

---

## 2. A 侧文件归属

| 文件 | 说明 |
|---|---|
| `src/data_loader.py` | M5 / 合成长表加载 + 清洗 + Replay 生成 + 长表↔宽表 |
| `src/forecasting/baseline_forecasters.py` | 3 种官方基线滚动预测 + 误差对比 + CSV 落盘 |
| `src/forecasting/custom_forecaster.py` | `WeightedMAForecaster` + `run_comparison_with_custom` |
| `tests/test_forecaster.py` | 34 用例：数据加载 / 基线 / 自研 / `(mean,std)` 契约 / 边界 |
| `src/visualization/dashboard.py` | **A 侧**预测绘图 3 函数（B 侧 3 函数归属 B） |
| `agent/agent_wrapper.py` | **A 侧** `prepare_demand` / `build_forecast_bundle`（B 侧已接入仿真） |
| `docs/solution.md` | 对外方案说明 |
| `docs/architecture.md` | 架构文档（A 章节；B 章节由 B 补充） |
| `docs/setup_guide.md` | 环境搭建指南 |
| `docs/acceptance_checklist.md` | 验收清单 |
| `docs/AB_interface_checklist.md` | A/B 对齐清单（A 主导维护） |
| `notebooks/01_quickstart_explore.ipynb`、`02_forecast_baseline.ipynb` | A 侧探索与预测基线调试 |
| `notebooks/05_final_demo.ipynb` | 最终演示（A 主笔整合 B 素材） |
| `scripts/quickstart_verify.py`、`fetch_source.sh`、`init_project_skeleton.sh` | 环境自检 / 源码查阅 / 骨架重建 |
| `requirements.txt`、`.gitignore`、`.gitattributes` | 依赖锁定与仓库配置 |

---

## 3. 交接给 B 的接口（冻结）

```python
# 需求：长表 [date, store_id, item_id, demand]
demand_df = prepare_demand(use_m5=False).demand_df

# 预测：dict[节点名, (fm, fs)]，两个 (T,) float64 数组；fs 允许为 0
forecast_map = build_forecast_bundle(demand_df)

# B 消费
result = net.simulate(demand, fm, fs)     # -> NetworkSimulationResult
```

改动上述任一契约必须提前通知对方。详见 [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md) §3。

---

## 4. 待 B 确认的三件事

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | 中间层预测走"路线甲"还是"路线乙" | 目前按路线甲跑通（中间层用框架的下游订单均值） | 若走路线乙，中间层也能吃 A 的多步预测，接口已预留 |
| 2 | `fill_rate` 最终用哪个口径提交 | 已双口径输出，对外主口径取 `true_fill_rate_excl_top` | 需在提交材料里统一说明 |
| 3 | `WeightedMAForecaster.forecast()` 返回数组而非标量 | 已实测确认框架调用路径安全（框架只调 `generate_forecasts`，且已重写） | 需 A/B 正式签字确认该偏差 |

---

## 5. 一键复验

```bash
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy=""   # 清失效代理
source .venv/Scripts/activate

python -m pytest -q                                                  # 79 passed, 1 skipped
python scripts/quickstart_verify.py --json-out outputs/results/env_check.json   # 8/8
python agent/agent_wrapper.py                                        # 4/4 PASS
python -m pytest tests/test_forecaster.py -q                         # 33 passed, 1 skipped
```

---

## 6. 事故记录

| 日期 | 事故 | 处理 |
|---|---|---|
| 2026-10-01 | git 拉取事故，`docs/setup_guide.md`、`docs/acceptance_checklist.md`、`scripts/` 全部丢失 | 2026-10-11 按当前实现重建，**以重建版为准**（见 README.md 文档索引） |
| 2026-10-10 | 本地代理 `127.0.0.1:7899` 失效，`git fetch` 报 TLS EOF | 用 `git -c http.proxy= -c https.proxy= ...` 绕过；已写入 `setup_guide.md` §2 |

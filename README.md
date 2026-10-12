# supply-chain-agent

> 赛题二 · 方向二「多级库存智能协同」
> 基于 [DeepBullwhip](https://pypi.org/project/deepbullwhip/) 0.4.1 供应链仿真框架

在「**区域仓 → 城市仓 → 终端门店**」三层供应链网络上，用**聚合缺口信号**实现跨节点补货协同，
对比基线策略与自研协同策略的 `Bullwhip Ratio`（牛鞭系数）、`Fill Rate`（订单满足率）、
`Total Cost`（总成本）。

**主结果**：总成本 **22,564 → 16,576（−26.5%）**，可优化层 fill rate **92.8% → 94.9%（+2.1pp）**，**不牺牲服务率**。

---

## 快速开始

```bash
# 0) 处理失效代理（本机必做，详见 docs/setup_guide.md 第 2 节）
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" NO_PROXY="*"

# 1) 建虚拟环境（必须 3.10 或 3.11，本机默认的 3.13 不合规）
py -3.10 -m venv .venv
source .venv/Scripts/activate        # Git Bash；macOS/Linux 用 .venv/bin/activate

# 2) 装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) 一键验证环境
python scripts/quickstart_verify.py
```

看到 `共 8 项，通过 8 项，失败 0 项 —— 环境就绪` 即环境就绪。

```bash
# 4) 跑测试与自测
python -m pytest -q                                    # 79 passed, 1 skipped
python agent/agent_wrapper.py                           # 4/4 PASS，A→B 全链路
```

---

## 目录结构

```
supply-chain-agent/
├── config/                 # network_config.yaml（拓扑参数）
├── data/
│   ├── raw/                # M5 原始数据（本次未启用；不进 git，仅 .gitkeep 占位）
│   └── processed/          # 处理好的时序数据
├── src/
│   ├── data_loader.py              # 需求加载与清洗（合成 AR(1) / M5）
│   ├── forecasting/
│   │   ├── baseline_forecasters.py     # 基线预测器误差对比
│   │   └── custom_forecaster.py        # 自研预测器，返回 (mean, std)
│   ├── network/
│   │   └── topology.py                 # 构建 3 层 NetworkSupplyChain
│   ├── coordination/               # 核心创新点
│   │   ├── signal_protocol.py          # 聚合缺口信号 + SignalBus
│   │   └── custom_policy.py            # 协同补货策略
│   ├── simulation/
│   │   └── run_benchmark.py            # 基线 vs 协同对比跑批
│   └── visualization/
│       └── dashboard.py                # 预测层 + 仿真对比层绘图
├── agent/agent_wrapper.py  # 一键全链路入口
├── notebooks/              # 01~05 探索与演示（05 为最终 Demo）
├── tests/                  # 单元测试
├── docs/                   # 文档（见下）
├── scripts/                # 环境自检 / 源码查阅 / 骨架重建
└── outputs/                # figures/ 图表、results/ 指标表
```

---

## 成果一览

| 指标 | baseline (OrderUpTo) | 协同 (alpha=0.8) | 变化 |
|---|---|---|---|
| 总成本 | 22,563.92 | **16,576.44** | **−26.5%** |
| 可优化层 fill rate（排除区域仓） | 92.79% | **94.86%** | **+2.07 pp** |
| 累计牛鞭系数 | 4.4828 | 5.0290 | +0.55 |

> 牛鞭系数升了但总成本降了 —— 三项指标必须联看。原因是协同把区域仓的消耗
> 压低（12,079 → 11,310），而缺货惩罚的下降幅度更大；只看牛鞭系数会得出
> "协同让系统更差"的错误结论。

---

## 文档索引

| 文档 | 内容 | 读者 |
|---|---|---|
| [docs/solution.md](docs/solution.md) | **对外方案说明**：问题定义、方法、实验设置与结果、局限 | 评委 / 全体 |
| [docs/setup_guide.md](docs/setup_guide.md) | **环境搭建完整指南**：版本、代理坑、安装、验证、需求数据、源码查阅、FAQ | 全体 |
| [docs/architecture.md](docs/architecture.md) | **架构文档**：数据流、模块清单、各模块详细设计 | 开发者 |
| [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md) | **A/B 对齐清单**：7 条实测框架事实 + 参数核对 + 接口契约 | A + B，**动手前必读** |
| [docs/acceptance_checklist.md](docs/acceptance_checklist.md) | **验收清单**：A~G 七组共 43 项硬指标 + 已知偏差 | 验收人 |
| [docs/risk_log.md](docs/risk_log.md) | **风险与排查日志**：框架行为排查记录 | 全体 |
| [docs/incident_log.md](docs/incident_log.md) | **事故记录**：两次影响交付的事故与处理方式 | 维护者 |
| [notebooks/05_final_demo.ipynb](notebooks/05_final_demo.ipynb) | **最终演示**（已执行，输出内嵌） | 评委 / 全体 |


---

## ⚠️ 开始写代码前，务必先看这两条

这两条推翻了原始需求文档里的写法，照抄会写出对不上的代码：

**1. `Forecaster.forecast()` 的真实签名返回标量，不是数组**

```python
# 真实抽象签名（deepbullwhip/forecast/base.py）
def forecast(self, demand_history: TimeSeries, steps_ahead: int = 1) -> tuple[float, float]:
    ...

# 要拿整段 (T,) 预测数组，用继承来的这个方法
fm, fs = forecaster.generate_forecasts(demand)
```

**2. M5 加载只有 `store` + `dept`，没有 `item`**

```python
load_dataset("m5", store="CA_1", dept="FOODS_1", freq="daily",
             data_root="data/raw")   # ← data_root 必须传，否则 FileNotFoundError
```

详细说明与其余 5 条实测结论见 [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md) 第 1 节。

---

## 常用命令

```bash
# 环境验证（可留档）
python scripts/quickstart_verify.py --json-out outputs/results/env_check.json

# 强制要求真实 M5（取不到则失败退出）
python scripts/quickstart_verify.py --require-m5

# 只测单条序列
python scripts/quickstart_verify.py --store CA_1 --dept FOODS_1

# 拉取框架源码备查（规避 GitHub 404），产物在 vendor/
bash scripts/fetch_source.sh --wheel

# 重建项目骨架（幂等，不覆盖已有文件）
bash scripts/init_project_skeleton.sh

# 跑测试
python -m pytest -q
```

---

## 关于框架 GitHub 仓库 404

`https://github.com/ai-vnv/deepbullwhip` 不可访问，但**不影响开发**：

- 装包走 PyPI：`pip install deepbullwhip` 正常（实测 0.4.1）
- 读源码用 `scripts/fetch_source.sh`（`pip download` 拿 wheel 解压，纯 Python 含全部源码）
- 官方文档站 `https://ai-vnv.github.io/deepbullwhip` 可访问

---

## 数据来源说明

本次实验的全部结果基于**合成 AR(1) 需求数据**生成：

| 项 | 值 |
|---|---|
| 生成方式 | `load_synthetic_demand(n_nodes=4, n_days=365, seed=42)` |
| 序列数 | 4（对应拓扑的 4 个终端门店） |
| 期数 | 365 天（daily，对齐 `simulation_days`） |
| 随机性 | AR(1)，`rho=0.6`、创新项 `sigma=2.4`，序列平稳 `std≈3`、`mean≈10` |
| 可复现性 | 固定 `seed=42`，同 seed 逐位一致 |

M5 真实数据的加载路径（本地 CSV → Kaggle 自动下载）已完整实现并保留，
**本次实验未启用**。拿到 `data/raw/m5/sales_train_evaluation.csv` 后，
把 `prepare_demand(use_m5=True)` 打开即可切换到真实数据重跑全流程。

---

## 模块接口

各模块之间只通过 [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md)
第 3 节定义的冻结契约交互，互不依赖内部实现：

| 交接物 | 形态 |
|---|---|
| 需求数据 | 长表 DataFrame（`date` / `store_id` / `item_id` / `demand`）+ `ReplayDemandGenerator` |
| 预测 | `dict[节点名, (fm, fs)]`，fm/fs 均为 `(T,)` float64；`fs` 允许为 0 |
| 仿真结果 | `NetworkSimulationResult`（`node_results` / `edge_flows` / `cumulative_bullwhip` / `total_cost`） |

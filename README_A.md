# supply-chain-agent

> 赛题二·方向二「多级库存智能协同」
> 基于 [DeepBullwhip](https://pypi.org/project/deepbullwhip/) 供应链仿真框架

多级库存智能协同系统：在「区域仓 → 城市仓 → 终端门店」三层供应链网络上，
用**聚合缺口信号**实现跨节点补货协同，并对比基线策略与自研协同策略的
`Bullwhip Ratio`（牛鞭系数）、`Fill Rate`（订单满足率）、`Total Cost`（总成本）。

---

## 快速开始

```bash
# 0) 处理失效代理（本机必做，详见 docs/setup_guide.md 第 2 节）
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" NO_PROXY="*"

# 1) 建虚拟环境（必须 3.10 或 3.11，本机默认的 3.13 不合规）
py -3.10 -m venv .venv
source .venv/Scripts/activate        # Git Bash

# 2) 装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) 一键验证环境
python scripts/quickstart_verify.py
```

看到 `共 8 项，通过 8 项，失败 0 项` 即环境就绪。

---

## 目录结构

```
supply-chain-agent/
├── config/                 # B同学：network_config.yaml（拓扑参数）
├── data/
│   ├── raw/                # M5 原始数据（不进 git）
│   └── processed/          # A同学：处理好的时序数据
├── src/
│   ├── data_loader.py              # 【A】M5 / 合成数据加载
│   ├── forecasting/                # 【A 全权】
│   │   ├── baseline_forecasters.py     # 基线预测器误差对比
│   │   └── custom_forecaster.py        # 自定义预测器，返回 (mean, std)
│   ├── network/                    # 【B 全权】
│   │   └── topology.py                 # 构建 3 层 NetworkSupplyChain
│   ├── coordination/               # 【B 全权】核心创新点
│   │   ├── signal_protocol.py          # 聚合缺口信号
│   │   ├── custom_policy.py            # 协同补货策略
│   │   └── manual_stepper.py           # 兜底手写仿真循环
│   ├── simulation/                 # 【A+B 联调】
│   │   └── run_benchmark.py
│   └── visualization/              # 【A+B 联调】
│       └── dashboard.py
├── agent/agent_wrapper.py  # 【A+B】全流程 Agent 封装
├── notebooks/              # 01~05 探索与演示
├── tests/                  # 单元测试
├── docs/                   # 文档（见下）
├── outputs/                # figures/ 图表、results/ 指标表
└── scripts/                # 环境验证与工具脚本
```

---

## 文档索引

| 文档 | 内容 | 读者 |
|---|---|---|
| [docs/setup_guide.md](docs/setup_guide.md) | **环境搭建完整指南**：版本、代理坑、安装、验证、M5 数据、源码查阅、FAQ | 全体 |
| [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md) | **A/B 对齐清单**：7 条实测框架事实 + 参数核对 + 接口契约 | A + B，**动手前必读** |
| [docs/acceptance_checklist.md](docs/acceptance_checklist.md) | **本阶段验收清单**：A~G 七组共 30+ 项硬指标 | 验收人 |
| [docs/risk_log.md](docs/risk_log.md) | **风险与排查日志**：9 条已查明的框架行为与坑 | 全体 |

> 状态（2026-10-01）：`AB_interface_checklist.md` 与 `architecture.md` 在库；
> `setup_guide.md` / `acceptance_checklist.md` / `risk_log.md` 在 git 拉取事故中丢失，待重建。

---

## ⚠️ 开始写代码前，务必先看这两条

这两条推翻了原始需求文档里的写法，照抄会写出对不上的代码：

**1. `Forecaster.forecast()` 返回标量，不是数组**

```python
# 真实签名（deepbullwhip/forecast/base.py）
def forecast(self, demand_history: TimeSeries, steps_ahead: int = 1) -> tuple[float, float]:
    ...

# 要拿整段 (T,) 预测数组，用继承来的这个方法（不用自己写）
fm, fs = forecaster.generate_forecasts(demand)
```

**2. M5 加载只有 `store` + `dept`，没有 `item`**

```python
load_dataset("m5", store="CA_1", dept="FOODS_1", freq="weekly",
             data_root="data/raw")   # ← data_root 必须传，否则 FileNotFoundError
```

详细说明与其余 5 条实测结论见 [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md) 第 1 节。

---

## 常用命令

> ⚠️ 注：`scripts/` 目录（quickstart_verify.py 等）在 2026-10-01 的 git 拉取事故中丢失，下列命令待脚本重建后可用。

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
pytest
```

---

## 关于框架 GitHub 仓库 404

`https://github.com/ai-vnv/deepbullwhip` 不可访问，但**不影响开发**：

- 装包走 PyPI：`pip install deepbullwhip` 正常（实测 0.4.1）
- 读源码用 `scripts/fetch_source.sh`（`pip download` 拿 wheel 解压，111KB 纯 Python 含全部源码）
- 官方文档站 `ai-vnv.github.io/deepbullwhip` 可访问

---

## 团队分工边界

| 角色 | 负责 |
|---|---|
| **A 同学** | 环境搭建、`data_loader.py`、`forecasting/*`、`tests/test_forecaster.py`、文档；**不写** `network/*` 与 `coordination/*` |
| **B 同学** | `config/network_config.yaml`、`network/topology.py`、`coordination/*`、`tests/test_policy.py` |
| **共同** | `simulation/run_benchmark.py`、`visualization/dashboard.py`、`agent/agent_wrapper.py` |

> 边界为硬性约束。A 不写协同策略实现，B 不写预测实现，双方只通过
> [docs/AB_interface_checklist.md](docs/AB_interface_checklist.md) 第 3 节的接口契约交互。

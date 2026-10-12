# 环境搭建完整指南

> 赛题二 · 方向二「多级库存智能协同」
> 读者：全体成员、验收人。**从零开始照做即可跑通全流程。**

---

## 1. 版本要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.10 或 3.11** | 赛题硬约束。本机默认的 3.13 不合规，必须显式指定版本建 venv |
| 操作系统 | Windows / macOS / Linux 均可 | 本机验证环境为 Windows 11 |
| DeepBullwhip | `0.4.1` | 走 PyPI 安装（其 GitHub 仓库 404，不影响安装） |

验证环境实测：**Python 3.10.4 + deepbullwhip 0.4.1**，`pytest` 79 passed / 1 skipped。

---

## 2. ⚠️ 先处理失效代理（本机必做）

这是本机最高频的故障源：**pip 装任何包都失败，但 curl / 浏览器正常。**

### 症状

```
pip install <任何包>
  ...
  SSLError: HTTPSConnectionPool(host='pypi.org', port=443)
  [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
```

### 原因

本机配了一个本地代理 `http://127.0.0.1:7899`（git 与 shell 环境变量里都有），
该代理进程已经不在运行，于是所有走代理的连接在 TLS 握手阶段被对端直接切断。

检查方式：

```bash
env | grep -i proxy
git config --get http.proxy      # 本机输出 http://127.0.0.1:7899
curl -sv https://pypi.org 2>&1 | grep -i "proxy\|connect"
```

### 解决

**每一步都用一次性清除，不改全局配置**（避免影响你其他项目的代理设置）：

```bash
# ① shell 环境变量（pip / curl 用这个）
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" NO_PROXY="*"

# ② git 的代理（git 不读上面的环境变量，走自己的 config）
git -c http.proxy= -c https.proxy= pull origin main
git -c http.proxy= -c https.proxy= fetch origin

# ③ 确认生效
pip config list
```

如果确认本地代理短期内不会恢复，可以一次性清掉 git 的全局配置：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

> **给队友的一句话**：`pip` 报 SSL/EOF 错，先清代理，别急着怀疑镜像源。

---

## 3. 创建虚拟环境

```bash
cd supply-chain-agent

# ① 用 3.10 建 venv（Windows 用 py 启动器指定版本）
py -3.10 -m venv .venv

# ② 激活
source .venv/Scripts/activate        # Git Bash（Windows）
# source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate             # PowerShell

# ③ 确认解释器是刚建的那个
which python
python -V                             # 必须是 3.10.x
```

---

## 4. 安装依赖

```bash
# 先清代理（见第 2 节）
export HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" NO_PROXY="*"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 是**锁版本的 `pip freeze` 输出**，按它重建的环境与"通过全部测试的验证环境"一致。

> 版本漂移的坑：本文件曾被按其它机器的环境重写过（`pandas==3.0.6` / `numpy==2.5.3`），
> 与实际验证环境不符，已改回实测版本（`pandas==2.3.3` / `numpy==2.2.6`）。
> **如果团队要升 pandas 3.x，先对 A 模块做回归测试再改。**

### 可选：跑 notebook 需要额外装

`requirements.txt` 只锁了 `ipykernel`。要用 `jupyter nbconvert` 命令行执行 notebook，
还需要装：

```bash
python -m pip install nbformat nbclient nbconvert
```

（在 Jupyter 界面里手动点开、逐 cell 运行则不需要。）

---

## 5. 一键验证环境

```bash
python scripts/quickstart_verify.py
```

看到下面这行即环境就绪：

```
  共 8 项，通过 8 项，失败 0 项 —— 环境就绪
```

八项检查：Python 版本 / 依赖包版本 / 关键 API 导入 / 数据集加载（合成 / M5 三级降级）/
`ReplayDemandGenerator` 转换 / Serial 仿真 / Network(DAG) 仿真 / 本项目 3 层拓扑。

常用参数：

```bash
# 留档 JSON（验收材料）
python scripts/quickstart_verify.py --json-out outputs/results/env_check.json

# 强制要求真实 M5，取不到就失败退出（CI / 验收用）
python scripts/quickstart_verify.py --require-m5

# 只测单条序列
python scripts/quickstart_verify.py --store CA_1 --dept FOODS_1

# 只输出末尾汇总
python scripts/quickstart_verify.py --quiet
```

退出码：`0` = 全过；`1` = 有失败项。

再跑一遍单元测试：

```bash
python -m pytest -q        # 期望 79 passed, 1 skipped
```

---

## 6. 需求数据

### 6.1 本次实验的数据来源：合成 AR(1)

**本次提交的全部实验结果都基于合成 AR(1) 数据**，不需要任何外部数据：

```python
from src.data_loader import load_synthetic_demand
df = load_synthetic_demand(n_nodes=4, n_days=365, seed=42)
```

| 项 | 值 |
|---|---|
| 序列数 | 4（对应拓扑的 4 个终端门店） |
| 期数 | 365 天（daily，对齐 `config.simulation_days`） |
| 统计量 | `mean≈10`、平稳 `std≈3`（AR(1)，`rho=0.6`、创新项 `sigma=2.4`） |
| 可复现性 | 固定 `seed=42`，同 seed 逐位一致 |
| 序列命名 | `store_id="SYN"`、`item_id="SKU_1..4"`，宽表列名 `SYN_SKU_1..4` |

因此 `quickstart_verify.py` 输出里的标签是 `SYN/SKU_1` 而不是 `CA_1/FOODS_1`——
这是正常的，说明当前跑的是合成数据。

### 6.2 M5 真实数据（加载路径已实现，本次未启用）

`src/data_loader.py` 保留了 M5 的完整三级降级路径，任何一级成功都能继续：

```
① data/raw/m5/sales_train_evaluation.csv（本地 CSV，最快）
        v 找不到
② Kaggle 自动下载（缓存 ~/.deepbullwhip/data/m5/）
        v 拿不到凭证
③ 合成 AR(1)（seed 可复现）—— 本次实验用的就是这一级
```

要用真实 M5 重跑，按下面任一条准备数据即可。

#### 走 ① 本地 CSV

```bash
mkdir -p data/raw/m5
# 把 Kaggle M5 数据集的 sales_train_evaluation.csv 放进去
ls data/raw/m5/sales_train_evaluation.csv
```

> `data/raw/` 被 `.gitignore` 忽略，**数据文件不进 git**（只有 `.gitkeep` 占位）。
> 换机器需要重新放置。

### 走 ② Kaggle 自动下载

```bash
python -m pip install kaggle
export KAGGLE_USERNAME="你的用户名"
export KAGGLE_KEY="你的 API Key"
```

### 走 ③ 合成数据（本次实验使用）

不用做任何事，见 §6.1。

### M5 数据口径（切换到真实数据时适用）

| 项 | 值 |
|---|---|
| 序列选择 | store `CA_1` × dept `FOODS_1` / `FOODS_2` / `HOUSEHOLD_1` / `HOUSEHOLD_2`（共 4 条） |
| 维度 | 框架 M5 加载**只有 store + dept**，没有 item 维度，`item_id` 承载 dept 级序列 ID |
| 频率 / 长度 | daily，截取前 **365 天**（对齐 `config.simulation_days`） |
| 零值与缺失 | NaN/Inf 记 0、负值截断为 0、**不做插值**（M5 无销量日即 0，是真实信息） |

切换方式：把 `prepare_demand(use_m5=True)` 打开，或用
`python scripts/quickstart_verify.py --require-m5` 强制校验 M5 路径。

---

## 7. 查阅框架源码

`https://github.com/ai-vnv/deepbullwhip` 返回 404，但**不影响开发**。用脚本从 PyPI 拉：

```bash
bash scripts/fetch_source.sh --wheel
```

产物：

```
vendor/deepbullwhip-0.4.1-py3-none-any.whl          # wheel 本体
vendor/deepbullwhip-wheel/deepbullwhip/...          # 解压后的纯 Python 源码
```

脚本会打印各实测结论对应的源码位置，例如：

| 源码 | 对应结论 |
|---|---|
| `forecast/base.py` | §1.1 `forecast()` 返回标量，不是数组 |
| `datasets/loader.py` | §1.2 M5 只有 store + dept |
| `chain/network_sim.py` | §1.4 forecasts 只对叶子节点生效；§1.5 反向拓扑序 ⇒ 跨节点信号可行 |
| `metrics/inventory.py` | §1.6 `fill_rate` 是"库存非负期数占比"，不是真满足率 |
| `chain/echelon.py` | §1.7 `EdgeConfig.lead_time` 对仿真结果无影响 |

（§ 编号对应 [`AB_interface_checklist.md`](AB_interface_checklist.md) 第 1 节。）

官方文档站 `https://ai-vnv.github.io/deepbullwhip` 可正常访问，适合查 API 用法。

---

## 8. 重建项目骨架

误删了空目录或空包时：

```bash
bash scripts/init_project_skeleton.sh              # 幂等，绝不覆盖已有文件
bash scripts/init_project_skeleton.sh --dry-run    # 只报告将要创建什么
```

它会补齐目录树、缺失的 `__init__.py`、以及 `data/raw/.gitkeep` / `data/processed/.gitkeep`
（后两者让空数据目录能被 git 跟踪）。

---

## 9. FAQ

**Q：`pip install` 报 `SSL: UNEXPECTED_EOF_WHILE_READING`？**
A：代理失效。见第 2 节，先 `export HTTP_PROXY="" ...`。

**Q：`git pull` 报 `TLS connect error: unexpected eof`？**
A：同上，但 git 不读环境变量，要用 `git -c http.proxy= -c https.proxy= pull`。

**Q：`python -V` 显示 3.13 / 3.12？**
A：venv 没用 3.10 建。删掉 `.venv` 重新 `py -3.10 -m venv .venv`。

**Q：`load_dataset("m5", ...)` 报 `FileNotFoundError`？**
A：pip 安装下框架的默认 `_DATA_ROOT` 指向 `site-packages` 里的错误位置。
必须显式传 `data_root`，或改用 `load_m5()`。本项目 `data_loader` 已封装好降级逻辑。

**Q：控制台中文显示成乱码？**
A：Windows 默认 GBK 控制台。本项目脚本内部已做 `errors="replace"` 兜底，
不影响功能。想彻底解决可执行 `chcp 65001`。

**Q：`jupyter nbconvert --execute` 报 `No module named 'deepbullwhip'`？**
A：nbconvert 用了系统里的另一个 Python 解释器，不是项目 venv。
显式指定内核：先注册 venv 内核，再用 `--ExecutePreprocessor.kernel_name=<名字>`。

**Q：`scripts/*.sh` 报 `bad interpreter: /usr/bin/env: No such file`？**
A：行尾被 checkout 成了 CRLF。仓库根有 `.gitattributes` 强制 `.sh` 为 LF；
若历史文件已被转换，执行 `git add --renormalize .` 后重新 checkout。

---

## 10. 相关文档

| 文档 | 内容 |
|---|---|
| [`AB_interface_checklist.md`](AB_interface_checklist.md) | A/B 对齐清单：7 条实测框架事实 + 参数核对 + 接口契约 —— **动手前必读** |
| [`architecture.md`](architecture.md) | 架构文档：数据流、模块清单、A 模块详细设计 |
| [`solution.md`](solution.md) | 对外方案说明：问题定义、方法、实验设置与结果、局限 |
| [`acceptance_checklist.md`](acceptance_checklist.md) | 本阶段验收清单（A~G 七组硬指标） |
| [`risk_log.md`](risk_log.md) | 框架行为排查日志 |

# -*- coding: utf-8 -*-
"""M5 数据集 / 合成需求数据加载，输出 ReplayDemandGenerator

职责
----
1. 封装 M5 数据集子集的加载与基础清洗（NaN 处理、时间序列连续性保证）；
2. 提供合成需求数据集生成函数，作为 M5 数据不可用时的调试兜底；
3. 把需求 DataFrame 转成 DeepBullwhip 框架的 ``ReplayDemandGenerator``，
   该对象是需求数据交给网络仿真模块的交接点。

> 本次实验的数据来源：**合成 AR(1)，seed=42**。M5 真实数据的加载路径已实现
> 并保留（见下），但本次未启用。

对外接口（参数名与顺序与任务书严格一致；返回格式按团队注意事项口径，见 §长表契约）
----
::

    load_m5_demand(store_ids, item_ids)          -> pd.DataFrame（长表）
    load_synthetic_demand(n_nodes, n_days, seed) -> pd.DataFrame（长表）
    build_replay_generator(demand_df)            -> ReplayDemandGenerator
    pivot_demand_wide(demand_long_df)            -> pd.DataFrame（长表 → 宽表工具）

长表契约（团队注意事项 §8：列名统一为 date / store_id / item_id / demand）
----
* 返回 DataFrame 固定四列：``date``（datetime64，期初日期）、``store_id``（str）、
  ``item_id``（str）、``demand``（float64），按 date → store_id → item_id 排序；
* **item_id 承载 M5 的 dept 级序列 ID**（FOODS_1..3 / HOBBIES_1..2 / HOUSEHOLD_1..2）
  ——框架 loader 只有 store+dept 两个维度、没有 item 维度（对齐清单 §1.2）；
* ``node_id`` 不在此列：门店/SKU → 拓扑节点的映射属拓扑配置，
  仿真层可用 ``pivot_demand_wide`` 得到"一列一个序列"的宽表后按列名映射；
* 供预测/绘图等内部流程使用的宽表：``pivot_demand_wide`` 一键转换
  （index = date，columns = "{store_id}_{item_id}"）。

合成数据默认参数（对齐 config/network_config.yaml 基准）
----
* ``sku_count=4``  → n_nodes 默认 4（对应 4 个终端门店）；
* ``simulation_days=365`` → n_days 默认 365，date 为逐日日期；
* ``demand.mean=10`` → AR(1) 均值 mu=10；
* ``demand.std=3`` → 需求序列平稳标准差 3。AR(1) 的平稳 std = sigma/sqrt(1-rho²)，
  rho=0.6 时取创新项 sigma=2.4 使序列 std≈3；
* 每个节点 seed+i 差异化，同 seed 逐位可复现。

框架事实（读本地 deepbullwhip 0.4.1 源码实测，非文档转述）
----
* ``deepbullwhip.datasets.load_dataset(name, freq, data_root, **kwargs)``：
  对 M5 只接受 ``store`` / ``dept`` 两个筛选维度，**没有 item 维度**；
  返回 1-D ``np.ndarray``（daily / weekly / monthly 聚合）。
* pip 安装下 ``load_dataset`` 默认数据目录指向 site-packages 内部，必然
  ``FileNotFoundError``；必须显式传 ``data_root=<项目>/data/raw``，
  读 ``data/raw/m5/sales_train_evaluation.csv``。
* ``ReplayDemandGenerator.__init__(data)`` 只接受 **1-D** 历史需求
  （``np.ndarray | list[float]``，见 deepbullwhip/demand/replay.py）；
  ``generate(T)`` 超长自动循环填充，批量路径叠加 5% 高斯噪声。
* ``deepbullwhip.datasets.synthetic.load_ar1(T, mu, rho, sigma, seed)``：
  框架自带 AR(1) 合成需求，内部对负值截断（下限 0.1）。

M5 日期与周期：M5 日历 d_1 = 2011-01-29（公开事实）；daily 口径下第 k 天的
date 记为锚点 + k 天，截取前 365 天对齐 config 的 simulation_days=365
（2026-10-04 拍板，旧的 156 周 weekly 口径作废，见对齐清单 §2.3/§3.1）。
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from deepbullwhip.datasets import load_dataset
from deepbullwhip.datasets.m5 import load_m5
from deepbullwhip.datasets.synthetic import load_ar1
from deepbullwhip.demand.replay import ReplayDemandGenerator

__all__ = [
    "load_m5_demand",
    "load_synthetic_demand",
    "build_replay_generator",
    "pivot_demand_wide",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 项目根目录（src/data_loader.py 的上一级），与 CWD 无关
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# M5 原始数据目录。可用环境变量 M5_DATA_ROOT 覆盖（不影响函数签名）。
M5_DATA_ROOT = Path(os.environ.get("M5_DATA_ROOT", str(PROJECT_ROOT / "data" / "raw")))

# 聚合频率与周期数：daily、截取前 365 天——对齐 config/network_config.yaml 的
# simulation_days=365（2026-10-04 拍板：task.md/对齐清单旧版的 156 周 weekly 口径作废）。
M5_FREQ = "daily"
M5_MAX_PERIODS = 365

# M5 日历锚点：d_1 = 2011-01-29（公开事实），daily 第 k 天 date = 锚点 + k 天
M5_ANCHOR_DATE = pd.Timestamp("2011-01-29")

# 长表固定列序
LONG_COLUMNS = ["date", "store_id", "item_id", "demand"]

# 合法取值（10 个门店 × 7 个品类）。框架 loader 对非法值不报错、
# 而是静默返回全 0 序列（空 mask 求和），所以这里必须先做白名单校验。
M5_STORES = frozenset(
    {f"CA_{i}" for i in range(1, 5)}
    | {f"TX_{i}" for i in range(1, 4)}
    | {f"WI_{i}" for i in range(1, 4)}
)
M5_DEPTS = frozenset(
    {f"FOODS_{i}" for i in range(1, 4)}
    | {f"HOBBIES_{i}" for i in range(1, 3)}
    | {f"HOUSEHOLD_{i}" for i in range(1, 3)}
)

# M5 完整 item_id 形如 FOODS_3_586（品类_序号_商品号），用于识别误传 item 级 ID
_ITEM_ID_RE = re.compile(r"^[A-Z]+_\d+_\d+$")

# 合成数据默认参数 —— 对齐 config/network_config.yaml 基准：
#   sku_count=4 -> n_nodes 默认 4（对应拓扑的 4 个终端门店）；
#   simulation_days=365 -> n_days 默认 365（逐日）；
#   demand.mean=10 -> AR(1) 均值 mu=10；
#   demand.std=3 -> 序列平稳 std=3。AR(1) 平稳 std = sigma/sqrt(1-rho²)，
#   rho=0.6 时创新项 sigma = 3 * sqrt(1-0.36) = 2.4。
_SYN_DEFAULT_NODES = 4
_SYN_DEFAULT_DAYS = 365
_SYN_MU = 10.0
_SYN_RHO = 0.6
_SYN_SIGMA = 2.4  # 创新项 std：2.4 / sqrt(1-0.36) = 3.0 = 序列平稳 std（config demand.std）
_SYN_STORE_ID = "SYN"
_SYN_START_DATE = "2024-01-01"  # 合成数据无真实日历，取固定锚点逐日生成


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _as_id_list(value: str | Sequence[str], what: str) -> list[str]:
    """把 str / 序列 统一成去重、去空、大写的 ID 列表。

    Parameters
    ----------
    value : str or Sequence[str]
        单个 ID 或 ID 序列。
    what : str
        参数名，仅用于报错信息。

    Returns
    -------
    list[str]
        清洗后的 ID 列表（保持原顺序，去重）。

    Raises
    ------
    TypeError
        ``value`` 不是 str 或字符串序列时。
    ValueError
        清洗后列表为空时。
    """
    if isinstance(value, str):
        raw: list[str] = [value]
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raise TypeError(f"{what} 应为 str 或 str 序列，收到 {type(value).__name__}")
    ids: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(f"{what} 中含有非字符串元素: {item!r}")
        token = item.strip().upper()
        if token and token not in ids:
            ids.append(token)
    if not ids:
        raise ValueError(f"{what} 不能为空")
    return ids


def _clean_series(values: np.ndarray | pd.Series | Sequence[float]) -> np.ndarray:
    """基础清洗：转 float64 一维，NaN/Inf 记 0，负值截断为 0。

    NaN 记 0 的口径与对齐清单 §2.4 第 22 条一致（M5 无销量日即 0，不做插值），
    同时保证时间轴上不出现空洞（date 连续由调用方按锚点日期生成保证）。

    Parameters
    ----------
    values : array-like
        原始需求序列。

    Returns
    -------
    np.ndarray
        shape (T,) 的 float64 数组，全部有限且 >= 0。
    """
    arr = np.array(values, dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = 0.0
    np.clip(arr, 0.0, None, out=arr)
    return arr


def _is_long_format(demand_df: pd.DataFrame) -> bool:
    """判断 DataFrame 是否为长表格式（date + demand 必备，且带 store_id/item_id 之一）。

    Parameters
    ----------
    demand_df : pd.DataFrame
        待判断的需求表。

    Returns
    -------
    bool
        True 表示长表（date / store_id / item_id / demand 四列口径）。
    """
    cols = set(demand_df.columns)
    return {"date", "demand"} <= cols and bool(
        {"store_id", "item_id"} & cols
    )


def _validate_m5_pairs(store_ids: str | Sequence[str],
                       item_ids: str | Sequence[str]) -> list[tuple[str, str]]:
    """校验并展开 (store, dept) 组合（笛卡尔积，保序去重）。

    Parameters
    ----------
    store_ids : str or Sequence[str]
        M5 门店 ID，如 ``"CA_1"`` 或 ``["CA_1", "TX_1"]``。
    item_ids : str or Sequence[str]
        M5 dept 级序列 ID，如 ``["FOODS_1", "HOUSEHOLD_1"]``。
        注意框架只支持到 dept 级（对齐清单 §1.2）。

    Returns
    -------
    list[tuple[str, str]]
        去重后的 (store, dept) 组合。

    Raises
    ------
    ValueError
        ID 非法、误传 item 级 ID 或组合为空时。
    """
    stores = _as_id_list(store_ids, "store_ids")
    depts = _as_id_list(item_ids, "item_ids")

    bad_stores = [s for s in stores if s not in M5_STORES]
    if bad_stores:
        raise ValueError(
            f"非法 M5 门店 ID: {bad_stores}。合法取值: {sorted(M5_STORES)}"
        )
    for d in depts:
        if _ITEM_ID_RE.match(d):
            raise ValueError(
                f"检测到 item 级 ID '{d}'。deepbullwhip 0.4.1 的 M5 loader 只有 "
                f"store+dept 两个维度（无 item 维度，见对齐清单 §1.2），"
                f"请改传 dept 级 ID，如 FOODS_1。"
            )
    bad_depts = [d for d in depts if d not in M5_DEPTS]
    if bad_depts:
        raise ValueError(
            f"非法 M5 dept ID: {bad_depts}。合法取值: {sorted(M5_DEPTS)}"
        )

    return [(s, d) for s in stores for d in depts]


def _load_one_m5_series(store: str, dept: str) -> np.ndarray:
    """加载单条 (store, dept) M5 序列：本地 CSV → Kaggle 下载，两级降级。

    本地路径即框架的推荐用法（对齐清单 §1.3）：必须显式传 ``data_root``，
    否则 pip 安装下框架默认路径指向 site-packages 内部、必然失败。

    Parameters
    ----------
    store : str
        M5 门店 ID，如 ``"CA_1"``。
    dept : str
        M5 dept ID，如 ``"FOODS_1"``。

    Returns
    -------
    np.ndarray
        按 ``M5_FREQ`` 聚合后的 1-D 需求序列。

    Raises
    ------
    FileNotFoundError
        本地 CSV 不存在且 Kaggle 路径不可用时，附带三类修复提示。
    """
    try:
        return load_dataset(
            "m5", store=store, dept=dept, freq=M5_FREQ, data_root=M5_DATA_ROOT
        )
    except FileNotFoundError as local_exc:
        # 路径 2：框架自带 Kaggle 下载（需凭证，成功后缓存到 ~/.deepbullwhip/data/m5）
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            try:
                return load_m5(store=store, dept=dept, freq=M5_FREQ)
            except Exception as kaggle_exc:  # noqa: BLE001 —— 保留两级原因再抛
                raise FileNotFoundError(
                    f"M5 数据获取失败（store={store}, dept={dept}）。\n"
                    f"  本地 CSV: {local_exc}\n"
                    f"  Kaggle 下载: {kaggle_exc}\n"
                    f"调试兜底请用 load_synthetic_demand()。"
                ) from kaggle_exc
        raise FileNotFoundError(
            f"未找到 M5 数据（store={store}, dept={dept}）。三种修复方式任选：\n"
            f"  1) 把 Kaggle 竞赛 m5-forecasting-accuracy 的 "
            f"sales_train_evaluation.csv 放到 {M5_DATA_ROOT / 'm5'}；\n"
            f"  2) 设置 KAGGLE_USERNAME / KAGGLE_KEY 后重试（自动下载并缓存）；\n"
            f"  3) 改用 load_synthetic_demand(n_nodes, n_days, seed) 合成兜底。"
        ) from local_exc


def _long_frame(store_id: str, item_id: str, dates: pd.DatetimeIndex,
                values: np.ndarray) -> pd.DataFrame:
    """把单条序列组装成标准长表片段（date/store_id/item_id/demand 四列）。"""
    return pd.DataFrame({
        "date": dates,
        "store_id": store_id,
        "item_id": item_id,
        "demand": values,
    })


# ---------------------------------------------------------------------------
# 对外函数（冻结契约）
# ---------------------------------------------------------------------------

def load_m5_demand(store_ids: str | Sequence[str],
                   item_ids: str | Sequence[str]) -> pd.DataFrame:
    """加载 M5 数据集子集（store × dept 级），清洗后返回**长表** DataFrame。

    内部调用 ``deepbullwhip.datasets.load_dataset("m5", store=..., dept=...,
    freq=..., data_root=...)``（框架真实 API，返回 1-D 数组）；本地 CSV 缺失时
    自动尝试框架的 Kaggle 下载路径 ``deepbullwhip.datasets.load_m5``。

    Parameters
    ----------
    store_ids : str or Sequence[str]
        M5 门店 ID，如 ``"CA_1"`` 或 ``["CA_1", "TX_1"]``。
        合法取值：CA_1..CA_4 / TX_1..TX_3 / WI_1..WI_3。
    item_ids : str or Sequence[str]
        M5 dept 级序列 ID，如 ``["FOODS_1", "FOODS_2"]``，写入长表 ``item_id`` 列。
        合法取值：FOODS_1..3 / HOBBIES_1..2 / HOUSEHOLD_1..2。
        注意：deepbullwhip 0.4.1 的 M5 loader 只有 store+dept 两个筛选维度，
        没有 item 维度（见模块 docstring 与对齐清单 §1.2）。

    Returns
    -------
    pd.DataFrame
        **长表**（团队注意事项 §8 口径），固定四列
        ``[date, store_id, item_id, demand]``：
        * ``date``：datetime64，daily 口径逐日日期（M5 日历锚点 2011-01-29 起，
        截取前 365 天对齐 config simulation_days=365）；
        * ``store_id``：M5 门店 ID（如 "CA_1"）；
        * ``item_id``：M5 dept 级序列 ID（如 "FOODS_1"，承载品类聚合粒度）；
        * ``demand``：float64 需求量（原始销量单位，不归一化）。
        按 date → store_id → item_id 排序，每 (store, dept) 序列日期连续无空洞。
        仿真层可用 :func:`pivot_demand_wide` 转宽表后按列名映射终端节点。

    Raises
    ------
    ValueError
        store/dept ID 非法、误传 item 级 ID、或某条序列清洗后全为 0 时。
    FileNotFoundError
        本地 CSV 与 Kaggle 路径都不可用时（附修复提示，可用
        ``load_synthetic_demand`` 兜底调试）。

    Notes
    -----
    清洗规则：NaN/Inf 记 0、负值截断为 0（M5 无销量日即 0，不插值，对齐清单
    §2.4 第 22 条）；daily 口径（框架 d_1..d_1969 逐日销量），截取前 365 天
    对齐 config 的 simulation_days=365，每条序列日期连续无空洞。
    """
    pairs = _validate_m5_pairs(store_ids, item_ids)

    frames: list[pd.DataFrame] = []
    n_lengths: set[int] = set()
    for store, dept in pairs:
        raw = _load_one_m5_series(store, dept)
        series = _clean_series(raw)
        if series.size == 0:
            raise ValueError(f"M5 序列 {store}/{dept} 为空，请检查数据文件完整性")
        if not series.any():
            raise ValueError(
                f"M5 序列 {store}/{dept} 聚合需求全为 0。通常是 (store, dept) "
                f"组合无数据或 CSV 不完整，请检查 {M5_DATA_ROOT / 'm5'} 下的数据。"
            )
        series = series[:M5_MAX_PERIODS]   # 截取前 365 天，对齐 config simulation_days
        n_lengths.add(series.size)
        dates = pd.DatetimeIndex(
            [M5_ANCHOR_DATE + pd.Timedelta(days=k) for k in range(series.size)],
            name="date",
        )
        frames.append(_long_frame(store, dept, dates, series))

    if len(n_lengths) > 1:
        # 同一份 CSV 聚合出的序列长度必然一致；不一致说明数据文件被改动过
        raise ValueError(f"各序列长度不一致 {sorted(n_lengths)}，M5 数据文件可能不完整")

    long_df = pd.concat(frames, ignore_index=True)
    return long_df.sort_values(["date", "store_id", "item_id"],
                               kind="stable").reset_index(drop=True)


def load_synthetic_demand(n_nodes: int = _SYN_DEFAULT_NODES,
                          n_days: int = _SYN_DEFAULT_DAYS,
                          seed: int = 42) -> pd.DataFrame:
    """生成合成时序需求数据（M5 不可用时的调试兜底，seed 保证可复现）。

    默认参数对齐 ``config/network_config.yaml`` 基准：
    ``sku_count=4``、``simulation_days=365``、``demand.mean=10``、``demand.std=3``。
    每个节点用框架自带的 AR(1) 需求过程 ``load_ar1`` 生成（内部对负值截断），
    节点间通过 ``seed + i`` 差异化，同 seed 逐位可复现。

    Parameters
    ----------
    n_nodes : int, default 4
        需求序列（SKU/节点）数量，>= 1。默认对齐 config ``sku_count=4``
        （对应拓扑的 4 个终端门店）。
    n_days : int, default 365
        期数（逐日日期），>= 2。默认对齐 config ``simulation_days=365``。
    seed : int, default 42
        随机种子。同 seed 生成结果逐位一致；第 i 个节点用 ``seed + i``。

    Returns
    -------
    pd.DataFrame
        **长表**，固定四列 ``[date, store_id, item_id, demand]``：
        * ``date``：datetime64，起自 2024-01-01 的逐日日期（合成数据无真实日历，
          取固定锚点）；
        * ``store_id``：恒为 "SYN"（合成数据统一占位门店）；
        * ``item_id``：``"SKU_1" ... f"SKU_{n_nodes}"``；
        * ``demand``：float64 非负需求，各序列均值≈10（config demand.mean）、
          平稳标准差≈3（config demand.std，由创新项 sigma=2.4 与 rho=0.6 换算）。

    Raises
    ------
    ValueError
        ``n_nodes < 1`` 或 ``n_days < 2`` 时。
    """
    n_nodes = int(n_nodes)
    n_days = int(n_days)
    if n_nodes < 1:
        raise ValueError(f"n_nodes 至少为 1，收到 {n_nodes}")
    if n_days < 2:
        raise ValueError(f"n_days 至少为 2，收到 {n_days}")

    dates = pd.date_range(start=_SYN_START_DATE, periods=n_days, freq="D",
                          name="date")
    frames: list[pd.DataFrame] = []
    for i in range(n_nodes):
        series = _clean_series(load_ar1(
            T=n_days,
            mu=_SYN_MU,
            rho=_SYN_RHO,
            sigma=_SYN_SIGMA,
            seed=int(seed) + i,
        ))
        frames.append(_long_frame(_SYN_STORE_ID, f"SKU_{i + 1}", dates, series))

    long_df = pd.concat(frames, ignore_index=True)
    return long_df.sort_values(["date", "store_id", "item_id"],
                               kind="stable").reset_index(drop=True)


def pivot_demand_wide(demand_df: pd.DataFrame) -> pd.DataFrame:
    """把长表需求 DataFrame 转成宽表（index=date，columns="{store_id}_{item_id}"）。

    供预测/绘图等内部流程使用（预测层按"一列一条序列"处理），也供仿真层
    做门店/SKU → 拓扑节点映射。重复的 (date, store, item) 组合按 demand 求和。

    Parameters
    ----------
    demand_df : pd.DataFrame
        长表（[date, store_id, item_id, demand] 四列口径，见
        :func:`load_m5_demand` / :func:`load_synthetic_demand`）。

    Returns
    -------
    pd.DataFrame
        宽表：index = date（DatetimeIndex，升序无重复），columns =
        ``"{store_id}_{item_id}"``（按列名排序），values = float64 demand。

    Raises
    ------
    TypeError
        ``demand_df`` 不是 DataFrame 时。
    ValueError
        缺少 date/demand/store_id/item_id 必要列，或表为空时。
    """
    if not isinstance(demand_df, pd.DataFrame):
        raise TypeError(
            f"demand_df 应为 pd.DataFrame，收到 {type(demand_df).__name__}"
        )
    missing = [c for c in LONG_COLUMNS if c not in demand_df.columns]
    if missing:
        raise ValueError(f"长表缺少必要列: {missing}（应含 {LONG_COLUMNS}）")
    if demand_df.empty:
        raise ValueError("demand_df 为空表，无法转换宽表")

    wide = demand_df.pivot_table(index="date", columns=["store_id", "item_id"],
                                 values="demand", aggfunc="sum")
    wide.columns = [f"{s}_{i}" for s, i in wide.columns]
    wide = wide[sorted(wide.columns)].astype(np.float64)
    wide.index.name = "date"
    return wide.sort_index()


def build_replay_generator(demand_df: pd.DataFrame | pd.Series) -> ReplayDemandGenerator:
    """把需求 DataFrame 转成框架的 ``ReplayDemandGenerator``（需求交接点）。

    ``ReplayDemandGenerator.__init__(data)`` 只接受 1-D 序列
    （np.ndarray | list[float]，deepbullwhip/demand/replay.py），因此：

    * **长表**输入（date/store_id/item_id/demand）→ 按 date 聚合（跨序列求和）
      成一条序列，即"整表聚合门店/网络需求"（对齐清单 §2.4 第 21 条）；
    * **宽表**输入（一列一条序列）→ 按行求和聚合成一条序列（向后兼容）；
    * **pd.Series** 输入 → 直接取该序列；
    * 需要给每个终端节点配一个 generator 时，先按序列过滤长表再传入，
      例如 ``build_replay_generator(demand_df[demand_df.item_id == "FOODS_1"])``，
      或对 :func:`pivot_demand_wide` 的每一列
      ``build_replay_generator(wide_df[[col]])``。

    Parameters
    ----------
    demand_df : pd.DataFrame or pd.Series
        长表（date/store_id/item_id/demand）、宽表（一列一条序列）或单条 Series。

    Returns
    -------
    ReplayDemandGenerator
        ``deepbullwhip.demand.replay.ReplayDemandGenerator`` 实例，
        ``generate(T)`` 在 T 超过数据长度时自动循环填充。

    Raises
    ------
    TypeError
        ``demand_df`` 不是 DataFrame / Series 时。
    ValueError
        表为空、缺必要列、含非数值列、整列全 NaN、或聚合后需求全为 0 时。
    """
    if isinstance(demand_df, pd.Series):
        frame = demand_df.to_frame(name=demand_df.name or "demand")
        values = frame.iloc[:, 0]
    elif isinstance(demand_df, pd.DataFrame):
        if _is_long_format(demand_df):
            values = pivot_demand_wide(demand_df).sum(axis=1)
        else:
            values = demand_df.sum(axis=1)
    else:
        raise TypeError(
            f"demand_df 应为 pd.DataFrame / pd.Series，收到 {type(demand_df).__name__}"
        )

    if len(values) == 0:
        raise ValueError("demand_df 为空（0 行），无法构造 ReplayDemandGenerator")

    all_nan = bool(values.isna().all())
    if all_nan:
        raise ValueError("需求序列整列均为 NaN，无有效数据")

    arr = _clean_series(values.to_numpy(dtype=np.float64))
    if arr.size == 0:
        raise ValueError("聚合后的需求序列为空")
    if not arr.any():
        raise ValueError("聚合后的需求序列全为 0，回放无意义，请检查输入数据")

    # ReplayDemandGenerator.__init__(data: np.ndarray | list[float]) —— 已核对 0.4.1 源码
    return ReplayDemandGenerator(arr)


# ---------------------------------------------------------------------------
# 内部自测（python src/data_loader.py）
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """依次自测 3 个对外函数 + pivot 工具：打印形状、验证对象与 config 对齐。

    M5 数据缺失（FileNotFoundError / ImportError）视为 SKIP，不算失败；
    其余任何异常记 FAIL。返回进程退出码：0 全部通过，1 有失败项。
    """
    # Windows 中文控制台默认 GBK，防 UnicodeEncodeError 导致自测中断
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    import deepbullwhip

    print("=" * 68)
    print("  src/data_loader.py 自测")
    print("=" * 68)
    print(f"解释器: {sys.executable}")
    print(f"deepbullwhip {getattr(deepbullwhip, '__version__', '?')} | "
          f"pandas {pd.__version__} | numpy {np.__version__} | "
          f"M5 数据目录: {M5_DATA_ROOT}")

    failures: list[str] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"       {line}")
        if not ok:
            failures.append(name)

    # ---------- 步骤 1：load_synthetic_demand（长表 + config 对齐） ----------
    df_syn: pd.DataFrame | None = None
    try:
        df_syn = load_synthetic_demand()  # 默认 4 SKU × 365 天（config 基准）
        df_same = load_synthetic_demand()
        df_other = load_synthetic_demand(seed=7)
        assert list(df_syn.columns) == LONG_COLUMNS, f"列错误: {list(df_syn.columns)}"
        assert df_syn.shape == (4 * 365, 4), f"形状异常: {df_syn.shape}"
        assert df_syn["date"].nunique() == 365, "日期数应为 simulation_days=365"
        assert sorted(df_syn["item_id"].unique()) == [f"SKU_{i}" for i in range(1, 5)]
        assert (df_syn["demand"] >= 0).all() and not df_syn.isna().any().any()
        assert df_syn.equals(df_same), "同 seed 两次生成结果应完全一致"
        assert not df_syn.equals(df_other), "不同 seed 应产生不同数据"
        report(
            "load_synthetic_demand()（默认 4 SKU × 365 天）", True,
            f"长表形状 {df_syn.shape}，列 {list(df_syn.columns)}\n"
            f"日期 {df_syn['date'].min().date()} ~ {df_syn['date'].max().date()}"
            f"（{df_syn['date'].nunique()} 天，对齐 simulation_days=365）\n"
            f"同 seed 可复现：OK；不同 seed 数据不同：OK\n"
            + df_syn.head(3).to_string(),
        )
    except Exception:  # noqa: BLE001
        report("load_synthetic_demand()（默认 4 SKU × 365 天）",
               False, traceback.format_exc())

    # ---------- 步骤 2：config 参数对齐（demand.mean=10 / demand.std=3） ----------
    if df_syn is None:
        report("config 参数对齐（demand.mean/std）", False, "依赖步骤 1 的数据")
    else:
        try:
            g = df_syn.groupby("item_id")["demand"]
            mean_dev = float((g.mean() - _SYN_MU).abs().max())
            std_dev = float((g.std() - 3.0).abs().max())
            assert mean_dev < 0.6, f"均值偏离 config demand.mean=10 过大: {mean_dev}"
            assert std_dev < 0.4, f"std 偏离 config demand.std=3 过大: {std_dev}"
            report("config 参数对齐（demand.mean=10 / demand.std=3）", True,
                   f"各 SKU 均值最大偏差 {mean_dev:.3f}（目标 10）；"
                   f"平稳 std 最大偏差 {std_dev:.3f}（目标 3）\n"
                   f"AR(1) 创新项 sigma={_SYN_SIGMA} / rho={_SYN_RHO} -> "
                   f"平稳 std={_SYN_SIGMA / (1 - _SYN_RHO ** 2) ** 0.5:.2f}")
        except Exception:  # noqa: BLE001
            report("config 参数对齐（demand.mean/std）", False, traceback.format_exc())

    # ---------- 步骤 3：pivot_demand_wide ----------
    if df_syn is None:
        report("pivot_demand_wide(demand_df)", False, "依赖步骤 1 的数据")
    else:
        try:
            wide = pivot_demand_wide(df_syn)
            assert wide.shape == (365, 4), f"宽表形状异常: {wide.shape}"
            assert list(wide.columns) == [f"SYN_SKU_{i}" for i in range(1, 5)]
            assert wide.index.name == "date"
            assert not wide.isna().any().any()
            report("pivot_demand_wide(demand_df)", True,
                   f"宽表形状 {wide.shape}（index=date，columns=SYN_SKU_1..4）")
        except Exception:  # noqa: BLE001
            report("pivot_demand_wide(demand_df)", False, traceback.format_exc())

    # ---------- 步骤 4：build_replay_generator（长表/宽表/Series 三态） ----------
    if df_syn is None:
        report("build_replay_generator(demand_df)", False, "依赖步骤 1 的数据")
    else:
        try:
            gen = build_replay_generator(df_syn)  # 长表：按 date 聚合求和
            assert isinstance(gen, ReplayDemandGenerator)
            assert gen.data.shape == (365,), f"长表聚合后长度应=365: {gen.data.shape}"

            wide = pivot_demand_wide(df_syn)
            gen_wide = build_replay_generator(wide)  # 宽表：按行求和（向后兼容）
            np.testing.assert_allclose(gen_wide.data, gen.data)

            gen_series = build_replay_generator(wide["SYN_SKU_1"])  # Series
            assert isinstance(gen_series, ReplayDemandGenerator)
            np.testing.assert_allclose(gen_series.data, wide["SYN_SKU_1"].to_numpy())

            s_short = gen.generate(T=20)
            s_long = gen.generate(T=600)  # 超出数据长度 → 框架自动循环填充
            assert s_short.shape == (20,) and s_long.shape == (600,)
            assert np.isfinite(s_short).all() and (s_short >= 0).all()
            report(
                "build_replay_generator(demand_df)", True,
                f"实例类型: {type(gen).__module__}.{type(gen).__name__}\n"
                f"长表按 date 聚合 -> 长度 {len(gen.data)}；宽表行求和逐值一致；"
                f"Series 直取一致\n"
                f"generate(20) -> {s_short.shape}；generate(600) -> {s_long.shape}"
                f"（循环填充 OK）",
            )
        except Exception:  # noqa: BLE001
            report("build_replay_generator(demand_df)", False, traceback.format_exc())

    # ---------- 步骤 5：load_m5_demand（数据缺失时 SKIP，不算失败） ----------
    m5_name = "load_m5_demand('CA_1', [FOODS_1, FOODS_2, HOUSEHOLD_1, HOUSEHOLD_2])"
    try:
        df_m5 = load_m5_demand("CA_1",
                               ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"])
    except (FileNotFoundError, ImportError) as exc:
        print(f"\n[SKIP] {m5_name}")
        first_line = str(exc).splitlines()[0]
        print(f"       原因: {first_line}")
        print("       说明: data/raw 下暂无 M5 CSV 属预期情况，本步骤不算失败；")
        print("       合成兜底路径已在步骤 1 验证。数据就位后重跑本文件即走真实数据。")
    except Exception:  # noqa: BLE001
        report(m5_name, False, traceback.format_exc())
    else:
        try:
            assert list(df_m5.columns) == LONG_COLUMNS
            assert df_m5["item_id"].nunique() == 4 and df_m5["store_id"].nunique() == 1
            n_days = df_m5["date"].nunique()
            assert n_days == M5_MAX_PERIODS, (
                f"应截取前 {M5_MAX_PERIODS} 天，实际 {n_days}")
            assert df_m5.shape == (n_days * 4, 4)
            assert not df_m5.isna().any().any()
            gen_m5 = build_replay_generator(df_m5[df_m5.item_id == "FOODS_1"])
            assert isinstance(gen_m5, ReplayDemandGenerator)
            report(
                m5_name, True,
                f"长表形状 {df_m5.shape}（{n_days} 天 × 4 条 dept 序列，"
                f"daily 365 对齐 config），NaN=0\n"
                f"列均值: "
                f"{ {k: round(float(v), 2) for k, v in df_m5.groupby('item_id')['demand'].mean().items()} }\n"
                f"逐序列回放 isinstance(ReplayDemandGenerator)：OK",
            )
        except Exception:  # noqa: BLE001
            report(m5_name, False, traceback.format_exc())

    print("\n" + "=" * 68)
    if failures:
        print(f"  自测失败 {len(failures)} 项: {failures}")
        print("=" * 68)
        return 1
    print("  自测完成：长表契约 + config 参数对齐 + 交接对象均无报错"
          "（M5 数据缺失时该步 SKIP，不算失败）。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

# -*- coding: utf-8 -*-
"""聚合缺口信号协议（核心创新点之一）

职责
----
定义跨节点上报的"聚合缺口信号"数据结构与共享总线 ``SignalBus``。
下级节点（门店/城市仓）每个时间步把自身的需求缺口聚合成一个信号上报，
上级节点读取下游信号后调整补货决策。信号只包含聚合量，不含单笔订单明细，
满足赛题的"隐私保护"要求。

对外接口
----
    AggregatedSignal               # 数据类：一次上报的信号
    SignalBus                      # 共享可变对象：各节点策略通过它互相读写

框架事实（见 docs/risk_log.md R1）
----
* ``OrderingPolicy.compute_order`` 是单节点接口，无跨节点参数；
* 但策略对象在 ``NetworkSupplyChain.__init__`` 里一次性构造、常驻整个仿真；
* 且 ``simulate()`` 内部按反向拓扑序处理节点（store -> city -> regional），
  所以**构造时注入共享对象**即可实现自下而上的信号传递。

隐私保护设计
----
信号只暴露"聚合缺口总量（gap）"与"不确定性（gap_std）"，不含：
* 单笔订单的 SKU/数量；
* 具体的库存明细；
* 需求序列本身。
上级只能看到"下游合计还需要 X 单位"这样一个模糊聚合量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["AggregatedSignal", "SignalBus"]


@dataclass
class AggregatedSignal:
    """聚合缺口信号——下级节点上报给上级的隐私保护信号。

    Attributes
    ----------
    node_name : str
        上报节点名（如 ``"store_1"``）；聚合信号用 ``"AGG(n)"`` 标记。
    timestep : int
        信号产生的时间步（0-based）。
    gap : float
        聚合缺口总量（target - current），>= 0；不含单笔订单明细。
    gap_std : float
        缺口的不确定性（沿用预测 std 口径），>= 0。
    n_sources : int
        参与聚合的来源节点数；叶子节点自己上报时为 1。
    """

    node_name: str
    timestep: int
    gap: float
    gap_std: float = 0.0
    n_sources: int = 1


class SignalBus:
    """共享信号总线——各节点策略通过它互相读写信号。

    用法
    ----
    构造一次，作为构造参数注入所有节点的策略对象::

        bus = SignalBus()
        policies = {name: CoordinatedOrderingPolicy(..., signal_bus=bus, ...)}

    每个时间步内，节点按 store -> city -> regional 顺序被处理：
    store 先写入自己的信号，city 立即就能读到并聚合，regional 同理。
    """

    def __init__(self) -> None:
        self._latest: dict[str, AggregatedSignal] = {}

    def write(self, signal: AggregatedSignal) -> None:
        """写入/更新一个节点的最新信号。"""
        self._latest[signal.node_name] = signal

    def read(self, node_name: str) -> AggregatedSignal | None:
        """读取某个节点的最新信号；若未上报过返回 None。"""
        return self._latest.get(node_name)

    def read_downstream(
        self, downstream_names: Sequence[str]
    ) -> AggregatedSignal | None:
        """聚合读取多个下游节点的信号。

        Parameters
        ----------
        downstream_names : Sequence[str]
            下游节点名列表（一般来自 ``graph.downstream_neighbors(name)``）。

        Returns
        -------
        AggregatedSignal or None
            * 全部下游都无信号 -> None；
            * 只有 1 个下游有信号 -> 直接返回该信号；
            * 多个下游 -> gap 求和、gap_std 按方差相加开方、n_sources 累加。
        """
        sigs = [self._latest[n] for n in downstream_names if n in self._latest]
        if not sigs:
            return None
        if len(sigs) == 1:
            return sigs[0]
        total_gap = float(sum(s.gap for s in sigs))
        combined_std = float(np.sqrt(sum(s.gap_std ** 2 for s in sigs)))
        latest_t = max(s.timestep for s in sigs)
        total_sources = sum(s.n_sources for s in sigs)
        return AggregatedSignal(
            node_name=f"AGG({len(sigs)})",
            timestep=latest_t,
            gap=total_gap,
            gap_std=combined_std,
            n_sources=total_sources,
        )

    def reset(self) -> None:
        """清空所有信号（仿真重置时调用）。"""
        self._latest.clear()

    def snapshot(self) -> dict[str, AggregatedSignal]:
        """返回当前所有信号的浅拷贝（只读用途，如调试/可视化）。"""
        return dict(self._latest)
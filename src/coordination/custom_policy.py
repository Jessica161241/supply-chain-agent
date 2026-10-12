# -*- coding: utf-8 -*-
"""协同订货策略（核心创新点之二）

职责
----
继承 ``OrderingPolicy``，在 OrderUpTo 基线上叠加"下游聚合缺口信号"补偿。

决策公式
----
::

    S          = (L+1) * fm + z * fs * sqrt(L+1)      # 同 OrderUpTo
    base_order = max(0, S - IP)
    coord      = alpha * downstream_gap               # 下游信号补偿
    order      = base_order + coord

* ``L``        = 补货提前期（节点级）
* ``fm, fs``   = 框架传入的预测均值/标准差
* ``IP``       = 库存头寸（inventory_position）
* ``z``        = 目标服务水平对应的正态分位数
* ``alpha``    = 协同强度系数（0 = 退化为纯 OrderUpTo；过大则成本上升）

框架事实（见 docs/risk_log.md R1）
----
* 策略对象常驻整个仿真，``compute_order`` 每个时间步被调用一次；
* 通过构造时注入 ``SignalBus``，同一步内可读到下游上报的信号；
* 本节点也会写自己的信号供上游读取，形成"自下而上"的信息流。

参数调优提示
----
R1 的 POC 实测表明：朴素的协同规则**不一定改善总成本**。
``alpha`` 需要扫描调优（一般从 0.1 ~ 0.5 开始），
不能默认"加了协同就一定更好"。
"""

from __future__ import annotations

import numpy as np
from deepbullwhip.policy.base import OrderingPolicy
from scipy import stats

from src.coordination.signal_protocol import AggregatedSignal, SignalBus

__all__ = ["CoordinatedOrderingPolicy"]


class CoordinatedOrderingPolicy(OrderingPolicy):
    """协同订货策略：OrderUpTo 基线 + 下游聚合缺口信号补偿。

    Parameters
    ----------
    lead_time : int
        补货提前期（应与节点 ``EchelonConfig.lead_time`` 一致）。
    service_level : float, default 0.95
        目标服务水平，用于计算安全库存系数 ``z_alpha``。
    signal_bus : SignalBus or None, default None
        共享信号总线。None 时退化为纯 OrderUpTo（用于对照实验）。
    node_name : str, default ""
        本节点名称（用于写入信号）。
    downstream_names : Sequence[str] or None, default None
        下游节点名列表（用于读取聚合信号）。叶子节点传空列表即可。
    alpha : float, default 0.3
        协同强度系数——下游缺口信号的权重。
        ``alpha=0`` 等价于纯 OrderUpTo；过大可能导致过量订货、成本上升。
    """

    def __init__(
        self,
        lead_time: int,
        service_level: float = 0.95,
        signal_bus: SignalBus | None = None,
        node_name: str = "",
        downstream_names: list[str] | None = None,
        alpha: float = 0.3,
    ) -> None:
        self.lead_time = int(lead_time)
        self.service_level = float(service_level)
        self.z_alpha = float(stats.norm.ppf(service_level))
        self.signal_bus = signal_bus
        self.node_name = node_name
        self.downstream_names = list(downstream_names or [])
        self.alpha = float(alpha)
        self._t = 0

    def compute_order(
        self,
        inventory_position: float,
        forecast_mean: float,
        forecast_std: float,
    ) -> float:
        """计算本时间步的下单量。

        Parameters
        ----------
        inventory_position : float
            库存头寸（在库 + 在途 - 缺货）。
        forecast_mean : float
            下游需求的预测均值（由框架传入：叶子节点来自外部预测，
            中间节点来自下游订单均值）。
        forecast_std : float
            预测标准差（允许为 0）。

        Returns
        -------
        float
            下单量（>= 0）。
        """
        # ---- 1. 基线 OrderUpTo 订货量 ----
        S = (self.lead_time + 1) * forecast_mean + (
            self.z_alpha * forecast_std * np.sqrt(self.lead_time + 1)
        )
        base_order = max(0.0, S - float(inventory_position))

        # ---- 2. 读取下游聚合信号，计算协同补偿 ----
        coordination = 0.0
        if self.signal_bus is not None and self.downstream_names:
            agg = self.signal_bus.read_downstream(self.downstream_names)
            if agg is not None:
                coordination = self.alpha * agg.gap

        order = base_order + coordination

        # ---- 3. 本节点向上报告自己的缺口信号 ----
        # 缺口定义 = 基线下单量（= 距离目标库存 S 的差距），
        # 含义是"如果只按本地需求，我还需要补多少"。
        if self.signal_bus is not None:
            self.signal_bus.write(
                AggregatedSignal(
                    node_name=self.node_name,
                    timestep=self._t,
                    gap=base_order,
                    gap_std=float(forecast_std),
                    n_sources=1,
                )
            )

        self._t += 1
        return order

    def reset(self) -> None:
        """重置时间步计数器（仿真重新开始时调用）。"""
        self._t = 0
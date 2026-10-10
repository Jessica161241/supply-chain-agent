#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检脚本 —— 一键验证开发环境是否就绪。

跑 8 项检查，每项独立捕获异常，**任一项失败不影响其余项继续执行**：

1. Python 版本（必须 3.10 / 3.11）
2. 依赖包版本（numpy / pandas / scipy / matplotlib / deepbullwhip）
3. 关键 API 导入（框架顶层导出 + 策略/数据集子模块）
4. 数据集加载（M5 本地 CSV → 合成兜底，三级降级）
5. ReplayDemandGenerator 转换
6. Serial（串行链）仿真跑通
7. Network（DAG）仿真跑通
8. 本项目 3 层拓扑（区域仓 → 城市仓 → 门店）校验

用法
----
::

    python scripts/quickstart_verify.py                      # 基本自检
    python scripts/quickstart_verify.py --json-out outputs/results/env_check.json
    python scripts/quickstart_verify.py --require-m5         # 拿不到真实 M5 就失败退出
    python scripts/quickstart_verify.py --store CA_1 --dept FOODS_1

退出码：0 = 全部通过；1 = 有失败项。
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
from pathlib import Path

# 项目根 = 本脚本的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 中文控制台默认 GBK，防止 ✓/表头等字符触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

SUPPORTED_PYTHON = ((3, 10), (3, 11))

DEFAULT_STORE = "CA_1"
DEFAULT_DEPTS = ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]

CONFIG_PATH = PROJECT_ROOT / "config" / "network_config.yaml"


# ---------------------------------------------------------------------------
# 报告收集
# ---------------------------------------------------------------------------

class Report:
    """逐项收集检查结果，最后统一打印 + 落盘 JSON。"""

    def __init__(self, quiet: bool = False) -> None:
        self.steps: list[dict] = []
        self.quiet = quiet

    def add(self, step: str, ok: bool, detail: str = "",
            skipped: bool = False, **extra) -> None:
        entry = {"step": step, "ok": bool(ok)}
        if skipped:
            entry["skipped"] = True
        if detail:
            entry["detail"] = detail
        entry.update(extra)
        self.steps.append(entry)

        if not self.quiet:
            tag = "SKIP" if skipped else ("PASS" if ok else "FAIL")
            print(f"[{tag}] {step}")
            for line in str(detail).splitlines():
                print(f"       {line}")
            print()

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if not s["ok"])

    def to_dict(self, **extra) -> dict:
        return {"steps": self.steps, "n_failed": self.n_failed, **extra}


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------

def check_python(rep: Report) -> None:
    """1. Python 版本。"""
    major, minor, micro = sys.version_info[:3]
    ver = f"{major}.{minor}.{micro}"
    ok = (major, minor) in SUPPORTED_PYTHON
    detail = (f"Python {ver} —— {'符合' if ok else '不符合'}赛题约束 (3.10 / 3.11)"
              f"  [{platform.system()}]")
    rep.add("Python 版本", ok, detail, python=ver, platform=platform.system())


def check_dependencies(rep: Report) -> None:
    """2. 依赖包版本。"""
    wanted = ["numpy", "pandas", "scipy", "matplotlib", "deepbullwhip"]
    lines, missing = [], []
    for name in wanted:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            lines.append(f"{name:<16s} {version}")
        except ImportError as exc:  # noqa: F841
            missing.append(name)
            lines.append(f"{name:<16s} MISSING")
    detail = "\n".join(lines)
    if missing:
        detail += ("\n修复：python -m pip install -r requirements.txt"
                   "（若 pip 报网络错误，先清空 HTTP_PROXY/HTTPS_PROXY）")
    rep.add("依赖包版本", not missing, detail, missing=missing)


def check_api_imports(rep: Report) -> None:
    """3. 关键 API 导入。"""
    targets = [
        ("deepbullwhip", "EchelonConfig"),
        ("deepbullwhip", "EdgeConfig"),
        ("deepbullwhip", "SupplyChainGraph"),
        ("deepbullwhip", "NetworkSupplyChain"),
        ("deepbullwhip", "SerialSupplyChain"),
        ("deepbullwhip", "Forecaster"),
        ("deepbullwhip", "ReplayDemandGenerator"),
        ("deepbullwhip", "SemiconductorDemandGenerator"),
        ("deepbullwhip.policy.base", "OrderingPolicy"),
        ("deepbullwhip.policy.order_up_to", "OrderUpToPolicy"),
        ("deepbullwhip.datasets.loader", "load_dataset"),
        ("deepbullwhip.datasets.m5", "load_m5"),
    ]
    lines, failed = [], []
    for module_name, attr in targets:
        try:
            module = __import__(module_name, fromlist=[attr])
            getattr(module, attr)
            lines.append(f"OK   {module_name}.{attr}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{module_name}.{attr}")
            lines.append(f"FAIL {module_name}.{attr}: {type(exc).__name__}: {exc}")

    # 常见的顶层导入陷阱
    try:
        from deepbullwhip import OrderingPolicy  # noqa: F401
        lines.append("注意: OrderingPolicy 顶层可用")
    except ImportError:
        lines.append("提示: OrderingPolicy 顶层未导出，"
                     "必须用 from deepbullwhip.policy.base import OrderingPolicy")

    rep.add("关键 API 导入", not failed, "\n".join(lines), failed=failed)


def _series_stats(df) -> list[tuple[str, int, float]]:
    """从长表按 (store_id, item_id) 提取 (标签, 期数, 均值) 三元组。"""
    out = []
    for (store, item), g in df.groupby(["store_id", "item_id"], sort=True):
        out.append((f"{store}/{item}", len(g), float(g["demand"].mean())))
    return out


def load_demand(store: str, depts: list[str], n_days: int,
                require_m5: bool, rep: Report) -> tuple[object, str]:
    """4. 数据集加载（M5 → 合成三级降级）。返回 (长表 DataFrame, 来源标签)。"""
    from src.data_loader import load_m5_demand, load_synthetic_demand

    try:
        df = load_m5_demand(store, depts)
        detail = "来源: M5 本地 CSV\n" + "\n".join(
            f"OK   {label}: T={n} 期, mean={m:.2f}"
            for label, n, m in _series_stats(df))
        rep.add("数据集加载（来源: m5）", True, detail, data_source="m5")
        return df, "m5"
    except FileNotFoundError as exc:
        first = str(exc).splitlines()[0]
        if require_m5:
            rep.add("数据集加载（要求 M5）", False,
                    f"未取到真实 M5 数据：{first}\n"
                    f"放置 data/raw/m5/sales_train_evaluation.csv "
                    f"或配置 KAGGLE_USERNAME / KAGGLE_KEY 后重试。",
                    data_source=None)
            return None, "missing"
        df = load_synthetic_demand(n_nodes=len(depts), n_days=n_days, seed=42)
        detail = ("M5 不可用，已降级为合成数据集："
                  f"{first}\n" + "\n".join(
                      f"OK   {label}: T={n} 期, mean={m:.2f}"
                      for label, n, m in _series_stats(df)))
        rep.add("数据集加载（来源: synthetic）", True, detail,
                data_source="synthetic")
        return df, "synthetic"


def check_replay(rep: Report, df) -> None:
    """5. ReplayDemandGenerator 转换。"""
    from src.data_loader import build_replay_generator

    if df is None:
        rep.add("ReplayDemandGenerator 转换", False,
                "上一步未取得需求数据，跳过", skipped=True)
        return
    try:
        gen = build_replay_generator(df)
        T_in = int(df["date"].nunique())
        T_out = 52
        series = gen.generate(T_out)
        ok = len(series) == T_out
        rep.add("ReplayDemandGenerator 转换", ok,
                f"输入 T={T_in} -> generate(T={T_out}) 输出 T={len(series)}，"
                f"mean={float(series.mean()):.2f}")
    except Exception as exc:  # noqa: BLE001
        rep.add("ReplayDemandGenerator 转换", False,
                f"{type(exc).__name__}: {exc}")


def check_serial(rep: Report) -> dict:
    """6. Serial（串行链）仿真跑通。"""
    import numpy as np
    from deepbullwhip import SerialSupplyChain, SemiconductorDemandGenerator
    try:
        T = 52
        gen = SemiconductorDemandGenerator()
        demand = np.asarray(gen.generate(T, seed=42), dtype=np.float64)
        fm = np.full(T, float(demand.mean()), dtype=np.float64)
        fs = np.full(T, max(float(demand.std()), 1e-9), dtype=np.float64)

        chain = SerialSupplyChain()
        result = chain.simulate(demand, fm, fs)

        lines = [f"{'echelon':<18s}{'BW':>8s}{'FillRate':>10s}{'Cost':>12s}"]
        for er in result.echelon_results:
            lines.append(f"{er.name:<18s}{er.bullwhip_ratio:>8.2f}"
                         f"{er.fill_rate:>9.0%}{er.total_cost:>12,.0f}")
        lines.append(f"{'cumulative BW':<18s}{result.cumulative_bullwhip:>8.2f}")
        detail = "\n".join(lines)
        ok = len(result.echelon_results) > 0 and np.isfinite(result.total_cost)
        stats = {"n_echelons": len(result.echelon_results),
                 "cumulative_bullwhip": float(result.cumulative_bullwhip),
                 "total_cost": float(result.total_cost)}
        rep.add("Serial 仿真跑通", ok, detail, serial=stats)
        return stats
    except Exception:  # noqa: BLE001
        import traceback
        rep.add("Serial 仿真跑通", False, traceback.format_exc())
        return {}


def check_network(rep: Report) -> dict:
    """7. Network（DAG）仿真跑通：Factory → Warehouse → Retail_A / Retail_B。"""
    import numpy as np
    from deepbullwhip import (EchelonConfig, EdgeConfig, NetworkSupplyChain,
                              SupplyChainGraph)
    try:
        T = 52
        graph = SupplyChainGraph(
            nodes={
                "Factory": EchelonConfig("Factory", lead_time=4,
                                         holding_cost=0.10, backorder_cost=0.40),
                "Warehouse": EchelonConfig("Warehouse", lead_time=2,
                                           holding_cost=0.15, backorder_cost=0.50),
                "Retail_A": EchelonConfig("Retail_A", lead_time=1,
                                          holding_cost=0.20, backorder_cost=0.60),
                "Retail_B": EchelonConfig("Retail_B", lead_time=1,
                                          holding_cost=0.20, backorder_cost=0.60),
            },
            edges={
                ("Factory", "Warehouse"): EdgeConfig(lead_time=3),
                ("Warehouse", "Retail_A"): EdgeConfig(lead_time=1),
                ("Warehouse", "Retail_B"): EdgeConfig(lead_time=1),
            },
        )
        net = NetworkSupplyChain(graph=graph)

        rng = np.random.default_rng(42)
        demand = {n: np.maximum(0.0, rng.normal(10.0, 3.0, T))
                  for n in graph.demand_nodes}
        fm = {n: np.full(T, float(v.mean())) for n, v in demand.items()}
        fs = {n: np.full(T, max(float(v.std()), 1e-9)) for n, v in demand.items()}

        result = net.simulate(demand, fm, fs)

        lines = [f"demand_nodes = {graph.demand_nodes}",
                 f"source_nodes = {graph.source_nodes}",
                 f"{'node':<18s}{'BW':>8s}{'FillRate':>10s}{'Cost':>12s}"]
        for name, er in result.node_results.items():
            lines.append(f"{name:<18s}{er.bullwhip_ratio:>8.2f}"
                         f"{er.fill_rate:>9.0%}{er.total_cost:>12,.0f}")
        lines.append(f"edge_flows: {len(result.edge_flows)} 条边")
        ok = len(result.node_results) == 4 and len(result.edge_flows) == 3
        stats = {"nodes": list(result.node_results),
                 "n_edges": len(result.edge_flows),
                 "cumulative_bullwhip": float(result.cumulative_bullwhip),
                 "total_cost": float(result.total_cost)}
        rep.add("Network(DAG) 仿真跑通", ok, "\n".join(lines), network=stats)
        return stats
    except Exception:  # noqa: BLE001
        import traceback
        rep.add("Network(DAG) 仿真跑通", False, traceback.format_exc())
        return {}


def check_project_topology(rep: Report) -> None:
    """8. 本项目 3 层拓扑校验（B 同学交付的 config + topology）。"""
    if not CONFIG_PATH.exists():
        rep.add("本项目 3 层拓扑校验", True,
                f"尚未交付: {CONFIG_PATH.relative_to(PROJECT_ROOT)}"
                f"（B 同学负责，交付后重跑本脚本即可）",
                skipped=True)
        return
    try:
        from src.network.topology import build_network_from_config
        net = build_network_from_config(CONFIG_PATH)
        nodes = list(net.graph.nodes)
        edges = list(net.graph.edges)
        expected_nodes = 7
        expected_edges = 6
        ok = len(nodes) == expected_nodes and len(edges) == expected_edges
        lines = [f"节点 {len(nodes)} 个 / 边 {len(edges)} 条",
                 f"拓扑序 {net.graph.topological_order()}"]
        for src, dst in edges:
            lines.append(f"  {src:22s} -> {dst}")
        if not ok:
            lines.append(f"预期 {expected_nodes} 节点 / {expected_edges} 边，"
                         f"实际 {len(nodes)} / {len(edges)}")
        rep.add("本项目 3 层拓扑校验", ok, "\n".join(lines),
                topology={"nodes": len(nodes), "edges": len(edges)})
    except Exception as exc:  # noqa: BLE001
        import traceback
        rep.add("本项目 3 层拓扑校验", False, traceback.format_exc())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepBullwhip 多级库存智能协同 —— 环境自检")
    p.add_argument("--store", default=DEFAULT_STORE,
                   help=f"M5 store 维度（默认 {DEFAULT_STORE}）")
    p.add_argument("--dept", action="append", dest="depts",
                   help=f"M5 dept 维度，可重复（默认 {DEFAULT_DEPTS}）")
    p.add_argument("--n-days", type=int, default=365,
                   help="合成兜底数据的天数（默认 365，对齐 simulation_days）")
    p.add_argument("--require-m5", action="store_true",
                   help="强制要求真实 M5 数据，取不到则以失败退出")
    p.add_argument("--json-out", default=None,
                   help="把结果写成 JSON（如 outputs/results/env_check.json）")
    p.add_argument("--quiet", action="store_true", help="只输出末尾汇总")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    depts = args.depts or list(DEFAULT_DEPTS)
    rep = Report(quiet=args.quiet)

    if not args.quiet:
        print("=" * 68)
        print("  DeepBullwhip 多级库存智能协同 —— 环境自检")
        print("=" * 68)
        print(f"解释器: {sys.executable}")
        print(f"项目根: {PROJECT_ROOT}")
        print()

    check_python(rep)
    check_dependencies(rep)
    check_api_imports(rep)

    df, source = load_demand(args.store, depts, args.n_days, args.require_m5, rep)
    check_replay(rep, df)

    serial = check_serial(rep)
    network = check_network(rep)
    check_project_topology(rep)

    # ---- 汇总 ----
    total = len(rep.steps)
    n_pass = total - rep.n_failed
    print("=" * 68)
    if rep.n_failed:
        print(f"  共 {total} 项，通过 {n_pass} 项，失败 {rep.n_failed} 项")
        for s in rep.steps:
            if not s["ok"]:
                print(f"    FAIL: {s['step']}")
    else:
        print(f"  共 {total} 项，通过 {total} 项，失败 0 项 —— 环境就绪")
    print("=" * 68)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = rep.to_dict(data_source=(source if source != "missing" else None),
                              serial=serial, network=network)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"结果已写入 {out}")

    return 1 if rep.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

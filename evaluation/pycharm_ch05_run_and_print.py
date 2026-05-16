"""PyCharm 入口：子进程运行 `run_batch_recommend_eval.py`（python -u），并打印 overall / per-query / paper 表。

与 `run_eval_from_pyc.py` 的关系：若需在**同一进程**加载冻结 bytecode，请用该脚本；本包装器调用**源码版**
`run_batch_recommend_eval.py`，避免 marshal，并保持 CLI argv 干净（子进程中 TotalCore 初始化更安全）。
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

# ----- PyCharm 里直接改这里 -----
RUN_MODE = "quick"  # "quick" | "full"
SPLIT = "ch30"  # all | train | dev | test | ch05 | ch20 | ch15 | ch30
TOPK = "5,10,20,50"
USE_KGAT = False
SAVE_DETAILS = False
HIT_REL_MIN = 1  # 1 或 2
SKIP_NEO4J_IN_EXPLAIN = False
OUTPUT_ROOT = os.path.join("evaluation", "batch_eval_outputs")
# --print-only 时：若填绝对/相对路径则直接读该次 run 目录；为 None 则在 OUTPUT_ROOT 下取最近修改的子目录
PRINT_RUN_DIR: Optional[str] = None
# 额外开关（不在 UI 块里常改时可关）
CAPTURE_STDOUT = True
EXTRA_SCRIPT_ARGS: List[str] = []  # 例如 ["--limit", "3", "--mc-n", "0"]


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT = os.path.join(os.path.dirname(__file__), "run_batch_recommend_eval.py")


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def _latest_child_dir(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    candidates: List[str] = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _resolve_run_dir(*, use_print_override: bool) -> str:
    if use_print_override and PRINT_RUN_DIR:
        d = (
            PRINT_RUN_DIR
            if os.path.isabs(PRINT_RUN_DIR)
            else os.path.abspath(os.path.join(_REPO_ROOT, PRINT_RUN_DIR))
        )
        if not os.path.isdir(d):
            raise SystemExit(f"PRINT_RUN_DIR 不是目录: {d}")
        return d
    out_abs = os.path.abspath(os.path.join(_REPO_ROOT, OUTPUT_ROOT))
    latest = _latest_child_dir(out_abs)
    if not latest:
        raise SystemExit(f"在 {out_abs} 下找不到任何 run 子目录（请先跑一次评估或设置 PRINT_RUN_DIR）")
    return latest


def _print_kv_block(title: str, row: Dict[str, str]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    items = sorted(row.items(), key=lambda x: x[0])
    w = max(len(k) for k, _ in items) if items else 0
    for k, v in items:
        print(f"  {k:{w}s}  {v}")


def _print_matrix(title: str, rows: Sequence[Dict[str, str]], max_rows: Optional[int] = None) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    if not rows:
        print("  (空)")
        return
    cols = list(rows[0].keys())
    use = list(rows) if max_rows is None else list(rows[: max(0, max_rows)])
    if max_rows is not None and len(rows) > len(use):
        note = f"\n… 仅显示前 {len(use)} / {len(rows)} 行，改 max_rows 可查看更多"
    else:
        note = ""
    widths = [max(len(c), max((len(str(r.get(c, ""))) for r in use), default=0)) for c in cols]
    head = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(head)
    print("-" * len(head))
    for r in use:
        print(" | ".join(str(r.get(c, "")).ljust(widths[i]) for i, c in enumerate(cols)))
    if note:
        print(note)


def print_tables_from_run(run_dir: str, *, per_query_max: int = 80) -> None:
    run_dir = os.path.abspath(run_dir)
    print("\n>>> 输出目录:", run_dir)

    overall_path = os.path.join(run_dir, "overall_metrics.csv")
    per_q_path = os.path.join(run_dir, "per_query_metrics.csv")
    paper_path = os.path.join(run_dir, "paper_thesis_metrics.csv")
    subgroup_path = os.path.join(run_dir, "subgroup_metrics.csv")

    o_rows = _read_csv_rows(overall_path)
    if len(o_rows) == 1:
        _print_kv_block("Overall（overall_metrics.csv）", o_rows[0])
    elif o_rows:
        _print_matrix("Overall（overall_metrics.csv 多行）", o_rows)
    else:
        print("\n(未找到 overall_metrics.csv)")

    pq = _read_csv_rows(per_q_path)
    _print_matrix("Per-query（per_query_metrics.csv）", pq, max_rows=per_query_max)

    if os.path.isfile(paper_path):
        pr = _read_csv_rows(paper_path)
        _print_matrix("Paper / thesis（paper_thesis_metrics.csv）", pr)
    elif os.path.isfile(subgroup_path):
        sg = _read_csv_rows(subgroup_path)
        _print_matrix("Subgroup（subgroup_metrics.csv，如 ch20）", sg)
    else:
        print("\n(无 paper_thesis_metrics.csv / subgroup_metrics.csv，可能非 ch15/ch20/ch30 或尚未生成)")


def _build_subprocess_cmd() -> List[str]:
    cmd: List[str] = [sys.executable, "-u", _SCRIPT]
    cmd += ["--mode", str(RUN_MODE), "--split", str(SPLIT), "--topk", str(TOPK)]
    cmd += ["--output-dir", os.path.join(_REPO_ROOT, OUTPUT_ROOT)]
    cmd += ["--hit-rel-min", str(int(HIT_REL_MIN))]
    if USE_KGAT:
        cmd.append("--use-kgat")
    if SAVE_DETAILS:
        cmd.append("--save-details")
    if SKIP_NEO4J_IN_EXPLAIN:
        cmd.append("--kgat-skip-neo4j-explain")
    if CAPTURE_STDOUT:
        cmd.append("--capture-stdout")
    cmd += EXTRA_SCRIPT_ARGS
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="PyCharm 包装：批量评估 + 表格打印")
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="不跑评估，只根据 PRINT_RUN_DIR 或 OUTPUT_ROOT 下最新 run 打印表",
    )
    ap.add_argument(
        "--per-query-max",
        type=int,
        default=80,
        help="打印 per_query 最多行数",
    )
    ns = ap.parse_args()

    if ns.print_only:
        run_dir = _resolve_run_dir(use_print_override=True)
        print_tables_from_run(run_dir, per_query_max=ns.per_query_max)
        return

    cmd = _build_subprocess_cmd()
    print(">>> cmd:", " ".join(cmd))
    print(">>> cwd:", _REPO_ROOT)
    r = subprocess.run(cmd, cwd=_REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    run_dir = _resolve_run_dir(use_print_override=False)
    print_tables_from_run(run_dir, per_query_max=ns.per_query_max)


if __name__ == "__main__":
    main()

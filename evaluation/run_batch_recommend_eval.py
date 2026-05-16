"""
批量推荐评估脚本（人工金标）：读取主库 gold_samples / picked_*，批量跑“总召回→候选池→(可选)KGAT-AX 精排”，
保存每条岗位的推荐结果与批量指标汇总。

- `--split ch05`：仅 dev+test 共 10 条；`--split ch20`：20 条（train 前 10 + dev+test 10，见 `subgroup_metrics.csv`）。
- `--split ch15`：Q001–Q015（1–10 为 train_dev，11–15 为 holdout），不依赖 `dataset_splits`；输出与 ch30 同型 `paper_thesis_metrics.csv` 等，子组名为 `all_15` / `holdout_5` / `train_dev_10`。
- `--split ch30`：Q001–Q030（1–20 为 train_dev，21–30 为 holdout），需 query_id 可解析为 1..30 号；另写
  `paper_thesis_metrics.csv`（Labeled- P@K 等）、`random_baseline_paper.csv`（金标内随机重排基线，Monte Carlo）、
  可选 `thesis_metrics_strict_g2.csv`（g>=2 严口径）。

- `--query-id Q027`（可重复）：在 split 之后仅评指定 query，用于单条复现/诊断（如 Q027 的 `kgat_failed:TransientError`）。

约束：不编造实验结果；不改动召回/精排主算法，仅组装入口与落盘。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import random
import re
import sqlite3
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import DB_PATH as _DEFAULT_DB_PATH, KGATAX_TRAIN_DATA_DIR
from src.core.recall.candidate_features import batch_load_top_works, extract_terms_from_label_evidence
from src.core.recall.total_recall import TotalRecallSystem
from src.core.total_core import TotalCore

logger = logging.getLogger("batch_eval")


def _now_tag() -> str:
    return _dt.now().strftime("%Y%m%dT%H%M%S")


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _parse_topk_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError("--topk 不能为空，例如 5,10,20,50")
    return out


def _gold_to_3class(raw: Any) -> Optional[int]:
    try:
        v = int(raw)
        if v >= 2:
            return 2
        if v == 1:
            return 1
        if v == 0:
            return 0
        return None
    except (TypeError, ValueError):
        return None


def _path_combo(from_v: bool, from_l: bool, from_c: bool) -> str:
    if from_v and from_l and from_c:
        return "all three"
    if from_v and from_l and not from_c:
        return "vector + label"
    if from_v and not from_l and from_c:
        return "vector + collab"
    if not from_v and from_l and from_c:
        return "label + collab"
    if from_v and not from_l and not from_c:
        return "vector only"
    if not from_v and from_l and not from_c:
        return "label only"
    if not from_v and not from_l and from_c:
        return "collab only"
    return "unknown"


def _dcg(rels: List[int], k: int) -> float:
    import math

    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        gain = (2**rel) - 1
        s += gain / math.log2(i + 1)
    return s


@dataclass
class QueryEvalResult:
    query_id: str
    security_id: Optional[str]
    judged_count: int
    judged_pos2_count: int
    judged_pos01_count: int
    rec_count: int
    success: bool
    used_kgat: bool
    fallback_reason: Optional[str]
    metrics: Dict[int, Dict[str, float]]


def _compute_metrics_for_query(
    ranked_author_ids: List[str],
    gold_by_author: Dict[str, int],
    topk_list: List[int],
    *,
    unjudged_policy: str = "ignore",
    hit_rel_min: int = 1,
) -> Dict[int, Dict[str, float]]:
    if hit_rel_min not in (1, 2):
        hit_rel_min = 1
    judged_authors = set(gold_by_author.keys())
    pos_hit = {a for a, g in gold_by_author.items() if g >= hit_rel_min}
    out: Dict[int, Dict[str, float]] = {}
    for k in topk_list:
        top = ranked_author_ids[:k]
        if unjudged_policy == "ignore":
            top_judged = [a for a in top if a in judged_authors]
        else:
            top_judged = top
        hit_n = len([a for a in top_judged if a in pos_hit])
        prec = (hit_n / len(top_judged)) if top_judged else 0.0
        rec = (hit_n / len(pos_hit)) if pos_hit else 0.0
        hit = 1.0 if hit_n > 0 else 0.0
        rels = [int(gold_by_author.get(a, 0)) for a in top_judged]
        dcg = _dcg(rels, len(rels))
        ideal_rels = sorted(list(gold_by_author.values()), reverse=True)
        idcg = _dcg(ideal_rels, min(k, len(ideal_rels)))
        ndcg = (dcg / idcg) if idcg > 1e-12 else 0.0
        out[k] = {
            "precision": float(prec),
            "recall": float(rec),
            "ndcg": float(ndcg),
            "hit": float(hit),
            "top_judged_n": float(len(top_judged)),
            "pos_hit_total": float(len(pos_hit)),
        }
    return out


def _detect_kgat_weights() -> Tuple[bool, Optional[str]]:
    weight_dir = os.path.join(KGATAX_TRAIN_DATA_DIR, "weights")
    if not os.path.isdir(weight_dir):
        return False, f"weights_dir_missing:{weight_dir}"
    files = [
        f
        for f in os.listdir(weight_dir)
        if str(f).startswith("best_model_epoch_") and str(f).endswith(".pth")
    ]
    if not files:
        return False, f"no_best_weights_in:{weight_dir}"
    last = sorted(files)[-1]
    return True, f"found:{os.path.join(weight_dir, last)}"


def _load_gold_queries(conn: sqlite3.Connection, mode: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    cur = conn.cursor()

    def _has_table(name: str) -> bool:
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return bool(cur.fetchone())

    def _cols(name: str) -> List[str]:
        cur.execute(f"PRAGMA table_info({name})")
        return [str(r[1]) for r in cur.fetchall()]

    def _pick_first(cols: List[str], cands: Sequence[str]) -> Optional[str]:
        s = set(cols)
        for c in cands:
            if c in s:
                return c
        return None

    if _has_table("gold_samples"):
        cols = _cols("gold_samples")
        q_col = _pick_first(cols, ["query_id", "qid", "queryId"])
        jd_col = _pick_first(cols, ["jd_text", "jd", "job_text", "query_text"])
        sid_col = _pick_first(cols, ["securityId", "security_id", "job_id", "jobId"])
        aid_col = _pick_first(cols, ["author_id", "aid", "authorId"])
        gl_col = _pick_first(cols, ["gold_label", "label", "gold"])
        if q_col and jd_col and aid_col and gl_col:
            if mode == "quick":
                cur.execute(
                    f"SELECT DISTINCT {q_col} FROM gold_samples ORDER BY {q_col}"
                )
                qids = [str(r[0]) for r in cur.fetchall()]
                if limit is not None:
                    qids = qids[: int(limit)]
                rows: List[Any] = []
                for qid in qids:
                    cur.execute(
                        f"SELECT {q_col}, "
                        f"{sid_col if sid_col else 'NULL'}, "
                        f"{jd_col}, {aid_col}, {gl_col} "
                        f"FROM gold_samples WHERE {q_col}=?",
                        (qid,),
                    )
                    rows.extend(cur.fetchall())
            else:
                if limit is None:
                    cur.execute(
                        f"SELECT {q_col}, {sid_col if sid_col else 'NULL'}, "
                        f"{jd_col}, {aid_col}, {gl_col} FROM gold_samples "
                        f"ORDER BY {q_col}, {aid_col}"
                    )
                    rows = cur.fetchall()
                else:
                    cur.execute(
                        f"SELECT DISTINCT {q_col} FROM gold_samples ORDER BY {q_col}"
                    )
                    qids = [str(r[0]) for r in cur.fetchall()]
                    qids = qids[: int(limit)]
                    rows = []
                    for qid in qids:
                        cur.execute(
                            f"SELECT {q_col}, "
                            f"{sid_col if sid_col else 'NULL'}, "
                            f"{jd_col}, {aid_col}, {gl_col} "
                            f"FROM gold_samples WHERE {q_col}=?",
                            (qid,),
                        )
                        rows.extend(cur.fetchall())
            out: List[Dict[str, Any]] = []
            for query_id, security_id, jd_text, author_id, gold_label in rows:
                out.append(
                    {
                        "query_id": str(query_id),
                        "securityId": security_id,
                        "jd_text": jd_text,
                        "author_id": str(author_id),
                        "gold_label_raw": gold_label,
                        "gold_label_3c": _gold_to_3class(gold_label),
                    }
                )
            if out:
                return out

    if not (_has_table("picked_jobs") and _has_table("picked_authors")):
        return []

    pj_cols = _cols("picked_jobs")
    pa_cols = _cols("picked_authors")
    qj_col = _pick_first(pj_cols, ["query_id", "qid", "queryId"])
    jd_col = _pick_first(pj_cols, ["jd_text", "jd", "description", "job_text", "query_text"])
    sid_col = _pick_first(pj_cols, ["securityId", "security_id", "job_id", "jobId"])
    qa_col = _pick_first(pa_cols, ["query_id", "qid", "queryId"])
    aid_col = _pick_first(pa_cols, ["author_id", "aid", "authorId"])
    gl_col = _pick_first(pa_cols, ["gold_label", "label", "gold"])
    if not (qj_col and jd_col and qa_col and aid_col and gl_col):
        return []

    if mode == "quick":
        cur.execute(f"SELECT DISTINCT {qj_col} FROM picked_jobs ORDER BY {qj_col}")
        qids = [str(r[0]) for r in cur.fetchall()]
        if limit is not None:
            qids = qids[: int(limit)]
    else:
        cur.execute(f"SELECT DISTINCT {qj_col} FROM picked_jobs ORDER BY {qj_col}")
        qids = [str(r[0]) for r in cur.fetchall()]
        if limit is not None:
            qids = qids[: int(limit)]

    out2: List[Dict[str, Any]] = []
    for qid in qids:
        cur.execute(
            f"SELECT {sid_col if sid_col else 'NULL'}, {jd_col} FROM picked_jobs "
            f"WHERE {qj_col}=? LIMIT 1",
            (qid,),
        )
        job_row = cur.fetchone()
        if job_row:
            security_id = job_row[0]
            jd_text = job_row[1] or ""
        else:
            security_id = None
            jd_text = ""
        cur.execute(
            f"SELECT {aid_col}, {gl_col} FROM picked_authors WHERE {qa_col}=?",
            (qid,),
        )
        arows = cur.fetchall()
        for author_id, gold_label in arows:
            out2.append(
                {
                    "query_id": str(qid),
                    "securityId": security_id,
                    "jd_text": jd_text,
                    "author_id": str(author_id),
                    "gold_label_raw": gold_label,
                    "gold_label_3c": _gold_to_3class(gold_label),
                }
            )
    return out2


def _load_dataset_splits_map(conn: sqlite3.Connection) -> Optional[Dict[str, str]]:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dataset_splits'"
        )
        if not cur.fetchone():
            return None
        cur.execute("SELECT query_id, split_name FROM dataset_splits")
        rows = cur.fetchall()
    except sqlite3.Error:
        return None
    return {str(a): str(b) for a, b in rows}


def _apply_split_filter(
    query_ids: List[str],
    split_map: Optional[Dict[str, str]],
    split: str,
) -> List[str]:
    if split == "all":
        return list(query_ids)
    if not split_map:
        raise SystemExit(
            "已指定 --split 但无法读取 dataset_splits 表。请先建表/对齐数据，或改用 --split all。"
        )
    if split == "ch05":
        allow = {"dev", "test"}
    else:
        allow = {split}
    out: List[str] = []
    missing: List[str] = []
    for qid in query_ids:
        sn = split_map.get(str(qid))
        if sn is None:
            missing.append(str(qid))
            continue
        if sn in allow:
            out.append(str(qid))
    if missing:
        logger.warning(
            "以下 query 不在 dataset_splits 中，已从本 run 排除（共 %s 个）：%s",
            len(missing),
            ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""),
        )
    if not out:
        raise SystemExit(
            f"--split {split} 过滤后没有可评估的 query。请检查 dataset_splits 与金标 query 是否一致。"
        )
    return sorted(out)


def _ch20_order_and_groups(
    gold_query_ids: List[str], split_map: Dict[str, str]
) -> Tuple[List[str], Dict[str, str]]:
    gset: Set[str] = {str(q) for q in gold_query_ids}
    train_all = sorted(
        str(q)
        for q, s in split_map.items()
        if s == "train" and str(q) in gset
    )
    dvt = sorted(
        str(q)
        for q, s in split_map.items()
        if s in ("dev", "test") and str(q) in gset
    )
    if len(train_all) < 10:
        raise SystemExit(
            f"--split ch20 需要金标中命中 train 至少 10 条，实际 {len(train_all)} 条。 "
            "请补 `dataset_splits` / 金标，或改用 --split ch05|all。"
        )
    if len(dvt) < 10:
        raise SystemExit(
            f"--split ch20 需要金标中 dev+test 共 10 条，实际 {len(dvt)} 条。"
        )
    if len(dvt) > 10:
        logger.warning(
            "ch20: dev+test 在库中有 %s 条，取 query_id 字典序前 10 条作为 holdout 子集。",
            len(dvt),
        )
    train10 = train_all[:10]
    dvt10 = dvt[:10]
    order = train10 + dvt10
    lab: Dict[str, str] = {}
    for q in train10:
        lab[q] = "train_seen"
    for q in dvt10:
        lab[q] = "holdout"
    return order, lab


_QID_NUM_RE = re.compile(r"^[Qq]0*([1-9]|[1-2][0-9]|30)\s*$")
_QID_NUM_RE_LOOSE = re.compile(r"^[Qq]0*(\d+)")


def _qnum_from_query_id(qid: str) -> Optional[int]:
    s = str(qid).strip()
    if not s:
        return None
    m = _QID_NUM_RE.match(s)
    if m:
        return int(m.group(1))
    m2 = _QID_NUM_RE_LOOSE.match(s)
    if m2:
        n = int(m2.group(1))
        if 1 <= n <= 30:
            return n
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 30:
            return n
    return None


def _filter_query_ids_to_wanted(
    query_ids: List[str],
    want_args: Optional[List[str]],
) -> List[str]:
    if not want_args:
        return query_ids
    tokens: List[str] = []
    for w in want_args:
        for part in str(w).replace(",", " ").split():
            p = part.strip()
            if p:
                tokens.append(p)
    if not tokens:
        return query_ids
    out: List[str] = []
    for q in query_ids:
        nq = _qnum_from_query_id(str(q))
        for t in tokens:
            if str(q) == str(t) or str(q).upper() == str(t).upper():
                out.append(q)
                break
            nt = _qnum_from_query_id(t)
            if nq is not None and nt is not None and nq == nt:
                out.append(q)
                break
    return out


def _ch30_order_and_groups(gold_query_ids: List[str]) -> Tuple[List[str], Dict[str, str]]:
    parsed: List[Tuple[int, str]] = []
    for q in gold_query_ids:
        n = _qnum_from_query_id(str(q))
        if n is None or not (1 <= n <= 30):
            continue
        parsed.append((n, str(q)))
    parsed.sort(key=lambda x: x[0])
    if len(parsed) < 30:
        have = {n for n, _ in parsed}
        missing = [i for i in range(1, 31) if i not in have]
        logger.warning(
            "ch30: 金标中可解析的 Q1–Q30 仅 %s/30 条，缺失 %s 个号（前 15: %s）",
            len(parsed),
            len(missing),
            ", ".join(str(x) for x in missing[:15])
            + ("..." if len(missing) > 15 else ""),
        )
    if not parsed:
        return [], {}
    order = [q for _n, q in parsed]
    group: Dict[str, str] = {}
    for n, q in parsed:
        group[q] = "train_dev" if n <= 20 else "holdout"
    return order, group


def _ch15_order_and_groups(gold_query_ids: List[str]) -> Tuple[List[str], Dict[str, str]]:
    parsed: List[Tuple[int, str]] = []
    for q in gold_query_ids:
        n = _qnum_from_query_id(str(q))
        if n is None or not (1 <= n <= 15):
            continue
        parsed.append((n, str(q)))
    parsed.sort(key=lambda x: x[0])
    if len(parsed) < 15:
        have = {n for n, _ in parsed}
        missing = [i for i in range(1, 16) if i not in have]
        logger.warning(
            "ch15: 金标中可解析的 Q1–Q15 仅 %s/15 条，缺失: %s",
            len(parsed),
            ", ".join(str(x) for x in missing[:20]),
        )
    if not parsed:
        return [], {}
    order = [q for _n, q in parsed]
    group: Dict[str, str] = {}
    for n, q in parsed:
        group[q] = "train_dev" if n <= 10 else "holdout"
    return order, group


def _subgroup_eval_rows(
    per_query: List[Dict[str, Any]],
    topk_list: List[int],
) -> List[Dict[str, Any]]:
    mets = ("Recall", "Precision", "NDCG", "Hit")
    out: List[Dict[str, Any]] = []
    specs: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("all", per_query),
        (
            "train_seen",
            [r for r in per_query if (r.get("eval_subgroup") or "") == "train_seen"],
        ),
        (
            "holdout",
            [r for r in per_query if (r.get("eval_subgroup") or "") == "holdout"],
        ),
    ]
    for label, rows in specs:
        if not rows:
            continue
        d: Dict[str, Any] = {"subgroup": label, "n_queries": len(rows)}
        for k in topk_list:
            for m in mets:
                key = f"{m}@{k}"
                s = sum(
                    float(x.get(key) or 0.0)
                    for x in rows
                )
                d[key] = s / max(1, len(rows))
        out.append(d)
    return out


def _mean_metrics_for_rows(
    per_query: List[Dict[str, Any]],
    topk_list: List[int],
) -> Optional[Dict[str, Any]]:
    mets = ("Recall", "Precision", "NDCG", "Hit")
    if not per_query:
        return None
    d: Dict[str, Any] = {"n_queries": len(per_query)}
    for k in topk_list:
        for m in mets:
            key = f"{m}@{k}"
            vals = [
                float(x.get(key) or 0.0)
                for x in per_query
            ]
            d[key] = sum(vals) / max(1, len(vals))
    return d


def _group_gold(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_q: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        qid = str(r["query_id"])
        bucket = by_q.setdefault(
            qid,
            {
                "query_id": qid,
                "securityId": r.get("securityId"),
                "jd_text": (r.get("jd_text") or "") or "",
                "gold_by_author": {},
            },
        )
        g = r.get("gold_label_3c")
        if g is None:
            continue
        aid = str(r["author_id"])
        bucket["gold_by_author"][aid] = int(g)
    return by_q


def _aggregate_strict_g2(
    gold_by_q: Dict[str, Dict[str, Any]],
    ranked_cache: Dict[str, List[str]],
    query_list: List[str],
    topk_list: List[int],
) -> Optional[Dict[str, float]]:
    mets = ("Hit", "Precision", "Recall", "NDCG")
    inner = ("hit", "precision", "recall", "ndcg")
    rows_flat: List[Dict[str, float]] = []
    for qid in query_list:
        rnk = ranked_cache.get(str(qid))
        if not rnk:
            continue
        item = gold_by_q.get(qid) or {}
        gmap: Dict[str, int] = item.get("gold_by_author") or {}
        if not gmap:
            continue
        m1 = _compute_metrics_for_query(
            rnk, gmap, topk_list, unjudged_policy="ignore", hit_rel_min=2
        )
        flat: Dict[str, float] = {}
        for k in topk_list:
            d = m1.get(k) or {}
            for m, mk in zip(mets, inner):
                flat[f"{m}@{k}"] = float(d.get(mk) or 0.0)
        rows_flat.append(flat)
    if not rows_flat:
        return None
    out: Dict[str, float] = {}
    for k in topk_list:
        for M in mets:
            key = f"{M}@{k}"
            out[key] = sum(p[key] for p in rows_flat) / len(rows_flat)
    return out


def _summarize_vector_evidence(ev: Any) -> Dict[str, Any]:
    if not isinstance(ev, dict):
        return {}
    summ = ev.get("summary") if isinstance(ev.get("summary"), dict) else {}
    papers = ev.get("top_papers") if isinstance(ev.get("top_papers"), list) else []
    titles: List[str] = []
    for p in papers[:3]:
        if isinstance(p, dict) and p.get("title"):
            titles.append(str(p["title"])[:120])
    out: Dict[str, Any] = {}
    if summ:
        for k in (
            "best_paper_score",
            "top_evidence_count",
            "max_query_hit_count",
            "max_clause_hit_count",
        ):
            if k in summ:
                out[k] = summ[k]
    if titles:
        out["top_titles"] = titles
    return out


def _summarize_label_evidence(ev: Any) -> Dict[str, Any]:
    terms = extract_terms_from_label_evidence(ev)
    if not terms:
        return {}
    core = sorted(
        (t for t in terms if str(t.get("bucket") or "").lower() == "core"),
        key=lambda x: -float(x.get("score") or 0.0),
    )
    sup = sorted(
        (t for t in terms if str(t.get("bucket") or "").lower() == "support"),
        key=lambda x: -float(x.get("score") or 0.0),
    )
    risky = sorted(
        (t for t in terms if str(t.get("bucket") or "").lower() == "risky"),
        key=lambda x: -float(x.get("score") or 0.0),
    )

    def _names(xs: List[Dict[str, Any]], n: int) -> List[str]:
        o: List[str] = []
        for x in xs[:n]:
            term = x.get("term")
            if term:
                o.append(str(term)[:60])
        return o

    return {
        "core_terms": _names(core, 5),
        "support_terms": _names(sup, 5),
        "risky_terms": _names(risky, 5),
    }


def _mc_random_baseline_for_queries(
    gold_by_q: Dict[str, Dict[str, Any]],
    query_list: List[str],
    topk_list: List[int],
    hit_rel_min: int,
    n_iter: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    if n_iter <= 0 or not query_list:
        return None
    rng = random.Random(int(seed))
    mets = ("recall", "precision", "ndcg", "hit")
    mcap = ("Recall", "Precision", "NDCG", "Hit")
    per_query_flat: List[Dict[str, float]] = []
    for qid in query_list:
        item = gold_by_q.get(qid) or {}
        gmap: Dict[str, int] = item.get("gold_by_author") or {}
        if not gmap:
            continue
        acc_k: Dict[int, Dict[str, float]] = {
            k: {me: 0.0 for me in mets} for k in topk_list
        }
        for _ in range(n_iter):
            order = list(gmap.keys())
            rng.shuffle(order)
            m1 = _compute_metrics_for_query(
                order,
                gmap,
                topk_list,
                unjudged_policy="ignore",
                hit_rel_min=int(hit_rel_min),
            )
            for k in topk_list:
                d = m1.get(k) or {}
                for me in mets:
                    acc_k[k][me] += float(d.get(me) or 0.0)
        n = float(n_iter)
        flat_q: Dict[str, float] = {}
        for k in topk_list:
            for j, me in enumerate(mets):
                cap = mcap[j]
                flat_q[f"{cap}@{k}"] = acc_k[k][me] / n
        per_query_flat.append(flat_q)
    if not per_query_flat:
        return None
    out: Dict[str, float] = {}
    for k in topk_list:
        for j, M in enumerate(mcap):
            key = f"{M}@{k}"
            out[key] = sum(p[key] for p in per_query_flat) / len(per_query_flat)
    return out


def _patch_parse_kgat_args() -> None:
    """TotalCore 初始化会调用 parse_kgat_args()，须避免吞掉本脚本的 sys.argv。"""
    import src.core.total_core as total_core_mod
    import src.infrastructure.database.kgat_ax.kgat_parser.parser_kgat as pkg

    orig = pkg.parse_kgat_args

    def _parse_kgat_no_argv():
        saved = sys.argv[:]
        sys.argv = ["kgat"]
        try:
            return orig()
        finally:
            sys.argv = saved

    pkg.parse_kgat_args = _parse_kgat_no_argv
    total_core_mod.parse_kgat_args = _parse_kgat_no_argv


def main() -> None:
    ap = argparse.ArgumentParser(description="批量推荐评估（人工金标）")
    ap.add_argument(
        "--db-path",
        default=None,
        help="SQLite 主库路径（默认使用 config.DB_PATH）",
    )
    ap.add_argument(
        "--output-dir",
        default=os.path.join("evaluation", "batch_eval_outputs"),
        help="输出目录",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多评估多少条 query（默认不限制；quick/full 均生效）",
    )
    ap.add_argument("--topk", default="5,10,20,50", help="K 值列表，例如 5,10,20,50")
    ap.add_argument("--use-kgat", action="store_true", help="启用 KGAT-AX 精排（不可用则自动降级）")
    ap.add_argument("--save-details", action="store_true", help="保存每条 query 的详细推荐结果")
    ap.add_argument(
        "--capture-stdout",
        action="store_true",
        help="将底层召回/标签路的 print 调试信息重定向到输出目录文件（避免刷屏；推荐开启）",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭按 query 的 tqdm 进度条（管道/重定向/纯日志场景）",
    )
    ap.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick：少量样例；full：完整/尽量全量",
    )
    ap.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "dev", "test", "ch05", "ch20", "ch15", "ch30"),
        help="按表 dataset_splits 过滤 query。ch05=dev+test（10 条）；ch20=20 条对比：train 中前 10 条(按 id) + dev+test 中前 10 条，并写 subgroup_metrics.csv（train_seen vs holdout）。需库中有 dataset_splits。ch15=Q001..Q015（1–10 train_dev，11–15 holdout），不依赖 dataset_splits，并写与 ch30 同型的 paper 表/随机基线/严口径；金标会按 full 全量加载。ch30=论文 30 条金标 Q001..Q030（1–20 train_dev，21–30 holdout），不依赖 dataset_splits，并写 paper_thesis_metrics.csv / random_baseline_paper.csv 等；金标会按 full 全量加载。",
    )
    ap.add_argument(
        "--mc-n",
        type=int,
        default=200,
        help="`--split ch15`/`ch30` 时金标内随机重排基线的 Monte Carlo 次数；0=不写 random_baseline_paper.csv。",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="随机基线用随机种子。",
    )
    ap.add_argument(
        "--hit-rel-min",
        type=int,
        default=1,
        choices=(1, 2),
        help="Hit/Precision/Recall 正例：三分类 g 满足 g>=此值。1=宽（金标 1/2 档算正例）；2=严（仅 2 档）。NDCG 不变。",
    )
    ap.add_argument(
        "--kgat-skip-neo4j-explain",
        action="store_true",
        help="精排解释里不查 Neo4j 补图路径（仍走 KGAT 前向+权重）。可避免 Neo4j 慢/TransientError 卡死，仅解释字段略简。",
    )
    ap.add_argument(
        "--query-id",
        action="append",
        default=None,
        metavar="QID",
        help="可重复传参，仅评估这些 query（在 --split 过滤之后再次收窄），如 --query-id Q027 或 --query-id 27，用于单条复现/诊断。",
    )
    args = ap.parse_args()
    _patch_parse_kgat_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if bool(getattr(args, "kgat_skip_neo4j_explain", False)):
        os.environ["RANKING_EXPLAIN_SKIP_NEO4J"] = "1"
    db_path = (
        os.path.abspath(args.db_path)
        if args.db_path
        else os.path.abspath(_DEFAULT_DB_PATH)
    )
    out_root = os.path.abspath(args.output_dir)
    run_id = _now_tag() + "_" + args.mode + (
        "" if args.split == "all" else "_" + str(args.split)
    )
    out_dir = _ensure_dir(os.path.join(out_root, run_id))
    _ensure_dir(out_dir)
    topk_list = _parse_topk_list(args.topk)
    logger.info("run_id=%s out_dir=%s", run_id, out_dir)
    logger.info("db_path=%s", db_path)
    mc_n = int(getattr(args, "mc_n", 0) or 0)
    random_seed = int(getattr(args, "random_seed", 42) or 42)
    logger.info(
        "mode=%s split=%s limit=%s topk=%s use_kgat=%s save_details=%s hit_rel_min=%s skip_neo4j_explain=%s mc_n=%s random_seed=%s",
        args.mode,
        args.split,
        args.limit,
        topk_list,
        bool(args.use_kgat),
        bool(args.save_details),
        args.hit_rel_min,
        bool(getattr(args, "kgat_skip_neo4j_explain", False)),
        mc_n,
        random_seed,
    )
    if args.capture_stdout:
        logger.info(
            "capture_stdout=True (per-query stdout/stderr will be redirected to files under out_dir)"
        )
    conn = sqlite3.connect(db_path)
    ch20_group: Dict[str, str] = {}
    ch15_group: Dict[str, str] = {}
    ch30_group: Dict[str, str] = {}
    gmode, glimit = str(args.mode), args.limit
    if args.split in ("ch15", "ch30"):
        gmode, glimit = "full", None
    gold_rows = _load_gold_queries(conn, gmode, glimit)
    gold_by_q = _group_gold(gold_rows)
    query_ids = sorted(gold_by_q.keys())
    split_map = _load_dataset_splits_map(conn)
    if args.split == "ch15":
        query_ids, ch15_group = _ch15_order_and_groups(list(query_ids))
        n_ch15 = len(ch15_group)
        if not query_ids:
            conn.close()
            raise SystemExit(
                "已指定 --split ch15 但金标中无可解析的 Q001..Q015（1–15 号）query_id。"
            )
        if args.limit is not None:
            query_ids = query_ids[: int(args.limit)]
            ch15_group = {q: ch15_group[q] for q in query_ids if q in ch15_group}
        logger.info(
            "split_filter=ch15: Q001..Q015 命中 %s 条，应用 limit 后评 %s 条",
            n_ch15,
            len(query_ids),
        )
    elif args.split == "ch30":
        query_ids, ch30_group = _ch30_order_and_groups(list(query_ids))
        n_ch30 = len(ch30_group)
        if not query_ids:
            conn.close()
            raise SystemExit(
                "已指定 --split ch30 但金标中无可解析的 Q001..Q030（1–30 号）query_id。"
            )
        if args.limit is not None:
            query_ids = query_ids[: int(args.limit)]
            ch30_group = {q: ch30_group[q] for q in query_ids if q in ch30_group}
        logger.info(
            "split_filter=ch30: Q001..Q030 命中 %s 条，应用 limit 后评 %s 条",
            n_ch30,
            len(query_ids),
        )
    elif args.split != "all":
        if split_map is None:
            conn.close()
            raise SystemExit(
                "已指定 --split 但主库中不存在可读的 dataset_splits 表。请建表后重试，或去掉 --split。"
            )
        n_before = len(query_ids)
        if args.split == "ch20":
            query_ids, ch20_group = _ch20_order_and_groups(
                list(query_ids), split_map
            )
            logger.info(
                "split_filter=ch20: query %s -> 20 (10 train_seen + 10 dev/test holdout), order fixed",
                n_before,
            )
        else:
            query_ids = _apply_split_filter(list(query_ids), split_map, args.split)
            logger.info(
                "split_filter=%s: query %s -> %s (dataset_splits 命中)",
                args.split,
                n_before,
                len(query_ids),
            )
    if args.query_id:
        n_before = len(query_ids)
        query_ids = _filter_query_ids_to_wanted(list(query_ids), list(args.query_id))
        if args.split == "ch15" and ch15_group:
            ch15_group = {q: ch15_group[q] for q in query_ids if q in ch15_group}
        if args.split == "ch30" and ch30_group:
            ch30_group = {q: ch30_group[q] for q in query_ids if q in ch30_group}
        if args.split == "ch20" and ch20_group:
            ch20_group = {q: ch20_group[q] for q in query_ids if q in ch20_group}
        logger.info(
            "query_id filter: %s -> %s 条: %s",
            n_before,
            len(query_ids),
            query_ids,
        )
    conn.close()
    logger.info("gold query_count=%s", len(query_ids))
    if not query_ids:
        logger.warning(
            "未加载到任何金标 query。请检查 gold_samples 或 picked_jobs/picked_authors 是否有数据。"
        )
    label_counter: Counter = Counter()
    judged_author_total = 0
    for qid in query_ids:
        gmap = gold_by_q[qid]["gold_by_author"]
        judged_author_total += len(gmap)
        label_counter.update(gmap.values())
    recall_sys: Optional[TotalRecallSystem] = None
    core: Optional[TotalCore] = None
    used_kgat_possible = False
    kgat_hint: Optional[str] = None
    if args.use_kgat and query_ids:
        ok, hint = _detect_kgat_weights()
        kgat_hint = hint
        if ok:
            try:
                core = TotalCore()
                recall_sys = core.recall_subsystem
                used_kgat_possible = True
            except Exception as e:
                logger.warning(
                    "KGAT 初始化失败，将降级到候选池排序。err=%s",
                    repr(e),
                )
                core = None
                recall_sys = None
                used_kgat_possible = False
        else:
            logger.warning("KGAT 权重不可用，将降级到候选池排序。reason=%s", hint)
    if recall_sys is None and query_ids:
        recall_sys = TotalRecallSystem()
    recommendations_path = os.path.join(out_dir, "recommendations.jsonl")
    errors_path = os.path.join(out_dir, "errors.jsonl")
    per_query_topk_path = os.path.join(out_dir, "per_query_topk.json")
    rec_rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []
    per_query_topk: Dict[str, Any] = {}
    per_query_metrics: List[Dict[str, Any]] = []
    ranked_cache: Dict[str, List[str]] = {}
    path_combo_counter: Counter = Counter()
    bucket_counter: Counter = Counter()
    success_n = fail_n = 0
    pbar: Optional[tqdm] = None
    if query_ids and not args.no_progress:
        pbar = tqdm(
            total=len(query_ids),
            desc="推荐评估",
            unit="条",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.3,
            ascii=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )

    def _pbar_one() -> None:
        if pbar is not None:
            pbar.update(1)

    for idx, qid in enumerate(query_ids, start=1):
        if pbar is not None:
            pbar.set_postfix_str(" " + str(qid), refresh=False)
        security_id: Optional[str] = None
        start_stage = "init"
        try:
            item = gold_by_q[qid]
            jd_text = (item.get("jd_text") or "").strip()
            security_id = item.get("securityId")  # type: ignore[assignment]
            gold_map: Dict[str, int] = dict(item.get("gold_by_author") or {})
            used_kgat = False
            fallback_reason: Optional[str] = None
            start_stage = "recall_execute"
            if args.capture_stdout:
                cap_dir = _ensure_dir(os.path.join(out_dir, "captured_stdout"))
                cap_path = os.path.join(cap_dir, f"{qid}.txt")
                with open(cap_path, "w", encoding="utf-8", errors="replace") as cap_f, contextlib.redirect_stdout(
                    cap_f
                ), contextlib.redirect_stderr(cap_f):
                    recall_out = recall_sys.execute(jd_text, domain_id=None)
            else:
                recall_out = recall_sys.execute(jd_text, domain_id=None)
            pool = recall_out.get("candidate_pool") if recall_out else None
            if pool is None or not getattr(pool, "candidate_records", None):
                raise RuntimeError("candidate_pool_missing_or_empty")
            records = list(pool.candidate_records)
            candidate_ids = [str(r.author_id) for r in records]
            cand_set = {str(c) for c in candidate_ids}
            pos_geq1 = {
                str(a)
                for a, g in gold_map.items()
                if int(g) >= 1
            }
            in_pool_geq1 = 1 if (pos_geq1 & cand_set) else 0
            ranked_ids = list(candidate_ids)
            if args.use_kgat and used_kgat_possible and core is not None:
                start_stage = "kgat_rank"
                try:
                    qvec, _enc_meta = recall_sys.encoder.encode(jd_text)
                    scores, indices = recall_sys.v_path.job_index.search(qvec, 3)
                    real_job_ids: List[str] = []
                    for i in indices[0]:
                        ii = int(i)
                        if 0 <= ii < len(recall_sys.v_path.job_id_map):
                            real_job_ids.append(recall_sys.v_path.job_id_map[ii])
                    if not real_job_ids:
                        raise RuntimeError("no_anchor_jobs")
                    final = core.ranking_engine.execute_rank(
                        real_job_ids,
                        candidate_ids,
                        filter_domain=None,
                        candidate_pool=pool,
                    )
                    ranked_ids = [
                        str(x.get("author_id"))
                        for x in final
                        if isinstance(x, dict) and x.get("author_id")
                    ]
                    if ranked_ids:
                        used_kgat = True
                    else:
                        fallback_reason = "kgat_returned_empty"
                except Exception as e:
                    used_kgat = False
                    fallback_reason = f"kgat_failed:{type(e).__name__}"
            start_stage = "metrics"
            m = _compute_metrics_for_query(
                ranked_ids,
                gold_map,
                topk_list,
                unjudged_policy="ignore",
                hit_rel_min=int(args.hit_rel_min),
            )
            qres = QueryEvalResult(
                query_id=str(qid),
                security_id=security_id if security_id is not None else None,
                judged_count=len(gold_map),
                judged_pos2_count=sum(1 for _a, g in gold_map.items() if g >= 2),
                judged_pos01_count=sum(1 for _a, g in gold_map.items() if g >= 1),
                rec_count=len(ranked_ids),
                success=True,
                used_kgat=used_kgat,
                fallback_reason=fallback_reason,
                metrics=m,
            )
            by_aid = {str(r.author_id): r for r in records}
            top_for_stats = ranked_ids[: max(topk_list)]
            for aid in top_for_stats:
                rec = by_aid.get(aid)
                if rec is None:
                    continue
                path_combo_counter[_path_combo(
                    bool(rec.from_vector),
                    bool(rec.from_label),
                    bool(rec.from_collab),
                )] += 1
                b = (getattr(rec, "bucket_type", None) or "").strip() or "Z"
                bucket_counter[b] += 1
            if args.save_details:
                topk_max = max(topk_list)
                top_ids = ranked_ids[:topk_max]
                top_records = [
                    by_aid[a]
                    for a in top_ids
                    if by_aid.get(a) is not None
                ]
                aids_for_works = [str(r.author_id) for r in top_records]
                top_works = batch_load_top_works(aids_for_works)
                for rank, aid in enumerate(top_ids, start=1):
                    rec = by_aid.get(aid)
                    if rec is None:
                        continue
                    works = top_works.get(aid) or []
                    rep = None
                    if works and isinstance(works[0], dict):
                        w0 = works[0]
                        rep = {
                            "work_id": w0.get("work_id") or w0.get("id"),
                            "title": w0.get("title"),
                            "year": w0.get("year"),
                        }
                    row = {
                        "run_id": run_id,
                        "query_id": qid,
                        "securityId": security_id,
                        "rank": rank,
                        "author_id": aid,
                        "final_score": None,
                        "candidate_pool_score": float(
                            getattr(rec, "candidate_pool_score", None) or 0.0
                        ),
                        "from_vector": bool(rec.from_vector),
                        "from_label": bool(rec.from_label),
                        "from_collab": bool(rec.from_collab),
                        "bucket_type": getattr(rec, "bucket_type", None),
                        "representative_work": rep,
                        "label_evidence": _summarize_label_evidence(
                            getattr(rec, "label_evidence", None)
                        ),
                        "vector_evidence": _summarize_vector_evidence(
                            getattr(rec, "vector_evidence", None)
                        ),
                        "gold_label": gold_map.get(aid),
                        "used_kgat": used_kgat,
                        "fallback_reason": fallback_reason,
                    }
                    rec_rows.append(row)
            per_query_topk[qid] = {
                "query_id": qid,
                "securityId": security_id,
                "topk_max": max(topk_list),
                "ranked_author_ids": ranked_ids,
                "used_kgat": used_kgat,
                "fallback_reason": fallback_reason,
            }
            subg = (
                (ch20_group.get(str(qid), "") if ch20_group else "")
                or (ch15_group.get(str(qid), "") if ch15_group else "")
                or (ch30_group.get(str(qid), "") if ch30_group else "")
            )
            row: Dict[str, Any] = {
                "query_id": qid,
                "securityId": security_id,
                "eval_subgroup": subg,
                "judged_authors": len(gold_map),
                "pos2_total": sum(1 for _a, g in gold_map.items() if g >= 2),
                "hit_relevant_total": sum(
                    1 for _a, g in gold_map.items() if g >= int(args.hit_rel_min)
                ),
                "used_kgat": int(bool(used_kgat)),
                "fallback_reason": fallback_reason or "",
            }
            for k in topk_list:
                mk = m[k]
                for met in ("recall", "precision", "ndcg", "hit"):
                    cap = met.capitalize()
                    row[f"{cap}@{k}"] = round(float(mk[met]), 6)
                row[f"TopJudged@{k}"] = int(mk["top_judged_n"])
            if args.split in ("ch15", "ch30"):
                row["in_pool_geq1"] = in_pool_geq1
                row["candidate_n"] = len(candidate_ids)
            per_query_metrics.append(row)
            if args.split in ("ch15", "ch30"):
                ranked_cache[str(qid)] = list(ranked_ids)
            success_n += 1
            logger.info(
                "[%s/%s] query_id=%s cand=%s topK=%s kgat=%s ok",
                idx,
                len(query_ids),
                qid,
                len(candidate_ids),
                max(topk_list),
                used_kgat,
            )
        except Exception as e:
            fail_n += 1
            err_rows.append(
                {
                    "run_id": run_id,
                    "query_id": qid,
                    "securityId": security_id,
                    "stage": start_stage,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )
            logger.error(
                "[%s/%s] query_id=%s failed stage=%s err=%s",
                idx,
                len(query_ids),
                qid,
                start_stage,
                repr(e),
            )
        _pbar_one()
    if pbar is not None:
        pbar.close()
    split_desc = "all"
    if args.split == "ch05":
        split_desc = "dev+test(10) from dataset_splits"
    elif args.split == "ch20":
        split_desc = "ch20: 10 train (first 10 by id) + 10 dev+test holdout; see subgroup_metrics.csv"
    elif args.split == "ch15":
        split_desc = "ch15: Q001..Q015 from gold (1-10=train_dev, 11-15=holdout); see paper_thesis_metrics.csv, in_pool_geq1 in per_query_metrics"
    elif args.split == "ch30":
        split_desc = "ch30: Q001..Q030 from gold (1-20=train_dev, 21-30=holdout); see paper_thesis_metrics.csv, in_pool_geq1 in per_query_metrics"
    elif args.split != "all":
        split_desc = f"{args.split} from dataset_splits"
    _pos_desc = (
        f"Hit/P/Recall: gold 3class g>={int(args.hit_rel_min)} "
        f"(1=rel1+2 wide, 2=rel2 only)"
    )
    _mn = (
        "main metrics on judged only; unjudged ignore; "
        + _pos_desc
        + "; NDCG=graded 2/1/0"
    )
    if args.split == "ch05":
        _mn += "; query_ids=dev∪test per dataset_splits"
    if args.split == "ch20":
        _mn += (
            "; ch20=10 train_seen + 10 holdout, subgroup table=subgroup_metrics.csv; "
            "NDCG/Hit 等同口径于 overall(20) 为简单平均、且含 train 与 holdout 分组平均"
        )
    if args.split == "ch15":
        _mn += (
            f"; ch15: paper 表=paper_thesis_metrics.csv, random={args.mc_n} 次/条（0=关） seed={random_seed}"
            "；Labeled- P@K 在已标作者上；in_pool=粗排池含 g>=1 正例；strict 严口径=thesis_metrics_strict_g2.csv 若生成了 ranked"
        )
    if args.split == "ch30":
        _mn += (
            f"; ch30: paper 表=paper_thesis_metrics.csv, random={args.mc_n} 次/条（0=关） seed={random_seed}"
            "；Labeled- P@K 在已标作者上；in_pool=粗排池含 g>=1 正例；strict 严口径=thesis_metrics_strict_g2.csv 若生成了 ranked"
        )
    if bool(getattr(args, "kgat_skip_neo4j_explain", False)):
        _mn += "; Neo4j explain skipped (KGAT forward still on)"
    overall_row: Dict[str, Any] = {
        "run_id": run_id,
        "mode": args.mode,
        "split_filter": args.split,
        "split_description": split_desc,
        "query_total": len(query_ids),
    }
    overall_row["query_success"] = success_n
    overall_row["query_failed"] = fail_n
    overall_row["judged_author_total"] = judged_author_total
    overall_row["label_2_count"] = int(label_counter.get(2, 0))
    overall_row["label_1_count"] = int(label_counter.get(1, 0))
    overall_row["label_0_count"] = int(label_counter.get(0, 0))
    overall_row["avg_judged_authors_per_query"] = (
        judged_author_total / len(query_ids) if query_ids else 0.0
    )
    overall_row["use_kgat_flag"] = int(bool(args.use_kgat))
    overall_row["kgat_weights_hint"] = kgat_hint or ""
    overall_row["hit_rel_min"] = int(args.hit_rel_min)
    overall_row["metrics_note"] = _mn
    if per_query_metrics:
        for k in topk_list:
            for met in ("Recall", "Precision", "NDCG", "Hit"):
                key = f"{met}@{k}"
                vals = [float(r.get(key) or 0.0) for r in per_query_metrics]
                overall_row[key] = sum(vals) / max(1, len(vals))
    overall_metrics = [overall_row]
    total_path = sum(path_combo_counter.values()) or 1
    path_rows: List[Dict[str, Any]] = []
    for name in (
        "vector only",
        "label only",
        "vector + label",
        "vector + collab",
        "label + collab",
        "all three",
        "collab only",
        "unknown",
    ):
        c = int(path_combo_counter.get(name, 0))
        path_rows.append({"path_combo": name, "count": c, "ratio": c / total_path})
    total_bucket = sum(bucket_counter.values()) or 1
    bucket_rows: List[Dict[str, Any]] = []
    for b in ("A", "B", "C", "D", "E", "F", "Z"):
        c = int(bucket_counter.get(b, 0))
        bucket_rows.append({"bucket_type": b, "count": c, "ratio": c / total_bucket})
    label_dist_rows = [
        {"gold_label_3class": int(lbl), "count": int(cnt)}
        for lbl, cnt in sorted(label_counter.items(), key=lambda t: t[0])
    ]
    _write_csv(
        os.path.join(out_dir, "overall_metrics.csv"),
        overall_metrics,
        list(overall_metrics[0].keys()),
    )
    pq_fields = (
        list(per_query_metrics[0].keys())
        if per_query_metrics
        else ["query_id"]
    )
    _write_csv(
        os.path.join(out_dir, "per_query_metrics.csv"),
        per_query_metrics,
        pq_fields,
    )
    if args.split == "ch20" and per_query_metrics and topk_list:
        sub_rows = _subgroup_eval_rows(per_query_metrics, topk_list)
        if sub_rows:
            _write_csv(
                os.path.join(out_dir, "subgroup_metrics.csv"),
                sub_rows,
                list(sub_rows[0].keys()),
            )
    if args.split in ("ch15", "ch30") and per_query_metrics and topk_list:

        def _in_ch30_pool(r: Dict[str, Any]) -> bool:
            v = r.get("in_pool_geq1")
            if v in (1, "1", True):
                return True
            if isinstance(v, (int, float)) and float(v) == 1.0:
                return True
            return False

        def _h(r: Dict[str, Any]) -> bool:
            return (r.get("eval_subgroup") or "") == "holdout"

        def _t(r: Dict[str, Any]) -> bool:
            return (r.get("eval_subgroup") or "") == "train_dev"

        p_all = [r for r in per_query_metrics if (_h(r) or _t(r))]
        hrm = int(args.hit_rel_min)
        nlab = "relaxed g>=1" if hrm == 1 else f"hit_rel_min={hrm}"
        if args.split == "ch30":
            ch_specs: List[Tuple[str, List[Dict[str, Any]]]] = [
                ("all_30", p_all),
                ("holdout_10", [r for r in per_query_metrics if _h(r)]),
                ("train_dev_20", [r for r in per_query_metrics if _t(r)]),
                ("in_pool_all", [r for r in p_all if _in_ch30_pool(r)]),
                ("in_pool_holdout", [r for r in per_query_metrics if _h(r) and _in_ch30_pool(r)]),
                ("in_pool_train_dev", [r for r in per_query_metrics if _t(r) and _in_ch30_pool(r)]),
            ]
        else:
            ch_specs = [
                ("all_15", p_all),
                ("holdout_5", [r for r in per_query_metrics if _h(r)]),
                ("train_dev_10", [r for r in per_query_metrics if _t(r)]),
                ("in_pool_all", [r for r in p_all if _in_ch30_pool(r)]),
                ("in_pool_holdout", [r for r in per_query_metrics if _h(r) and _in_ch30_pool(r)]),
                ("in_pool_train_dev", [r for r in per_query_metrics if _t(r) and _in_ch30_pool(r)]),
            ]
        thesis_rows: List[Dict[str, Any]] = []
        for name, sub in ch_specs:
            if not sub:
                continue
            mm = _mean_metrics_for_rows(sub, topk_list)
            if not mm:
                continue
            line = {
                "subset": name,
                "method": "system",
                "hit_rel_min": hrm,
                "note": f"Labeled-P/Hit/Recall/NDCG, {nlab}, unjudged ignore; see per_query in_pool_geq1",
            }
            line.update(mm)
            thesis_rows.append(line)
        if thesis_rows:
            _write_csv(
                os.path.join(out_dir, "paper_thesis_metrics.csv"),
                thesis_rows,
                list(thesis_rows[0].keys()),
            )
        mcn = int(getattr(args, "mc_n", 0) or 0)
        rseed = int(getattr(args, "random_seed", 42) or 42)
        if mcn > 0:
            rand_out: List[Dict[str, Any]] = []
            for name, sub in ch_specs:
                if not sub:
                    continue
                qlist = [
                    str(r.get("query_id") or "")
                    for r in sub
                    if str(r.get("query_id") or "")
                ]
                qlist = [q for q in qlist if q]
                rm = _mc_random_baseline_for_queries(
                    gold_by_q, qlist, topk_list, hrm, mcn, rseed
                )
                n_eff = len(
                    [
                        q
                        for q in qlist
                        if (gold_by_q.get(q) or {}).get("gold_by_author")
                    ]
                )
                if not rm or n_eff == 0:
                    continue
                rrow: Dict[str, Any] = {
                    "subset": name,
                    "method": "random_baseline",
                    "hit_rel_min": hrm,
                    "mc_n": mcn,
                    "random_seed": rseed,
                    "n_queries": n_eff,
                    "note": "Per-query: shuffle list(gold_by_author keys), mean over shuffles, then over queries",
                }
                for kk, v in rm.items():
                    if isinstance(v, (int, float)):
                        rrow[kk] = round(float(v), 6)
                    else:
                        rrow[kk] = v
                rand_out.append(rrow)
            if rand_out:
                _write_csv(
                    os.path.join(out_dir, "random_baseline_paper.csv"),
                    rand_out,
                    list(rand_out[0].keys()),
                )
        if ranked_cache:
            s2: List[Dict[str, Any]] = []
            for name, sub in ch_specs:
                if not sub:
                    continue
                qlist = [
                    str(r.get("query_id") or "")
                    for r in sub
                    if str(r.get("query_id") or "")
                ]
                qlist = [q for q in qlist if q]
                g2d = _aggregate_strict_g2(
                    gold_by_q, ranked_cache, qlist, topk_list
                )
                if not g2d:
                    continue
                n_ok = len([q for q in qlist if q in ranked_cache])
                srow = {
                    "subset": name,
                    "method": "strict_g2_system",
                    "hit_rel_min": 2,
                    "n_queries": n_ok,
                    "note": "Same ranked list; Hit/P/Rec need g>=2; NDCG with graded 2/1/0",
                }
                for kk, v in g2d.items():
                    srow[kk] = round(float(v), 6)
                s2.append(srow)
            if s2:
                _write_csv(
                    os.path.join(out_dir, "thesis_metrics_strict_g2.csv"),
                    s2,
                    list(s2[0].keys()),
                )
    _write_csv(
        os.path.join(out_dir, "path_breakdown.csv"),
        path_rows,
        ["path_combo", "count", "ratio"],
    )
    _write_csv(
        os.path.join(out_dir, "bucket_breakdown.csv"),
        bucket_rows,
        ["bucket_type", "count", "ratio"],
    )
    _write_csv(
        os.path.join(out_dir, "label_distribution.csv"),
        label_dist_rows,
        ["gold_label_3class", "count"],
    )
    if args.save_details:
        _write_jsonl(recommendations_path, rec_rows)
        with open(per_query_topk_path, "w", encoding="utf-8") as f:
            json.dump(per_query_topk, f, ensure_ascii=False, indent=2)
    if err_rows:
        _write_jsonl(errors_path, err_rows)
    else:
        _write_jsonl(errors_path, [])
    summary_md = os.path.join(out_dir, "run_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Batch recommend eval run summary\n\n")
        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- db_path: `{db_path}`\n")
        f.write(f"- mode: `{args.mode}`\n")
        f.write(f"- split: `{args.split}`\n")
        f.write(f"- limit: `{args.limit}`\n")
        f.write(f"- topk: `{args.topk}`\n")
        f.write(f"- use_kgat_flag: `{bool(args.use_kgat)}`\n")
        f.write(f"- kgat_weights_hint: `{kgat_hint or ''}`\n")
        f.write(f"- query_total: `{len(query_ids)}`\n")
        f.write(f"- query_success: `{success_n}`\n")
        f.write(f"- query_failed: `{fail_n}`\n")
        f.write(f"- judged_author_total: `{judged_author_total}`\n")
        f.write(f"- label_distribution(3class): `{dict(label_counter)}`\n\n")
        f.write("## Notes\n")
        f.write(
            "- 本次评估金标来自 SQLite `gold_samples`（字段 `gold_label` 可能为 3/2/1/0）；脚本映射为三分类 2/1/0。\n"
        )
        f.write("- 主指标默认只在已标注作者上计算（unjudged ignore）。\n")
        f.write(
            f"- Hit/Precision/Recall：`--hit-rel-min={args.hit_rel_min}`，即三分类 g>="
            f"{args.hit_rel_min} 的作者算正例（1=含边缘档，2=仅强档）；NDCG 仍用 2/1/0 分档。\n"
        )
        if args.split == "ch20":
            f.write(
                "- `--split ch20`：20 条=train 前 10(按 id)+dev+test 前 10；`subgroup_metrics.csv` 含 train_seen / holdout 分组平均。\n"
            )
        if args.split == "ch15":
            f.write(
                f"- `--split ch15`：Q001..Q015 金标（1–10 train_dev，11–15 holdout）；`paper_thesis_metrics.csv` 中子组为 `all_15` / `holdout_5` / `train_dev_10` 等；其余与 ch30 相同（`random_baseline_paper` 在 mc_n>0 时；`thesis_metrics_strict_g2`；`in_pool_geq1`）。当前 mc_n={args.mc_n}。\n"
            )
        if args.split == "ch30":
            f.write(
                f"- `--split ch30`：Q001..Q030 金标（1–20 train_dev，21–30 holdout）；`paper_thesis_metrics.csv` 为分组平均；`random_baseline_paper.csv` 在 `--mc-n>0` 时写出（当前 mc_n={args.mc_n}）；`thesis_metrics_strict_g2.csv` 为 g>=2 正例口径，`in_pool_geq1` 表示粗排候选池是否出现至少一名 g>=1 正例作者。\n"
            )
        if bool(getattr(args, "kgat_skip_neo4j_explain", False)):
            f.write(
                "- `--kgat-skip-neo4j-explain`：已启用；精排不查 Neo4j 补证，KGAT 前向及权重仍使用。\n"
            )
        f.write(
            "\n- bucket A--F 仅代表来源组合，不代表质量等级。\n"
        )
        if query_ids:
            f.write("\n## Errors\n")
    logger.info("done. outputs in %s", out_dir)


if __name__ == "__main__":
    main()

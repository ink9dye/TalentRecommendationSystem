"""
KGAT-AX 训练数据生成器。

职责（对齐 README / ch04）：
- 训练样本入口优先基于 candidate_pool.training_pool_records（与 total_recall 精排/训练入口一致）；
  若为空则回退 candidate_records 前 100；再无则 final_top_500 仅作末级兼容。
- 随机大池 train.txt/test.txt：分层正负样本（Strong/Weak Positive 等），与线上「全库随机岗位」扩展集兼容。
- 池监督 JSONL（主口径）：以 training_pool_records 为样本主体；picked_authors（若存在）或 v_gold_samples 为金标覆盖；
  简单弱标签 + sample_weight；可选金标未命中注入（INJECT_MISSING_GOLD）；评估仅用 label_source=gold。
- 可选导出四分支侧车到 train_four_branch.json；池监督行内附带 kgatax sidecar（按 author_id 对齐）。
- 多进程：子进程各持一套 TotalRecallSystem，仅并行 execute(is_training=True)。
"""
import os
import re
import json
import sqlite3
import faiss
import numpy as np
import pandas as pd
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from py2neo import Graph
from config import (
    DB_PATH, KGATAX_TRAIN_DATA_DIR, FEATURE_INDEX_PATH,
    JOB_INDEX_PATH, JOB_MAP_PATH, NEO4J_URI, NEO4J_USER,
    NEO4J_PASSWORD, NEO4J_DATABASE
)
from src.core.recall.total_recall import TotalRecallSystem

# ---------------------------------------------------------------------------
# 多进程召回 worker（Windows spawn 下需在模块顶层可 pickle）
# ---------------------------------------------------------------------------
_MP_RECALL_SYSTEM: Optional[TotalRecallSystem] = None
_MP_JOB_IDX: Optional[Dict[str, int]] = None


def _kgat_mp_worker_init() -> None:
    global _MP_RECALL_SYSTEM, _MP_JOB_IDX
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with open(JOB_MAP_PATH, "r", encoding="utf-8") as f:
        sid_list = json.load(f)
    _MP_JOB_IDX = {str(sid): i for i, sid in enumerate(sid_list)}
    _MP_RECALL_SYSTEM = TotalRecallSystem()
    _MP_RECALL_SYSTEM.v_path.verbose = False
    _MP_RECALL_SYSTEM.l_path.verbose = False


def _kgat_mp_worker_task(spec: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """返回 (securityId, execute 结果)；无索引的岗位返回 (id, None)。"""
    global _MP_RECALL_SYSTEM, _MP_JOB_IDX
    assert _MP_RECALL_SYSTEM is not None and _MP_JOB_IDX is not None
    job = spec["job"]
    jid = str(job["securityId"])
    if jid not in _MP_JOB_IDX:
        return jid, None
    query_text = f"{job.get('job_name') or ''} {job.get('description') or ''} {job.get('skills') or ''}"
    target_domain = spec.get("target_domain")
    res = _MP_RECALL_SYSTEM.execute(query_text, domain_id=target_domain, is_training=True)
    return jid, res


def _kgat_mp_worker_task_gold(spec: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """金标支路：按 query_text（jd_text）与 target_domain=None 执行召回；返回 (query_id, execute 结果)。"""
    global _MP_RECALL_SYSTEM, _MP_JOB_IDX
    assert _MP_RECALL_SYSTEM is not None and _MP_JOB_IDX is not None
    qid = str(spec["query_id"])
    jid = str(spec["securityId"])
    if jid not in _MP_JOB_IDX:
        return qid, None
    query_text = spec["query_text"]
    res = _MP_RECALL_SYSTEM.execute(query_text, domain_id=None, is_training=True)
    return qid, res


def parallel_total_recall_for_specs(
    specs: List[Dict[str, Any]],
    workers: int,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    对 specs 并行执行 TotalRecallSystem.execute(is_training=True)。
    specs 每项: {"job": dict, "target_domain": Optional[str]}
    """
    if not specs:
        return {}
    n = max(1, int(workers))
    n = min(n, cpu_count() or 1)
    if n < 2:
        raise ValueError("parallel_total_recall_for_specs requires workers >= 2")
    chunksize = max(1, len(specs) // (n * 4))
    with Pool(processes=n, initializer=_kgat_mp_worker_init) as pool:
        pairs = pool.map(_kgat_mp_worker_task, specs, chunksize=chunksize)
    return {jid: res for jid, res in pairs}


def parallel_gold_recall_for_specs(
    specs: List[Dict[str, Any]],
    workers: int,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    对金标 query specs 并行执行 TotalRecallSystem.execute(is_training=True, domain_id=None)。
    specs 每项至少: query_id, securityId, query_text
    返回 query_id -> execute 结果。
    """
    if not specs:
        return {}
    n = max(1, int(workers))
    n = min(n, cpu_count() or 1)
    if n < 2:
        raise ValueError("parallel_gold_recall_for_specs requires workers >= 2")
    chunksize = max(1, len(specs) // (n * 4))
    with Pool(processes=n, initializer=_kgat_mp_worker_init) as pool:
        pairs = pool.map(_kgat_mp_worker_task_gold, specs, chunksize=chunksize)
    return {qid: res for qid, res in pairs}


# 分层正负样本标签（README 5.5）
LABEL_STRONG_POS = "strong_pos"
LABEL_WEAK_POS = "weak_pos"
LABEL_HARD_NEG = "hard_neg"
LABEL_COLLAB_NEG = "collab_neg"
LABEL_FIELD_NEG = "field_neg"
LABEL_EASY_NEG = "easy_neg"


# ---------------------------------------------------------------------------
# 金标监督增强支路：dataset_splits + v_gold_samples → JSONL（独立于 train.txt / test.txt）
# ---------------------------------------------------------------------------


def load_dataset_splits(db_path: str) -> Dict[str, str]:
    """
    读取 SQLite 表 dataset_splits。
    返回 query_id -> split_name（'train' | 'dev' | 'test'）。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT query_id, split_name FROM dataset_splits").fetchall()
    finally:
        conn.close()
    return {str(r["query_id"]): str(r["split_name"]) for r in rows}


def load_gold_samples(db_path: str) -> pd.DataFrame:
    """读取视图 v_gold_samples 为 DataFrame。"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM v_gold_samples", conn)
    finally:
        conn.close()
    return df


def validate_gold_splits(split_map: Dict[str, str], gold_df: pd.DataFrame) -> None:
    """
    校验 dataset_splits 为 20/5/5（共 30 query），且 v_gold_samples 共 900 行，
    且 splits 与金标的 query_id 集合一致。
    不通过时抛出 ValueError。
    """
    if len(gold_df) != 900:
        raise ValueError(f"v_gold_samples 期望 900 行，实际为 {len(gold_df)}")
    by_split = Counter(split_map.values())
    if (
        by_split.get("train") != 20
        or by_split.get("dev") != 5
        or by_split.get("test") != 5
    ):
        raise ValueError(
            f"dataset_splits 期望 train/dev/test = 20/5/5，实际计数为 {dict(by_split)}"
        )
    if len(split_map) != 30:
        raise ValueError(f"dataset_splits 期望 30 条 query，实际为 {len(split_map)}")
    gq = set(gold_df["query_id"].astype(str))
    sq = set(split_map.keys())
    if gq != sq:
        raise ValueError(
            "dataset_splits 与 v_gold_samples 的 query_id 集合不一致："
            f"仅 splits 有 {sq - gq}，仅 gold 有 {gq - sq}"
        )


def build_gold_query_specs(
    split_map: Dict[str, str], gold_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    对每个 gold query 组织 spec，包含：
    query_id, securityId, jd_text, split_name, gold_rows（该 query 的金标行）,
    query_text（优先 jd_text）, target_domain=None。
    """
    specs: List[Dict[str, Any]] = []
    for qid, split_name in split_map.items():
        qid_s = str(qid)
        sub = gold_df[gold_df["query_id"].astype(str) == qid_s]
        if sub.empty:
            raise ValueError(f"v_gold_samples 中缺少 query_id={qid_s}")
        row0 = sub.iloc[0]
        jd_text = row0.get("jd_text")
        if jd_text is None or (isinstance(jd_text, float) and np.isnan(jd_text)):
            jd_text = ""
        else:
            jd_text = str(jd_text)
        specs.append(
            {
                "query_id": qid_s,
                "securityId": str(row0["securityId"]),
                "jd_text": jd_text,
                "split_name": split_name,
                "gold_rows": sub.copy(),
                "query_text": jd_text,
                "target_domain": None,
            }
        )
    order = {"train": 0, "dev": 1, "test": 2}
    specs.sort(key=lambda s: (order.get(s["split_name"], 9), s["query_id"]))
    return specs


def _is_nullish(val: Any) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and np.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _series_gold_fields(row: pd.Series) -> Dict[str, Any]:
    """从金标行提取 gold_label / label_level / author_bucket / manual_rank / paper titles。视图列 rank → manual_rank。"""
    gl = row["gold_label"] if "gold_label" in row.index else None
    if _is_nullish(gl):
        gl = None
    ll = row["label_level"] if "label_level" in row.index else None
    if _is_nullish(ll):
        ll = None
    ab = row["author_bucket"] if "author_bucket" in row.index else None
    if _is_nullish(ab):
        ab = None
    mr = row["rank"] if "rank" in row.index else None
    if _is_nullish(mr):
        manual_rank = None
    else:
        try:
            manual_rank = int(mr)
        except (TypeError, ValueError):
            manual_rank = None
    p1 = row["paper_1_title"] if "paper_1_title" in row.index else None
    if _is_nullish(p1):
        p1 = None
    else:
        p1 = str(p1)
    p2 = row["paper_2_title"] if "paper_2_title" in row.index else None
    if _is_nullish(p2):
        p2 = None
    else:
        p2 = str(p2)
    return {
        "gold_label": gl,
        "label_level": ll,
        "author_bucket": ab,
        "manual_rank": manual_rank,
        "paper_1_title": p1,
        "paper_2_title": p2,
    }


def _pool_debug_query_limit() -> int:
    """KGATAX_DEBUG_QUERY_LIMIT 优先；未设置时回退 KGATAX_DEBUG_LIMIT（与随机大池调试一致）。"""
    v = int(os.environ.get("KGATAX_DEBUG_QUERY_LIMIT", "0") or "0")
    if v > 0:
        return v
    return int(os.environ.get("KGATAX_DEBUG_LIMIT", "0") or "0")


def _inject_missing_gold_default() -> bool:
    raw = os.environ.get("INJECT_MISSING_GOLD", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _weak_sample_weight() -> float:
    try:
        return float(os.environ.get("KGATAX_WEAK_SAMPLE_WEIGHT", "0.25") or "0.25")
    except (TypeError, ValueError):
        return 0.25


GOLD_SAMPLE_WEIGHT: float = 1.0


def infer_weak_label(record: Any) -> Tuple[Optional[int], str]:
    """
    轻量弱标签（无 teacher / 蒸馏）。返回 (label_grade, weak_label_type)。
    label_grade: 2 弱正强 / 1 边缘 / 0 弱负；无法标注时为 (None, 'unlabeled')。
    """
    tier = str(getattr(record, "candidate_trust_tier", "") or "")
    bucket = str(getattr(record, "bucket_type", "") or "")
    path_count = int(getattr(record, "path_count", 0) or 0)
    from_label = bool(getattr(record, "from_label", False))
    from_vector = bool(getattr(record, "from_vector", False))
    risk_flags = set(getattr(record, "risk_flags", []) or [])

    serious_risk = bool(
        risk_flags
        & {
            "single_paper_dominated",
            "thin_job_axis_coverage",
            "weak_job_axis_coverage",
            "possible_off_context_dense_match",
            "off_context_dense_match",
            "label_only_low_domain_consistency",
            "label_only_weak_paper_hit",
        }
    )

    if tier in {"strong_consensus", "strong_label"}:
        return 2, "weak_positive"

    if tier in {"strong_vector", "weak_consensus"}:
        return 1, "weak_borderline"

    if tier == "risky_vector" or serious_risk:
        return 0, "weak_negative"

    if from_label:
        return 1, "weak_borderline"

    if from_vector and not from_label and path_count == 1:
        return 0, "weak_negative"

    _ = bucket  # 保留与规范一致；当前规则未直接用 bucket
    return None, "unlabeled"


def train_y_from_grade(label_grade: Optional[int]) -> int:
    """二分类训练目标：grade 1/2 为正，0 与缺失为负。"""
    if label_grade is None:
        return 0
    return 1 if label_grade in (1, 2) else 0


def load_gold_author_map(db_path: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], str]:
    """
    金标映射 (query_id, author_id) -> {gold_label, label_level, note, ...}。
    优先表 picked_authors；不存在则使用 v_gold_samples。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    gold_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source = "v_gold_samples"
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='picked_authors'"
        ).fetchone()
        if row:
            source = "picked_authors"
            q = (
                "SELECT query_id, author_id, gold_label, label_level, note "
                "FROM picked_authors WHERE gold_label IS NOT NULL"
            )
            try:
                rows = conn.execute(q).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    "SELECT query_id, author_id, gold_label, label_level, NULL AS note "
                    "FROM picked_authors WHERE gold_label IS NOT NULL"
                ).fetchall()
            for r in rows:
                key = (str(r["query_id"]), str(r["author_id"]))
                note_v = None
                try:
                    note_v = r["note"]
                except (KeyError, IndexError):
                    pass
                gold_map[key] = {
                    "gold_label": int(r["gold_label"]),
                    "label_level": r["label_level"],
                    "note": note_v,
                }
        else:
            rows = conn.execute(
                "SELECT query_id, author_id, gold_label, label_level FROM v_gold_samples "
                "WHERE gold_label IS NOT NULL"
            ).fetchall()
            for r in rows:
                key = (str(r["query_id"]), str(r["author_id"]))
                gold_map[key] = {
                    "gold_label": int(r["gold_label"]),
                    "label_level": r["label_level"],
                    "note": None,
                }
    finally:
        conn.close()
    return gold_map, source


def binary_label_from_gold(gold_label: Any) -> Any:
    """
    binary_label = 1 when gold_label >= 2
    binary_label = 0 when gold_label == 0
    binary_label = null when gold_label == 1 或其它无法判定情形
    """
    if _is_nullish(gold_label):
        return None
    try:
        g = int(gold_label)
    except (TypeError, ValueError):
        return None
    if g >= 2:
        return 1
    if g == 0:
        return 0
    if g == 1:
        return None
    return None


def attach_gold_labels_to_candidates(
    query_id: str,
    candidate_records: Optional[List[Any]],
    gold_df_query: pd.DataFrame,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    将召回 candidate_records 与金标按 author_id merge，并追加「未被召回覆盖」的金标作者行。
    securityId / jd_text 取自 gold_df_query 首行（与 v_gold_samples 一致）。

    返回 (jsonl 行列表, 召回覆盖的金标作者数, 未覆盖的金标作者数)。
    """
    if gold_df_query.empty:
        raise ValueError(f"金标子表为空: query_id={query_id}")
    _r0 = gold_df_query.iloc[0]
    security_id = str(_r0["securityId"])
    _jd = _r0.get("jd_text")
    if _jd is None or (isinstance(_jd, float) and np.isnan(_jd)):
        jd_text = ""
    else:
        jd_text = str(_jd)

    gold_by_aid: Dict[str, pd.Series] = {}
    for _, row in gold_df_query.iterrows():
        aid = str(row["author_id"])
        gold_by_aid[aid] = row

    cand_aids: set = set()
    rows_out: List[Dict[str, Any]] = []
    records = candidate_records or []

    for r in records:
        aid = str(r.author_id)
        cand_aids.add(aid)
        base: Dict[str, Any] = {
            "split": split_name,
            "query_id": query_id,
            "securityId": security_id,
            "jd_text": jd_text,
            "author_id": aid,
            "author_name": getattr(r, "author_name", None),
            "candidate_pool_score": getattr(r, "candidate_pool_score", None),
            "path_count": getattr(r, "path_count", None),
            "from_vector": bool(getattr(r, "from_vector", False)),
            "from_label": bool(getattr(r, "from_label", False)),
            "from_collab": bool(getattr(r, "from_collab", False)),
            "vector_rank": getattr(r, "vector_rank", None),
            "label_rank": getattr(r, "label_rank", None),
            "collab_rank": getattr(r, "collab_rank", None),
            "vector_score_raw": getattr(r, "vector_score_raw", None),
            "label_score_raw": getattr(r, "label_score_raw", None),
            "collab_score_raw": getattr(r, "collab_score_raw", None),
        }
        g_row = gold_by_aid.get(aid)
        if g_row is not None:
            gf = _series_gold_fields(g_row)
            base.update(gf)
            base["binary_label"] = binary_label_from_gold(base.get("gold_label"))
        else:
            base["gold_label"] = None
            base["label_level"] = None
            base["author_bucket"] = None
            base["manual_rank"] = None
            base["paper_1_title"] = None
            base["paper_2_title"] = None
            base["binary_label"] = None
        rows_out.append(base)

    covered = len(set(gold_by_aid.keys()) & cand_aids)
    missing_aids = [a for a in gold_by_aid.keys() if a not in cand_aids]

    for aid in missing_aids:
        grow = gold_by_aid[aid]
        gf = _series_gold_fields(grow)
        name = grow["author_name"] if "author_name" in grow.index else None
        if _is_nullish(name):
            an = None
        else:
            an = str(name)
        rows_out.append(
            {
                "split": split_name,
                "query_id": query_id,
                "securityId": security_id,
                "jd_text": jd_text,
                "author_id": aid,
                "author_name": an,
                **gf,
                "candidate_pool_score": None,
                "path_count": None,
                "from_vector": None,
                "from_label": None,
                "from_collab": None,
                "vector_rank": None,
                "label_rank": None,
                "collab_rank": None,
                "vector_score_raw": None,
                "label_score_raw": None,
                "collab_score_raw": None,
                "binary_label": binary_label_from_gold(gf.get("gold_label")),
            }
        )

    missed = len(missing_aids)
    return rows_out, covered, missed


def generate_gold_supervised_exports(
    generator: "KGATAXTrainingGenerator",
    recall_workers: int = 0,
) -> Dict[str, Any]:
    """
    金标监督增强：基于 dataset_splits + v_gold_samples 对 30 个 query 跑三路召回，
    merge 后写出 train_gold_supervised.jsonl / dev_gold_supervised.jsonl / test_gold_supervised.jsonl。

    不改变 train.txt / test.txt。返回统计信息供主流程汇总日志使用。
    """
    out_dir = generator.output_dir
    split_map = load_dataset_splits(DB_PATH)
    gold_df = load_gold_samples(DB_PATH)
    validate_gold_splits(split_map, gold_df)
    specs = build_gold_query_specs(split_map, gold_df)

    by_split_lines: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    query_counts = Counter()
    stats_split: Dict[str, Dict[str, Any]] = {
        "train": {"hits": 0, "missed": 0, "queries": 0},
        "dev": {"hits": 0, "missed": 0, "queries": 0},
        "test": {"hits": 0, "missed": 0, "queries": 0},
    }

    rw = int(recall_workers or 0)
    results_by_qid: Dict[str, Optional[Dict[str, Any]]] = {}

    if rw >= 2:
        print(
            f"[*] 金标支路多进程召回: workers={min(rw, cpu_count() or 1)}，"
            f"queries={len(specs)}，is_training=True, domain_id=None",
            flush=True,
        )
        gold_specs_parallel = [
            {
                "query_id": s["query_id"],
                "securityId": s["securityId"],
                "query_text": s["query_text"],
            }
            for s in specs
        ]
        results_by_qid = parallel_gold_recall_for_specs(gold_specs_parallel, rw)
    else:
        rs = generator._ensure_recall_system()
        for s in tqdm(specs, desc="Gold-supervised recall"):
            jid = str(s["securityId"])
            if jid not in generator.job_id_to_idx:
                print(
                    f"[WARN] 金标 query_id={s['query_id']} securityId={jid} 不在 JOB_MAP，跳过召回。",
                    flush=True,
                )
                results_by_qid[s["query_id"]] = None
                continue
            results_by_qid[s["query_id"]] = rs.execute(
                s["query_text"], domain_id=None, is_training=True
            )

    for s in tqdm(specs, desc="Gold-supervised merge & export"):
        sp_name = s["split_name"]
        qid = s["query_id"]
        query_counts[sp_name] += 1
        stats_split[sp_name]["queries"] += 1

        rec = results_by_qid.get(qid)
        pool = rec.get("candidate_pool") if rec else None
        records = (
            generator._select_training_pool_records(pool)
            if pool is not None
            else []
        )
        if rec is None or pool is None:
            print(
                f"[WARN] 金标 query_id={qid} 无有效召回结果，仅导出金标行（召回字段为空）。",
                flush=True,
            )
            records = []

        gsub = s["gold_rows"]
        lines, hit_n, miss_n = attach_gold_labels_to_candidates(
            qid,
            records,
            gsub,
            sp_name,
        )
        stats_split[sp_name]["hits"] += hit_n
        stats_split[sp_name]["missed"] += miss_n
        by_split_lines[sp_name].extend(lines)

    out_names = {
        "train": os.path.join(out_dir, "train_gold_supervised.jsonl"),
        "dev": os.path.join(out_dir, "dev_gold_supervised.jsonl"),
        "test": os.path.join(out_dir, "test_gold_supervised.jsonl"),
    }
    for split_k, path in out_names.items():
        with open(path, "w", encoding="utf-8") as wf:
            for obj in by_split_lines[split_k]:
                wf.write(
                    json.dumps(obj, ensure_ascii=False, default=str) + "\n"
                )

    dist_per_split: Dict[str, Counter] = {}
    for split_k in ("train", "dev", "test"):
        dist_per_split[split_k] = Counter()
        for obj in by_split_lines[split_k]:
            gl = obj.get("gold_label")
            if not _is_nullish(gl):
                try:
                    dist_per_split[split_k][int(gl)] += 1
                except (TypeError, ValueError):
                    dist_per_split[split_k][str(gl)] += 1

    return {
        "query_counts": dict(query_counts),
        "line_counts": {k: len(v) for k, v in by_split_lines.items()},
        "stats_split": stats_split,
        "gold_label_dist": {k: dict(v) for k, v in dist_per_split.items()},
        "paths": out_names,
    }


def _training_pool_supervised_row_from_record(
    *,
    split_name: str,
    query_id: str,
    security_id: str,
    jd_text: str,
    record: Any,
    label_grade: Optional[int],
    train_y: int,
    label_source: str,
    sample_weight: float,
    recall_hit: int,
    forced_gold_injection: int,
    weak_label_type: Optional[str],
) -> Dict[str, Any]:
    aid = str(getattr(record, "author_id", ""))
    rf = getattr(record, "risk_flags", None) or []
    sf = getattr(record, "sampleability_flags", None) or []
    row: Dict[str, Any] = {
        "split": split_name,
        "query_id": query_id,
        "securityId": security_id,
        "jd_text": jd_text,
        "author_id": aid,
        "author_name": getattr(record, "author_name", None),
        "label_grade": label_grade,
        "train_y": train_y,
        "label_source": label_source,
        "sample_weight": float(sample_weight),
        "recall_hit": int(recall_hit),
        "forced_gold_injection": int(forced_gold_injection),
        "weak_label_type": weak_label_type,
        "from_vector": bool(getattr(record, "from_vector", False)),
        "from_label": bool(getattr(record, "from_label", False)),
        "from_collab": bool(getattr(record, "from_collab", False)),
        "path_count": int(getattr(record, "path_count", 0) or 0),
        "vector_rank": getattr(record, "vector_rank", None),
        "label_rank": getattr(record, "label_rank", None),
        "collab_rank": getattr(record, "collab_rank", None),
        "candidate_pool_score": getattr(record, "candidate_pool_score", None),
        "bucket_type": getattr(record, "bucket_type", None),
        "candidate_trust_tier": str(getattr(record, "candidate_trust_tier", "") or ""),
        "risk_flags": list(rf) if isinstance(rf, (list, tuple, set)) else [],
        "sampleability_flags": list(sf) if isinstance(sf, (list, tuple, set)) else [],
    }
    return row


def _injected_gold_training_row(
    *,
    split_name: str,
    query_id: str,
    security_id: str,
    jd_text: str,
    author_id: str,
    author_name: Optional[str],
    label_grade: int,
    train_y: int,
    gold_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "split": split_name,
        "query_id": query_id,
        "securityId": security_id,
        "jd_text": jd_text,
        "author_id": author_id,
        "author_name": author_name,
        "label_grade": int(label_grade),
        "train_y": int(train_y),
        "label_source": "gold",
        "sample_weight": float(GOLD_SAMPLE_WEIGHT),
        "recall_hit": 0,
        "forced_gold_injection": 1,
        "weak_label_type": None,
        "from_vector": None,
        "from_label": None,
        "from_collab": None,
        "path_count": None,
        "vector_rank": None,
        "label_rank": None,
        "collab_rank": None,
        "candidate_pool_score": None,
        "bucket_type": None,
        "candidate_trust_tier": "",
        "risk_flags": [],
        "sampleability_flags": [],
        "gold_label_level": gold_meta.get("label_level"),
        "gold_note": gold_meta.get("note"),
    }


def generate_training_pool_supervised_exports(
    generator: "KGATAXTrainingGenerator",
    recall_workers: int = 0,
) -> Dict[str, Any]:
    """
    池内精排监督 JSONL：主体为 training_pool_records；金标覆盖弱标；可选注入未命中金标作者。
    写出 train/dev/test_pool_supervised.jsonl；返回汇总统计。
    """
    out_dir = generator.output_dir
    q_lim = _pool_debug_query_limit()
    inject_miss = _inject_missing_gold_default()
    weak_w = _weak_sample_weight()

    split_map = load_dataset_splits(DB_PATH)
    gold_df = load_gold_samples(DB_PATH)
    validate_gold_splits(split_map, gold_df)
    gold_map, gold_src = load_gold_author_map(DB_PATH)

    specs = build_gold_query_specs(split_map, gold_df)
    if q_lim > 0:
        # 调试时尽量覆盖 train/dev/test，避免 dev/test 为空导致评估口径无法验证
        by = {"train": [], "dev": [], "test": []}
        for s in specs:
            sp = str(s.get("split_name") or "")
            if sp in by:
                by[sp].append(s)
        picked: List[Dict[str, Any]] = []
        # round-robin: train -> dev -> test
        order = ["train", "dev", "test"]
        while len(picked) < q_lim and any(by[k] for k in order):
            for k in order:
                if len(picked) >= q_lim:
                    break
                if by[k]:
                    picked.append(by[k].pop(0))
        specs = picked
        print(
            f"[*] 池监督: DEBUG query limit → 处理 query 数={len(specs)} "
            f"(KGATAX_DEBUG_QUERY_LIMIT / KGATAX_DEBUG_LIMIT)",
            flush=True,
        )

    print(
        f"[*] 池监督: 金标映射来源={gold_src}，条目={len(gold_map)}，"
        f"INJECT_MISSING_GOLD={inject_miss}，weak_sample_weight={weak_w}",
        flush=True,
    )

    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    rw = int(recall_workers or 0)
    results_by_qid: Dict[str, Optional[Dict[str, Any]]] = {}

    if rw >= 2:
        gold_specs_parallel = [
            {
                "query_id": s["query_id"],
                "securityId": s["securityId"],
                "query_text": s["query_text"],
            }
            for s in specs
        ]
        print(
            f"[*] 池监督多进程召回: workers={min(rw, cpu_count() or 1)}，queries={len(specs)}",
            flush=True,
        )
        results_by_qid = parallel_gold_recall_for_specs(gold_specs_parallel, rw)
    else:
        rs = generator._ensure_recall_system()
        for s in tqdm(specs, desc="Pool-supervised recall"):
            jid = str(s["securityId"])
            if jid not in generator.job_id_to_idx:
                print(
                    f"[WARN] 池监督 query_id={s['query_id']} securityId={jid} 不在 JOB_MAP，跳过召回。",
                    flush=True,
                )
                results_by_qid[s["query_id"]] = None
                continue
            results_by_qid[s["query_id"]] = rs.execute(
                s["query_text"], domain_id=None, is_training=True
            )

    gold_recall_hit = 0
    gold_recall_miss = 0
    forced_gold_injection_count = 0
    training_pool_total = 0
    gold_sample_total = 0
    label_source_counts: Counter = Counter()
    label_grade_counts: Counter = Counter()
    weak_label_type_counts: Counter = Counter()
    sample_w_gold: List[float] = []
    sample_w_weak: List[float] = []
    per_query_stats: List[Dict[str, Any]] = []

    for s in tqdm(specs, desc="Pool-supervised build"):
        qid = str(s["query_id"])
        sp_name = str(s["split_name"])
        rec = results_by_qid.get(qid)
        pool = rec.get("candidate_pool") if rec else None
        records = generator._select_training_pool_records(pool) if pool else []
        training_pool_total += len(records)

        if rec is None or pool is None:
            print(
                f"[WARN] 池监督 query_id={qid} 无有效召回；仅处理金标注入（若有）。",
                flush=True,
            )

        sidecar_list = (rec or {}).get("kgatax_sidecar_rows") or []
        side_by_aid = {
            str(x.get("author_id")): x
            for x in sidecar_list
            if x.get("author_id") is not None
        }

        security_id = str(s["securityId"])
        jd_text = str(s.get("jd_text") or "")
        gsub = s["gold_rows"]

        gold_aids_set = {
            str(r["author_id"])
            for _, r in gsub.iterrows()
            if not _is_nullish(r.get("gold_label"))
        }
        pool_aids = {str(getattr(r, "author_id", "")) for r in records}
        pool_aids.discard("")

        gold_in_pool = gold_aids_set & pool_aids
        gold_not_in_pool = gold_aids_set - pool_aids
        gold_recall_hit += len(gold_in_pool)
        gold_recall_miss += len(gold_not_in_pool)

        name_by_aid: Dict[str, Optional[str]] = {}
        for _, grow in gsub.iterrows():
            aid_g = str(grow["author_id"])
            nm = grow.get("author_name")
            if not _is_nullish(nm):
                name_by_aid[aid_g] = str(nm)
            elif aid_g not in name_by_aid:
                name_by_aid[aid_g] = None

        weak_count = 0
        rows_out: List[Dict[str, Any]] = []

        for r in records:
            aid = str(r.author_id)
            key = (qid, aid)
            ginfo = gold_map.get(key)

            if ginfo is not None:
                label_grade = int(ginfo["gold_label"])
                label_source = "gold"
                sample_weight = GOLD_SAMPLE_WEIGHT
                train_y = train_y_from_grade(label_grade)
                weak_label_type: Optional[str] = None
                gold_sample_total += 1
                sample_w_gold.append(sample_weight)
            else:
                wg, wtype = infer_weak_label(r)
                weak_label_type = wtype
                if wg is None:
                    label_grade = None
                    label_source = "unlabeled"
                    sample_weight = 0.0
                    train_y = 0
                else:
                    label_source = "weak"
                    label_grade = wg
                    sample_weight = weak_w
                    train_y = train_y_from_grade(wg)
                    weak_count += 1
                    sample_w_weak.append(sample_weight)

            gmeta = gold_map.get((qid, aid))
            row_d = _training_pool_supervised_row_from_record(
                split_name=sp_name,
                query_id=qid,
                security_id=security_id,
                jd_text=jd_text,
                record=r,
                label_grade=label_grade,
                train_y=train_y,
                label_source=label_source,
                sample_weight=sample_weight,
                recall_hit=1,
                forced_gold_injection=0,
                weak_label_type=weak_label_type,
            )
            if gmeta is not None:
                row_d["gold_label_level"] = gmeta.get("label_level")
                row_d["gold_note"] = gmeta.get("note")
            sc = side_by_aid.get(aid)
            if sc is not None:
                row_d["kgatax_sidecar"] = sc

            label_source_counts[label_source] += 1
            if label_grade is not None:
                label_grade_counts[int(label_grade)] += 1
            if label_source in ("weak", "unlabeled"):
                weak_label_type_counts[str(weak_label_type or "unlabeled")] += 1
            rows_out.append(row_d)

        if inject_miss:
            for aid in sorted(gold_not_in_pool):
                key = (qid, aid)
                ginfo = gold_map.get(key)
                if ginfo is None:
                    continue
                label_grade = int(ginfo["gold_label"])
                train_y = train_y_from_grade(label_grade)
                row_d = _injected_gold_training_row(
                    split_name=sp_name,
                    query_id=qid,
                    security_id=security_id,
                    jd_text=jd_text,
                    author_id=aid,
                    author_name=name_by_aid.get(aid),
                    label_grade=label_grade,
                    train_y=train_y,
                    gold_meta=ginfo,
                )
                rows_out.append(row_d)
                forced_gold_injection_count += 1
                gold_sample_total += 1
                sample_w_gold.append(GOLD_SAMPLE_WEIGHT)
                label_source_counts["gold"] += 1
                label_grade_counts[label_grade] += 1

        by_split[sp_name].extend(rows_out)
        per_query_stats.append(
            {
                "query_id": qid,
                "training_pool_count": len(records),
                "gold_count": len(gold_aids_set),
                "gold_hit": len(gold_in_pool),
                "gold_miss": len(gold_not_in_pool),
                "weak_count": weak_count,
            }
        )

    out_names = {
        "train": os.path.join(out_dir, "train_pool_supervised.jsonl"),
        "dev": os.path.join(out_dir, "dev_pool_supervised.jsonl"),
        "test": os.path.join(out_dir, "test_pool_supervised.jsonl"),
    }
    for split_k, path in out_names.items():
        with open(path, "w", encoding="utf-8") as wf:
            for obj in by_split[split_k]:
                wf.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def _summ_weights(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(xs),
            "min": float(min(xs)),
            "max": float(max(xs)),
            "mean": float(sum(xs) / len(xs)),
        }

    train_queries = {str(s["query_id"]) for s in specs if s["split_name"] == "train"}
    test_queries = {str(s["query_id"]) for s in specs if s["split_name"] == "test"}
    train_sample_count = len(by_split["train"])
    test_gold_sample_count = sum(
        1
        for o in by_split["test"]
        if str(o.get("label_source")) == "gold"
    )

    print("\n" + "=" * 60, flush=True)
    print("[池监督 JSONL] 统计汇总", flush=True)
    print("=" * 60, flush=True)
    print(f"  query_count (已处理): {len(specs)}", flush=True)
    print(f"  training_pool_total (行基数合计): {training_pool_total}", flush=True)
    print(f"  gold_sample_total (金标样本行): {gold_sample_total}", flush=True)
    print(f"  gold_recall_hit (金标作者 ∩ 训练池): {gold_recall_hit}", flush=True)
    print(f"  gold_recall_miss (金标作者 \\ 训练池): {gold_recall_miss}", flush=True)
    print(f"  forced_gold_injection_count: {forced_gold_injection_count}", flush=True)
    print(f"  label_source_counts: {dict(label_source_counts)}", flush=True)
    print(f"  label_grade_counts: {dict(label_grade_counts)}", flush=True)
    print(f"  weak_label_type_counts: {dict(weak_label_type_counts)}", flush=True)
    print(f"  sample_weight_summary: gold={_summ_weights(sample_w_gold)} weak={_summ_weights(sample_w_weak)}", flush=True)
    print(f"  train_query_count: {len(train_queries)}", flush=True)
    print(f"  test_query_count: {len(test_queries)}", flush=True)
    print(f"  train_sample_count: {train_sample_count}", flush=True)
    print(f"  test_gold_sample_count (评估锚点): {test_gold_sample_count}", flush=True)
    print("  --- per query (前 20 条) ---", flush=True)
    for row in per_query_stats[:20]:
        print(f"    {row}", flush=True)
    if len(per_query_stats) > 20:
        print(f"    ... 其余 {len(per_query_stats) - 20} 条省略", flush=True)
    print("=" * 60 + "\n", flush=True)

    return {
        "gold_map_source": gold_src,
        "query_count": len(specs),
        "training_pool_total": training_pool_total,
        "gold_sample_total": gold_sample_total,
        "gold_recall_hit": gold_recall_hit,
        "gold_recall_miss": gold_recall_miss,
        "forced_gold_injection_count": forced_gold_injection_count,
        "label_source_counts": dict(label_source_counts),
        "label_grade_counts": dict(label_grade_counts),
        "weak_label_type_counts": dict(weak_label_type_counts),
        "sample_weight_summary": {
            "gold": _summ_weights(sample_w_gold),
            "weak": _summ_weights(sample_w_weak),
        },
        "train_query_count": len(train_queries),
        "test_query_count": len(test_queries),
        "train_sample_count": train_sample_count,
        "test_gold_sample_count": test_gold_sample_count,
        "per_query_stats": per_query_stats,
        "paths": out_names,
        "by_split_line_counts": {k: len(v) for k, v in by_split.items()},
    }


class KGATAXTrainingGenerator:
    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. 初始化 Neo4j 连接
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DATABASE)

        # 2. 载入岗位索引
        print(f"[*] 正在载入预计算岗位索引: {JOB_INDEX_PATH}")
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_to_idx = {sid: i for i, sid in enumerate(json.load(f))}

        # 3. 召回系统（顺序模式懒加载；多进程模式下每 worker 自建）
        self.recall_system: Optional[TotalRecallSystem] = None

        # 4. 载入学术质量映射
        self.author_quality_map = {}
        self._preload_author_quality_from_index()

        # 5. ID 映射管理
        self.user_to_int = {}
        self.entity_to_int = {}
        self.user_counter = 0
        self.entity_counter = 0
        self.ENTITY_OFFSET = 0

    def _preload_author_quality_from_index(self):
        if not os.path.exists(FEATURE_INDEX_PATH):
            return
        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)
        author_features = feature_bundle.get('author', {})
        for aid, feats in author_features.items():
            h_norm = feats.get('h_index', 0.0)
            c_norm = feats.get('cited_by_count', 0.0)
            w_norm = feats.get('works_count', 0.0)
            self.author_quality_map[str(aid)] = 0.4 * h_norm + 0.4 * c_norm + 0.2 * w_norm

    def _ensure_recall_system(self) -> TotalRecallSystem:
        if self.recall_system is None:
            self.recall_system = TotalRecallSystem()
            self.recall_system.v_path.verbose = False
            self.recall_system.l_path.verbose = False
        return self.recall_system

    def _select_training_pool_records(self, pool: Any) -> List[Any]:
        """与 ch04 一致：训练入口优先使用 training_pool_records。"""
        if pool is None:
            return []
        tpr = getattr(pool, "training_pool_records", None) or []
        if tpr:
            return list(tpr)
        cr = getattr(pool, "candidate_records", None) or []
        if not cr:
            return []
        return list(cr)[:100]

    def _training_pool_min_for_build(self) -> int:
        """默认至少 100 人再写分层样本；调试模式下可降为 1（环境变量覆盖）。"""
        if int(os.environ.get("KGATAX_DEBUG_LIMIT", "0") or "0") > 0:
            return int(os.environ.get("KGATAX_TRAINING_MIN_POOL", "1") or "1")
        return int(os.environ.get("KGATAX_TRAINING_MIN_POOL", "100") or "100")

    def _log_kgatax_sidecar_alignment(
        self,
        job_raw_id: str,
        training_records: List[Any],
        recall_results: Dict[str, Any],
    ) -> None:
        """核对 execute 返回的 kgatax_sidecar_rows 是否覆盖 training_pool 作者（不静默）。"""
        t_ids = {str(getattr(r, "author_id", "")) for r in training_records}
        t_ids.discard("")
        sidecar = recall_results.get("kgatax_sidecar_rows") or []
        sc_ids = {str(row.get("author_id")) for row in sidecar if row.get("author_id") is not None}
        missing_ct = len(t_ids - sc_ids)
        matched_ct = len(t_ids & sc_ids)
        print(
            f"[KGATAX sidecar] job={job_raw_id} training_pool_count={len(training_records)} "
            f"training_unique_author_ids={len(t_ids)} sidecar_rows_count={len(sidecar)} "
            f"matched_sidecar_author_count={matched_ct} missing_sidecar_author_count={missing_ct}",
            flush=True,
        )
        if missing_ct > 0 and t_ids:
            sample = sorted(t_ids - sc_ids)[:15]
            print(
                f"[KGATAX sidecar] 警告: {missing_ct} 个 training_pool 作者未出现在 kgatax_sidecar_rows "
                f"（示例 author_id: {sample}）",
                flush=True,
            )

    def get_user_id(self, raw_id):
        raw_id = str(raw_id)
        if raw_id not in self.user_to_int:
            self.user_to_int[raw_id] = self.user_counter
            self.user_counter += 1
        return self.user_to_int[raw_id]

    def get_ent_id(self, raw_id):
        raw_id = str(raw_id).strip().lower()
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.entity_counter
            self.entity_counter += 1
        return self.entity_to_int[raw_id] + self.ENTITY_OFFSET

    def generate_refined_train_data(self, train_size=3000, test_size=300, recall_workers: int = 0):
        """
        任务 1：生成精排训练样本（优先基于 candidate_pool.candidate_records，分层正负样本，README 5.5）。
        同时写入 train_four_branch.json / test_four_branch.json（四分支字段），供 DataLoader 与四分支模型使用。
        返回：被抽样用于训练的岗位 ID 列表 (锚点)。

        :param recall_workers: 0=顺序执行；>=2 时用多进程并行 execute（每进程一套 TotalRecallSystem），
               主进程顺序做 ID 映射；与顺序模式一致地传 is_training=True。
        """
        print(f"\n>>> 任务 1: 生成混合排名精排数据（候选池入口 + 分层正负样本 + 四分支导出）...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        debug_lim = int(os.environ.get("KGATAX_DEBUG_LIMIT", "0") or "0")
        if debug_lim > 0:
            ts = max(1, debug_lim - 1)
            train_size = min(train_size, ts)
            test_size = min(test_size, max(1, debug_lim - train_size))
            print(
                f"[*] KGATAX_DEBUG_LIMIT={debug_lim} → train_size={train_size} test_size={test_size}",
                flush=True,
            )

        all_jobs = conn.execute("SELECT securityId, job_name, description, skills, domain_ids FROM jobs").fetchall()
        sampled = random.sample(all_jobs, min(len(all_jobs), train_size + test_size))
        sampled_job_ids = [str(j['securityId']) for j in sampled]

        train_lines, test_lines = [], []
        train_four_branch, test_four_branch = {}, {}

        train_slice = sampled[:train_size]
        test_slice = sampled[train_size:]
        rng = random.Random(42)

        if recall_workers and recall_workers >= 2:
            def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
                return {k: row[k] for k in row.keys()}

            train_specs = [
                {
                    "job": _row_to_dict(job),
                    "target_domain": job["domain_ids"] if rng.random() < 0.75 else None,
                }
                for job in train_slice
            ]
            test_rng = random.Random(43)
            test_specs = [
                {
                    "job": _row_to_dict(job),
                    "target_domain": job["domain_ids"] if test_rng.random() < 0.75 else None,
                }
                for job in test_slice
            ]
            print(
                f"[*] 多进程召回: workers={min(recall_workers, cpu_count() or 1)} "
                f"(train={len(train_specs)} test={len(test_specs)})，is_training=True",
                flush=True,
            )
            train_by = parallel_total_recall_for_specs(train_specs, recall_workers)
            test_by = parallel_total_recall_for_specs(test_specs, recall_workers)

            for job in tqdm(train_slice, desc="Generating Train (finalize)"):
                line, aux = self._finalize_job_from_recall(job, train_by.get(str(job["securityId"])))
                if line:
                    train_lines.append(line)
                    if aux:
                        train_four_branch.update(aux)

            for job in tqdm(test_slice, desc="Generating Test (finalize)"):
                line, aux = self._finalize_job_from_recall(job, test_by.get(str(job["securityId"])))
                if line:
                    test_lines.append(line)
                    if aux:
                        test_four_branch.update(aux)
        else:
            for job in tqdm(train_slice, desc="Generating Train"):
                line, aux = self._process_single_job(job)
                if line:
                    train_lines.append(line)
                    if aux:
                        train_four_branch.update(aux)

            for job in tqdm(test_slice, desc="Generating Test"):
                line, aux = self._process_single_job(job)
                if line:
                    test_lines.append(line)
                    if aux:
                        test_four_branch.update(aux)

        train_path = os.path.join(self.output_dir, "train.txt")
        test_path = os.path.join(self.output_dir, "test.txt")
        with open(train_path, "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(test_path, "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))

        if train_four_branch:
            with open(os.path.join(self.output_dir, "train_four_branch.json"), "w", encoding='utf-8') as f:
                json.dump(train_four_branch, f, ensure_ascii=False, indent=2)
        if test_four_branch:
            with open(os.path.join(self.output_dir, "test_four_branch.json"), "w", encoding='utf-8') as f:
                json.dump(test_four_branch, f, ensure_ascii=False, indent=2)

        conn.close()
        print(f"[OK] 任务 1 完成，共生成 {len(train_lines) + len(test_lines)} 条阶梯样本；四分支条目 train={len(train_four_branch)} test={len(test_four_branch)}。")
        return sampled_job_ids

    def _process_single_job(self, job, use_domain: Optional[bool] = None):
        """
        顺序模式：单次召回 + 组装训练行。use_domain=None 时按 75% 概率使用岗位 domain_ids。
        """
        job_raw_id = str(job["securityId"])
        if job_raw_id not in self.job_id_to_idx:
            return None, None
        query_text = f"{job['job_name'] or ''} {job['description'] or ''} {job['skills'] or ''}"
        if use_domain is None:
            use_domain = random.random() < 0.75
        target_domain = job["domain_ids"] if use_domain else None
        recall_results = self._ensure_recall_system().execute(
            query_text, domain_id=target_domain, is_training=True
        )
        return self._finalize_job_from_recall(job, recall_results)

    def _finalize_job_from_recall(
        self,
        job,
        recall_results: Optional[Dict[str, Any]],
    ):
        """
        在已有 execute 结果上组装训练行（供多进程召回后主进程顺序调用，保证 get_user_id/get_ent_id 一致）。
        """
        if not recall_results:
            return None, None
        job_raw_id = str(job["securityId"])
        if job_raw_id not in self.job_id_to_idx:
            return None, None

        pool = recall_results.get("candidate_pool")
        records = self._select_training_pool_records(pool)
        min_need = self._training_pool_min_for_build()

        if records and len(records) >= min_need:
            self._log_kgatax_sidecar_alignment(job_raw_id, records, recall_results)
            line, aux = self._build_from_candidate_pool(job_raw_id, records)
            return line, aux

        if records:
            print(
                f"[KGATAX] job={job_raw_id} training_pool 仅 {len(records)} 人 "
                f"< min={min_need}，跳过 candidate_pool 入口，尝试末级兼容。",
                flush=True,
            )

        # 末级兼容：final_top_500 / final_top_200（deprecated，与展示池同源；非正式训练入口）
        candidates = recall_results.get("final_top_500", []) or recall_results.get("final_top_200", [])
        if len(candidates) < 100:
            return None, None

        recall_ranks = {str(aid): i for i, aid in enumerate(candidates)}
        quality_list = sorted(
            [(str(aid), self.author_quality_map.get(str(aid), 0.0)) for aid in candidates],
            key=lambda x: x[1], reverse=True,
        )
        quality_ranks = {aid: i for i, (aid, _) in enumerate(quality_list)}
        fused = sorted(
            [(str(aid), 0.6 * recall_ranks[str(aid)] + 0.4 * quality_ranks[str(aid)]) for aid in candidates],
            key=lambda x: x[1],
        )
        num_cand = len(fused)
        pos_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused[: min(100, num_cand)]]
        fair_ids = [
            str(self.get_ent_id(f"a_{a[0]}"))
            for a in random.sample(
                fused[min(100, num_cand) : min(400, num_cand)],
                min(100, max(0, min(400, num_cand) - 100)),
            )
        ]
        neutral_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused[min(400, num_cand) :]]
        potential_pool = list(self.author_quality_map.keys())
        easy_neg_ids = [str(self.get_ent_id(f"a_{aid}")) for aid in random.sample(potential_pool, 100)]
        u_id = self.get_user_id(job_raw_id)
        line = f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        return line, None

    def _classify_record(self, r, rank_by_score):
        """
        训练标签分配：多特征一致性优先，不再默认信标签路。
        vector+label 高质量 -> strong_pos；label only 且 topic/domain/activity 弱 -> hard_neg；
        vector only 但整体强 -> weak_pos。
        """
        if not getattr(r, "passed_hard_filter", True):
            return LABEL_FIELD_NEG
        bucket = (getattr(r, "bucket_type") or "").strip().upper()
        from_label = getattr(r, "from_label", False)
        from_collab = getattr(r, "from_collab", False)
        from_vector = getattr(r, "from_vector", False)
        path_count = getattr(r, "path_count", 0) or 0
        domain_consistency = getattr(r, "domain_consistency", None)
        paper_hit_strength = getattr(r, "paper_hit_strength", None)
        recent_activity_match = getattr(r, "recent_activity_match", None)
        label_risky = getattr(r, "label_risky_term_count", 0) or 0
        label_core = getattr(r, "label_core_term_count", 0) or 0

        if from_vector and from_label and (domain_consistency is None or domain_consistency >= 0.25):
            return LABEL_STRONG_POS
        if from_vector and (from_label or path_count >= 2) or (bucket in ("A", "B", "C") and from_vector):
            return LABEL_WEAK_POS
        if from_collab and path_count == 1 and not from_label and not from_vector:
            return LABEL_COLLAB_NEG
        if from_label and path_count == 1:
            if domain_consistency is not None and domain_consistency < 0.25:
                return LABEL_HARD_NEG
            if paper_hit_strength is not None and paper_hit_strength < 0.15:
                return LABEL_HARD_NEG
            if recent_activity_match is not None and recent_activity_match < 0.20:
                return LABEL_HARD_NEG
            if label_risky >= 2 and label_core == 0:
                return LABEL_HARD_NEG
        if (from_label or path_count >= 2) and rank_by_score.get(r.author_id, 999) > 100:
            return LABEL_HARD_NEG
        return LABEL_FIELD_NEG

    def _build_from_candidate_pool(self, job_raw_id, records):
        """基于 candidate_records 构建分层正负样本并生成 train 行与四分支侧车。"""
        u_id = self.get_user_id(job_raw_id)
        pool_aids = {r.author_id for r in records}
        max_score = max((r.candidate_pool_score or 0.0) for r in records) or 1.0
        rank_by_score = {}
        for i, r in enumerate(sorted(records, key=lambda x: -(x.candidate_pool_score or 0.0))):
            rank_by_score[r.author_id] = i

        strong_pos, weak_pos, hard_neg, collab_neg, field_neg = [], [], [], [], []
        for r in records:
            label = self._classify_record(r, rank_by_score)
            if label == LABEL_STRONG_POS:
                strong_pos.append(r)
            elif label == LABEL_WEAK_POS:
                weak_pos.append(r)
            elif label == LABEL_HARD_NEG:
                hard_neg.append(r)
            elif label == LABEL_COLLAB_NEG:
                collab_neg.append(r)
            else:
                field_neg.append(r)

        pos_cap = min(100, len(strong_pos) + len(weak_pos))
        pos_pool = sorted(strong_pos + weak_pos, key=lambda x: -(x.candidate_pool_score or 0.0))
        pos_records = pos_pool[:pos_cap]
        if len(pos_records) < 100:
            all_by_score = sorted(records, key=lambda x: -(x.candidate_pool_score or 0.0))
            pos_set = {r.author_id for r in pos_records}
            for r in all_by_score:
                if len(pos_records) >= 100:
                    break
                if r.author_id not in pos_set:
                    pos_records.append(r)
                    pos_set.add(r.author_id)
        rest_weak = [r for r in pos_pool if r not in pos_records]
        fair_records = (rest_weak + hard_neg)[:100]
        neutral_records = field_neg[:150]
        easy_neg_aids = [aid for aid in random.sample(list(self.author_quality_map.keys()), 200) if aid not in pool_aids][:100]
        if len(easy_neg_aids) < 100:
            easy_neg_aids += [r.author_id for r in collab_neg[: 100 - len(easy_neg_aids)]]

        pos_ids = [str(self.get_ent_id(f"a_{r.author_id}")) for r in pos_records]
        fair_ids = [str(self.get_ent_id(f"a_{r.author_id}")) for r in fair_records]
        neutral_ids = [str(self.get_ent_id(f"a_{r.author_id}")) for r in neutral_records]
        easy_neg_ids = [str(self.get_ent_id(f"a_{aid}")) for aid in easy_neg_aids]
        if len(easy_neg_ids) < 100:
            easy_neg_ids += [str(self.get_ent_id(f"a_{r.author_id}")) for r in collab_neg[: 100 - len(easy_neg_ids)]]

        four_branch = {}
        for eid, r in zip(pos_ids, pos_records):
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)
        for eid, r in zip(fair_ids, fair_records):
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)
        for eid, r in zip(neutral_ids, neutral_records):
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)
        for aid in easy_neg_aids:
            eid = str(self.get_ent_id(f"a_{aid}"))
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(None, max_score)
        for r in collab_neg[: max(0, 100 - len(easy_neg_aids))]:
            eid = str(self.get_ent_id(f"a_{r.author_id}"))
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)

        line = f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        return line, four_branch

    def _encode_bucket(self, bucket_type):
        m = {"A": 0.25, "B": 0.35, "C": 0.5, "D": 0.6, "E": 0.75, "F": 0.9, "Z": 1.0}
        bt = (bucket_type or "D").strip().upper() if bucket_type else "D"
        return m.get(bt, 0.5)

    def _export_four_branch_row(self, record, max_pool_score):
        """导出一条 (job, author) 的四分支特征：recall 13 维、author_aux 12 维、interaction 8 维（含标签路摘要）。"""
        if record is None:
            return {"recall": [0.0] * 13, "author_aux": [0.0] * 12, "interaction": [0.0] * 8}
        max_rank = 500.0
        r = record
        v_rank = getattr(r, "vector_rank", None) or 0
        l_rank = getattr(r, "label_rank", None) or 0
        c_rank = getattr(r, "collab_rank", None) or 0
        recall = [
            1.0 if getattr(r, "from_vector", False) else 0.0,
            1.0 if getattr(r, "from_label", False) else 0.0,
            1.0 if getattr(r, "from_collab", False) else 0.0,
            (getattr(r, "path_count", 0) or 0) / 3.0,
            1.0 - min(v_rank / max_rank, 1.0) if v_rank else 0.0,
            1.0 - min(l_rank / max_rank, 1.0) if l_rank else 0.0,
            1.0 - min(c_rank / max_rank, 1.0) if c_rank else 0.0,
            (float(r.candidate_pool_score or 0.0) / max_pool_score) if max_pool_score else 0.0,
            self._encode_bucket(getattr(r, "bucket_type", None)),
            1.0 if getattr(r, "is_multi_path_hit", False) else 0.0,
            float(getattr(r, "vector_score_raw") or 0.0),
            float(getattr(r, "label_score_raw") or 0.0),
            float(getattr(r, "collab_score_raw") or 0.0),
        ]
        h = getattr(r, "h_index", None)
        w = getattr(r, "works_count", None)
        c = getattr(r, "cited_by_count", None)
        rw = getattr(r, "recent_works_count", None)
        rc = getattr(r, "recent_citations", None)
        inst = getattr(r, "institution_level", None)
        tq = getattr(r, "top_work_quality", None)
        author_aux = [
            np.log1p(h) if h is not None else 0.0,
            np.log1p(c) if c is not None else 0.0,
            np.log1p(w) if w is not None else 0.0,
            np.log1p(rw) if rw is not None else 0.0,
            np.log1p(rc) if rc is not None else 0.0,
            float(inst) if inst is not None else 0.0,
            float(tq) if tq is not None else 0.0,
        ] + [0.0] * 5
        interaction = [
            float(getattr(r, "topic_similarity") or 0.0),
            float(getattr(r, "skill_coverage_ratio") or 0.0),
            float(getattr(r, "domain_consistency") or 0.0),
            float(getattr(r, "paper_hit_strength") or 0.0),
            float(getattr(r, "recent_activity_match") or 0.0),
            (getattr(r, "label_term_count", 0) or 0) / 10.0,
            (getattr(r, "label_core_term_count", 0) or 0) / 10.0,
            float(getattr(r, "label_best_term_score") or 0.0),
        ]
        return {
            "recall": recall,
            "author_aux": author_aux,
            "interaction": interaction,
            "pool_meta": {
                "candidate_trust_tier": str(getattr(record, "candidate_trust_tier", "") or ""),
                "bucket_type": (getattr(record, "bucket_type", "") or "").strip() or "Z",
                "risk_flags": list(getattr(record, "risk_flags", None) or []),
                "sampleability_flags": list(getattr(record, "sampleability_flags", None) or []),
                "from_vector": bool(getattr(record, "from_vector", False)),
                "from_label": bool(getattr(record, "from_label", False)),
                "from_collab": bool(getattr(record, "from_collab", False)),
                "path_count": int(getattr(record, "path_count", 0) or 0),
                "candidate_pool_score": float(getattr(record, "candidate_pool_score", None) or 0.0),
                "vector_rank": getattr(record, "vector_rank", None),
                "label_rank": getattr(record, "label_rank", None),
                "collab_rank": getattr(record, "collab_rank", None),
            },
        }

    def generate_kg_topology(self, sampled_job_ids: list):
        """
        任务 2：全量加权拓扑收割
        修正点：提取 pos_weight 和 score，并将权重持久化到 kg_final.txt 中。
        """
        print(f"\n>>> 任务 2: 执行全量 ID 登记与加权拓扑收割...")



        #  导出锚点 (保持原逻辑)
        anchor_path = os.path.join(self.output_dir, "trained_anchors.json")
        with open(anchor_path, "w", encoding='utf-8') as f:
            json.dump(sampled_job_ids, f, indent=4, ensure_ascii=False)

        kg_triplets = []  # 现在存储 (h, r, t, weight)
        rel_map = {"AUTHORED": 1, "PRODUCED_BY": 2, "PUBLISHED_IN": 3, "HAS_TOPIC": 4, "REQUIRE_SKILL": 5,
                   "SIMILAR_TO": 6}

        for rel_name, rel_id in rel_map.items():
            print(f"[*] 正在收割加权关系: {rel_name} (ID: {rel_id})")
            h_l, t_l = self._get_labels(rel_name)
            if not h_l: continue

            # --- 核心修改：增加对 weight (pos_weight) 和 score 的提取 ---
            query = f"""
            MATCH (n:{h_l})-[r:{rel_name}]->(m:{t_l}) 
            RETURN n.id as hid, m.id as tid, r.pos_weight as weight, r.score as score
            """
            res = self.graph.run(query).data()

            for r in tqdm(res, desc=f"Mapping {rel_name}", leave=False):
                h_raw, t_raw = str(r['hid']), str(r['tid'])

                # 处理 ID 映射
                if rel_id == 5:
                    h_int = self.get_user_id(h_raw)
                else:
                    h_int = self.get_ent_id(f"{self._get_prefix_by_label(h_l)}{h_raw}")
                t_int = self.get_ent_id(f"{self._get_prefix_by_label(t_l)}{t_raw}")

                # --- 核心修改：确定该边的最终权重 ---
                # 1. 如果是 AUTHORED，取 pos_weight (内含时序衰减)
                # 2. 如果是 SIMILAR_TO，取 score
                # 3. 其他默认 1.0
                edge_w = r.get('weight') or r.get('score') or 1.0

                kg_triplets.append((h_int, rel_id, t_int, round(float(edge_w), 4)))

        # 3. 持久化输出加权三元组
        print(f"[*] 正在写入加权拓扑文件 kg_final.txt...")
        # 去重时保留权重
        df = pd.DataFrame(kg_triplets, columns=['h', 'r', 't', 'w']).drop_duplicates(subset=['h', 'r', 't'])
        final_list = df[df['h'] != df['t']].values.tolist()

        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            for h, r, t, w in final_list:
                # 格式改为: h r t w
                f.write(f"{int(h)} {int(r)} {int(t)} {w}\n")

        self._save_mapping()
        print(f"[OK] 拓扑构建完成，包含权重信息。最终边数: {len(final_list)}")

    def _get_labels(self, rel):
        m = {"AUTHORED": ("Author", "Work"), "PRODUCED_BY": ("Work", "Institution"), "PUBLISHED_IN": ("Work", "Source"),
             "HAS_TOPIC": ("Work", "Vocabulary"), "REQUIRE_SKILL": ("Job", "Vocabulary"),
             "SIMILAR_TO": ("Vocabulary", "Vocabulary")}
        return m.get(rel, (None, None))

    def _get_prefix_by_label(self, label):
        return {"Author": "a_", "Work": "w_", "Institution": "i_", "Source": "s_", "Vocabulary": "v_", "Job": ""}.get(
            label, "")

    def _save_mapping(self):
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items(): full_mapping[k] = v + self.ENTITY_OFFSET
        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump({"entity": full_mapping, "offset": self.ENTITY_OFFSET,
                       "total_nodes": self.user_counter + self.entity_counter,
                       "user_count": self.user_counter, "entity_count": self.entity_counter}, f, indent=4,
                      ensure_ascii=False)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gen = KGATAXTrainingGenerator()
    # 多进程召回：设置环境变量 KGATAX_RECALL_WORKERS=4（或传参扩展）；0=顺序
    _rw = int(os.environ.get("KGATAX_RECALL_WORKERS", "0") or "0")
    _debug_skip_topo = os.environ.get("KGATAX_DEBUG_SKIP_TOPOLOGY", "").strip() in ("1", "true", "True", "yes", "YES")
    _debug_skip_gold = os.environ.get("KGATAX_DEBUG_SKIP_GOLD", "").strip() in ("1", "true", "True", "yes", "YES")
    _skip_pool_sup = os.environ.get("KGATAX_DEBUG_SKIP_POOL_SUPERVISED", "").strip() in (
        "1", "true", "True", "yes", "YES",
    )
    _legacy_gold_jsonl = os.environ.get("KGATAX_LEGACY_GOLD_JSONL", "").strip() in (
        "1", "true", "True", "yes", "YES",
    )

    # 第一步：锁定全局 ID 偏移量 (从 Neo4j 读取全量岗位)
    print("\n[*] 正在从 Neo4j 锁定全局 ID 偏移量 (执行人口普查)...")
    # 这里的查询必须覆盖所有可能的 Job 节点
    all_job_res = gen.graph.run("MATCH (j:Job) RETURN j.id as jid").data()

    for res in tqdm(all_job_res, desc="登记全量岗位 ID"):
        # 强制将 Neo4j 里的所有 Job ID 映射到 User 空间 (0 ~ OFFSET-1)
        gen.get_user_id(str(res['jid']))

    # 核心锁：一旦赋值，后面绝不能再改 self.ENTITY_OFFSET
    gen.ENTITY_OFFSET = gen.user_counter
    print(f"[OK] ENTITY_OFFSET 已锁定为: {gen.ENTITY_OFFSET}")

    # 第二步：生成精排样本 (此时使用已锁定的 OFFSET)
    trained_anchors = gen.generate_refined_train_data(
        train_size=3000, test_size=300, recall_workers=_rw
    )

    _train_txt = os.path.join(gen.output_dir, "train.txt")
    _test_txt = os.path.join(gen.output_dir, "test.txt")
    def _count_nonempty_lines(p: str) -> int:
        if not os.path.isfile(p):
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    _train_lines_n = _count_nonempty_lines(_train_txt)
    _test_lines_n = _count_nonempty_lines(_test_txt)

    # 第三步：执行全量拓扑收割
    if _debug_skip_topo:
        print("[*] KGATAX_DEBUG_SKIP_TOPOLOGY 已设置：跳过 generate_kg_topology（仅验收训练数据生成）", flush=True)
    else:
        gen.generate_kg_topology(sampled_job_ids=trained_anchors)

    # 第四步：池监督 JSONL（training_pool + 金标覆盖 + 弱标签）；旧版 train_gold_supervised 需 LEGACY 开关
    _pool_stats: Dict[str, Any] = {}
    if _skip_pool_sup:
        print("[*] KGATAX_DEBUG_SKIP_POOL_SUPERVISED：跳过池监督 JSONL", flush=True)
    else:
        _pool_stats = generate_training_pool_supervised_exports(gen, recall_workers=_rw)

    if _legacy_gold_jsonl and not _debug_skip_gold:
        _gold_stats = generate_gold_supervised_exports(gen, recall_workers=_rw)
    else:
        if _debug_skip_gold:
            print(
                "[*] KGATAX_DEBUG_SKIP_GOLD：跳过旧版 train_gold_supervised 等导出（主口径为 *_pool_supervised.jsonl）",
                flush=True,
            )
        _gold_stats = {
            "query_counts": {},
            "line_counts": {},
            "gold_label_dist": {},
            "stats_split": {},
        }

    from src.infrastructure.database.kgat_ax.pipeline_state import write_stage_done

    if not _debug_skip_topo:
        write_stage_done(1)
    else:
        print("[*] 未写入 pipeline_stage1.done（调试跳过拓扑时避免误标阶段完成）", flush=True)

    print("\n" + "=" * 60)
    print("[汇总] 阶段 1 数据产出")
    print("=" * 60)
    print(f"  [旧主链] train.txt 样本行数: {_train_lines_n}")
    print(f"  [旧主链] test.txt  样本行数: {_test_lines_n}")
    if _pool_stats:
        print(
            "  [池监督] JSONL 行数 train/dev/test: "
            f"{_pool_stats.get('by_split_line_counts', {}).get('train', 0)} / "
            f"{_pool_stats.get('by_split_line_counts', {}).get('dev', 0)} / "
            f"{_pool_stats.get('by_split_line_counts', {}).get('test', 0)}",
            flush=True,
        )
        print(
            f"  [池监督] gold_recall_hit / miss / injection: "
            f"{_pool_stats.get('gold_recall_hit', 0)} / "
            f"{_pool_stats.get('gold_recall_miss', 0)} / "
            f"{_pool_stats.get('forced_gold_injection_count', 0)}",
            flush=True,
        )
    print("  [旧版金标 JSONL] 各 split query 数 (train/dev/test): "
          f"{_gold_stats['query_counts'].get('train', 0)} / "
          f"{_gold_stats['query_counts'].get('dev', 0)} / "
          f"{_gold_stats['query_counts'].get('test', 0)}")
    print("  [旧版金标 JSONL] 各 split 行数: "
          f"train={_gold_stats['line_counts'].get('train', 0)}, "
          f"dev={_gold_stats['line_counts'].get('dev', 0)}, "
          f"test={_gold_stats['line_counts'].get('test', 0)}")
    for sk in ("train", "dev", "test"):
        dist = _gold_stats["gold_label_dist"].get(sk, {})
        print(f"  [旧版金标 JSONL] split={sk} gold_label 分布: {dist}")
    for sk in ("train", "dev", "test"):
        st = _gold_stats["stats_split"].get(sk, {})
        print(
            f"  [旧版金标 JSONL] split={sk} 召回命中金标作者数(累计): {st.get('hits', 0)} | "
            f"未覆盖金标作者数(累计): {st.get('missed', 0)}"
        )
    print("=" * 60 + "\n")
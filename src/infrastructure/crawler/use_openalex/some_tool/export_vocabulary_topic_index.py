# -*- coding: utf-8 -*-
"""
方案 B：从 vocabulary_topic 表导出「词 → topic + 业务 domain」索引为 JSON。

用途：供召回/标签路按 voc_id 或 term 查三级领域（field / subfield / topic）及 our_domain_id，
      便于后续词汇索引使用 field/subfield/topic，并与 17 业务领域配合。

依赖：backfill_vocabulary_topic 已跑完，主库中存在 vocabulary_topic 表。
路径：直接使用 config.DB_PATH、config.DATA_DIR；输出默认 config.DATA_DIR/vocabulary_topic_index.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3

from config import DATA_DIR, DB_PATH

from src.utils.domain_config import NAME_EN_TO_DOMAIN_ID, NAME_TO_DOMAIN_ID

DEFAULT_OUTPUT = os.path.join(DATA_DIR, "vocabulary_topic_index.json")


def _resolve_our_domain_id(domain_name: str | None, domain_id_openalex: str | None) -> str | None:
    """
    将 OpenAlex 的 domain 名称（或 id）映射为项目内 17 业务领域 ID（1–17）。
    先尝试英文名（name_en 归一化），再尝试中文名。
    """
    if not domain_name and not domain_id_openalex:
        return None
    raw = (domain_name or "").strip() or (domain_id_openalex or "").strip()
    if not raw:
        return None
    key_en = raw.replace("_", " ").lower()
    if key_en in NAME_EN_TO_DOMAIN_ID:
        return NAME_EN_TO_DOMAIN_ID[key_en]
    if raw in NAME_TO_DOMAIN_ID:
        return NAME_TO_DOMAIN_ID[raw]
    return None


def run(db_path: str, output_path: str, by_term: bool = False, verbose: bool = True) -> None:
    """
    从 vocabulary_topic 读取并导出为 JSON。

    :param db_path: 主库路径（含 vocabulary + vocabulary_topic）
    :param output_path: 输出 JSON 路径
    :param by_term: 若 True，额外生成一份以 term 为键的索引（同目录下，后缀 _by_term.json）
    :param verbose: 是否打印条数等信息
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT vt.voc_id, vt.topic_id, vt.topic_display_name,
               vt.domain_id AS domain_id_oa, vt.domain_name AS domain_name_oa,
               vt.field_id, vt.field_name, vt.subfield_id, vt.subfield_name,
               vt.hierarchy_path, vt.openalex_topic_url, vt.updated_at,
               v.term
        FROM vocabulary_topic vt
        JOIN vocabulary v ON vt.voc_id = v.voc_id
        ORDER BY vt.voc_id
    """
    rows = conn.execute(sql).fetchall()
    conn.close()

    index_by_voc_id = {}
    index_by_term = {} if by_term else None

    for row in rows:
        our_domain_id = _resolve_our_domain_id(
            row["domain_name_oa"], row["domain_id_oa"]
        )
        entry = {
            "term": row["term"],
            "topic_id": row["topic_id"],
            "topic_display_name": row["topic_display_name"],
            "field_id": row["field_id"],
            "field_name": row["field_name"],
            "subfield_id": row["subfield_id"],
            "subfield_name": row["subfield_name"],
            "our_domain_id": our_domain_id,
            "hierarchy_path": row["hierarchy_path"],
            "openalex_topic_url": row["openalex_topic_url"],
            "domain_id_oa": row["domain_id_oa"],
            "domain_name_oa": row["domain_name_oa"],
            "updated_at": row["updated_at"],
        }
        vid = int(row["voc_id"])
        index_by_voc_id[str(vid)] = entry
        if by_term and row["term"]:
            term_key = (row["term"] or "").strip().lower()
            if term_key:
                index_by_term[term_key] = entry

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_by_voc_id, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[*] 已导出 voc_id 索引: {os.path.abspath(output_path)}，共 {len(index_by_voc_id)} 条")

    if by_term and index_by_term is not None:
        base, ext = os.path.splitext(output_path)
        term_path = f"{base}_by_term{ext}"
        with open(term_path, "w", encoding="utf-8") as f:
            json.dump(index_by_term, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"[*] 已导出 term 索引: {os.path.abspath(term_path)}，共 {len(index_by_term)} 条")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="方案 B：从 vocabulary_topic 导出 vocabulary_topic_index.json（voc_id → topic + our_domain_id）"
    )
    parser.add_argument("--db", default=DB_PATH, help="主库路径（默认使用 config.DB_PATH）")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    parser.add_argument("--by-term", action="store_true", help="额外生成以 term 为键的 JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印信息")
    args = parser.parse_args()
    run(db_path=args.db, output_path=args.out, by_term=args.by_term, verbose=not args.quiet)


if __name__ == "__main__":
    main()

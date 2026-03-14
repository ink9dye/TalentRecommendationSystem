from __future__ import annotations

import argparse
import os
import re
import sqlite3
import json
from datetime import datetime

# 尝试导入配置
try:
    from config import BASE_DIR, DATA_DIR, DB_PATH as CONFIG_DB_PATH
except Exception:
    CONFIG_DB_PATH = None
    DATA_DIR = "data"

from src.infrastructure.crawler.use_openalex.db_config import (
    DB_PATH as OPENALEX_DB_PATH,
)

DB_PATH = CONFIG_DB_PATH or OPENALEX_DB_PATH
LOCAL_TOPIC_FILE = os.path.abspath(os.path.join(DATA_DIR, "topic"))


def _short_id(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    m = re.search(r"/([A-Za-z0-9]+)$", url.rstrip("/"))
    return m.group(1) if m else None


def clean_term_logic(term: str) -> list[str]:
    """
    对术语进行多级启发式清洗，返回一个尝试列表。
    例如: "Neural Networks (Deep Learning) 2.0"
    -> ["neural networks (deep learning) 2.0", "neural networks", "neural network"]
    """
    variants = []
    t = term.strip().lower()
    variants.append(t)  # 1. 原始词

    # 2. 去噪音：去掉括号内容及末尾的版本号
    # 去掉 (...)
    t_no_paren = re.sub(r"\(.*?\)", "", t).strip()
    # 去掉末尾版本号 (如 v1.0, 2.0.1, .3)
    t_clean = re.sub(r"v?\d+(\.\d+)*$", "", t_no_paren).strip()

    if t_clean and t_clean != variants[0]:
        variants.append(t_clean)

    # 3. 简单词干化/去后缀 (Plural to Singular)
    # 处理常见复数及基于XX的描述
    stemmed = re.sub(r"-based$|-oriented$", "", t_clean).strip()
    # 简单的复数还原逻辑
    if stemmed.endswith("ies"):
        stemmed = stemmed[:-3] + "y"
    elif stemmed.endswith("ses"):  # matches bases, losses
        stemmed = stemmed[:-2]
    elif stemmed.endswith("s") and not stemmed.endswith("ss"):
        stemmed = stemmed[:-1]

    if stemmed and stemmed not in variants:
        variants.append(stemmed)

    return variants


def ensure_vocabulary_topic_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS vocabulary_topic
                 (
                     voc_id
                     INTEGER
                     PRIMARY
                     KEY,
                     topic_id
                     TEXT,
                     topic_display_name
                     TEXT,
                     domain_id
                     TEXT,
                     domain_name
                     TEXT,
                     field_id
                     TEXT,
                     field_name
                     TEXT,
                     subfield_id
                     TEXT,
                     subfield_name
                     TEXT,
                     hierarchy_path
                     TEXT,
                     openalex_topic_url
                     TEXT,
                     updated_at
                     TEXT,
                     FOREIGN
                     KEY
                 (
                     voc_id
                 ) REFERENCES vocabulary
                 (
                     voc_id
                 )
                     )
                 """)
    conn.commit()


def load_local_topics(file_path: str) -> dict:
    topic_index = {}
    if not os.path.exists(file_path):
        print(f"[!] 警告: 找不到本地快照文件 {file_path}")
        return {}

    print(f"[*] 正在从本地加载快照数据并构建智能索引...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                t = json.loads(line)
                # 记录标准名称
                raw_name = t.get("display_name", "").strip().lower()
                if not raw_name: continue

                domain = t.get("domain") or {}
                field = t.get("field") or {}
                subfield = t.get("subfield") or {}

                info = {
                    "topic_id": _short_id(t.get("id")),
                    "topic_display_name": t.get("display_name"),
                    "openalex_topic_url": t.get("id"),
                    "domain_id": _short_id(domain.get("id")),
                    "domain_name": domain.get("display_name"),
                    "field_id": _short_id(field.get("id")),
                    "field_name": field.get("display_name"),
                    "subfield_id": _short_id(subfield.get("id")),
                    "subfield_name": subfield.get("display_name"),
                    "hierarchy_path": " > ".join(filter(None, [
                        domain.get("display_name"),
                        field.get("display_name"),
                        subfield.get("display_name"),
                        t.get("display_name")
                    ])),
                }

                # 建立多向索引：主名称 + 清洗后的主名称 + 关键词
                topic_index[raw_name] = info
                # 同样对快照里的词也做简单清洗以增加碰撞概率
                for name_variant in clean_term_logic(raw_name):
                    if name_variant not in topic_index:
                        topic_index[name_variant] = info

                for kw in t.get("keywords", []):
                    kw_clean = kw.lower().strip()
                    topic_index[kw_clean] = info
                    # 对关键词也尝试词干化
                    if kw_clean.endswith('s'):
                        topic_index[kw_clean[:-1]] = info
            except:
                continue
    print(f"[*] 加载完成，本地索引规模: {len(topic_index)} 条路径")
    return topic_index


def run(db_path: str, limit: int | None = None, skip_existing: bool = True,
        concept_only: bool = False, verbose: bool = True) -> None:
    ensure_vocabulary_topic_table(sqlite3.connect(db_path))
    topic_index = load_local_topics(LOCAL_TOPIC_FILE)
    if not topic_index: return

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        where_clause = "WHERE vt.voc_id IS NULL" if skip_existing else "WHERE 1=1"
        if concept_only:
            where_clause += " AND LOWER(COALESCE(v.entity_type,'')) = 'concept'"
        else:
            where_clause += " AND LOWER(COALESCE(v.entity_type,'')) <> 'industry'"

        sql = f"""
            SELECT v.voc_id, v.term FROM vocabulary v
            {"LEFT JOIN vocabulary_topic vt ON v.voc_id = vt.voc_id" if skip_existing else ""}
            {where_clause}
            ORDER BY v.voc_id
        """
        if limit is not None: sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()

    total = len(rows)
    if total == 0:
        print("没有需要回填的记录。")
        return

    print(f"待处理: {total} 条 (模式: 增强版启发式匹配)")
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok, fail = 0, 0
    batch_updates = []

    for row in rows:
        raw_term = row["term"] or ""
        # 尝试多种清洗后的变体
        search_variants = clean_term_logic(raw_term)

        found_rec = None
        for variant in search_variants:
            if variant in topic_index:
                found_rec = topic_index[variant]
                break

        if found_rec:
            batch_updates.append((
                row["voc_id"], found_rec["topic_id"], found_rec["topic_display_name"],
                found_rec["domain_id"], found_rec["domain_name"],
                found_rec["field_id"], found_rec["field_name"],
                found_rec["subfield_id"], found_rec["subfield_name"],
                found_rec["hierarchy_path"], found_rec["openalex_topic_url"], updated_at
            ))
            ok += 1
        else:
            fail += 1

    if batch_updates:
        with sqlite3.connect(db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO vocabulary_topic
                (voc_id, topic_id, topic_display_name, domain_id, domain_name,
                 field_id, field_name, subfield_id, subfield_name, hierarchy_path, 
                 openalex_topic_url, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_updates)
            conn.commit()

    print(f"完成: 成功回填 {ok} 条 (含清洗后匹配), 未命中 {fail} 条")


def main():
    parser = argparse.ArgumentParser(description="增强匹配版: 通过本地快照回填 OpenAlex Topic")
    parser.add_argument("--db", default=DB_PATH, help="DB 路径")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument("--full", action="store_true", help="全量重跑")
    parser.add_argument("--concept-only", action="store_true", help="只处理 concept 词条")
    args = parser.parse_args()

    run(db_path=args.db, limit=args.limit, skip_existing=not args.full,
        concept_only=args.concept_only)


if __name__ == "__main__":
    main()
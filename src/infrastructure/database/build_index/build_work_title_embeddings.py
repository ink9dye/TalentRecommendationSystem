# -*- coding: utf-8 -*-
"""
离线构建论文标题向量库，供标签路 Stage5「标题↔JD」门控查表加速。

- 编码与线上一致：`QueryEncoder.encode_batch`（动态共振 + L2 归一化）。
- 输出：`config.WORK_TITLE_EMB_DB_PATH`（默认 `data/build_index/work_title_embeddings.db`）。
- 表：`work_title_embeddings(work_id, dim, embedding, updated_at)`，与 `work_title_emb_store.py` 读取一致。
- **断点续跑**：输出库内 `work_title_emb_build_progress` 记录主库 `works.rowid` 检查点；
  每批 **commit 成功后** 更新检查点；`--resume` 从上次 `rowid` 之后继续（不整表加载已有 work_id）。

用法（项目根目录）：
  python src/infrastructure/database/build_index/build_work_title_embeddings.py
  python .../build_work_title_embeddings.py --limit 10000
  python .../build_work_title_embeddings.py --out E:/tmp/titles.db
  python .../build_work_title_embeddings.py --resume          # 断点续跑
  python .../build_work_title_embeddings.py                   # 从头扫（重置进度，INSERT OR REPLACE 覆盖同 work_id）
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# 项目根：.../TalentRecommendationSystem
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config import DB_PATH, WORK_TITLE_EMB_DB_PATH  # noqa: E402
from src.core.recall.input_to_vector import QueryEncoder  # noqa: E402

_TITLE_MAX_CHARS = 2000
_PROGRESS_ID = 1


def _ensure_output_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS work_title_embeddings (
            work_id TEXT PRIMARY KEY,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            updated_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_wte_updated ON work_title_embeddings(updated_at);

        CREATE TABLE IF NOT EXISTS work_title_emb_build_progress (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_rowid INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP
        );
        """
    )
    conn.commit()


def _check_embedding_dim(conn: sqlite3.Connection, expected_dim: int) -> None:
    """输出库中已有向量时，dim 须与当前模型一致。"""
    try:
        dims = conn.execute(
            "SELECT DISTINCT dim FROM work_title_embeddings LIMIT 5"
        ).fetchall()
    except sqlite3.Error:
        return
    if not dims:
        return
    bad = [int(r[0]) for r in dims if int(r[0]) != int(expected_dim)]
    if bad:
        raise SystemExit(
            f"[Fatal] 输出库中已有 dim={dims}，与当前模型 dim={expected_dim} 不一致。"
            "请删除输出库或换 --out。"
        )


def _get_last_rowid(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT last_rowid FROM work_title_emb_build_progress WHERE id = ?",
        (_PROGRESS_ID,),
    ).fetchone()
    if not row or row[0] is None:
        return 0
    return int(row[0])


def _set_last_rowid(conn: sqlite3.Connection, last_rowid: int) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn.execute(
        """
        INSERT INTO work_title_emb_build_progress (id, last_rowid, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            last_rowid = excluded.last_rowid,
            updated_at = excluded.updated_at
        """,
        (_PROGRESS_ID, int(last_rowid), now),
    )


def _reset_progress(conn: sqlite3.Connection) -> None:
    """非 --resume：从主表头重新扫描。"""
    conn.execute("DELETE FROM work_title_emb_build_progress")
    conn.commit()


def _count_remaining_titles(main_conn: sqlite3.Connection, after_rowid: int) -> int:
    row = main_conn.execute(
        """
        SELECT COUNT(*) FROM works
        WHERE title IS NOT NULL AND TRIM(title) != ''
          AND rowid > ?
        """,
        (after_rowid,),
    ).fetchone()
    return int(row[0] or 0)


def _iter_work_rows_stream(
    main_conn: sqlite3.Connection,
    after_rowid: int,
) -> Iterator[sqlite3.Row]:
    """按主库 rowid 升序流式迭代，用于断点续跑。"""
    cur = main_conn.execute(
        """
        SELECT rowid, work_id, title FROM works
        WHERE title IS NOT NULL AND TRIM(title) != ''
          AND rowid > ?
        ORDER BY rowid
        """,
        (after_rowid,),
    )
    yield from cur


def run(
    out_path: str,
    limit: int = 0,
    resume: bool = False,
    batch_size: int = 32,
) -> None:
    if not os.path.isfile(DB_PATH):
        raise SystemExit(f"[Fatal] 主库不存在: {DB_PATH}")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"[*] 主库: {DB_PATH}", flush=True)
    print(f"[*] 输出: {out_path}", flush=True)

    encoder = QueryEncoder()
    dim = int(encoder.model.get_sentence_embedding_dimension())

    out_conn = sqlite3.connect(out_path, timeout=120)
    _ensure_output_schema(out_conn)
    _check_embedding_dim(out_conn, dim)

    if resume:
        start_rowid = _get_last_rowid(out_conn)
        print(
            f"[*] --resume：从主库 works.rowid > {start_rowid} 继续（断点表 work_title_emb_build_progress）",
            flush=True,
        )
    else:
        _reset_progress(out_conn)
        start_rowid = 0
        print("[*] 全量扫描：检查点已重置为 0（同 work_id 将 INSERT OR REPLACE 覆盖）", flush=True)

    main_conn = sqlite3.connect(DB_PATH, timeout=120)
    main_conn.row_factory = sqlite3.Row

    remaining = _count_remaining_titles(main_conn, start_rowid)
    if limit > 0:
        todo = min(remaining, limit)
    else:
        todo = remaining

    if todo <= 0:
        print("[OK] 无待编码记录（主库无标题或已在检查点之后无行）。")
        main_conn.close()
        out_conn.close()
        return

    print(
        f"[*] 待处理约 {todo} 条（remaining={remaining}, limit={limit or '∞'}），batch_size={batch_size}",
        flush=True,
    )

    ins_sql = """
        INSERT OR REPLACE INTO work_title_embeddings (work_id, dim, embedding, updated_at)
        VALUES (?, ?, ?, ?)
    """

    t0 = time.time()
    n_written = 0
    batch_rows: List[Tuple[int, str, str]] = []
    stream = _iter_work_rows_stream(main_conn, start_rowid)

    def flush_batch() -> None:
        nonlocal n_written, batch_rows
        if not batch_rows:
            return
        wids = [x[1] for x in batch_rows]
        titles = [x[2] for x in batch_rows]
        max_rowid = max(x[0] for x in batch_rows)
        vecs = encoder.encode_batch(titles)
        if vecs.shape[0] != len(batch_rows):
            print(f"[Warn] batch 行数不一致，跳过本批 rowid<={max_rowid}")
            batch_rows.clear()
            return
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for j, wid in enumerate(wids):
            row_v = np.asarray(vecs[j], dtype=np.float32).flatten()
            out_conn.execute(ins_sql, (wid, dim, row_v.tobytes(), now))
        _set_last_rowid(out_conn, max_rowid)
        out_conn.commit()
        n_written += len(batch_rows)
        batch_rows.clear()

    with tqdm(total=todo, desc="encode_batch", unit="条") as pbar:
        for row in stream:
            if limit > 0 and n_written >= limit:
                break
            rid = int(row["rowid"])
            wid = str(row["work_id"] or "").strip()
            title = (row["title"] or "").strip()
            if not wid or not title:
                continue
            if len(title) > _TITLE_MAX_CHARS:
                title = title[:_TITLE_MAX_CHARS]

            remaining_quota = (limit - n_written) if limit > 0 else batch_size
            if remaining_quota <= 0:
                break
            target = min(batch_size, remaining_quota)

            batch_rows.append((rid, wid, title))
            if len(batch_rows) >= target:
                k = len(batch_rows)
                flush_batch()
                pbar.update(k)

        if batch_rows and (limit <= 0 or n_written < limit):
            if limit > 0:
                allow = limit - n_written
                if allow <= 0:
                    batch_rows.clear()
                elif len(batch_rows) > allow:
                    batch_rows[:] = batch_rows[:allow]
            if batch_rows:
                k = len(batch_rows)
                flush_batch()
                pbar.update(k)

    main_conn.close()
    out_conn.close()
    elapsed = time.time() - t0
    print(
        f"[OK] 本 run 写入 {n_written} 条，dim={dim}，耗时 {elapsed:.1f}s "
        f"({n_written / max(elapsed, 1e-6):.1f} 条/s)；检查点已更新至最后成功批次。",
        flush=True,
    )


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="构建论文标题向量 SQLite（标签路 Stage5，支持断点续跑）")
    p.add_argument("--limit", type=int, default=0, help="本 run 最多新写入多少条（0=不限制）")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 SQLite 路径，默认 config.WORK_TITLE_EMB_DB_PATH",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="从输出库 work_title_emb_build_progress.last_rowid 继续（须与当前 SBERT dim 一致）",
    )
    p.add_argument("--batch-size", type=int, default=32, help="encode_batch 批大小")
    args = p.parse_args(argv)
    out = args.out.strip() or WORK_TITLE_EMB_DB_PATH
    run(out_path=out, limit=args.limit, resume=args.resume, batch_size=max(1, args.batch_size))


if __name__ == "__main__":
    main()

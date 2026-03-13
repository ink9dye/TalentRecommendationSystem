import os
import sys
import sqlite3
import json
from collections import defaultdict

# ---------- 引入 config ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from config import DB_PATH, VOCAB_STATS_DB_PATH


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "vocab_check")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, "industry_vocab_quality_report.txt")


def load_industry_vocab(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT voc_id, term
        FROM vocabulary
        WHERE entity_type='industry'
    """)
    return cursor.fetchall()


def load_vocab_stats(conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT voc_id, work_count, domain_span, domain_dist
        FROM vocabulary_domain_stats
    """)

    stats = {}

    for vid, work_count, span, dist in cursor.fetchall():

        try:
            dist = json.loads(dist)
        except:
            dist = {}

        stats[vid] = {
            "work_count": work_count,
            "domain_span": span,
            "domain_dist": dist
        }

    return stats


def compute_job_coverage(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT skills
        FROM jobs
        WHERE skills IS NOT NULL
    """)

    job_terms = defaultdict(int)

    for (skills,) in cursor.fetchall():

        if not skills:
            continue

        for t in skills.replace("，", ",").split(","):
            term = t.strip().lower()
            if term:
                job_terms[term] += 1

    return job_terms


def main():

    print("读取数据库...")

    main_conn = sqlite3.connect(DB_PATH)
    stats_conn = sqlite3.connect(VOCAB_STATS_DB_PATH)

    vocab = load_industry_vocab(main_conn)
    stats = load_vocab_stats(stats_conn)
    job_terms = compute_job_coverage(main_conn)

    main_conn.close()
    stats_conn.close()

    paper_list = []
    span_list = []
    job_list = []

    for vid, term in vocab:

        stat = stats.get(vid)

        if stat:

            paper_list.append((term, stat["work_count"]))
            span_list.append((term, stat["domain_span"]))

        job_count = job_terms.get(term, 0)
        job_list.append((term, job_count))

    paper_list.sort(key=lambda x: x[1], reverse=True)
    span_list.sort(key=lambda x: x[1], reverse=True)
    job_list.sort(key=lambda x: x[1], reverse=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        f.write("========= 工业词污染分析报告 =========\n\n")

        f.write("====== 论文覆盖率 TOP 100 （最危险泛词）======\n")
        for term, count in paper_list[:100]:
            f.write(f"{term}\t{count}\n")

        f.write("\n====== 跨领域 TOP 100 （语义污染源）======\n")
        for term, span in span_list[:100]:
            f.write(f"{term}\tspan={span}\n")

        f.write("\n====== JD 覆盖率 TOP 100 （岗位泛词）======\n")
        for term, count in job_list[:100]:
            f.write(f"{term}\t{count}\n")

    print("分析完成")
    print("报告位置:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
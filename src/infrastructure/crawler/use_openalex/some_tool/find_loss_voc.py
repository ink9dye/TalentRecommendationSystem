import sqlite3
import os
import re
from collections import Counter
from config import DB_PATH


def find_clean_missing_voc(db_path):
    if not os.path.exists(db_path):
        print(f"[!] 错误: 找不到数据库:\n    {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        print(f"[*] 正在分析数据库: {db_path}")
        print("[*] 正在筛选：既无括号标注、又未完成 Topic 分层的词条...")

        # 1. 加载候选词：无 Topic 且名称中不含括号
        missing_voc = {}
        cursor = conn.execute("""
                              SELECT v.voc_id, v.term
                              FROM vocabulary v
                                       LEFT JOIN vocabulary_topic vt ON v.voc_id = vt.voc_id
                              WHERE vt.voc_id IS NULL
                                AND v.entity_type <> 'industry'
                              """)

        for row in cursor:
            term = row['term'] or ""
            # 关键过滤：排除掉包含括号的词条 (如 "work (physics)")
            if '(' not in term and ')' not in term:
                missing_voc[term.lower().strip()] = row['voc_id']

        if not missing_voc:
            print("[!] 未发现符合条件的漏网词条。")
            return

        print(f"[*] 待检测目标数: {len(missing_voc)} 条")

        # 2. 扫描论文词频
        counter = Counter()
        cursor = conn.execute("SELECT concepts_text, keywords_text FROM works")

        for row in cursor:
            content = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
            terms = content.split('|')
            for t in terms:
                t_clean = t.strip().lower()
                if t_clean in missing_voc:
                    counter[t_clean] += 1

        # 3. 结果展现
        print("\n" + "=" * 70)
        print(f"{'VOC_ID':<10} | {'频次':<8} | {'纯净缺失词条 (无括号/无分类)'}")
        print("-" * 70)

        results = [
            (term, count, missing_voc[term])
            for term, count in counter.items() if count >= 1
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        for term, count, voc_id in results[:100]:
            print(f"{voc_id:<10} | {count:<8} | {term}")

        print("=" * 70)
        print(f"[*] 统计完成。共找到 {len(results)} 个纯净漏网词。")

    except sqlite3.OperationalError as e:
        print(f"[!] SQLite 错误: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    find_clean_missing_voc(DB_PATH)
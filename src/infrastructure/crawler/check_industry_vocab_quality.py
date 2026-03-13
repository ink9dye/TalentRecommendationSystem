import os
import re
import sqlite3
import sys
from collections import Counter

# 让脚本能够从 src/infrastructure/crawler 下导入项目根目录的 config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DB_PATH


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "vocab_check")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_list(output_dir: str, filename: str, data: list[str]) -> None:
    path = os.path.join(output_dir, filename)
    unique_sorted = sorted(set(data))
    with open(path, "w", encoding="utf-8") as f:
        for item in unique_sorted:
            f.write(item + "\n")
    print(f"{filename} -> {len(unique_sorted)} 条")


def main() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库不存在: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"正在读取工业词汇，数据库路径: {DB_PATH}")

    cursor.execute("""
        SELECT term
        FROM vocabulary
        WHERE entity_type = 'industry'
          AND term IS NOT NULL
          AND TRIM(term) != ''
    """)

    rows = cursor.fetchall()
    conn.close()

    terms = [row[0].strip() for row in rows if row[0]]
    print(f"工业词数量: {len(terms)}")

    term_counter = Counter(terms)

    space_abnormal = []
    punctuation_terms = []
    number_terms = []
    long_terms = []
    single_terms = []
    keyword_terms=[]

    # 含标点：保留中文、英文、数字、下划线、空格，其他都判为“含标点/特殊符号”
    punctuation_pattern = re.compile(r"[^\w\u4e00-\u9fa5 ]")
    digit_pattern = re.compile(r"\d")
    keyword_pattern = re.compile(r"(论文|竞赛|相关|经验)")
    for term in terms:
        # 1. 包含空格异常
        # 这里定义为：连续空格，或空格数量 >= 2
        if "  " in term or term.count(" ") >= 2:
            space_abnormal.append(term)

        # 2. 包含标点 / 特殊符号
        if punctuation_pattern.search(term):
            punctuation_terms.append(term)

        # 3. 包含数字
        if digit_pattern.search(term):
            number_terms.append(term)

        # 4. 过长词
        # 阈值你可以后续再调，比如 20 / 25 / 30
        if len(term) > 30:
            long_terms.append(term)

        # 5. 只出现 1 次词
        if term_counter[term] == 1:
            single_terms.append(term)
        if keyword_pattern.search(term):
            keyword_terms.append(term)
    print("\n开始生成检测结果...")
    save_list(OUTPUT_DIR, "space_abnormal.txt", space_abnormal)
    save_list(OUTPUT_DIR, "punctuation_terms.txt", punctuation_terms)
    save_list(OUTPUT_DIR, "number_terms.txt", number_terms)
    save_list(OUTPUT_DIR, "long_terms.txt", long_terms)
    save_list(OUTPUT_DIR, "single_terms.txt", single_terms)
    save_list(OUTPUT_DIR, "keyword_terms.txt", keyword_terms)

    print("\n检测完成，结果目录:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
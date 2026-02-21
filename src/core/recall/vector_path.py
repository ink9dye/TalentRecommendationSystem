import faiss
import json
import sqlite3
import time
import numpy as np
from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH


class VectorPath:
    """
    向量路召回：实现基于 SBERT 的语义召回
    """

    def __init__(self, recall_limit=300):
        self.search_k = 500  # 检索深度
        self.recall_limit = recall_limit

        # 1. 加载论文摘要索引
        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

        # 2. 加载岗位描述索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

    def recall(self, query_vector, target_domains=None, verbose=False):
        """
        向量路召回：实现基于领域 ID 的论文级硬过滤，且 domain_id 为可选参数
        :param query_vector: 输入向量
        :param target_domains: 领域 ID 列表或字符串，例如 ['1'] 或 '1'
        :param verbose: 是否打印中间调试信息
        """
        start_t = time.time()
        conn = sqlite3.connect(DB_PATH)

        # 统一处理 target_domains 格式
        if target_domains:
            if isinstance(target_domains, str):
                target_set = {target_domains}
            else:
                target_set = set(target_domains)
        else:
            target_set = None

        try:
            # --- 步骤 1: 语义检索相似论文 (Faiss 获取最相关的论文 ID 序列) ---
            _, indices = self.index.search(query_vector, self.search_k)
            raw_work_ids = [self.id_map[idx] for idx in indices[0] if 0 <= idx < len(self.id_map)]

            if not raw_work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 2: 获取论文的领域标签用于过滤 ---
            placeholders = ','.join(['?'] * len(raw_work_ids))
            sql = f"SELECT work_id, domain_ids FROM works WHERE work_id IN ({placeholders})"
            work_data = conn.execute(sql, raw_work_ids).fetchall()
            domain_dict = {row[0]: row[1] for row in work_data}

            # --- 步骤 3: 领域硬过滤（只有对应领域的论文才能发挥作用） ---
            filtered_work_ids = []
            for wid in raw_work_ids:
                if wid not in domain_dict:
                    continue

                # 如果提供了 domain_id，执行交集校验
                if target_set:
                    work_domains_str = domain_dict[wid]
                    if work_domains_str:
                        actual_domains = set(work_domains_str.split('|'))
                        if not (actual_domains & target_set):
                            continue  # 领域不匹配，跳过此论文
                    else:
                        continue  # 论文无领域标签，跳过

                # 如果没有 domain_id 或匹配成功，则加入列表
                filtered_work_ids.append(wid)

            if not filtered_work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 4: 映射到作者，并保持原始论文的语义排名顺序 ---
            # 我们使用过滤后的论文列表顺序来查询作者
            work_placeholders = ','.join(['?'] * len(filtered_work_ids))
            ordered_work_str = ",".join(filtered_work_ids)

            # SQL 解析：按该作者关联的最高质量(排名最靠前)论文进行排序
            author_query = f"""
                SELECT author_id 
                FROM authorships 
                WHERE work_id IN ({work_placeholders})
                GROUP BY author_id
                ORDER BY MIN(instr(?, work_id))
            """

            query_params = [ordered_work_str] + filtered_work_ids
            author_ids = [row[0] for row in conn.execute(author_query, query_params).fetchall()]

        finally:
            conn.close()

        duration = (time.time() - start_t) * 1000
        return author_ids[:self.recall_limit], duration


# 修改 vector_path.py 最后的 if __name__ == "__main__": 部分
if __name__ == "__main__":
    from src.core.recall.input_to_vector import QueryEncoder  # 确保路径正确

    v_path = VectorPath(recall_limit=300)
    encoder = QueryEncoder()

    fields = {
        "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
        "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
        "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
        "16": "地质学", "17": "经济学"
    }


    def get_work_title(author_id):
        """简单从 SQLite 获取该作者的一篇论文标题用于展示"""
        conn = sqlite3.connect(DB_PATH)
        res = conn.execute("""
                           SELECT w.title
                           FROM works w
                                    JOIN authorships a ON w.work_id = a.work_id
                           WHERE a.author_id = ? LIMIT 1
                           """, (author_id,)).fetchone()
        conn.close()
        return res[0] if res else "无论文数据"


    print("\n" + "=" * 115)
    print("🚀 向量路 (Vector Path) 独立语义召回测试")
    print("-" * 115)
    f_list = list(fields.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 115)

    try:
        domain_choice = input("\n请选择领域编号 (1-17, 0跳过): ").strip() or "0"
        current_field = fields.get(domain_choice, "全领域")

        while True:
            user_input = input(f"\n[{current_field}] 请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q': break

            # 1. 文本转向量
            query_vec, _ = encoder.encode(user_input)
            faiss.normalize_L2(query_vec)

            # 2. 执行召回
            author_ids, duration = v_path.recall(query_vec,
                                                 target_domains=domain_choice if domain_choice != "0" else None)

            # 3. 打印报告
            print(f"\n[召回报告] 耗时: {duration:.2f}ms | 命中人数: {len(author_ids)}")
            print("-" * 115)
            print(f"{'排名':<6} | {'作者 ID':<12} | {'检索路径':<15} | {'代表作标题 (数据源: SQLite)'}")
            print("-" * 115)

            for rank, aid in enumerate(author_ids[:20], 1):
                title = get_work_title(aid)
                if len(title) > 70: title = title[:67] + "..."
                print(f"#{rank:<5} | {aid:<12} | {'Vector (V)':<15} | {title}")

            print("-" * 115)

    except KeyboardInterrupt:
        print("\n[!] 测试结束")
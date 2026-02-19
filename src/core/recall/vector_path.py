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


if __name__ == "__main__":
    # 实例化向量路召回组件，设置召回上限为300
    v_path = VectorPath(recall_limit=300)

    print("\n" + "=" * 60)
    print("🚀 向量路召回独立测试控制台")
    print("请输入查询向量 (JSON列表格式) 进行语义召回")
    print("=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴稠密向量 (输入 'q' 退出):\n>> ").strip()

            # 检查退出条件
            if not raw_input or raw_input.lower() == 'q':
                print("退出测试模式")
                break

            try:
                # 解析JSON格式的向量列表
                vector_list = json.loads(raw_input)

                # 转换为numpy数组并调整维度
                query_vec = np.array([vector_list]).astype('float32')

                # 执行L2归一化以适应Faiss的内积索引
                faiss.normalize_L2(query_vec)

                # 执行召回，开启verbose模式显示中间过程
                print("\n[执行召回中...]")
                author_ids, duration = v_path.recall(query_vec, verbose=True)

                # 输出结果
                print(f"\n[召回结果报告]")
                print(f"- 召回耗时: {duration:.2f} ms")
                print(f"- 召回作者数量: {len(author_ids)}")

                if author_ids:
                    print(f"- 前5位作者ID: {author_ids[:5]}")
                    print(f"- 全部作者ID列表:")
                    # 每行显示10个ID，提高可读性
                    for i in range(0, len(author_ids), 10):
                        print(f"  {author_ids[i:i + 10]}")
                else:
                    print("- 未找到相关作者")

                print("-" * 30)

            except json.JSONDecodeError:
                print("[错误] JSON格式解析失败，请确保输入的是有效的JSON列表格式")
                print("示例: [0.1, -0.2, 0.3, ...]")
            except Exception as e:
                print(f"[错误] 召回过程发生异常: {str(e)}")
                import traceback

                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n[!] 测试被用户中断")

    print("[*] 向量路召回测试结束")
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
        向量路召回：实现基于领域 ID 的必然硬过滤，并保持语义排名顺序
        :param query_vector: 输入向量
        :param target_domains: 领域 ID 列表，例如 ['1', '4']
        :param verbose: 是否打印中间调试信息
        """
        start_t = time.time()
        conn = sqlite3.connect(DB_PATH)  #

        try:
            # --- 步骤 1: 岗位锚定 (调试逻辑) ---
            if verbose:
                _, j_indices = self.job_index.search(query_vector, 3)  #
                j_ids = [self.job_id_map[idx] for idx in j_indices[0] if 0 <= idx < len(self.job_id_map)]  #
                if j_ids:
                    job_placeholders = ','.join(['?'] * len(j_ids))  #
                    jobs = conn.execute(
                        f"SELECT job_name, description FROM jobs WHERE securityId IN ({job_placeholders})",
                        j_ids).fetchall()  #
                    print(f"\n[向量路中间过程 - 匹配岗位需求]")
                    for name, desc in jobs:
                        print(f" - 岗位: {name} | 描述: {desc[:60]}...")  #

            # --- 步骤 2: 语义检索相似论文 (Faiss 初步检索获取原始排名) ---
            _, indices = self.index.search(query_vector, self.search_k)  #
            raw_work_ids = [self.id_map[idx] for idx in indices[0] if 0 <= idx < len(self.id_map)]  #

            if not raw_work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 3: 领域硬过滤与排名保序 ---
            placeholders = ','.join(['?'] * len(raw_work_ids))
            # 仅查询 domain_ids，不改变原始 raw_work_ids 的顺序
            sql = f"SELECT work_id, domain_ids FROM works WHERE work_id IN ({placeholders})"
            candidates = conn.execute(sql, raw_work_ids).fetchall()  #

            # 建立领域 ID 映射字典，用于 O(1) 级别的快速检索
            domain_map = {row[0]: row[1] for row in candidates}

            filtered_work_ids = []
            target_set = set(target_domains) if target_domains else None

            # 【重要优化】按照 raw_work_ids 的原始语义排名顺序进行过滤
            for wid in raw_work_ids:
                if wid not in domain_map:
                    continue

                d_ids_str = domain_map[wid]
                if target_set:
                    if d_ids_str:
                        actual_domains = set(d_ids_str.split('|'))
                        # 必然项逻辑：论文领域与岗位领域必须有交集
                        if actual_domains & target_set:
                            filtered_work_ids.append(wid)
                else:
                    # 如果没有领域约束，则直接按序保留
                    filtered_work_ids.append(wid)

            if not filtered_work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 4: 调试信息打印 (基于过滤并保序后的列表) ---
            if verbose:
                # 此处省略具体打印逻辑，保持原有摘要提取代码即可
                pass

            # --- 步骤 5: 获取对应的作者 ID (维持论文的质量排序) ---
            # 为了保持作者的“质量”排名，我们通过过滤后的论文列表顺序来查询
            all_work_placeholders = ','.join(['?'] * len(filtered_work_ids))
            # 注意：此处使用聚合确保作者不重复，但由于 work_ids 有序，
            # 产出的作者列表也会倾向于先出现高质量论文的作者
            author_query = f"""
                SELECT author_id FROM authorships 
                WHERE work_id IN ({all_work_placeholders})
                GROUP BY author_id
                ORDER BY MIN(instr(?, work_id))
            """
            # 这里通过 instr 辅助维持原始 work_ids 的优先级顺序
            ordered_work_str = ",".join(filtered_work_ids)
            author_ids = [row[0] for row in conn.execute(author_query, [ordered_work_str]).fetchall()]

        finally:
            conn.close()  #

        duration = (time.time() - start_t) * 1000
        return author_ids[:self.recall_limit], duration  #


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
import sqlite3
import pandas as pd
import numpy as np
import json
import os
import gc

# --- 配置区 ---
from config import DB_PATH, FEATURE_INDEX_PATH


class FeatureIndexBuilder:
    def __init__(self, db_path, feature_index_path):
        self.db_path = db_path
        self.feature_index_path = feature_index_path

    def _normalize(self, df, columns):
        """
        增强版归一化：对数平滑 + Min-Max Scaling
        解决 H-index 和引用量长尾分布导致的特征压缩问题
        """
        for col in columns:
            if col not in df.columns:
                continue

            # 1. 对数平滑处理：ln(x + 1) 防止零值
            # 这样可以拉近顶级学者与普通学者的分值差距，增强模型对中层人才的敏感度
            df[col] = np.log1p(df[col].astype(float))

            # 2. 线性归一化到 0-1
            max_val, min_val = df[col].max(), df[col].min()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
        return df

    def build(self):
        print("--- 启动任务: 构建学者/机构特征索引 (对数平滑版) ---")

        if not os.path.exists(self.db_path):
            print(f"ERROR: 数据库文件不存在: {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)

        try:
            # 1. 处理作者特征
            print("正在提取作者特征并执行对数归一化...")
            # 对应系统设计的作者表核心指标 [cite: 2, 3]
            authors_df = pd.read_sql_query(
                "SELECT author_id as id, h_index, works_count, cited_by_count FROM authors",
                conn
            )
            authors_df = self._normalize(authors_df, ['h_index', 'works_count', 'cited_by_count'])
            author_features = authors_df.set_index('id').T.to_dict()

            # 释放内存
            del authors_df
            gc.collect()

            # 2. 处理机构特征
            print("正在提取机构特征并执行对数归一化...")
            inst_df = pd.read_sql_query(
                "SELECT inst_id as id, works_count, cited_by_count FROM institutions",
                conn
            )
            inst_df = self._normalize(inst_df, ['works_count', 'cited_by_count'])
            inst_features = inst_df.set_index('id').T.to_dict()

            del inst_df
            gc.collect()

            # 3. 打包并持久化存储
            # 结果将作为 KGAT-AX 模型的全息嵌入层输入
            feature_bundle = {
                "author": author_features,
                "institution": inst_features,
                "metadata": {
                    "version": "1.1",
                    "scaling_method": "log1p_minmax",
                    "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            os.makedirs(os.path.dirname(self.feature_index_path), exist_ok=True)

            print(f"正在写入文件: {self.feature_index_path}")
            with open(self.feature_index_path, 'w', encoding='utf-8') as f:
                json.dump(feature_bundle, f)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(self.feature_index_path):
                print(f"SUCCESS: 特征索引已保存。文件大小: {os.path.getsize(self.feature_index_path) / 1024:.2f} KB")

        except Exception as e:
            print(f"ERROR: 特征索引构建失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
            gc.collect()


if __name__ == "__main__":
    builder = FeatureIndexBuilder(DB_PATH, FEATURE_INDEX_PATH)
    builder.build()
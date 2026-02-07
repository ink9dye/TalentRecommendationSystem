import sqlite3
import pandas as pd
import json
import os
import gc

# --- 配置区 ---
from config import DB_PATH, FEATURE_INDEX_PATH

class FeatureIndexBuilder:
    def __init__(self, db_path, FEATURE_INDEX_PATH):
        self.db_path = db_path
        self.FEATURE_INDEX_PATH = FEATURE_INDEX_PATH

    def _normalize(self, df, columns):
        """归一化特征到 0-1 之间"""
        for col in columns:
            max_val, min_val = df[col].max(), df[col].min()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
        return df

    def build(self):
        print("--- 启动任务: 构建学者/机构特征索引 ---")
        conn = sqlite3.connect(self.db_path)

        try:
            # 1. 处理作者特征
            print("正在提取作者特征并归一化...")
            # 修正字段：id -> author_id
            authors_df = pd.read_sql_query(
                "SELECT author_id as id, h_index, works_count, cited_by_count FROM authors",
                conn
            )
            authors_df = self._normalize(authors_df, ['h_index', 'works_count', 'cited_by_count'])
            author_features = authors_df.set_index('id').T.to_dict()

            # 释放中间变量内存
            del authors_df
            gc.collect()

            # 2. 处理机构特征
            print("正在提取机构特征并归一化...")
            # 修正字段：id -> inst_id
            inst_df = pd.read_sql_query(
                "SELECT inst_id as id, works_count, cited_by_count FROM institutions",
                conn
            )
            inst_df = self._normalize(inst_df, ['works_count', 'cited_by_count'])
            inst_features = inst_df.set_index('id').T.to_dict()

            del inst_df
            gc.collect()

            # 3. 打包并保存
            feature_bundle = {"author": author_features, "institution": inst_features}

            # 确保目录存在
            os.makedirs(os.path.dirname(self.FEATURE_INDEX_PATH), exist_ok=True)

            print(f"正在写入文件: {self.FEATURE_INDEX_PATH}")
            with open(self.FEATURE_INDEX_PATH, 'w', encoding='utf-8') as f:
                json.dump(feature_bundle, f)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(self.FEATURE_INDEX_PATH):
                print(f"SUCCESS: 特征索引已保存。大小: {os.path.getsize(self.FEATURE_INDEX_PATH) / 1024:.2f} KB")

        except Exception as e:
            print(f"ERROR: 特征索引构建失败: {e}")
        finally:
            conn.close()
            gc.collect()


# --- 独立运行入口 ---
if __name__ == "__main__":
    builder = FeatureIndexBuilder(DB_PATH, FEATURE_INDEX_PATH)
    builder.build()
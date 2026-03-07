import json
import logging
from datetime import datetime
from alex_utils import generate_work_id, clean_id, safe_get
from database import DatabaseManager
from src.infrastructure.crawler.use_openalex.db_config import FIELDS # 导入 17 领域配置

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.today = datetime.now().date()

        # --- 关键新增：预构建领域名称到 ID 的映射表，提升匹配效率 ---
        # 映射示例: {"computer science": "1", "medicine": "2", ...}
        # 将 FIELDS 中的下划线替换为空格并转为小写，以匹配 OpenAlex 的 display_name
        self.domain_name_to_id = {
            v[0].replace("_", " ").lower(): k
            for k, v in FIELDS.items()
        }

    def _extract_domain_ids(self, concepts: list) -> str:
        """
        新增辅助方法：从论文原始概念列表中提取匹配的项目标准 1-17 领域 ID
        """
        matched_ids = []
        for c_name in concepts:
            c_lower = c_name.lower()
            for std_name, d_id in self.domain_name_to_id.items():
                # 如果论文概念（如 "Artificial Intelligence"）包含标准领域名（如 "computer science"）
                # 或者属于该大类，则标记该领域 ID
                if std_name in c_lower:
                    matched_ids.append(d_id)

        if not matched_ids:
            return ""
        # 去重并排序，生成标准格式如 "1|4"
        return "|".join(sorted(list(set(matched_ids))))

    def restore_abstract(self, inverted_index: dict) -> str:
        """从 OpenAlex 反向索引中还原英文纯文本摘要（保持原有逻辑，增加健壮性）"""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        try:
            word_map = {}
            for word, pos_list in inverted_index.items():
                for pos in pos_list:
                    word_map[pos] = word
            if not word_map:
                return ""
            # 按照位置索引顺序排列并拼接
            max_index = max(word_map.keys())
            abstract_list = [word_map.get(i, "") for i in range(max_index + 1)]
            return " ".join(abstract_list).strip()
        except Exception as e:
            logger.error(f"摘要还原失败: {e}")
            return ""

    def process_work(self, work: dict, field_id: str, field_name: str):
        """
        核心处理逻辑：已适配 11 字段数据库架构，支持自动提取 1-17 领域 ID
        """
        # 1. 日期校验与拦截
        pub_date_str = work.get('publication_date')
        if not pub_date_str:
            return {"status": "no_date"}

        try:
            pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
            if pub_date > self.today:
                return {"status": "filtered_future_date"}
        except Exception:
            return {"status": "date_format_error"}

        work_id = generate_work_id(work)
        is_new = not self.db.work_exists(work_id)

        # 记录学科归属关系 (保持现有逻辑)
        is_field_new = self.db.save_work_field_relation(work_id, field_id, field_name)

        source_raw = safe_get(work, ['primary_location', 'source'])
        s_id = clean_id(source_raw.get('id')) if source_raw else None

        # 2. 处理论文主表、摘要及词汇 (针对新论文执行 11 字段封装)
        if is_new:
            concepts = [c.get('display_name') for c in (work.get('concepts') or []) if c.get('display_name')]
            keywords = [k.get('display_name') for k in (work.get('keywords', []) or []) if k.get('display_name')]

            # --- 核心改进：自动计算领域 ID 串 ---
            domain_ids = self._extract_domain_ids(concepts)

            # 适配 11 个字段的 works 表
            # 顺序: id, doi, title, year, date, citation, concepts_text, domain_ids, keywords_text, type, language
            work_row = (
                work_id,
                work.get('doi'),
                work.get('display_name'),
                work.get('publication_year'),
                pub_date_str,
                work.get('cited_by_count', 0),
                "|".join(concepts),
                domain_ids,  # <--- 新增的第 8 个字段：项目标准领域 ID
                "|".join(keywords),
                work.get('type'),
                work.get('language')
            )

            # 摘要处理
            abstract_raw = work.get('abstract_inverted_index')
            full_text = self.restore_abstract(abstract_raw)
            abstract_row = (work_id, json.dumps(abstract_raw), full_text)

            # 语义词汇
            vocab_terms = [(c, "concept") for c in concepts] + [(k, "keyword") for k in keywords]

            # 确保 database.py 中的此方法已更新为 11 个占位符
            self.db.save_work_bundle(work_row, abstract_row, vocab_terms)

        # 3. 解析并准备作者、机构及关系数据 (保持 8 字段 authorships 逻辑不变)
        authors_data, insts_data, rels_data, sources_data = [], [], [], []

        # 保存 Source 基础信息
        if s_id and source_raw:
            sources_data.append((
                s_id, source_raw.get('display_name'), source_raw.get('type'),
                source_raw.get('works_count', 0), source_raw.get('cited_by_count', 0), None
            ))

        for idx, authorship in enumerate(work.get('authorships', []) or []):
            author_raw = authorship.get('author', {})
            a_id = clean_id(author_raw.get('id'))
            if not a_id: continue

            authors_data.append((
                a_id, clean_id(author_raw.get('orcid')), author_raw.get('display_name') or "Unknown Author",
                author_raw.get('works_count', 0), author_raw.get('cited_by_count', 0),
                0, clean_id(safe_get(author_raw, ['last_known_institution', 'id'])), None
            ))

            insts = authorship.get('institutions', []) or []
            is_corr = 1 if authorship.get('is_corresponding') else 0
            pos_label = authorship.get('author_position')
            is_alphabetical = 1 if authorship.get('is_alphabetical') else 0

            if not insts:
                rels_data.append((work_id, a_id, None, s_id, idx + 1, pos_label, is_corr, is_alphabetical))
            else:
                for inst in insts:
                    i_id = clean_id(inst.get('id'))
                    if i_id:
                        insts_data.append((
                            i_id, inst.get('display_name'), inst.get('country_code'), inst.get('type'),
                            inst.get('works_count', 0), inst.get('cited_by_count', 0), None
                        ))
                        rels_data.append((work_id, a_id, i_id, s_id, idx + 1, pos_label, is_corr, is_alphabetical))

        # 4. 执行批量保存
        if sources_data: self.db.save_source_batch(sources_data)
        if authors_data: self.db.save_author_batch(authors_data)
        if insts_data: self.db.save_institution_batch(insts_data)
        if rels_data: self.db.save_authorship_batch(rels_data)

        return {"work_id": work_id, "is_new": is_new, "is_field_new": is_field_new}
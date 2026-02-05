import json
import logging
from datetime import datetime
from utils import generate_work_id, clean_id, safe_get
from database import DatabaseManager

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.today = datetime.now().date()

    def restore_abstract(self, inverted_index: dict) -> str:
        """从 OpenAlex 反向索引中还原英文纯文本摘要"""
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
        核心处理逻辑：整合主表存储、摘要还原、作者画像提取及 HIN 关系构建
        """
        # 1. 日期校验与拦截
        pub_date_str = work.get('publication_date')
        if not pub_date_str:
            return {"status": "no_date"}

        try:
            pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
            if pub_date > self.today:
                return {"status": "filtered_future_date"}  # 拦截未来日期的论文
        except Exception:
            return {"status": "date_format_error"}

        work_id = generate_work_id(work)
        is_new = not self.db.work_exists(work_id)  # 论文本身是否为库里没有的新记录

        # 【关键修改】无论是不是新论文，都记录其学科归属
        # 这个方法会告诉我们：该论文是否是该学科下“新”发现的
        is_field_new = self.db.save_work_field_relation(work_id, field_id, field_name)

        # 提前提取 Source (期刊/会议) ID，用于注入后续的关系表
        source_raw = safe_get(work, ['primary_location', 'source'])
        s_id = clean_id(source_raw.get('id')) if source_raw else None

        # 2. 处理论文主表、摘要及词汇 (仅针对新论文)
        if is_new:
            concepts = [c.get('display_name') for c in (work.get('concepts') or []) if c.get('display_name')]
            keywords = [k.get('display_name') for k in (work.get('keywords', []) or []) if k.get('display_name')]

            # 适配 11 个字段的 works 表（source_id 已迁移至关系表）
            work_row = (
                work_id,
                work.get('doi'),
                work.get('display_name'),
                work.get('publication_year'),
                pub_date_str,
                work.get('cited_by_count', 0),
                "|".join(concepts),
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

            self.db.save_work_bundle(work_row, abstract_row, vocab_terms)

        # 3. 解析并准备作者、机构及关系数据 (无论是否为新论文，都尝试更新关系)
        authors_data, insts_data, rels_data, sources_data = [], [], [], []

        # 保存 Source 基础信息
        if s_id and source_raw:
            sources_data.append((
                s_id,
                source_raw.get('display_name'),
                source_raw.get('type'),
                source_raw.get('works_count', 0),
                source_raw.get('cited_by_count', 0),
                None
            ))

        # 遍历作者列表，提取画像并建立多维关联
        for idx, authorship in enumerate(work.get('authorships', []) or []):
            author_raw = authorship.get('author', {})
            a_id = clean_id(author_raw.get('id'))
            if not a_id: continue

            # 提取作者基本画像，增加 h_index（初始给 0，等 Phase 3 精确补全）
            authors_data.append((
                a_id,
                clean_id(author_raw.get('orcid')),
                author_raw.get('display_name') or "Unknown Author",
                author_raw.get('works_count', 0),
                author_raw.get('cited_by_count', 0),
                0,  # h_index
                clean_id(safe_get(author_raw, ['last_known_institution', 'id'])),
                None  # 关键：补齐第 8 个字段 last_updated
            ))

            # 提取机构及关系属性
            insts = authorship.get('institutions', []) or []
            is_corr = 1 if authorship.get('is_corresponding') else 0
            pos_label = authorship.get('author_position')

            # 新增：获取作者排序方式
            is_alphabetical = 1 if authorship.get('is_alphabetical') else 0

            # 如果没有机构信息，仍记录"作者-论文-期刊"关系
            if not insts:
                rels_data.append((
                    work_id, a_id, None, s_id, idx + 1, pos_label, is_corr, is_alphabetical  # 新增第8个字段
                ))
            else:
                for inst in insts:
                    i_id = clean_id(inst.get('id'))
                    if i_id:
                        insts_data.append((
                            i_id,
                            inst.get('display_name'),
                            inst.get('country_code'),
                            inst.get('type'),
                            inst.get('works_count', 0),
                            inst.get('cited_by_count', 0),
                            None
                        ))
                        # 核心：建立含 source_id 和 is_alphabetical 的八元组关联记录
                        rels_data.append((
                            work_id, a_id, i_id, s_id, idx + 1, pos_label, is_corr, is_alphabetical  # 新增第8个字段
                        ))

        # 4. 执行批量保存到数据库
        if sources_data:
            self.db.save_source_batch(sources_data)
        if authors_data:
            self.db.save_author_batch(authors_data)
        if insts_data:
            self.db.save_institution_batch(insts_data)
        if rels_data:
            self.db.save_authorship_batch(rels_data)

        return {"work_id": work_id, "is_new": is_new, "is_field_new": is_field_new}
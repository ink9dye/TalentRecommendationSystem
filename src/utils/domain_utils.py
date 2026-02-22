import re
import numpy as np

class DomainProcessor:
    """
    领域逻辑处理器：集中处理领域 ID 的拆解、清洗及 Cypher/Python 正则构建。
    针对格式：'1|4|14', '1,4', ['1|4', '14'] 等复杂情况。
    """

    @staticmethod
    def to_set(domain_input):
        """
        将各种格式的领域输入（字符串、列表、带竖线的复合字符串）统一解析为 ID 集合。
        """
        if not domain_input or domain_input == "0":
            return set()

        target_set = set()
        # 统一转为列表处理
        if isinstance(domain_input, str):
            items = [domain_input]
        elif isinstance(domain_input, (list, set, tuple, np.ndarray)):
            items = domain_input
        else:
            return set()

        for item in items:
            # 兼容竖线 |、逗号 ,、空格 \s 分隔符
            parts = re.split(r'[|,\s]+', str(item))
            for p in parts:
                p_clean = p.strip()
                if p_clean:
                    target_set.add(p_clean)
        return target_set

    @staticmethod
    def build_neo4j_regex(domain_set):
        """
        为 Neo4j 构建正则表达式。
        匹配模式：确保 ID 是独立的，例如匹配 '1' 时不会误命 '14'。
        """
        if not domain_set:
            return None
        # 对 ID 排序保证正则稳定性，并进行转义
        escaped_ids = [re.escape(str(d)) for d in sorted(list(domain_set))]
        # 正则逻辑：(起始/逗号/竖线) + ID + (结尾/逗号/竖线)
        return rf"(^|[|,])({'|'.join(escaped_ids)})([|,]|$)"

    @staticmethod
    def build_python_regex(domain_set):
        """
        为 Python (re 模块) 构建正则表达式，常用于 RankingEngine 的 SQLite 结果修剪。
        """
        if not domain_set:
            return None
        pattern = "|".join([re.escape(str(d)) for d in sorted(list(domain_set))])
        # 对应你 ranking_engine.py 中的逻辑
        return re.compile(fr"(^|,|\|)({pattern})(,|$|\|)")

    @staticmethod
    def has_intersect(paper_domain_input, active_domain_set):
        """
        Python 侧的高性能集合交集判定。
        """
        if not active_domain_set:
            return True
        paper_set = DomainProcessor.to_set(paper_domain_input)
        # 如果论文没标签，为了召回率通常设为 True，或者根据需求设为 False
        if not paper_set:
            return True
        return bool(paper_set & active_domain_set)
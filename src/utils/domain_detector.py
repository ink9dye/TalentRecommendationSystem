import collections
from typing import Optional, Sequence, Tuple, Dict, Any

from src.utils.domain_utils import DomainProcessor


class DomainDetector:
    """
    领域探测器：对外提供统一的「给定 Query → 推断领域 ID 集合」接口。

    默认实现可以基于 Job Faiss + Neo4j 独立完成领域探测；
    若未提供图与索引资源，则回退到 Label 路的 _stage1_domain_and_anchors 逻辑，
    用于兼容 TotalRecall / VectorPath 现有调用方式。
    """

    def __init__(
        self,
        label_path=None,
        graph=None,
        job_index=None,
        job_id_map: Optional[Sequence] = None,
        detect_jobs_top_k: int = 20,
        candidate_domains_top_k: int = 5,
        active_domains_top_k: int = 3,
    ):
        """
        :param label_path: 可选 LabelRecallPath 实例（具备 _stage1_domain_and_anchors 方法），
                           仅在缺少 Job 索引 / 图资源时作为回退方案使用。
        :param graph: 可选 Neo4j Graph 对象，用于 Job 侧领域统计。
        :param job_index: 可选 Faiss 索引，表示 Job 向量空间。
        :param job_id_map: 可选，从 Faiss 行号到 Job 节点 ID 的映射。
        :param detect_jobs_top_k: 领域探测时在 Job 空间检索的 Top-K。
        :param candidate_domains_top_k: 候选领域数（基于 Job 统计）。
        :param active_domains_top_k: 最终激活领域上限（用于简单裁剪）。
        """
        self._label_path = label_path
        self._graph = graph
        self._job_index = job_index
        self._job_id_map = list(job_id_map) if job_id_map is not None else []
        self._detect_jobs_top_k = int(detect_jobs_top_k)
        self._candidate_domains_top_k = int(candidate_domains_top_k)
        self._active_domains_top_k = int(active_domains_top_k)

    # ------------------------------------------------------------------
    # Job 空间领域探测与调试工具
    # ------------------------------------------------------------------

    def detect_from_jobs(
        self, query_vector
    ) -> Tuple[Sequence, Sequence[str], float]:
        """
        通过 Job 向量空间与 Neo4j 统计领域分布。

        逻辑对应 Label 路中的 `_detect_domain_context`：
          - 在 Job 索引中检索 Top-K 相似岗位；
          - 根据这些岗位的 domain_ids 频次统计候选领域；
          - 返回候选 Job ID 列表、候选领域 ID 列表以及主导领域占比。
        """
        if self._job_index is None or not self._job_id_map or self._graph is None:
            return [], [], 0.0

        k = self._detect_jobs_top_k
        _, indices = self._job_index.search(query_vector, k)
        candidate_ids = [
            self._job_id_map[idx]
            for idx in indices[0]
            if 0 <= idx < len(self._job_id_map)
        ]

        domain_counter = collections.Counter()
        cursor = self._graph.run(
            "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.domain_ids AS d_ids",
            j_ids=candidate_ids,
        )
        for row in cursor:
            if row["d_ids"]:
                for d in DomainProcessor.to_set(row["d_ids"]):
                    domain_counter[d] += 1

        n_candidate = self._candidate_domains_top_k
        inferred = [d for d, _ in domain_counter.most_common(n_candidate)]
        dominance = (
            domain_counter.most_common(1)[0][1] / float(k) if domain_counter else 0.0
        )
        return candidate_ids, inferred, dominance

    def get_job_previews(
        self, job_ids: Sequence, max_snippet: int = 200
    ) -> Sequence[Dict[str, Any]]:
        """
        查询命中岗位的名称与描述片段，用于诊断「TopK 是否真是目标领域岗位」。
        返回: [{"id": id, "name": name, "description_snippet": desc[:max_snippet]}, ...]
        """
        if not job_ids or self._graph is None:
            return []
        try:
            cursor = self._graph.run(
                "MATCH (j:Job) WHERE j.id IN $j_ids "
                "RETURN j.id AS id, j.name AS name, j.description AS desc",
                j_ids=list(job_ids)[:20],
            )
            out = []
            for row in cursor:
                desc = (row.get("desc") or "") or ""
                if isinstance(desc, str) and len(desc) > max_snippet:
                    desc = desc[:max_snippet] + "..."
                out.append(
                    {
                        "id": row.get("id"),
                        "name": (row.get("name") or "")[:80],
                        "description_snippet": desc,
                    }
                )
            return out
        except Exception:
            return []

    def get_anchor_debug_stats(
        self, job_ids: Sequence, total_j: float
    ) -> Dict[str, Any]:
        """
        统计参与锚点提取的岗位的 REQUIRE_SKILL 数量，以及熔断前后词数、被熔断词样例。
        返回: {
            "per_job_skill_count": [...],
            "skills_before_melt": N,
            "skills_after_melt": M,
            "melted_terms_sample": [...],
        }
        """
        if not job_ids or self._graph is None or total_j <= 0:
            return {}
        try:
            cursor = self._graph.run(
                """
                MATCH (j:Job) WHERE j.id IN $j_ids
                MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
                WITH j.id AS jid, count(v) AS skill_count
                RETURN jid, skill_count ORDER BY jid
                """,
                j_ids=list(job_ids)[:20],
            )
            per_job = [{"jid": r["jid"], "skill_count": r["skill_count"]} for r in cursor]

            cypher_all = """
            MATCH (j:Job) WHERE j.id IN $j_ids
            MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
            WITH v, (COUNT { (v)<-[:REQUIRE_SKILL]-() } * 1.0 / $total_j) AS cov_j
            RETURN v.id AS vid, v.term AS term, cov_j
            """
            rows = self._graph.run(
                cypher_all, j_ids=list(job_ids)[:20], total_j=float(total_j)
            ).data()
            before_melt = len(rows)
            after_melt = len(
                [r for r in rows if r["cov_j"] < 0.03 and len((r.get("term") or "")) > 1]
            )
            melted = [r["term"] for r in rows if r["cov_j"] >= 0.03][:20]
            return {
                "per_job_skill_count": per_job,
                "skills_before_melt": before_melt,
                "skills_after_melt": after_melt,
                "melted_terms_sample": melted,
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # 对外统一接口
    # ------------------------------------------------------------------

    def detect(
        self,
        query_vector,
        query_text: Optional[str] = None,
        user_domain: Optional[str] = None,
    ):
        """
        推断当前查询的领域集合。

        优先级：
        1. 若 user_domain 不为空：直接使用用户领域（单元素集合）。
        2. 否则：自动探测：
           - 若 graph + job_index 可用：走 Job 空间统计；
           - 否则：复用 Label 路的 _stage1_domain_and_anchors。

        :return: (active_set, applied_str, debug_info)
                 - active_set: set[str]，推断出的领域 ID 集合（字符串形式）
                 - applied_str: 便于日志展示的领域字符串（如 "1|4|14" 或 "All Fields"）
                 - debug_info: 调试信息字典，包含来源、active_set 以及 Stage1 的 debug。
        """
        # 1. 用户显式指定领域时，直接使用
        if user_domain is not None and str(user_domain).strip() not in ("", "0"):
            dom_id = str(user_domain).strip()
            active = {dom_id}
            applied_str = dom_id
            debug = {
                "source": "user",
                "active_set": sorted(active),
                "stage1_debug": None,
            }
            return active, applied_str, debug

        # 2. 自动领域探测
        active = set()
        stage1_debug: Dict[str, Any] = {}

        # 2.1 优先使用独立 Job 探测模式（graph + job_index 可用时）
        if self._job_index is not None and self._graph is not None and self._job_id_map:
            try:
                job_ids, inferred, dominance = self.detect_from_jobs(query_vector)
                # 激活领域集合：简单地裁剪到 active_domains_top_k
                active = set(list(inferred)[: self._active_domains_top_k])
                job_previews = self.get_job_previews(job_ids)
                anchor_debug = self.get_anchor_debug_stats(
                    job_ids, total_j=float(len(self._job_id_map) or 1.0)
                )
                stage1_debug = {
                    "source": "auto_job_space",
                    "job_ids": job_ids,
                    "candidate_domains": inferred,
                    "dominance": dominance,
                    "job_previews": job_previews,
                    "anchor_debug": anchor_debug,
                }
            except Exception as e:  # 防御性降级
                active = set()
                stage1_debug = {"error": str(e), "source": "auto_job_space"}

        # 2.2 否则退化为复用 Label 路的阶段一逻辑（兼容旧用法，如 TotalRecall / VectorPath）
        elif self._label_path is not None:
            try:
                active_set, _, _, debug1 = self._label_path._stage1_domain_and_anchors(  # type: ignore[attr-defined]
                    query_vector,
                    query_text=query_text,
                    domain_id=None,
                )
                # 统一转为字符串 ID 集合
                active = {str(d) for d in (active_set or [])}
                stage1_debug = debug1 or {}
                stage1_debug["source"] = "auto_label_path"
            except Exception as e:  # 防御性降级
                active = set()
                stage1_debug = {"error": str(e), "source": "auto_label_path"}

        if active:
            applied_str = "|".join(sorted(active))
        else:
            applied_str = "All Fields"

        debug = {
            "source": stage1_debug.get("source", "auto"),
            "active_set": sorted(active),
            "stage1_debug": stage1_debug,
        }
        return active, applied_str, debug


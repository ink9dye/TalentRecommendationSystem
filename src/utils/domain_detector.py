import typing as _t


class DomainDetector:
    """
    领域探测器：对外提供统一的「给定 Query → 推断领域 ID 集合」接口。

    实现上复用 Label 路的阶段一逻辑（_stage1_domain_and_anchors），
    但将其封装成独立组件，便于在多路召回（TotalRecall、VectorPath CLI 等）
    之间共享，而不用各自直接依赖 Label 路的内部实现细节。
    """

    def __init__(self, label_path):
        """
        :param label_path: 一个具备 _stage1_domain_and_anchors 方法的实例，
                           通常为 LabelRecallPath。
        """
        self._label_path = label_path

    def detect(
        self,
        query_vector,
        query_text: _t.Optional[str] = None,
        user_domain: _t.Optional[str] = None,
    ):
        """
        推断当前查询的领域集合。

        优先级：
        1. 若 user_domain 不为空：直接使用用户领域（单元素集合）。
        2. 否则：调用 Label 路的 _stage1_domain_and_anchors 自动推断 active_domain_set。

        :return: (active_set, applied_str, debug_info)
                 - active_set: set[str]，推断出的领域 ID 集合（字符串形式）
                 - applied_str: 便于日志展示的领域字符串（如 "1|4|14" 或 "All Fields"）
                 - debug_info: 调试信息字典，包含来源、active_set 以及 Label 路 stage1 的 debug。
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

        # 2. 自动领域探测：复用 Label 路的阶段一逻辑
        active = set()
        stage1_debug = None
        try:
            active_set, _, _, debug1 = self._label_path._stage1_domain_and_anchors(  # type: ignore[attr-defined]
                query_vector,
                query_text=query_text,
                domain_id=None,
            )
            # 统一转为字符串 ID 集合
            active = {str(d) for d in (active_set or [])}
            stage1_debug = debug1
        except Exception as e:  # pragma: no cover - 仅用于防御性降级
            active = set()
            stage1_debug = {"error": str(e)}

        if active:
            applied_str = "|".join(sorted(active))
        else:
            applied_str = "All Fields"

        debug = {
            "source": "auto_label_path",
            "active_set": sorted(active),
            "stage1_debug": stage1_debug,
        }
        return active, applied_str, debug
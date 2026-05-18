import re
import sys
import os
import html
import traceback
import streamlit as st

# 1. 确保能找到项目根目录下的 src 和 config
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
    sys.path.append(root)

from src.core.total_core import TotalCore
from config import DOMAIN_MAP, NAME_TO_DOMAIN_ID


def _inject_layout_styles():
    st.markdown(
        """
        <style>
        #MainMenu { visibility: hidden; }
        header[data-testid="stHeader"] { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        div[data-testid="stDecoration"] { display: none; }
        footer { visibility: hidden; }
        section[data-testid="stSidebar"] { display: none; }
        div[data-testid="stSidebarCollapseButton"] { display: none; }

        .main .block-container {
            max-width: 1150px;
            margin-left: auto;
            margin-right: auto;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .trs-page-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.35rem;
        }
        .trs-hint { color: #888; font-size: 0.82rem; margin-bottom: 1rem; }
        .trs-section {
            font-size: 0.86rem;
            font-weight: 600;
            color: #333;
            margin-top: 0.75rem;
            margin-bottom: 0.3rem;
            padding-bottom: 0.15rem;
            border-bottom: 1px solid #eaeaea;
        }
        .trs-results-head {
            display: flex;
            align-items: baseline;
            gap: 0.75rem;
            margin-bottom: 0.65rem;
        }
        .trs-results-title { font-size: 1rem; font-weight: 600; color: #222; }
        .trs-results-meta { color: #888; font-size: 0.78rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #fff !important;
            border-color: #dcdcdc !important;
            border-radius: 8px !important;
        }
        .trs-row-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
        .trs-score { font-size: 0.88rem; color: #222; white-space: nowrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_engine():
    return TotalCore()


def build_manual_domain_pattern(selected_names):
    """将多选领域名称转为 backend 所需的 '|' 拼接 ID；未选则 None。"""
    if not selected_names:
        return None
    ids = []
    for name in selected_names:
        domain_id = NAME_TO_DOMAIN_ID.get(name)
        if domain_id is not None:
            ids.append(str(domain_id))
    return "|".join(ids) if ids else None


def format_score(value):
    """分数格式化为 4 位小数；非数值则原样或 '-'。"""
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        s = str(value).strip()
        return s if s else "-"


def format_publication_source(representative_work: dict) -> str:
    """发表平台：优先 representative_work.source，兼容旧字段 published_at。"""
    if not isinstance(representative_work, dict):
        return "未知"
    for key in ("source", "published_at"):
        value = representative_work.get(key)
        if value is None:
            continue
        s = str(value).strip()
        if s:
            return s
    return "未知"


_RANKING_BOILERPLATE_PHRASES = (
    "精排阶段将该候选保留在当前排序结果中，说明图结构证据与召回证据没有明显冲突。",
    "精排阶段将该候选保留在当前排序结果中，说明图结构与候选池证据没有明显冲突。",
    "精排阶段未发现与候选池证据明显冲突的信号。",
)


def clean_recommendation_reason(value):
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    for phrase in _RANKING_BOILERPLATE_PHRASES:
        s = s.replace(phrase, "")

    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


# True：展示「查看详细信息」及召回分/精排分等排序信号
_SHOW_DEBUG_EXPANDER = False


def _safe_details(item):
    d = item.get("details")
    return d if isinstance(d, dict) else {}


def _safe_metrics(item):
    m = item.get("metrics")
    return m if isinstance(m, dict) else {}


def _safe_representative_work(item):
    w = item.get("representative_work")
    return w if isinstance(w, dict) else {}


def render_debug_expander(item, query_domain_pattern=None):
    with st.expander("查看详细信息", expanded=False):
        st.markdown("**author_id**")
        st.text(str(item.get("author_id", "")))

        d = _safe_details(item)
        st.markdown("**排序信号**")
        s1, s2 = st.columns(2)
        s1.markdown(f"召回分：{format_score(d.get('recall_score'))}")
        s2.markdown(f"精排分：{format_score(d.get('kgat_score'))}")
        if d.get("rule_stability") is not None:
            st.markdown(f"稳定项：{format_score(d.get('rule_stability'))}")
        if d.get("candidate_pool_score") is not None:
            st.markdown(f"候选池原始分：{format_score(d.get('candidate_pool_score'))}")

        st.markdown("**details**")
        if d:
            st.json(d)
        else:
            st.caption("—")

        m = _safe_metrics(item)
        st.markdown("**metrics**")
        if m:
            st.json(m)
        else:
            st.caption("—")

        w = _safe_representative_work(item)
        st.markdown("**representative_work**")
        if w:
            st.json(w)
        else:
            st.caption("—")

        st.markdown("**score**")
        st.text(format_score(item.get("score")))

        core_keys = {"author_id", "score", "details", "metrics", "representative_work"}
        rest = {k: v for k, v in item.items() if k not in core_keys}
        if rest:
            st.markdown("**其他字段**")
            st.json(rest)

        if query_domain_pattern:
            st.markdown("**手动领域 ID**")
            st.code(str(query_domain_pattern), language=None)


def render_author_card(item, list_index: int, query_domain_pattern=None):
    rank = item.get("rank")
    if rank is None:
        try:
            rank = int(list_index) + 1
        except (TypeError, ValueError):
            rank = list_index + 1
    name = item.get("name") or "未知作者"
    score_str = format_score(item.get("score"))

    reason = clean_recommendation_reason(item.get("recommendation_reason"))
    if reason is None or not str(reason).strip():
        reason_text = "暂无推荐依据"
    else:
        reason_text = str(reason)

    rw = _safe_representative_work(item)
    title = rw.get("title")
    title_display = (
        str(title).strip()
        if title is not None and str(title).strip()
        else "未知论文"
    )
    link = rw.get("link")
    publication_source = format_publication_source(rw)

    m = _safe_metrics(item)
    name_safe = html.escape(str(name))

    with st.container(border=True):
        st.markdown(
            f'<div class="trs-row-head"><span><strong>#{rank}</strong>　{name_safe}</span>'
            f'<span class="trs-score">综合分　{score_str}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="trs-section">推荐依据</div>', unsafe_allow_html=True)
        st.markdown(reason_text)

        st.markdown('<div class="trs-section">代表论文</div>', unsafe_allow_html=True)
        st.markdown(title_display)
        if link:
            st.markdown(f"[OpenAlex]({link})")
        st.caption(f"发表平台：{publication_source}")

        st.markdown('<div class="trs-section">作者画像</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"H-index：{m.get('h_index', '—')}")
        c2.markdown(f"论文数：{m.get('total_papers', '—')}")
        c3.markdown(f"引用量：{m.get('citations', '—')}")

        if _SHOW_DEBUG_EXPANDER:
            render_debug_expander(item, query_domain_pattern=query_domain_pattern)


def main():
    st.set_page_config(
        page_title="科技人才推荐系统",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_layout_styles()

    with st.spinner("加载中…"):
        core = load_engine()

    st.markdown(
        '<div class="trs-page-title">科技人才推荐系统</div>'
        '<div class="trs-hint">请输入岗位需求，系统将返回相关候选作者。</div>',
        unsafe_allow_html=True,
    )

    if "jd_query" not in st.session_state:
        st.session_state.jd_query = ""

    with st.container(border=True):
        c_left, c_right = st.columns([1.65, 1.0], gap="medium")
        with c_left:
            st.markdown("**岗位需求**")
            query_text = st.text_area(
                "岗位需求输入",
                height=240,
                key="jd_query",
                placeholder="请输入岗位职责、研究方向、技能要求等文本",
                label_visibility="collapsed",
            )
            run_recommend = st.button("生成推荐结果")
        with c_right:
            st.markdown("**推荐设置**")
            selected_names = st.multiselect(
                "领域限定",
                options=sorted(DOMAIN_MAP.values()),
            )
            top_n = st.selectbox(
                "展示数量",
                options=[10, 20, 30, 50],
                index=1,
            )

    manual_id_pattern = build_manual_domain_pattern(selected_names)

    if run_recommend:
        if not (query_text or "").strip():
            st.warning("请先输入岗位需求。")
        else:
            results = []
            err_tb = None
            try:
                with st.spinner("正在生成推荐结果…"):
                    results = core.suggest(query_text, manual_domain_id=manual_id_pattern) or []
            except Exception:
                err_tb = traceback.format_exc()
                st.error("推荐流程执行失败")
                with st.expander("错误信息", expanded=False):
                    st.code(err_tb)

            if err_tb is None:
                if not results:
                    st.info("未找到合适候选作者。可以尝试补充岗位技术描述，或放宽领域限制。")
                else:
                    tn = int(top_n)
                    st.markdown(
                        f'<div class="trs-results-head">'
                        f'<span class="trs-results-title">推荐结果</span>'
                        f'<span class="trs-results-meta">展示 Top {tn}</span></div>',
                        unsafe_allow_html=True,
                    )
                    displayed = results[: min(tn, len(results))]
                    for idx, item in enumerate(displayed):
                        if not isinstance(item, dict):
                            continue
                        render_author_card(item, idx, query_domain_pattern=manual_id_pattern)


if __name__ == "__main__":
    main()

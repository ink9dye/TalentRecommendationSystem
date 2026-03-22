import sys
import os
import streamlit as st

# 1. 确保能找到项目根目录下的 src 和 config
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
    sys.path.append(root)

from src.core.total_core import TotalCore
# 直接使用你 config.py 中定义的统一映射
from config import DOMAIN_MAP, NAME_TO_DOMAIN_ID

# 页面基础配置
st.set_page_config(page_title="TalentAI 深度推荐", layout="wide")


@st.cache_resource
def load_engine():
    return TotalCore()


# 初始化引擎
with st.spinner("🚀 系统引擎加载中..."):
    core = load_engine()

# --- 侧边栏：配置中心 ---
with st.sidebar:
    st.title("🧩 匹配配置")
    st.markdown("---")

    # 【核心修改点】使用 multiselect 替代 selectbox 实现多选
    selected_names = st.multiselect(
        "目标岗位领域 (可多选)",
        options=list(DOMAIN_MAP.values()),
        help="支持多选。若不选择，系统将根据 JD 自动计算领域并集。"
    )

    # 处理多选逻辑：将选中的名称转换为 ID 并用 "|" 拼接
    manual_id_pattern = None
    if selected_names:
        selected_ids = [NAME_TO_DOMAIN_ID[name] for name in selected_names]
        manual_id_pattern = "|".join(selected_ids)  # 拼接成类似 "1|4|14" 的正则格式
        st.success(f"已锁定领域 ID: {manual_id_pattern}")
    else:
        st.info("当前模式：自动岗位并集匹配")

    st.markdown("---")
    st.caption("Powered by KGATAX & Streamlit")

# --- 主界面 ---
st.title("🕵️‍♂️ TalentAI 专家推荐系统")
st.write("输入岗位需求，通过知识图谱与精排模型锁定最优人才。")

# JD 输入
query_text = st.text_area(
    "岗位需求描述 (JD Content)",
    placeholder="在此输入完整的岗位职责与任职要求...",
    height=250
)

if st.button("开始深度匹配", type="primary"):
    if not query_text.strip():
        st.warning("请先输入岗位描述。")
    else:
        with st.status("正在穿越知识图谱寻访专家...", expanded=True) as status:
            st.write("执行多路召回与语义对齐...")
            # 将拼接好的 "|" 字符串传给后端
            results = core.suggest(query_text, manual_domain_id=manual_id_pattern)
            status.update(label="匹配完成！", state="complete", expanded=False)

        if not results:
            st.error("未找到符合条件的候选人，请尝试扩大领域范围。")
        else:
            st.success(f"为您找到 {len(results)} 名最匹配的专家")

            # 渲染结果列表
            for item in results[:20]:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 5])

                    with c1:
                        # 综合得分展示
                        st.metric("综合分", item['score'])
                        st.caption(f"召回: {item['details']['recall_score']}")
                        st.caption(f"精排: {item['details']['kgat_score']}")

                    with c2:
                        st.markdown(f"### {item['rank']}. {item['name']} :blue[[ID: {item['author_id']}]]")

                        # 推荐理由
                        st.chat_message("assistant").write(f"**推荐理由：** {item['recommendation_reason']}")

                        # 代表作及链接
                        work = item['representative_work']
                        st.markdown(f"📖 **代表作：** 《{work['title']}》")
                        st.markdown(f"🔗 **[OpenAlex 预览]({work['link']})** | **发表平台：** {work['published_at']}")

                        # 核心指标平铺
                        m = item['metrics']
                        cols = st.columns(3)
                        cols[0].markdown(f"**H-Index** \n `{m.get('h_index', 0)}`")
                        cols[1].markdown(f"**总论文数** \n `{m.get('total_papers', 0)}`")
                        cols[2].markdown(f"**总引用量** \n `{m.get('citations', 0)}`")
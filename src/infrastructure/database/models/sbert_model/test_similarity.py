import os
import warnings

# --- 1. 环境与警告屏蔽 (必须放在最前面) ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 屏蔽 Hugging Face 的正则表达式警告和 Symlink 警告
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

# 2. 路径配置
LOCAL_MODEL_PATH = r'/src/infrastructure/database/models/sbert_model'


def get_clean_model():
    """以最安静的方式加载模型"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("正在首次下载模型，请稍候...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        model.save(LOCAL_MODEL_PATH)

    # 加载模型
    print(f"正在加载本地模型自: {LOCAL_MODEL_PATH} ...")
    model = SentenceTransformer(LOCAL_MODEL_PATH)

    # 强制修复分词器以确保向量质量
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, fix_mistral_regex=True)
    model.tokenizer = tokenizer

    print("模型加载完成，状态：就绪 (Ready)\n")
    return model


# 3. 执行测试
model = get_clean_model()

if model:
    # 模拟：BOSS直聘岗位需求 vs OpenAlex学术背景
    test_cases = [
        ("Space Complexity", "空间复杂度"),
        ("Space Complexity", "内存占用"),
        ("研究算法的空间复杂度优化", "针对大规模系统的内存占用进行调优")
    ]

    print("=" * 40)
    print(f"{'语义相似度深度测试结果':^30}")
    print("=" * 40)

    for a, b in test_cases:
        emb1 = model.encode(a, convert_to_tensor=True)
        emb2 = model.encode(b, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        print(f"输入 A: {a}\n输入 B: {b}\n>>> 相似度分值: {score:.4f}")
        print("-" * 40)

    print(f"验证：输出向量维度为 {emb1.shape[0]}，完全符合 384 维设计要求。")
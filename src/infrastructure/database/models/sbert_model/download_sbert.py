from sentence_transformers import SentenceTransformer
import os

# 定义项目内的模型存储路径
local_model_path = r'/src/infrastructure/database/models/sbert_model'

# 检查目录下是否存在核心权重文件
if not os.path.exists(os.path.join(local_model_path, 'pytorch_model.bin')) and \
        not os.path.exists(os.path.join(local_model_path, 'model.safetensors')):

    print("本地模型文件不完整，正在从 Hugging Face 下载并保存到项目目录...")

    # 1. 从网络加载模型
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 2. 确保目录存在
    os.makedirs(local_model_path, exist_ok=True)

    # 3. 将模型完整保存到该目录
    model.save(local_model_path)
    print(f"模型已成功保存至: {local_model_path}")
else:
    print("检测到完整的本地模型，正在加载...")
    model = SentenceTransformer(local_model_path)

# 测试运行
sentences = ["科技人才推荐", "Academic Research"]
embeddings = model.encode(sentences)
print(f"向量生成成功，形状为: {embeddings.shape}")  # 预期输出 (2, 384)
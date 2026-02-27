import os
import time
from huggingface_hub import snapshot_download

# 设置国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 增加重试次数
os.environ["requests_max_retries"] = "20"

repo_id = "Alibaba-NLP/gte-multilingual-base"
local_dir = r"E:\PythonProject\TalentRecommendationSystem\data\build_sbert\gte-multilingual-base"

def download_with_retry():
    retry_count = 0
    while True:
        try:
            print(f"[*] 正在尝试下载模型 (第 {retry_count + 1} 次尝试)...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,  # 强制开启断点续传
                max_workers=4          # 开启多线程下载
            )
            print("[+] 模型集齐！所有文件下载成功。")
            break
        except Exception as e:
            retry_count += 1
            print(f"[-] 下载中途崩溃: {e}")
            print(f"[*] 5秒后自动重启下载...")
            time.sleep(5)

if __name__ == "__main__":
    download_with_retry()
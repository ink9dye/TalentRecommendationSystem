import os
import subprocess
import sys
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_command(command, description):
    """运行 shell 命令并监控输出"""
    logging.info(f"=== 启动阶段: {description} ===")
    start_time = time.time()

    # 使用 sys.executable 确保使用当前环境的 Python 解释器
    process = subprocess.Popen(
        [sys.executable] + command.split(),
        stdout=None,
        stderr=subprocess.STDOUT
    )

    exit_code = process.wait()
    duration = (time.time() - start_time) / 60

    if exit_code == 0:
        logging.info(f"--- 阶段完成: {description} (耗时: {duration:.2f} 分钟) ---\n")
    else:
        logging.error(f"!!! 阶段失败: {description} (错误码: {exit_code}) !!!")
        sys.exit(exit_code)


def main():
    # 1. 环境准备
    logging.info("开始执行 KGATAX 全流程流水线...")

    # 2. 步骤一：生成训练数据
    # 职责：ID压缩映射、岗位/人才交互对提取、构建物理分区
    run_command("generate_training_data.py", "训练数据生成 (Generate)")

    # 3. 步骤二：构建 KG 索引
    # 职责：将生成的 kg_final.txt 结构化为 SQLite，建立高效率覆盖索引
    # 这是 DataLoaderKGAT 进行秒级子图采样的前置条件
    run_command("build_kg_index.py", "KG 离线索引构建 (Indexing)")

    # 4. 步骤三：启动模型训练
    # 职责：加载数据、执行 CF+KG 联合训练、模型评估
    # 注意：这里你可以根据需要添加特定的命令行参数，如 --n_epoch 100
    run_command("trainer.py", "KGAT-AX 模型训练 (Training)")

    logging.info("=" * 50)
    logging.info("所有流水线任务已成功执行完毕！")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
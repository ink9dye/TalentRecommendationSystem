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
    logging.info("开始执行 KGATAX 全流程流水线...")


    # 步骤 1：生成训练数据
    # 职责：执行 500 人召回 -> RRF 融合重排 -> 四级梯度持久化
    run_command("generate_training_data.py", "四级梯度训练数据生成")

    # 步骤 2：构建 KG 索引
    # 职责：为 3200 万条边构建 SQLite 覆盖索引，支持秒级子图采样
    run_command("build_kg_index.py", "KG 离线索引构建")

    # 步骤 3：启动训练
    # 职责：执行 CF + KG 联合训练，并利用阶梯采样学习排序逻辑
    # 建议增加参数控制，例如训练 50 轮，每 5 轮评估一次
    run_command("trainer.py --n_epoch 50 --evaluate_every 5", "KGAT-AX 阶梯对比训练")

    logging.info("=" * 50)
    logging.info("所有流水线任务已成功执行完毕！")


if __name__ == "__main__":
    main()
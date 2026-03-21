import argparse
import logging
import os
import subprocess
import sys
import time

# 与 generate_training_data / trainer 一致：保证能 import config
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_command(argv: list, description: str) -> None:
    """在 kgat_ax 目录下执行子进程（与脚本内相对导入一致）。"""
    logging.info(f"=== 启动阶段: {description} ===")
    start_time = time.time()

    process = subprocess.Popen(
        [sys.executable] + argv,
        cwd=_SCRIPT_DIR,
        stdout=None,
        stderr=subprocess.STDOUT,
    )

    exit_code = process.wait()
    duration = (time.time() - start_time) / 60

    if exit_code == 0:
        logging.info(f"--- 阶段完成: {description} (耗时: {duration:.2f} 分钟) ---\n")
    else:
        logging.error(f"!!! 阶段失败: {description} (错误码: {exit_code}) !!!")
        sys.exit(exit_code)


def main():
    from src.infrastructure.database.kgat_ax.pipeline_state import (
        clear_all_markers,
        stage1_complete,
        stage2_complete,
        stage3_complete,
        verify_stage1_artifacts,
        verify_stage2_artifacts,
    )

    parser = argparse.ArgumentParser(description="KGAT-AX 全流程流水线（支持断点续跑）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略完成标记，强制重跑所有阶段（会先删除 .done 标记文件）",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="从第几阶段开始：2 要求阶段 1 已完成；3 要求阶段 1、2 已完成（默认 1）",
    )
    parser.add_argument(
        "--skip-stage3",
        action="store_true",
        help="即使阶段 3 未完成也不运行训练（仅生成数据 + 建索引）",
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=50,
        help="传给 trainer.py 的训练轮数",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=5,
        help="传给 trainer.py 的评估间隔",
    )
    parser.add_argument(
        "--no-four-branch",
        action="store_true",
        help="仅训练图塔 + 全局 3 维学术特征（FEATURE_INDEX），不加载 train_four_branch 侧车的 13/12/8 维塔",
    )
    args = parser.parse_args()

    logging.info("开始执行 KGATAX 全流程流水线...")

    if args.force:
        clear_all_markers()
        logging.info("已使用 --force：已清除各阶段 .done 标记，将依次执行全部阶段。")

    if args.start_from >= 2 and not args.force:
        if not stage1_complete():
            logging.error(
                "无法从阶段 %s 开始：阶段 1 未完成（缺少 train/test/kg_final/id_map 或非空校验失败，或缺少 pipeline_stage1.done）。"
                "请先跑完阶段 1，或使用 --force。",
                args.start_from,
            )
            sys.exit(2)

    if args.start_from >= 3 and not args.force:
        if not stage2_complete():
            logging.error(
                "无法从阶段 %s 开始：阶段 2 未完成（kg_index.db 无效或缺少 pipeline_stage2.done）。",
                args.start_from,
            )
            sys.exit(2)

    # ---------- 阶段 1 ----------
    if args.start_from <= 1:
        if not args.force and stage1_complete():
            logging.info(
                "检测到阶段 1 已完成（产物 + pipeline_stage1.done），跳过 generate_training_data.py"
            )
        else:
            if not args.force and verify_stage1_artifacts():
                logging.warning(
                    "存在 train/kg_final/id_map 等产物，但阶段 1 未标记完成；将重新执行以写入 pipeline_stage1.done。"
                )
            run_command(["generate_training_data.py"], "四级梯度训练数据生成")

    # ---------- 阶段 2 ----------
    if args.start_from <= 2:
        if not args.force and stage2_complete():
            logging.info(
                "检测到阶段 2 已完成（kg_index.db + pipeline_stage2.done），跳过 build_kg_index.py"
            )
        else:
            if not args.force and verify_stage2_artifacts():
                logging.warning(
                    "存在有效 kg_index.db，但阶段 2 未标记完成；将执行 build_kg_index（通常会快速跳过重建）。"
                )
            run_command(["build_kg_index.py"], "KG 离线索引构建")

    # ---------- 阶段 3 ----------
    if args.skip_stage3:
        logging.info("已使用 --skip-stage3，不运行 trainer。")
        logging.info("=" * 50)
        logging.info("流水线已按配置结束。")
        return

    if args.start_from <= 3:
        if not args.force and stage3_complete():
            logging.info(
                "检测到阶段 3 已完成（pipeline_stage3.done），跳过 trainer.py。"
                "若需重新训练请使用 --force 或删除 data/kgatax_train_data/pipeline_stage3.done"
            )
        else:
            # 与 generate_training_data._export_four_branch_row 及 parser_kgat / model 默认维度对齐：
            # - n_aux_features=3：全图 h_index / cited_by_count / works_count（FEATURE_INDEX）
            # - 13/12/8：侧车 Recall / Author(含 log 硬指标) / Interaction 塔，训练时走 calc_score_v2
            trainer_argv = [
                "trainer.py",
                "--n_epoch",
                str(args.n_epoch),
                "--evaluate_every",
                str(args.evaluate_every),
                "--n_aux_features",
                "3",
            ]
            if args.no_four_branch:
                trainer_argv += [
                    "--n_recall_features",
                    "0",
                    "--n_author_aux",
                    "0",
                    "--n_interaction_features",
                    "0",
                ]
            else:
                trainer_argv += [
                    "--n_recall_features",
                    "13",
                    "--n_author_aux",
                    "12",
                    "--n_interaction_features",
                    "8",
                ]
            stage3_desc = (
                "KGAT-AX 阶梯对比训练（图塔 + 全局 3 维学术特征）"
                if args.no_four_branch
                else "KGAT-AX 阶梯对比训练（图塔 + 全局 3 维 + 四分支侧车 13/12/8）"
            )
            run_command(trainer_argv, stage3_desc)

    logging.info("=" * 50)
    logging.info("所有流水线任务已成功执行完毕！")


if __name__ == "__main__":
    main()

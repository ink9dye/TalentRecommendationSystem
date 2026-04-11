# -*- coding: utf-8 -*-
"""
标签路一键调试：零交互运行，等价于交互式 CLI 中选用
  - 领域编号 0（跳过显式领域）
  - 详细打印 y
  - 下方默认岗位 JD

运行方式（任选）:
  - 项目根目录: python src/core/recall/label_path_run.py
  - 或在 IDE 中直接运行本文件（已自动把项目根加入 sys.path）
"""
from __future__ import annotations

import os
import sys
import traceback

# 从 recall 目录上溯到项目根，避免「运行当前文件」时找不到 src.*
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from src.core.recall.label_means.label_debug_cli import run_single_label_debug
from src.core.recall.label_path import LabelRecallPath

# 与交互示例一致：0=跳过领域选择
DEFAULT_DOMAIN_CHOICE = "0"
# True=详细诊断打印；False=仅最终人选列表
DEFAULT_VERBOSE = True

# 默认岗位需求（可改此常量或设置环境变量 LABEL_PATH_JD 覆盖）
DEFAULT_JOB_TEXT = """运动学与动力学算法研发：负责机器人运动学、动力学建模以及运动控制算法的设计与优化；建立高性能、可扩展的机器人运动控制与状态估计模块。轨迹规划与全身控制算法开发：研发与优化机器人轨迹生成与全身控制算法，包括但不限于RRT/PRM/CHOMP/MPC/iLQR等；针对复杂场景进行约束优化、时序规划与碰撞规避设计，确保系统的平滑性、稳定性与可执行性。仿真平台构建与验证：利用Isaac Sim/Gazebo/MuJoCo等平台搭建仿真环境，进行算法快速验证与评估；推动仿真到实机的一致性优化，包括动力学一致性、摩擦模型校准、控制频率匹配等。系统集成与性能调优：参与机器人控制系统全流程开发，从底层控制到高层规划架构；开展实时控制性能调优（延迟、抖动、稳定性分析），提升系统在复杂任务下的执行效率与鲁棒性。技术追踪与创新：持续关注运动控制、机器人动力学建模及规划领域的前沿研究，对新算法进行调研、实现与落地；推动技术创新并形成知识沉淀。任职要求核心要求● 掌握机器人正逆运动学、动力学建模与算法推导，熟悉常用运动学求解器。● 熟练掌握运动规划（RRT/PRM）、轨迹优化与最优控制（MPC/iLQR/DDP 等）的原理与应用。● 熟练使用 C++/Python，熟悉 ROS/ROS2、MoveIt、Pinocchio、OCS2、Drake 等机器人常用开发库。● 理解实时控制系统，有控制律设计经验。● 有扎实的数学基础，精通线性代数、优化理论、数值方法等。加分项● 深度参与过机器人真机项目（机械臂、移动机器人、双臂协作等），具备从仿真到实机调试的经验。● 在机器人、控制、自动化相关顶会/顶刊（ICRA、IROS、RSS 等）发表论文者优先。● 熟悉强化学习及深度学习优先。● 具备扎实的工程实现能力，对系统架构、性能调优、代码质量有高要求。● 具备优秀的技术文档能力、沟通协作能力和系统化思维"""


def main() -> None:
    jd = (os.environ.get("LABEL_PATH_JD") or "").strip() or DEFAULT_JOB_TEXT
    domain = (os.environ.get("LABEL_PATH_DOMAIN") or "").strip() or DEFAULT_DOMAIN_CHOICE
    verbose_env = os.environ.get("LABEL_PATH_VERBOSE", "").strip().lower()
    if verbose_env in ("0", "n", "no", "false"):
        verbose = False
    elif verbose_env in ("1", "y", "yes", "true"):
        verbose = True
    else:
        verbose = DEFAULT_VERBOSE

    try:
        l_path = LabelRecallPath(recall_limit=200, verbose=verbose, silent=False)
        run_single_label_debug(l_path, domain, verbose, jd)
    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

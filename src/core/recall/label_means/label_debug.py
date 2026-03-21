# -*- coding: utf-8 -*-
"""
标签路分区 debug 打印：按 Step2 / Stage2A / Stage2B / Stage3 分层，便于定位故障。
"""
# 0: 不打印  1: 只打印汇总  2: 打印汇总+top 明细  3: 打印汇总+top+rejected/borderline/risky
DEBUG_LABEL_PIPELINE = True
DEBUG_LABEL_PIPELINE_LEVEL = 2


def debug_print(level: int, msg: str, label_or_recall=None) -> None:
    """level 1=汇总 2=+明细 3=+rejected/borderline/risky。verbose=True 时视为 level>=1 均打印。"""
    verbose = label_or_recall is not None and getattr(label_or_recall, "verbose", False)
    if verbose and level >= 1:
        print(msg)
        return
    # 已传入 LabelRecallPath 且 verbose=False：不走全局 DEBUG_LABEL_PIPELINE，避免 CLI 选「非详细」仍刷屏
    if label_or_recall is not None and not getattr(label_or_recall, "verbose", False):
        return
    if not DEBUG_LABEL_PIPELINE or DEBUG_LABEL_PIPELINE_LEVEL < level:
        return
    print(msg)

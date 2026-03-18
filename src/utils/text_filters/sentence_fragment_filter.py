# -------------------------------------------------
# 句子残片过滤：拦截泛用叙述句式，避免进入技能/锚点
# -------------------------------------------------

import re
from typing import List, Optional


# 1. 进行……（以「进行」开头，后面还有内容）
FRAGMENT_PREFIX_JINXING = "进行"

# 2. 从……到……（中间任意内容）
FRAGMENT_PATTERN_CONG_DAO = re.compile(r"从.+到")

# 3. 对……进行……（中间任意内容）
FRAGMENT_PATTERN_DUI_JINXING = re.compile(r"对.+进行")

# 4. 推动……（以「推动」开头，后面还有内容）
FRAGMENT_PREFIX_TUIDONG = "推动"

# 5. 实现与落地（含该固定短语即拦）
FRAGMENT_PHRASE_SHIXIAN_LUODI = "实现与落地"


def is_sentence_fragment(text: str) -> bool:
    """
    判断文本是否为需要拦截的句子残片。

    拦截规则：
    - 进行……
    - 从……到……
    - 对……进行……
    - 推动……
    - 实现与落地
    """
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False

    if s.startswith(FRAGMENT_PREFIX_JINXING) and len(s) > len(FRAGMENT_PREFIX_JINXING):
        return True
    if FRAGMENT_PATTERN_CONG_DAO.search(s):
        return True
    if FRAGMENT_PATTERN_DUI_JINXING.search(s):
        return True
    if s.startswith(FRAGMENT_PREFIX_TUIDONG) and len(s) > len(FRAGMENT_PREFIX_TUIDONG):
        return True
    if FRAGMENT_PHRASE_SHIXIAN_LUODI in s:
        return True

    return False


def filter_sentence_fragments(
    items: List[str],
    *,
    remove: bool = True,
) -> List[str]:
    """
    从列表中过滤掉命中「句子残片」规则的项。
    remove=True 返回保留项，remove=False 返回被判定为残片的项（调试用）。
    """
    if remove:
        return [x for x in items if not is_sentence_fragment(x)]
    return [x for x in items if is_sentence_fragment(x)]

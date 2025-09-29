# -*- coding: utf-8 -*-
"""
scoring/extract.py — 从模型原文中抽取答案字符串
- 规则：
  1) 优先匹配唯一一次 <ANS> ... </ANS>（跨行）
  2) 若无 <ANS>，兜底：取“最后一次”【答案】... 的内容
  3) 轻量清洗：全角->半角、去首尾空白/标点、统一空格
- 输出：
  extract_ans(text) -> (ok: bool, value: str|None, why: str)
依赖：仅标准库
"""

from __future__ import annotations
import re
import unicodedata
from typing import Tuple, Optional


# ---------- 公共入口 ----------

def extract_ans(text: str) -> Tuple[bool, Optional[str], str]:
    """
    从模型输出文本中提取答案字符串
    """
    if not isinstance(text, str) or not text.strip():
        return False, None, "empty_text"

    # 1) <ANS>…</ANS>，要求唯一
    ok, val, why = _extract_from_tags(text)
    if ok:
        return True, _post_clean(val), "from_tags"
    if why == "multiple_tags":
        # 多个 <ANS> 判为失败（最小性在 policy 层再兜底也可以加严）
        return False, None, "multiple_tags"

    # 2) 回退：最后一行的【答案】…（允许中括/小括等变体）
    ok, val = _extract_from_last_answer_line(text)
    if ok:
        return True, _post_clean(val), "from_last_answer_line"

    return False, None, "no_match"


# ---------- 具体实现 ----------

_TAG_RE = re.compile(r"<ANS>\s*(.*?)\s*</ANS>", re.IGNORECASE | re.DOTALL)

def _extract_from_tags(text: str) -> Tuple[bool, Optional[str], str]:
    matches = _TAG_RE.findall(text)
    if not matches:
        return False, None, "no_tags"
    if len(matches) > 1:
        return False, None, "multiple_tags"
    return True, matches[0], "ok"


# 允许的“答案行”标签文本（最后一行优先）
_ANSWER_LINE_RE = re.compile(
    r"^[\s　]*[【\[\(（]?答\s*案[】\]\)）]?\s*[:：]?\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE
)

def _extract_from_last_answer_line(text: str) -> Tuple[bool, Optional[str]]:
    # 找到所有匹配，取最后一个
    all_matches = list(_ANSWER_LINE_RE.finditer(text))
    if not all_matches:
        return False, None
    last = all_matches[-1]
    return True, last.group(1)


# ---------- 清洗/规范化 ----------

def _to_halfwidth(s: str) -> str:
    """全角->半角"""
    return unicodedata.normalize("NFKC", s)

def _strip_outer_punct(s: str) -> str:
    """去掉首尾常见标点/引号/句号"""
    return s.strip().strip(".,;:，。；：、\"'“”‘’`")

_WS_RE = re.compile(r"\s+")

def _post_clean(s: str) -> str:
    """轻量清洗：全角->半角，去首尾标点，统一空格"""
    s = _to_halfwidth(s)
    s = _strip_outer_punct(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------- 简单自测 ----------
if __name__ == "__main__":
    samples = [
        "过程...\n<ANS> 24/7 </ANS>\n谢谢",
        "推导...\n【答案】 2",
        "【答案】3\n说明\n【答案】 4/5",  # 应取最后一个
        "无效\n<ANS>1</ANS>\n中间\n<ANS>2</ANS>",
        ""
    ]
    for i, t in enumerate(samples, 1):
        ok, v, why = extract_ans(t)
        print(f"#{i} ok={ok} why={why} value={v!r}")

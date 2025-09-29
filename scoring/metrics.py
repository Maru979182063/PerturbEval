# -*- coding: utf-8 -*-
"""
scoring/metrics.py — 指标聚合
- 统一由 runner 调用，给定每题各环判定结果，产出整体指标/明细
- 只用标准库
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# ========== 输入数据结构 ==========
# runner 组织好后传进来；保持极简以方便落地。

@dataclass
class PerQidResult:
    qid: str
    rings_order: List[str]                 # e.g. ["R1","R2","R3","R4"]（按实际运行顺序）
    ring_ok: Dict[str, Optional[bool]]     # e.g. {"R1": True, "R2": False, "R3": None, "R4": None}
    final_ring: Optional[str]              # 命中的环；若全错/未命中则 None
    final_ok: bool                         # 该题最终是否判对（与 final_ring 一致；全错时 False）


# ========== 核心计算 ==========

def compute_global_metrics(results: List[PerQidResult]) -> Dict:
    """对所有题做聚合，返回一个 dict（可直接写入 JSON/MD）"""
    total = len(results)
    acc_n = sum(1 for r in results if r.final_ok)
    acc = acc_n / total if total else 0.0

    # 各环命中分布
    ring_hits: Dict[str, int] = {}
    for r in results:
        if r.final_ring:
            ring_hits[r.final_ring] = ring_hits.get(r.final_ring, 0) + 1

    # 救活率
    rescue_r1_r2 = _rescue_rate(results, src="R1", dst="R2")
    rescue_r1_r3 = _rescue_rate(results, src="R1", dst="R3", require_prev_false=True)

    # 形成一个简洁的“概览”
    overview = {
        "total": total,
        "acc_n": acc_n,
        "acc": round(acc, 6),
        "ring_hits": ring_hits,
        "rescue": {
            "R1_to_R2": rescue_r1_r2,
            "R1_to_R3": rescue_r1_r3,
        },
    }
    return {
        "overview": overview,
        "details": [asdict(r) for r in results],
    }


def _rescue_rate(results: List[PerQidResult], src: str, dst: str, require_prev_false: bool = False) -> Dict:
    """
    计算“从 src 到 dst 的救活率”：
      - 分母：src 错（或未命中）且后续环中包含 dst 的题数
      - 分子：上述题里 dst 对
    参数：
      - require_prev_false=True 时，要求所有中间环（(src, dst) 之间）也必须错/未命中，
        例如 R1→R3 救活率时，R2 必须是错/未命中
    """
    denom = 0
    numer = 0
    for r in results:
        if src not in r.rings_order or dst not in r.rings_order:
            continue
        # 题目是否“从 src 继续尝试了 dst”
        idx_s = r.rings_order.index(src)
        idx_d = r.rings_order.index(dst)
        if idx_d <= idx_s:
            continue

        ok_src = r.ring_ok.get(src, None) is True
        if ok_src:
            continue  # src 已经对了，就不算“救活”

        # 分母候选：src 错/未命中，且尝试到了 dst
        # （是否“尝试到了 dst”由 runner 决定：如果 dst 没跑，将 ring_ok[dst] 设为 None 即视作尝试但未命中）
        denom += 1

        if require_prev_false:
            # 需要保证 src 和 dst 之间的环都不是 True
            mid_ok = False
            for k in r.rings_order[idx_s + 1: idx_d]:
                if r.ring_ok.get(k, None) is True:
                    mid_ok = True
                    break
            if mid_ok:
                # 中间已被命中，不属于“直接从 src 跨到 dst 的救活”
                continue

        if r.ring_ok.get(dst, None) is True:
            numer += 1

    rate = (numer / denom) if denom else 0.0
    return {"numer": numer, "denom": denom, "rate": round(rate, 6)}


# ========== Markdown 报告（可选） ==========

def to_markdown_report(metrics: Dict, exp_name: str) -> str:
    """
    把 compute_global_metrics 的结果转成简洁 Markdown，方便写入 results/reports/summary.md
    """
    ov = metrics["overview"]
    lines = []
    lines.append(f"# 实验报告 · {exp_name}")
    lines.append("")
    lines.append("## 概览")
    lines.append(f"- 题目总数：**{ov['total']}**")
    lines.append(f"- 最终正确：**{ov['acc_n']}**  （ACC = **{ov['acc']:.2%}**）")
    if ov["ring_hits"]:
        sorted_hits = sorted(ov["ring_hits"].items(), key=lambda x: x[0])
        hits_str = "，".join([f"{k}:{v}" for k, v in sorted_hits])
        lines.append(f"- 各环命中：{hits_str}")
    lines.append("")
    lines.append("## 救活率")
    r12 = ov["rescue"]["R1_to_R2"]
    r13 = ov["rescue"]["R1_to_R3"]
    lines.append(f"- R1→R2：{r12['numer']}/{r12['denom']}  （**{r12['rate']:.2%}**）")
    lines.append(f"- R1→R3：{r13['numer']}/{r13['denom']}  （**{r13['rate']:.2%}**）")
    lines.append("")
    return "\n".join(lines)

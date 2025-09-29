#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从评测 CSV 汇总成 metrics.json，兼容你截图里的列名：
qid, tag, final_ok, final_ring, R1_ans, R1_why, R2_ans, R2_why, R3_ans, R3_why, ...

统计口径：
- 某规则 Rk 的正确：对应列 Rk_why == 'ok'（大小写不敏感，前后空白忽略）
- 缺失/空值不计入分母
- 输出 by_rule 里至少包含 S0_acc（基线本次跑的正确率）
- 如果你未来在 CSV 里加入 stage 列（S0/S1/S2），脚本会自动按 stage 细分出 S1_acc、S2_acc

用法：
python scripts/csv2metrics.py --csv results/reports/run.csv --out results/reports/metrics.json
"""

import argparse, json, math
import pandas as pd

def norm_ok(x):
    if isinstance(x, str):
        return x.strip().lower() == "ok"
    return False

def acc(series):
    # 仅统计非空条目
    s = series.dropna()
    if len(s) == 0:
        return None
    return float((s.apply(norm_ok)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入 CSV（含 R1_why/R2_why/R3_why 列）")
    ap.add_argument("--out", default="results/reports/metrics.json", help="输出 metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    rules = ["R1","R2","R3"]
    why_cols = {r: f"{r}_why" for r in rules if f"{r}_why" in df.columns}

    # 是否存在 stage 列（S0/S1/S2）
    has_stage = "stage" in df.columns

    by_rule = {}

    if has_stage:
        for r, col in why_cols.items():
            sub = df[[col, "stage"]].copy()
            for s in ["S0","S1","S2"]:
                mask = sub["stage"].astype(str).str.upper().eq(s)
                val = acc(sub.loc[mask, col])
                if r not in by_rule: by_rule[r] = {}
                if val is not None:
                    by_rule[r][f"{s}_acc"] = val
    else:
        # 没有 stage：至少给出 S0_acc（把本次视作 S0）
        for r, col in why_cols.items():
            val = acc(df[col])  # 基于 *_why 是否为 ok
            if r not in by_rule: by_rule[r] = {}
            if val is not None:
                by_rule[r]["S0_acc"] = val

    # 派生量（若 S2/S0 齐全则计算 RobustSlope；否则仅保留已知）
    derived = {"RescueRate": {}, "PathPenalty": {}}
    robust = {}
    stab = {}
    base = {}

    for r, vals in by_rule.items():
        s0 = vals.get("S0_acc")
        s2 = vals.get("S2_acc")
        if s0 is not None: base[r] = s0
        if s0 is not None and s2 is not None and s0 > 1e-12:
            slope = max(0.0, min(1.0, 1.0 - (s2 / s0)))
            robust[r] = slope
            stab[r] = 1.0 - slope

    out = {
        "by_rule": by_rule,
        "derived": {
            "RobustSlope": robust,
            "Stability": stab,
            "BaselineAcc": base,
            **derived
        }
    }

    # 确保输出目录存在
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] metrics.json -> {args.out}")

if __name__ == "__main__":
    main()

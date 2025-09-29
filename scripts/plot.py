#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 metrics.json 生成两张图：
- 有 S2_acc：画 RobustSlope + 四象限
- 无 S2_acc：画 S0_acc 柱状图 + S0_acc 散点（退化模式）

用法：
python scripts/plot.py --metrics results/reports/metrics.json --outdir assets/ --title "PerturbEval"
"""

import argparse, json, os
import matplotlib.pyplot as plt

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_bar(xlabs, ys, ylabel, title, outpath):
    plt.figure(figsize=(6,4))
    xs = range(len(xlabs))
    plt.bar(xs, ys)
    plt.xticks(list(xs), xlabs)
    plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.title(title)
    for i, v in enumerate(ys):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_scatter(xs, ys, labels, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(6,4))
    plt.scatter(xs, ys)
    for k, x, y in zip(labels, xs, ys):
        plt.text(x, y, f" {k}", va="center", ha="left")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/reports/metrics.json")
    ap.add_argument("--outdir", default="assets/")
    ap.add_argument("--title", default="PerturbEval")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    m = load_metrics(args.metrics)
    by_rule = m.get("by_rule", {})
    derived = m.get("derived", {})

    # 规则顺序优先 R1/R2/R3
    rules = [r for r in ["R1","R2","R3"] if r in by_rule] or list(by_rule.keys())

    # 判断是否有 S2
    has_s2 = any("S2_acc" in by_rule[r] for r in rules)

    if has_s2 and "RobustSlope" in derived and derived["RobustSlope"]:
        # 图1：RobustSlope
        slope = [derived["RobustSlope"].get(r, 0.0) for r in rules]
        plot_bar(rules, slope, "RobustSlope (↓ 稳定性下降率越小越好)", args.title,
                 os.path.join(args.outdir, "perf_by_prompt.png"))
        # 图2：四象限（S0 vs Stability）
        base = [derived.get("BaselineAcc", {}).get(r, by_rule[r].get("S0_acc", 0.0)) for r in rules]
        stab = [derived.get("Stability", {}).get(r, 1.0 - s) for r, s in zip(rules, slope)]
        plot_scatter(base, stab, rules, "Baseline Accuracy @ S0 (→)", "Stability (↑)", args.title,
                     os.path.join(args.outdir, "robustness_quadrant.png"))
    else:
        # 退化模式：仅有 S0_acc
        s0 = [by_rule[r].get("S0_acc", 0.0) for r in rules]
        plot_bar(rules, s0, "Accuracy @ S0", args.title,
                 os.path.join(args.outdir, "perf_by_prompt.png"))
        # 用同一数据画散点（横纵同为 S0_acc），仅做占位
        plot_scatter(s0, s0, rules, "Accuracy @ S0 (→)", "Accuracy @ S0 (↑)", args.title,
                     os.path.join(args.outdir, "robustness_quadrant.png"))

    print("[OK] saved:", os.path.join(args.outdir, "perf_by_prompt.png"))
    print("[OK] saved:", os.path.join(args.outdir, "robustness_quadrant.png"))

if __name__ == "__main__":
    main()

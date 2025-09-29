#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CSV / XLS / XLSX 汇总成 metrics.json
兼容列名（大小写/空格不敏感，自动规范化）：
  qid, tag, final_ok, final_ring,
  R1_ans, R1_why, R2_ans, R2_why, R3_ans, R3_why,
  (可选) stage  ∈ {S0,S1,S2}

用法：
  python scripts/tabular2metrics.py --in results/reports/run.xlsx --sheet 0 --out results/reports/metrics.json
  python scripts/tabular2metrics.py --in results/reports/run.csv --out results/reports/metrics.json
"""

import argparse
import os
import json
from typing import Optional, Dict

import pandas as pd


def _normalize_colname(c: str) -> str:
    """列名标准化：去空格、全小写。"""
    return "".join(str(c).strip().split()).lower()


def read_table(path: str, sheet: Optional[str]):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif ext in (".xlsx", ".xls"):
        # 若未指定 sheet，默认读取第一个工作表（sheet_name=0）
        sheet_name = 0 if sheet in (None, "",) else sheet
        # 支持 --sheet 传入 "0"/"1" 这样的数字字符串
        if isinstance(sheet_name, str) and sheet_name.isdigit():
            sheet_name = int(sheet_name)
        df = pd.read_excel(path, sheet_name=sheet_name)
        # 如果未指定 sheet 且 read_excel 返回 dict（某些版本行为），取第一个
        if isinstance(df, dict):
            df = next(iter(df.values()))
    else:
        raise ValueError(f"不支持的表格格式：{ext}")
    return df


def norm_ok(x) -> bool:
    """把各种“正确”标记规整为布尔 True。"""
    if isinstance(x, str):
        v = x.strip().lower()
        return v in {"ok", "true", "yes", "pass", "正确", "是", "对"}
    if isinstance(x, bool):
        return x
    return False


def acc(series: pd.Series) -> Optional[float]:
    """按 *_why 是否为 ok 计算准确率；空值不计入分母。"""
    s = series.dropna()
    if len(s) == 0:
        return None
    return float((s.apply(norm_ok)).mean())


def build_metrics(df: pd.DataFrame) -> Dict:
    # 列名标准化映射
    orig_cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in orig_cols}
    df = df.rename(columns={v: k for k, v in norm_map.items()})  # 用规范名作为列名

    # 我们关心的列（可能不存在）
    # 规则 why 列的规范名：r1_why / r2_why / r3_why
    rules = ["r1", "r2", "r3"]
    why_cols = {r: f"{r}_why" for r in rules if f"{r}_why" in df.columns}

    # 可选 stage 列（规范名：stage）
    has_stage = "stage" in df.columns

    by_rule: Dict[str, Dict[str, float]] = {}

    if has_stage:
        stage_series = df["stage"].astype(str).str.upper()
        for r, col in why_cols.items():
            for s in ["S0", "S1", "S2"]:
                mask = stage_series.eq(s)
                val = acc(df.loc[mask, col])
                R = r.upper()
                if R not in by_rule:
                    by_rule[R] = {}
                if val is not None:
                    by_rule[R][f"{s}_acc"] = val
    else:
        # 没有 stage：把这次视作 S0
        for r, col in why_cols.items():
            val = acc(df[col])
            R = r.upper()
            if R not in by_rule:
                by_rule[R] = {}
            if val is not None:
                by_rule[R]["S0_acc"] = val

    # 派生量（若 S0/S2 齐全就计算 RobustSlope）
    derived = {"RobustSlope": {}, "Stability": {}, "BaselineAcc": {}}
    for R, vals in by_rule.items():
        s0 = vals.get("S0_acc")
        s2 = vals.get("S2_acc")
        if s0 is not None:
            derived["BaselineAcc"][R] = s0
        if s0 is not None and s2 is not None and s0 > 1e-12:
            slope = max(0.0, min(1.0, 1.0 - (s2 / s0)))  # 1 - S2/S0
            derived["RobustSlope"][R] = slope
            derived["Stability"][R] = 1.0 - slope

    return {"by_rule": by_rule, "derived": derived}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入：CSV/XLS/XLSX 路径")
    ap.add_argument("--sheet", default=None, help="Excel 工作表（索引或名称；可省略=第一个）")
    ap.add_argument("--out", default="results/reports/metrics.json", help="输出 metrics.json")
    args = ap.parse_args()

    df = read_table(args.inp, args.sheet)
    m = build_metrics(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"[OK] metrics.json -> {args.out}")


if __name__ == "__main__":
    main()

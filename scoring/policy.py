# -*- coding: utf-8 -*-
"""
scoring/policy.py — 评测口径与判定（加强版）
功能:
  - 单位一致（含中文单位）+ 可选“金标缺单位时宽松”
  - 最小性（只拦截多个候选，不再按“数字个数”误杀）
  - 禁四舍五入/禁上浮（对纯数值）
  - 数值等价（整数/小数/分数）
  - 符号表达式等价（SymPy 优先；√/π/frac/绝对值；回退多点代入）
  - 赋值剥离：S=...、S(t)=...、t=4 等视作右侧表达式
  - 中文外衣剥离：第8天/8天/第10题 → 8（仅金标为纯数值时启用）
  - 容差 abs_tol / rel_tol
  - 字母型答案（A/B/C/D）
依赖：标准库 + 可选 sympy（已安装更佳）
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, getcontext, InvalidOperation
from fractions import Fraction
import math, re, unicodedata, ast
from typing import Optional, Tuple, Dict, List

# ==== 可选：SymPy ====
_SYMPY_OK = True
try:
    import sympy as sp
    from sympy import sqrt, Abs
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
except Exception:
    _SYMPY_OK = False

# ========= 配置 =========
@dataclass
class JudgeConfig:
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    allow_fraction: bool = True
    require_unit_match: bool = True
    forbid_rounding: bool = True
    forbid_up_round: bool = True
    minimal_answer: bool = True
    # 金标没有单位时，预测多写单位不判错（例如 "8.3分" vs 金标 "8.3"）
    relax_unit_when_gold_missing: bool = True

# ========= 预处理辅助（新增） =========
_ANS_RE = re.compile(r"<\s*ANS\s*>(.*?)<\s*/\s*ANS\s*>", re.I | re.S)
_SUP_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

def _strip_ans_tags(s: str) -> str:
    """提取 <ANS>...</ANS> 内的内容；无标签则原样返回"""
    if not s:
        return s
    m = _ANS_RE.search(s)
    return m.group(1).strip() if m else s.strip()

def _normalize_scientific(s: str) -> str:
    """
    统一“科学计数法/乘号/上标数字”写法，保证后续可解析：
      6.0 × 10^4 / 3.2×10⁻⁵ / 1.2*10^3 / 1.2×10⁴  →  6.0*10**4
    """
    if not s:
        return s
    t = _to_halfwidth(s).translate(_SUP_DIGITS)
    t = t.replace("·", "*").replace("×", "*").replace("✖", "*").replace("＊", "*")
    # 规范 ^ 为 ** ： *10^k -> *10**k
    t = re.sub(r"(\*?\s*10)\s*\^\s*([-+]?\d+)", r"\1**\2", t)
    return t.strip()

# ========= 公共入口 =========
def judge(pred_text: str, gold_answer: str, gold_unit: Optional[str],
          cfg: JudgeConfig,
          alt_answers: Optional[List[str]] = None) -> Tuple[bool, str, Dict]:
    """
    主判定入口
    返回: (is_correct, reason, details)
    reason: ok / unit_mismatch / not_minimal / not_numeric_equal / not_alpha_equal /
            looks_rounded / looks_up_round / invalid_format /
            symbolic_ok / symbolic_not_equal / symbolic_parse_error
    """
    alt_answers = alt_answers or []

    # (-1) 先做“剥标签 + 科学计数法归一化”（新增）
    pred_text = _normalize_scientific(_strip_ans_tags(pred_text))
    gold_answer = _normalize_scientific(_strip_ans_tags(gold_answer))

    # 0) 赋值剥离（S=... / S(t)=... / t=4）
    pred_text = _strip_assignment(pred_text)
    gold_answer = _strip_assignment(gold_answer)

    # 1) 最小性（放宽：仅拦截明显多个候选）
    ok_min, reason_min = _check_minimality(pred_text) if cfg.minimal_answer else (True, "skip_minimal")
    if not ok_min:
        return False, reason_min, {"how": "minimality"}

    # 1.5) “中文外衣” → 仅当金标是纯数值时，剥离“第8天/8天/第10题”等
    if _looks_like_pure_numeric_expr(gold_answer):
        core = _strip_cn_wrappers_to_number(pred_text)
        if core is not None:
            pred_text = core

    # 2) 拆单位
    pv_raw, pu = _split_value_unit(pred_text)
    gv_raw, gu = _split_value_unit(gold_answer)
    if gold_unit is not None:
        gu = _norm_unit(gold_unit)

    # 3) 单位一致
    if cfg.require_unit_match:
        if gu is None and cfg.relax_unit_when_gold_missing:
            pass  # 金标没写单位 → 宽松，不因预测多写单位判错
        else:
            if not _unit_equal(pu, gu):
                return False, "unit_mismatch", {
                    "unit_pred": pu, "unit_gold": gu, "parsed_pred": pv_raw, "parsed_gold": gv_raw, "how": "unit"
                }

    # 3.5) 有变量/符号标记（而非单字母选项）→ 走符号等价
    if (_has_symbolic_markers(pv_raw) or _has_symbolic_markers(gv_raw)) and not _is_alpha_answer(gv_raw):
        ok_sym, why_sym = _symbolic_equal(pv_raw, gv_raw)
        if ok_sym is not None:
            return ok_sym, ("ok" if ok_sym else why_sym), {
                "parsed_pred": pv_raw, "parsed_gold": gv_raw,
                "unit_pred": pu, "unit_gold": gu, "how": ("symbolic" if ok_sym else why_sym)
            }

    # 4) 单字母答案
    if _is_alpha_answer(gv_raw):
        ok_alpha = _alpha_equal(pv_raw, gv_raw, alt_answers)
        return ok_alpha, "ok" if ok_alpha else "not_alpha_equal", {
            "parsed_pred": pv_raw, "parsed_gold": gv_raw, "unit_pred": pu, "unit_gold": gu, "how": "alpha"
        }

    # 5) 数值等价（含分数）
    ok_num, reason_num, det = _numeric_equal_with_policies(pv_raw, gv_raw, alt_answers, cfg)
    det.update({"unit_pred": pu, "unit_gold": gu})
    return ok_num, ("ok" if ok_num else reason_num), det

# ========= 字母/最小性/单位 =========
def _to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _has_letter(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", _to_halfwidth(s or "")))

def _is_alpha_answer(s: str) -> bool:
    s = _to_halfwidth(s or "").strip()
    return bool(re.fullmatch(r"[A-Za-z]", s))

def _alpha_equal(pred: str, gold: str, alts: List[str]) -> bool:
    p = _to_halfwidth(pred).strip().upper()
    g = _to_halfwidth(gold).strip().upper()
    if p == g:
        return True
    for a in alts:
        if _to_halfwidth(a).strip().upper() == p:
            return True
    return False

# 明显多个候选才拦截；允许分数/多项式/赋值等
def _check_minimality(text: str) -> Tuple[bool, str]:
    t = _to_halfwidth(text or "").strip()
    if re.search(r"(,|，|;|；|\bor\b|\band\b|或)", t, flags=re.I):
        if not re.search(r"\d,\d{3}(?:\D|$)", t):  # 允许 1,234
            return False, "not_minimal"
    return True, "ok"

# 赋值剥离：S=... / S(t)=...
_ASSIGN_RE = re.compile(r"^\s*[A-Za-z_]\w*(?:\s*\([^\)]*\))?\s*=\s*(.+)$")
def _strip_assignment(s: str) -> str:
    s0 = _to_halfwidth(s or "").strip()
    m = _ASSIGN_RE.match(s0)
    return m.group(1).strip() if m else s0

# 中文外衣：第8天/8天/第10题 → 8
_CN_WRAP_RE = re.compile(
    r"^\s*第?\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:天|日|号|周|星期|月|年|题|页|步|次|元|米|厘米|毫米|公里|分钟|分|小时|时|秒)?\s*$"
)
def _strip_cn_wrappers_to_number(s: str) -> Optional[str]:
    if not s:
        return None
    s0 = _to_halfwidth(s).strip()
    m = _CN_WRAP_RE.match(s0)
    if m:
        return m.group(1)
    return None

def _looks_like_pure_numeric_expr(s: str) -> bool:
    ss = _to_halfwidth(s or "").strip()
    return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?", ss)) or _FRACTION_RE.match(ss) is not None

def _split_value_unit(s: str) -> Tuple[str, Optional[str]]:
    s = _to_halfwidth(s or "").strip()
    s = re.sub(r"^\s*第\s*", "", s)  # 去“第”
    # 末尾单位（中/英/符号）
    m = re.search(r"([\u4e00-\u9fffA-Za-z%°℃]+)$", s)
    if m and not _FRACTION_RE.match(s):
        u = m.group(1)
        val = s[: -len(u)].strip()
        return val, _norm_unit(u)
    return s, None

def _norm_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    u = _to_halfwidth(u).strip()
    aliases = {
        "秒": "s", "分钟": "min", "分": "min", "小时": "h",
        "米": "m", "厘米": "cm", "毫米": "mm", "公里": "km",
        "度": "°", "°": "°", "百分号": "%", "摄氏度": "℃",
    }
    return aliases.get(u, u)

def _unit_equal(u1: Optional[str], u2: Optional[str]) -> bool:
    return (_norm_unit(u1) or None) == (_norm_unit(u2) or None)

# ========= 数值判等 =========
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_FRACTION_RE = re.compile(r"^[-+]?\d+/\d+$")

def _parse_number(s: str, allow_fraction: bool):
    s = _to_halfwidth(s or "").strip()
    if allow_fraction and _FRACTION_RE.match(s):
        num, den = s.split("/", 1)
        try:
            f = Fraction(int(num), int(den))
            return Decimal(f.numerator) / Decimal(f.denominator), f, "frac"
        except Exception:
            return None, None, "invalid"
    getcontext().prec = 50
    try:
        d = Decimal(s)
        return d, None, "dec"
    except (InvalidOperation, ValueError):
        return None, None, "invalid"

def _numeric_equal_with_policies(pred: str, gold: str, alts: List[str], cfg: JudgeConfig) -> Tuple[bool, str, Dict]:
    for a in [gold] + list(alts):
        ok, reason, det = _numeric_equal_single(pred, a, cfg)
        if ok:
            det["match_to"] = a
            return True, "ok", det
    ok, reason, det = _numeric_equal_single(pred, gold, cfg)
    return False, reason, det

def _numeric_equal_single(pred: str, gold: str, cfg: JudgeConfig) -> Tuple[bool, str, Dict]:
    pd, pf, pk = _parse_number(pred, cfg.allow_fraction)
    gd, gf, gk = _parse_number(gold, cfg.allow_fraction)
    if pd is None or gd is None:
        p = _to_halfwidth(pred or "").strip()
        g = _to_halfwidth(gold or "").strip()
        return (p == g), ("ok" if p == g else "invalid_format"), {"pred": pred, "gold": gold, "how": "string"}
    diff = abs(pd - gd)
    abs_ok = diff <= Decimal(str(cfg.abs_tol))
    rel_ok = (gd == 0 and diff == 0) or (gd != 0 and (diff / abs(gd)) <= Decimal(str(cfg.rel_tol)))
    equal_by_tol = abs_ok or rel_ok
    if cfg.forbid_rounding:
        if pd != gd:
            if _looks_like_rounding(pred, gd):
                return False, "looks_rounded", {"pred": str(pd), "gold": str(gd), "how": "rounding_like"}
            if equal_by_tol:
                return False, "not_numeric_equal", {"pred": str(pd), "gold": str(gd), "how": "tol_forbidden"}
    if cfg.forbid_up_round and pd > gd:
        if _looks_like_up_round(pred, gd):
            return False, "looks_up_round", {"pred": str(pd), "gold": str(gd), "how": "up_round_like"}
    if pd == gd or (equal_by_tol and not cfg.forbid_rounding):
        return True, "ok", {"pred": str(pd), "gold": str(gd), "how": "numeric"}
    return False, "not_numeric_equal", {"pred": str(pd), "gold": str(gd), "how": "numeric"}

def _decimal_places(s: str) -> int:
    s = _to_halfwidth(s or "").strip()
    if _FRACTION_RE.match(s):
        return 99
    if "." in s:
        return len(s.split(".", 1)[1].rstrip("0"))
    return 0

def _looks_like_rounding(pred_str: str, gold_dec: Decimal) -> bool:
    try:
        getcontext().prec = 50
        pd = Decimal(pred_str)
    except Exception:
        return False
    d = _decimal_places(pred_str)
    if d >= 99:
        return False
    step = Decimal(1) / (Decimal(10) ** d)
    return abs(pd - gold_dec) < (step / 2)

def _looks_like_up_round(pred_str: str, gold_dec: Decimal) -> bool:
    try:
        getcontext().prec = 50
        pd = Decimal(pred_str)
    except Exception:
        return False
    d = _decimal_places(pred_str)
    if d >= 99:
        return False
    step = Decimal(1) / (Decimal(10) ** d)
    return (pd > gold_dec) and ((pd - gold_dec) <= step)

# ========= 符号表达式等价 =========
def _normalize_expr_latexish(s: str) -> str:
    """
    把 LaTeX/紧挨写法 归一为更易解析的表达式：
      - \frac{a}{b} -> (a/b)
      - \sqrt{a} / √a -> sqrt(a)
      - \pi -> pi
      - |expr| -> Abs(expr)
      - t2 -> t**2
      - )t / 2t / (a/b)t -> 插入 *
    """
    s = _to_halfwidth(s or "")
    # 空白 / 各种乘号
    s = s.replace(" ", "").replace("·", "*").replace("×", "*").replace("✖", "*").replace("＊", "*")
    # LaTeX
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace(r"\pi", "pi")
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    # 绝对值
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    # √x -> sqrt(x)
    s = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)
    # 紧挨乘
    s = re.sub(r"\)(?=[A-Za-z_])", r")*", s)
    s = re.sub(r"(\d)(?=[A-Za-z_])", r"\1*", s)
    # 幂：t2 -> t**2
    s = re.sub(r"([A-Za-z_])(\d+)", r"\1**\2", s)
    # +- / --
    s = s.replace("+-", "-").replace("--", "+")
    return s

def _has_symbolic_markers(s: str) -> bool:
    s0 = _to_halfwidth(s or "")
    return bool(re.search(r"(\\frac|\\sqrt|√|\\pi|\^|\||Abs\(|[A-Za-z_])", s0))

def _sympy_equal(pred: str, gold: str) -> Tuple[Optional[bool], str]:
    if not _SYMPY_OK:
        return None, "symbolic_parse_error:sympy_unavailable"
    try:
        p = _normalize_expr_latexish(pred)
        g = _normalize_expr_latexish(gold)
        # 自动变量收集（含下划线）
        vars_set = sorted(set(re.findall(r"[A-Za-z_]\w*", p + g)))
        symbols = {name: sp.symbols(name) for name in vars_set} if vars_set else {}
        pe = parse_expr(p, transformations=_TRANSFORMS, local_dict=symbols, evaluate=True)
        ge = parse_expr(g, transformations=_TRANSFORMS, local_dict=symbols, evaluate=True)
        eq = sp.simplify(pe - ge) == 0
        return (True, "symbolic_ok") if eq else (False, "symbolic_not_equal")
    except Exception as e:
        return None, f"symbolic_parse_error:{type(e).__name__}"

# 回退：多点代入（Fraction 精确）
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)

def _safe_eval_expr(expr: str, var_name: str, var_value: Fraction) -> Fraction:
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        if isinstance(n, ast.Num): return Fraction(n.n)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, int): return Fraction(n.value)
            raise ValueError("const-not-int")
        if isinstance(n, ast.Name):
            if n.id == var_name: return var_value
            raise ValueError("unknown-var")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, _ALLOWED_UNARY):
            v = _eval(n.operand); return +v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(n.op, _ALLOWED_BINOPS):
            l = _eval(n.left); r = _eval(n.right)
            if isinstance(n.op, ast.Add):  return l + r
            if isinstance(n.op, ast.Sub):  return l - r
            if isinstance(n.op, ast.Mult): return l * r
            if isinstance(n.op, ast.Div):  return l / r
            if isinstance(n.op, ast.Pow):
                if isinstance(n.right, ast.Constant) and isinstance(n.right.value, int) and n.right.value >= 0:
                    return l ** n.right.value
                raise ValueError("pow-int-only")
        raise ValueError("bad-node")
    return _eval(node)

def _normalize_expr_basic(s: str, var: str = "t") -> str:
    return _normalize_expr_latexish(s)

def _polynomial_equal_basic(pred: str, gold: str, var: str = "t") -> Tuple[bool, str]:
    try:
        p = _normalize_expr_basic(pred, var=var)
        g = _normalize_expr_basic(gold, var=var)
        test_points = [-3, -2, -1, 0, 1, 2, 3, 5]
        for x in test_points:
            vx = Fraction(x, 1)
            pv = _safe_eval_expr(p, var, vx)
            gv = _safe_eval_expr(g, var, vx)
            if pv != gv:
                return False, f"poly_not_equal_at_{var}={x}"
        return True, "symbolic_ok"
    except Exception as e:
        return False, f"symbolic_parse_error:{type(e).__name__}"

def _symbolic_equal(pred: str, gold: str) -> Tuple[Optional[bool], str]:
    ok, why = _sympy_equal(pred, gold)
    if ok is not None:
        return ok, why
    ok2, why2 = _polynomial_equal_basic(pred, gold, var="t")
    return ok2, why2

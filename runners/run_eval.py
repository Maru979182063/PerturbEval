# -*- coding: utf-8 -*-
"""
runners/run_eval.py — 入口调度（加 watchdog：单题硬截止）

用法：
  python -m runners.run_eval --config run.yaml
可选覆盖：
  --rings R1,R2,R3    --start_idx 0   --max_items 20

输出：
  results/
    ├─ preds/<exp_name>.csv
    ├─ logs/<exp_name>/*.json                # adapters/openai_client.py 已写
    ├─ logs/<exp_name>/raw_txt/<qid>_<r>.txt # 本文件可选落文本/复用
    └─ reports/{summary.json, summary.md}
"""

from __future__ import annotations
import os, sys, json, csv, argparse, time
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from collections import defaultdict, OrderedDict

import yaml
from tqdm import tqdm

# 本地模块
from adapters.openai_client import ClientConfig, OpenAIClient
from scoring.extract import extract_ans
from scoring.policy import JudgeConfig, judge
from scoring.metrics import PerQidResult, compute_global_metrics, to_markdown_report


# ----------------- 基础工具 -----------------

def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return

    fields = list(rows[0].keys())

    def _norm_cell(v):
        if v is None:
            return ""
        s = str(v)
        return s.replace("\r", " ").replace("\n", " ")

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: _norm_cell(r.get(k, "")) for k in fields})


# ----------------- 配置构造 -----------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def build_client(cfg: Dict[str, Any], exp_name: str) -> OpenAIClient:
    m = cfg.get("model", {}) or {}
    run = cfg.get("run", {}) or {}
    out_root = cfg.get("output_dir", "results")
    log_root = os.path.join(out_root, "logs", exp_name)

    cc = ClientConfig(
        model=m.get("name", "gpt-4o"),
        api_key_env=m.get("api_key_env", "OPENAI_API_KEY"),
        base_url=m.get("base_url", None),
        timeout_s=int(run.get("timeout_s", 60)),
        temperature=float(m.get("temperature", 0.0)),
        max_output_tokens=int(run.get("max_output_tokens", 1024)),
        max_retries=int(run.get("max_retries", 3)),
        retry_backoff=float(run.get("retry_backoff", 0.75)),
        extra_headers=m.get("extra_headers", None),
        log_root=log_root,
    )
    return OpenAIClient(cc)

def build_judge_cfg(cfg: Dict[str, Any]) -> JudgeConfig:
    j = cfg.get("judge", {}) or {}
    return JudgeConfig(
        abs_tol=float(j.get("abs_tol", 0.0)),
        rel_tol=float(j.get("rel_tol", 0.0)),
        allow_fraction=bool(j.get("allow_fraction", True)),
        require_unit_match=bool(j.get("require_unit_match", True)),
        forbid_rounding=bool(j.get("forbid_rounding", True)),
        forbid_up_round=bool(j.get("forbid_up_round", True)),
        minimal_answer=bool(j.get("minimal_answer", True)),
        relax_unit_when_gold_missing=bool(j.get("relax_unit_when_gold_missing", True)),
    )


# ----------------- 数据编组 -----------------

def group_tasks_by_qid(tasks: List[Dict[str, Any]], rings_order: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for t in tasks:
        qid = t.get("qid")
        ring = t.get("ring")
        if not qid or not ring:
            continue
        if ring not in rings_order:
            continue
        groups[qid][ring] = t

    out: Dict[str, List[Dict[str, Any]]] = {}
    for qid, ring_map in groups.items():
        seq = [ring_map[r] for r in rings_order if r in ring_map]
        if seq:
            out[qid] = seq
    return out


# ----------------- 单题执行 -----------------

def _pick_tag_from_seq(seq_items: List[Dict[str, Any]], gold: Dict[str, Any]) -> Optional[str]:
    """
    从任务或 gold 中提取 tag/label/category/tags；
    如果是数组类型，拼接成逗号分隔字符串。
    """
    keys = ("tag", "tags", "label", "category")
    for item in seq_items:
        for k in keys:
            v = item.get(k)
            if v:
                if isinstance(v, list):
                    return ",".join(map(str, v))
                return str(v)
    for k in keys:
        v = gold.get(k)
        if v:
            if isinstance(v, list):
                return ",".join(map(str, v))
            return str(v)
    return None

def _read_cached_raw(dump_raw_dir: Optional[str], qid: str, ring: str) -> Optional[str]:
    if not dump_raw_dir:
        return None
    fp = os.path.join(dump_raw_dir, f"{qid}_{ring}.txt")
    if os.path.exists(fp):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None

def _write_raw(dump_raw_dir: Optional[str], qid: str, ring: str, text: str) -> None:
    if not dump_raw_dir:
        return
    _ensure_dir(dump_raw_dir)
    with open(os.path.join(dump_raw_dir, f"{qid}_{ring}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

def run_one_qid(
    qid: str,
    seq: List[Dict[str, Any]],
    client: OpenAIClient,
    jcfg: JudgeConfig,
    gold: Dict[str, Any],
    dump_raw_dir: Optional[str],
    early_stop: bool,
    resume_from_logs: bool,
) -> Tuple[PerQidResult, Dict[str, Any]]:
    ring_ok: Dict[str, Optional[bool]] = {t["ring"]: None for t in seq}
    ring_pred_value: Dict[str, Optional[str]] = {t["ring"]: None for t in seq}
    ring_reason: Dict[str, Optional[str]] = {t["ring"]: None for t in seq}

    final_ring = None
    final_ok = False
    tag_value = _pick_tag_from_seq(seq, gold)

    for t in seq:
        ring = t["ring"]
        prompt = t.get("prompt", "")
        images = t.get("images", None)
        log_tag = f"{qid}_{ring}"

        # 1) 空 prompt 拦截：不请求 API，直接跳过
        if not prompt.strip():
            raw = "<ANS></ANS>"
            _write_raw(dump_raw_dir, qid, ring, raw)
        else:
            raw: Optional[str] = None
            if resume_from_logs:
                raw = _read_cached_raw(dump_raw_dir, qid, ring)
            if raw is None:
                raw = client.generate(prompt=prompt, images=images, log_tag=log_tag)
                _write_raw(dump_raw_dir, qid, ring, raw)

        # 2) 抽取答案
        ok_ex, value, why_ex = extract_ans(raw)
        if not ok_ex:
            ring_ok[ring] = False
            ring_reason[ring] = f"extract_fail:{why_ex}"
            continue
        ring_pred_value[ring] = value

        # 3) 判分
        is_ok, reason, _details = judge(
            pred_text=value or "",
            gold_answer=gold.get("answer", ""),
            gold_unit=gold.get("unit", None),
            cfg=jcfg,
            alt_answers=gold.get("alt_answers", []),
        )
        ring_ok[ring] = bool(is_ok)
        ring_reason[ring] = reason

        if is_ok and final_ring is None:
            final_ring = ring
            final_ok = True
            if early_stop:
                break

    if final_ring is None:
        final_ok = False

    rings_order = [t["ring"] for t in seq]
    per = PerQidResult(
        qid=qid,
        rings_order=rings_order,
        ring_ok=ring_ok,
        final_ring=final_ring,
        final_ok=final_ok,
    )

    detail = {
        "qid": qid,
        "tag": tag_value,
        "gold": gold,
        "rings_order": rings_order,
        "ring_pred_value": ring_pred_value,
        "ring_reason": ring_reason,
        "final_ring": final_ring,
        "final_ok": final_ok,
    }
    return per, detail


# ----------------- watchdog 包装 -----------------

def run_one_with_deadline(deadline_s: int, *args, **kwargs):
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as inner:
            fut = inner.submit(run_one_qid, *args, **kwargs)
            return fut.result(timeout=deadline_s)
    except FutureTimeout:
        qid = args[0] if args else kwargs.get("qid", "UNKNOWN_QID")
        seq = args[1] if len(args) > 1 else kwargs.get("seq", [])
        rings_order = [t.get("ring", "R") for t in (seq or [])]
        ring_reason = {r: "error:item_deadline" for r in rings_order}
        per = PerQidResult(
            qid=qid,
            rings_order=rings_order or ["R1"],
            ring_ok={r: None for r in (rings_order or ["R1"])},
            final_ring=None,
            final_ok=False,
        )
        detail = {
            "qid": qid,
            "tag": _pick_tag_from_seq(seq or [], kwargs.get("gold", {})) or "",
            "gold": kwargs.get("gold", {}),
            "rings_order": rings_order,
            "ring_pred_value": {},
            "ring_reason": ring_reason,
            "final_ring": None,
            "final_ok": False,
            "elapsed_s": round(time.time() - start, 3),
        }
        return per, detail


# ----------------- 主流程 -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="run.yaml")
    ap.add_argument("--rings", default=None)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    exp_name = cfg.get("exp_name") or time.strftime("exp_%Y%m%d_%H%M%S")
    out_root = cfg.get("output_dir", "results")
    preds_dir = os.path.join(out_root, "preds")
    reports_dir = os.path.join(out_root, "reports")
    dump_raw = bool(cfg.get("run", {}).get("dump_raw", True))
    dump_raw_dir = os.path.join(out_root, "logs", exp_name, "raw_txt") if dump_raw else None

    tasks_path = cfg.get("data", {}).get("tasks_path", "data/tasks.jsonl")
    gold_path = cfg.get("data", {}).get("gold_path", "data/gold.jsonl")
    _ensure_dir(out_root); _ensure_dir(preds_dir); _ensure_dir(reports_dir)

    rings = cfg.get("run", {}).get("rings", ["R1", "R2", "R3"])
    if args.rings:
        rings = [x.strip() for x in args.rings.split(",") if x.strip()]

    concurrency = int(cfg.get("run", {}).get("concurrency", 3))
    early_stop = bool(cfg.get("run", {}).get("early_stop", True))
    resume_from_logs = bool(cfg.get("run", {}).get("resume_from_logs", False))
    fail_soft = bool(cfg.get("run", {}).get("fail_soft", True))

    timeout_s = int(cfg.get("run", {}).get("timeout_s", 60))
    item_deadline_s = int(cfg.get("run", {}).get("item_deadline_s", max(90, timeout_s + 30)))

    tasks = _read_jsonl(tasks_path)
    gold_list = _read_jsonl(gold_path)
    gold_map: Dict[str, Dict[str, Any]] = {g["qid"]: g for g in gold_list if "qid" in g}

    grouped = group_tasks_by_qid(tasks, rings)
    qids = sorted(grouped.keys())

    start_idx = args.start_idx if args.start_idx is not None else int(cfg.get("run", {}).get("start_idx", 0))
    max_items = args.max_items if args.max_items is not None else (cfg.get("run", {}).get("max_items", None))
    if max_items is not None:
        max_items = int(max_items)
    qids = qids[start_idx: (start_idx + max_items) if max_items is not None else None]

    client = build_client(cfg, exp_name=exp_name)
    jcfg = build_judge_cfg(cfg)

    per_list: List[PerQidResult] = []
    details: List[Dict[str, Any]] = []

    preds_csv = os.path.join(preds_dir, f"{exp_name}.csv")
    summ_json = os.path.join(reports_dir, "summary.json")
    summ_md = os.path.join(reports_dir, "summary.md")

    def _flush_outputs() -> None:
        try:
            metrics = compute_global_metrics(per_list)
            by_tag: Dict[str, Dict[str, Any]] = {}
            for det in details:
                tag = det.get("tag") or "UNSPECIFIED"
                s = by_tag.setdefault(tag, {"total": 0, "acc_n": 0})
                s["total"] += 1
                s["acc_n"] += 1 if det.get("final_ok") else 0
            for tag, s in by_tag.items():
                s["acc"] = (s["acc_n"] / s["total"]) if s["total"] else 0.0

            rows = []
            desired_rings = rings
            for det in sorted(details, key=lambda x: x["qid"]):
                row = OrderedDict()
                row["qid"] = det["qid"]
                row["tag"] = det.get("tag") or ""
                row["final_ok"] = det["final_ok"]
                row["final_ring"] = det["final_ring"] or ""
                rp = det.get("ring_pred_value") or {}
                ry = det.get("ring_reason") or {}
                for r in desired_rings:
                    row[f"{r}_ans"] = rp.get(r, "") or ""
                    row[f"{r}_why"] = ry.get(r, "") or ""
                rows.append(row)

            _write_csv(preds_csv, rows)
            _write_json(summ_json, {
                "exp_name": exp_name,
                "rings": rings,
                "early_stop": early_stop,
                "resume_from_logs": resume_from_logs,
                "item_deadline_s": item_deadline_s,
                "metrics": metrics,
                "by_tag": by_tag,
                "details": details,
            })
            md = to_markdown_report(metrics, exp_name=exp_name)
            _ensure_dir(os.path.dirname(summ_md))
            with open(summ_md, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            print(f"[warn] flush outputs failed: {e}", file=sys.stderr)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(
                    run_one_with_deadline,
                    item_deadline_s,
                    qid,
                    grouped[qid],
                    client,
                    jcfg,
                    gold_map.get(qid, {"qid": qid, "answer": "", "unit": None, "alt_answers": []}),
                    dump_raw_dir,
                    early_stop,
                    resume_from_logs
                ): qid
                for qid in qids
            }

            for fut in tqdm(as_completed(futs), total=len(futs), desc="Evaluating", ncols=88):
                try:
                    per, det = fut.result()
                    per_list.append(per)
                    details.append(det)
                except Exception as e:
                    qid = futs[fut]
                    msg = f"run_one_qid_error:{type(e).__name__}:{e}"
                    if fail_soft:
                        rings_order = [t["ring"] for t in grouped[qid]]
                        per_list.append(PerQidResult(
                            qid=qid,
                            rings_order=rings_order,
                            ring_ok={r: None for r in rings_order},
                            final_ring=None,
                            final_ok=False,
                        ))
                        details.append({
                            "qid": qid,
                            "tag": _pick_tag_from_seq(grouped[qid], gold_map.get(qid, {})) or "",
                            "gold": gold_map.get(qid, {}),
                            "rings_order": rings_order,
                            "ring_pred_value": {},
                            "ring_reason": {r: ("error:" + msg) for r in rings_order},
                            "final_ring": None,
                            "final_ok": False,
                        })
                        print(f"[warn] {qid} failed but continue: {msg}", file=sys.stderr)
                    else:
                        raise
                finally:
                    _flush_outputs()
    finally:
        _flush_outputs()
        print("\n== Done ==")
        print(f"Predictions: {preds_csv}")
        print(f"Report JSON: {summ_json}")
        print(f"Report MD  : {summ_md}")


if __name__ == "__main__":
    main()

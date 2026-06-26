#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_synthetic_abclean_factorial.py

Analyzer for Synthetic A/B/clean factorial outputs.

It compares each Synthetic treatment trace against the existing Synthetic off
baseline under strict iso-time coverage. Lower exact_score is better.
Gap convention throughout:
  gap = base_score - treatment_score
  gap > 0 means the treatment is better than the base.

Main outputs:
  pricecal_treatment_manifest_synth.csv
  final_pairwise_iso60_detailed_synth.csv
  clean_gate_summary_synth.csv
  A_effect_w0_vs_clean_summary_synth.csv
  B_only_effect_vs_clean_summary_synth.csv
  B_dose_effect_vs_w0_summary_synth.csv
  check2_w0_vs_off_summary_synth.csv
  level0_pressure_regression_synth.csv
  synthetic_abclean_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

W_EXACT = {
    "exact_high_deficit": 1_500_000.0,
    "exact_volume_deficit": 200_000.0,
    "exact_volume_excess": 20_000.0,
    "exact_violation": 4_500.0,
    "exact_blocked": 1_000.0,
    "exact_ward_excess": 2_500.0,
    "exact_peak": 900.0,
    "exact_icu_excess": 250.0,
    "exact_overtime": 8.0,
}
COMPONENTS = list(W_EXACT.keys())
ARMS = ["off", "price_off_clean", "A_only", "B_only", "AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"]
DEFAULT_SIZES = [50, 70, 100, 150]
DEFAULT_SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
DEFAULT_SCENARIOS = ["nominal", "transfer_bottleneck"]


def as_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        elif isinstance(v, list):
            out[kk] = "|".join(map(str, v))
        else:
            out[kk] = v
    return out


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_metadata_near_trace(trace_path: Path) -> Dict[str, Any]:
    candidates = []
    p = trace_path.parent
    for _ in range(6):
        candidates += [
            p / "spiral_pr_glns_metadata.json",
            p / "abclean_run_manifest.json",
            p / "metadata.json",
            p / "run_metadata.json",
        ]
        p = p.parent
    merged: Dict[str, Any] = {}
    for c in candidates:
        if c.exists():
            merged.update(flatten(read_json(c)))
    return merged


def choose_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower.get(str(c).lower())
        if hit is not None:
            return hit
    for col in df.columns:
        cl = str(col).lower()
        for key in candidates:
            if str(key).lower() in cl:
                return col
    return None


def value_from_row(row: pd.Series, keys: Iterable[str]) -> float:
    lower = {str(c).lower(): c for c in row.index}
    for key in keys:
        c = lower.get(str(key).lower())
        if c is not None:
            v = as_float(row.get(c))
            if np.isfinite(v):
                return v
    for col in row.index:
        cl = str(col).lower()
        for key in keys:
            if str(key).lower() in cl:
                v = as_float(row.get(col))
                if np.isfinite(v):
                    return v
    return np.nan


def value_from_meta(meta: Dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        if key in meta:
            v = as_float(meta.get(key))
            if np.isfinite(v):
                return v
    for mk, mv in meta.items():
        ml = str(mk).lower()
        for key in keys:
            kl = str(key).lower()
            if ml.endswith(kl) or kl in ml:
                v = as_float(mv)
                if np.isfinite(v):
                    return v
    return np.nan


def value_row_or_meta(row: pd.Series, meta: Dict[str, Any], row_keys: Iterable[str], meta_keys: Optional[Iterable[str]] = None) -> float:
    v = value_from_row(row, row_keys)
    if np.isfinite(v):
        return v
    return value_from_meta(meta, meta_keys if meta_keys is not None else row_keys)


def infer_case_fields(path: Path) -> Dict[str, Any]:
    s = str(path).replace("\\", "/")
    out: Dict[str, Any] = {"dataset": "Synthetic", "n": np.nan, "seed": np.nan, "scenario": "", "arm": ""}
    m = re.search(r"case_(\d+)_seed(\d+)", s)
    if m:
        out["n"] = int(m.group(1))
        out["seed"] = int(m.group(2))
    m2 = re.search(r"/n(\d+)(?:/|$)", s)
    if m2:
        out["n"] = int(m2.group(1))
    parts = s.split("/")
    for sc in DEFAULT_SCENARIOS + ["ward_pressure", "transfer_bottleneck", "nominal"]:
        if sc in parts:
            out["scenario"] = sc
    for arm in sorted(ARMS, key=len, reverse=True):
        if arm in parts or f"/{arm}/" in s:
            out["arm"] = arm
            break
    if not out["arm"] and "spiral_off" in parts:
        out["arm"] = "off"
    return out


def select_iso_row(df: pd.DataFrame, iso_t: float) -> Tuple[Optional[pd.Series], str, float]:
    time_col = choose_col(df, ["elapsed_sec", "wallclock_elapsed_s", "elapsed_s", "time_s", "elapsed_seconds", "runtime_sec"])
    if time_col is None:
        return None, "missing_elapsed_column", np.nan
    tmp = df.copy()
    tmp["_elapsed_sec"] = pd.to_numeric(tmp[time_col], errors="coerce")
    tmp = tmp[np.isfinite(tmp["_elapsed_sec"])]
    if tmp.empty:
        return None, "no_finite_elapsed", np.nan
    max_elapsed = float(tmp["_elapsed_sec"].max())
    if max_elapsed + 1e-9 < iso_t:
        return None, "insufficient_horizon", max_elapsed
    before = tmp[tmp["_elapsed_sec"] <= iso_t + 1e-9]
    if before.empty:
        return None, "no_row_at_or_before_iso", max_elapsed
    row = before.sort_values("_elapsed_sec").iloc[-1]
    return row, "ok", max_elapsed


def exact_components(row: pd.Series, meta: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
    target_volume = value_row_or_meta(row, meta, ["target_volume", "best_target_volume"], ["target_volume", "abclean_manifest.target_volume"])
    high_target = value_row_or_meta(row, meta, ["high_target", "best_high_target"], ["high_target", "abclean_manifest.high_target"])
    n_sched = value_row_or_meta(row, meta, ["best_n_scheduled", "n_scheduled"], ["best_evaluator_summary.n_scheduled", "n_scheduled"])
    high_sched = value_row_or_meta(row, meta, ["best_high", "best_high_scheduled", "best_n_high_priority_scheduled", "n_high_priority_scheduled"], ["best_evaluator_summary.n_high_priority_scheduled", "n_high_priority_scheduled"])
    violation = value_row_or_meta(row, meta, ["best_violation", "violation_count"], ["best_evaluator_summary.violation_count", "violation_count"])
    overtime = value_row_or_meta(row, meta, ["best_overtime", "or_overtime_min"], ["best_evaluator_summary.or_overtime_min", "or_overtime_min"])
    blocked = value_row_or_meta(row, meta, ["best_stage3_blocked", "blocked_transfer_patient_days_stage3", "best_eval_blocked", "blocked_transfer_patient_days"], ["best_stage3_summary.blocked_transfer_patient_days_stage3", "blocked_transfer_patient_days_stage3", "best_evaluator_summary.blocked_transfer_patient_days"])
    ward_excess = value_row_or_meta(row, meta, ["best_stage3_ward_excess", "ward_excess_bed_days_stage3", "best_eval_ward_excess", "ward_excess_bed_days_blocking"], ["best_stage3_summary.ward_excess_bed_days_stage3", "ward_excess_bed_days_stage3", "best_evaluator_summary.ward_excess_bed_days_blocking"])
    peak = value_row_or_meta(row, meta, ["best_stage3_peak", "best_stage3_peak_blocked", "peak_icu_ready_blocked_stage3", "best_eval_peak_blocked", "peak_icu_ready_blocked"], ["best_stage3_summary.peak_icu_ready_blocked_stage3", "peak_icu_ready_blocked_stage3", "best_evaluator_summary.peak_icu_ready_blocked"])
    icu_excess = value_row_or_meta(row, meta, ["best_stage3_icu_excess", "icu_excess_bed_days_stage3", "best_eval_icu_excess", "icu_excess_bed_days_blocking"], ["best_stage3_summary.icu_excess_bed_days_stage3", "icu_excess_bed_days_stage3", "best_evaluator_summary.icu_excess_bed_days_blocking"])
    pressure_cost = value_row_or_meta(row, meta, ["best_pressure_cost", "pressure_cost"], ["best_evaluator_summary.pressure_cost", "pressure_cost"])

    comps = {
        "exact_high_deficit": max(0.0, high_target - high_sched) if np.isfinite(high_target) and np.isfinite(high_sched) else np.nan,
        "exact_volume_deficit": max(0.0, target_volume - n_sched) if np.isfinite(target_volume) and np.isfinite(n_sched) else np.nan,
        "exact_volume_excess": max(0.0, n_sched - target_volume) if np.isfinite(target_volume) and np.isfinite(n_sched) else np.nan,
        "exact_violation": violation,
        "exact_blocked": blocked,
        "exact_ward_excess": ward_excess,
        "exact_peak": peak,
        "exact_icu_excess": icu_excess,
        "exact_overtime": overtime,
        "pressure_cost": pressure_cost,
        "target_volume": target_volume,
        "high_target": high_target,
        "n_scheduled": n_sched,
        "n_high_priority_scheduled": high_sched,
    }
    missing = [k for k in COMPONENTS if not np.isfinite(as_float(comps.get(k)))]
    return comps, "|".join(missing)


def score_from_components(comps: Dict[str, float]) -> float:
    total = 0.0
    for c, w in W_EXACT.items():
        v = as_float(comps.get(c))
        if not np.isfinite(v):
            return np.nan
        total += w * v
    return float(total)


def load_trace_record(trace_path: Path, iso_t: float, arm_override: Optional[str] = None) -> Dict[str, Any]:
    info = infer_case_fields(trace_path)
    if arm_override:
        info["arm"] = arm_override
    rec: Dict[str, Any] = {**info, "trace_path": str(trace_path), "status": "", "error": ""}
    try:
        df = pd.read_csv(trace_path)
    except Exception as e:
        rec.update({"status": "read_error", "error": repr(e)})
        return rec
    if df.empty:
        rec.update({"status": "empty_trace"})
        return rec
    meta = read_metadata_near_trace(trace_path)
    row, st, max_elapsed = select_iso_row(df, iso_t)
    rec["max_elapsed_sec"] = max_elapsed
    rec["iso_t_sec"] = iso_t
    if row is None:
        rec.update({"status": st})
        return rec
    comps, missing = exact_components(row, meta)
    score = score_from_components(comps)
    rec.update({
        "status": "ok" if np.isfinite(score) else "missing_components",
        "missing_components": missing,
        "elapsed_sec_at_iso": as_float(row.get("_elapsed_sec")),
        "exact_nopressure_score": score,
    })
    for c in COMPONENTS:
        rec[c] = comps.get(c, np.nan)
        rec[f"weighted_{c}"] = as_float(comps.get(c)) * W_EXACT[c] if np.isfinite(as_float(comps.get(c))) else np.nan
    for k in ["pressure_cost", "target_volume", "high_target", "n_scheduled", "n_high_priority_scheduled"]:
        rec[k] = comps.get(k, np.nan)
    # Manifest/meta fields.
    for k in [
        "actual_pressure_weight", "pressure_weight_unit", "b_multiplier", "operator_on", "guidance_mode",
        "lns_seed", "cycles", "proposals_per_cycle", "exact_every", "exact_top_k", "max_or_overtime",
        "recourse_pressure_weight_fast", "abclean_manifest.actual_pressure_weight", "abclean_manifest.operator_on",
    ]:
        if k in meta:
            safe = k.replace(".", "_")
            rec[safe] = meta[k]
    return rec


def discover_treatment_traces(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("spiral_trace.csv"))


def discover_manifest(root: Path) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for p in sorted(root.rglob("abclean_run_manifest.json")):
        d = read_json(p)
        if d:
            d["manifest_path"] = str(p)
            rows.append(d)
    # Also include run log if present.
    for p in sorted(root.rglob("synthetic_abclean_run_log.csv")):
        try:
            df = pd.read_csv(p)
            df["run_log_path"] = str(p)
            rows += df.to_dict("records")
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=[c for c in ["dataset", "n", "scenario", "seed", "arm", "output_dir"] if c in pd.DataFrame(rows).columns], keep="last")


def find_off_trace(off_roots: List[Path], scenario: str, n: int, seed: int) -> Optional[Path]:
    cname = f"case_{int(n)}_seed{int(seed)}"
    direct_patterns = []
    for root in off_roots:
        direct_patterns += [
            root / "Synthetic" / scenario / cname / "spiral_off" / "spiral_trace.csv",
            root / scenario / cname / "spiral_off" / "spiral_trace.csv",
            root / "Synthetic" / f"n{int(n)}" / "off" / scenario / cname / "off" / "spiral_trace.csv",
            root / "Synthetic" / f"n{int(n)}" / "price_off_clean" / scenario / cname / "price_off_clean" / "spiral_trace.csv",
        ]
    for p in direct_patterns:
        if p.exists():
            return p
    candidates: List[Tuple[int, int, Path]] = []
    for root in off_roots:
        if not root.exists():
            continue
        for p in root.rglob("spiral_trace.csv"):
            s = str(p).replace("\\", "/").lower()
            if cname.lower() not in s or scenario.lower() not in s:
                continue
            score = 0
            if "/spiral_off/" in s or s.endswith("/spiral_off/spiral_trace.csv"):
                score += 10
            if "/off/" in s or "price_off" in s:
                score += 3
            if "synthetic" in s:
                score += 1
            candidates.append((score, len(str(p)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def summarize_gap(df: pd.DataFrame, gap_col: str, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for keys, g in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = {"grouping": "+".join(group_cols) if group_cols else "overall"}
        for c, v in zip(group_cols, keys):
            row[c] = v
        vals = pd.to_numeric(g[gap_col], errors="coerce").dropna()
        row.update({
            "num_cases": int(len(vals)),
            "wins": int((vals > 1e-9).sum()),
            "losses": int((vals < -1e-9).sum()),
            "ties": int((vals.abs() <= 1e-9).sum()),
            "win_rate": float((vals > 1e-9).mean()) if len(vals) else np.nan,
            "mean_gap": float(vals.mean()) if len(vals) else np.nan,
            "median_gap": float(vals.median()) if len(vals) else np.nan,
            "min_gap": float(vals.min()) if len(vals) else np.nan,
            "max_gap": float(vals.max()) if len(vals) else np.nan,
        })
        for comp in COMPONENTS:
            c = f"{gap_col}_component__{comp}"
            if c in g.columns:
                vv = pd.to_numeric(g[c], errors="coerce").dropna()
                row[f"mean_component_{comp}"] = float(vv.mean()) if len(vv) else np.nan
                row[f"median_component_{comp}"] = float(vv.median()) if len(vv) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_pairwise(treat: pd.DataFrame, off_index: Dict[Tuple[int, str, int], Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for _, r in treat.iterrows():
        if r.get("status") != "ok":
            continue
        key = (int(r["n"]), str(r["scenario"]), int(r["seed"]))
        off = off_index.get(key)
        row = {"dataset": "Synthetic", "n": key[0], "scenario": key[1], "seed": key[2], "arm": r["arm"]}
        row["treatment_trace_path"] = r["trace_path"]
        if off is None:
            row["status"] = "off_trace_not_found"
            rows.append(row)
            continue
        if off.get("status") != "ok":
            row["status"] = "bad_off_trace:" + str(off.get("status"))
            rows.append(row)
            continue
        row["status"] = "ok"
        row["off_trace_path"] = off["trace_path"]
        row["off_exact_nopressure_score"] = off["exact_nopressure_score"]
        row["treatment_exact_nopressure_score"] = r["exact_nopressure_score"]
        row["gap_exact_nopressure"] = off["exact_nopressure_score"] - r["exact_nopressure_score"]
        row["treatment_pressure_cost"] = r.get("pressure_cost", np.nan)
        row["actual_pressure_weight"] = r.get("actual_pressure_weight", r.get("abclean_manifest_actual_pressure_weight", np.nan))
        row["operator_on"] = r.get("operator_on", r.get("abclean_manifest_operator_on", np.nan))
        row["guidance_mode"] = r.get("guidance_mode", "")
        for comp in COMPONENTS:
            row[f"off_{comp}"] = off.get(comp, np.nan)
            row[f"treatment_{comp}"] = r.get(comp, np.nan)
            row[f"gap_raw__{comp}"] = off.get(comp, np.nan) - r.get(comp, np.nan)
            row[f"gap_exact_nopressure_component__{comp}"] = W_EXACT[comp] * (off.get(comp, np.nan) - r.get(comp, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_arms(records: pd.DataFrame, base_arm: str, treat_arm: str, out_gap: str) -> pd.DataFrame:
    key = ["dataset", "n", "scenario", "seed"]
    b = records[(records["arm"] == base_arm) & (records["status"] == "ok")].copy()
    t = records[(records["arm"] == treat_arm) & (records["status"] == "ok")].copy()
    if b.empty or t.empty:
        return pd.DataFrame()
    keep = key + ["exact_nopressure_score"] + COMPONENTS + ["pressure_cost", "trace_path"]
    b = b[[c for c in keep if c in b.columns]].rename(columns={"exact_nopressure_score": "base_score", "pressure_cost": "base_pressure_cost", "trace_path": "base_trace_path"})
    t = t[[c for c in keep if c in t.columns]].rename(columns={"exact_nopressure_score": "treatment_score", "pressure_cost": "treatment_pressure_cost", "trace_path": "treatment_trace_path"})
    for comp in COMPONENTS:
        if comp in b.columns:
            b = b.rename(columns={comp: f"base_{comp}"})
        if comp in t.columns:
            t = t.rename(columns={comp: f"treatment_{comp}"})
    m = b.merge(t, on=key, how="inner")
    m["base_arm"] = base_arm
    m["treatment_arm"] = treat_arm
    m[out_gap] = m["base_score"] - m["treatment_score"]
    for comp in COMPONENTS:
        cb, ct = f"base_{comp}", f"treatment_{comp}"
        if cb in m.columns and ct in m.columns:
            m[f"{out_gap}_component__{comp}"] = W_EXACT[comp] * (m[cb] - m[ct])
    return m


def fit_level0_slope(pairwise: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    # Use operator-on guided arms when possible; fall back to any treatment with finite pressure cost.
    df = pairwise[pairwise["status"].eq("ok")].copy()
    if df.empty:
        return pd.DataFrame([{"dataset": "Synthetic", "status": "insufficient", "n_points": 0}])
    preferred = df[df["arm"].astype(str).isin(["A_only", "AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"])].copy()
    use = preferred if len(preferred) >= 3 else df.copy()
    use["x_pressure_cost"] = pd.to_numeric(use.get("treatment_pressure_cost"), errors="coerce")
    # y is treatment score minus same-seed off score: larger means worse downstream exact objective relative to off.
    use["y_exact_delta_vs_off"] = pd.to_numeric(use["treatment_exact_nopressure_score"], errors="coerce") - pd.to_numeric(use["off_exact_nopressure_score"], errors="coerce")
    use = use[np.isfinite(use["x_pressure_cost"]) & np.isfinite(use["y_exact_delta_vs_off"])]
    if len(use) < 3 or float(use["x_pressure_cost"].var()) <= 1e-12:
        return pd.DataFrame([{
            "dataset": "Synthetic", "status": "insufficient", "n_points": int(len(use)),
            "reason": "need >=3 points with nonzero pressure-cost variance",
        }])
    x = use["x_pressure_cost"].to_numpy(dtype=float)
    y = use["y_exact_delta_vs_off"].to_numpy(dtype=float)
    xm, ym = x.mean(), y.mean()
    slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    intercept = float(ym - slope * xm)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return pd.DataFrame([{
        "dataset": "Synthetic", "status": "ok", "slope_syn": slope, "intercept_syn": intercept,
        "R2_syn": r2, "n_points": int(len(use)),
        "x": "treatment_pressure_cost", "y": "treatment_exact_nopressure_score - off_exact_nopressure_score",
        "note": "If R2_syn < threshold, treat B as weak/uninformative and prioritize A_only interpretation.",
    }])


def concat_summaries(detailed: pd.DataFrame, gap_col: str) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame()
    parts = [
        summarize_gap(detailed, gap_col, ["dataset", "n"]),
        summarize_gap(detailed, gap_col, ["dataset", "scenario", "n"]),
        summarize_gap(detailed, gap_col, ["n"]),
        summarize_gap(detailed, gap_col, []),
    ]
    return pd.concat(parts, ignore_index=True)


def write_report(outdir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Synthetic A/B/clean factorial report", "",
        "Gap convention: `gap = base_score - treatment_score`; positive means treatment is better.", "",
        "Evaluation weights are frozen to the GermanOR/Mannino exact-score weights.", "",
    ]
    for title, df in tables.items():
        lines.append(f"## {title}")
        lines.append("")
        if df is None or df.empty:
            lines.append("No rows.")
        else:
            show = df.head(30)
            try:
                lines.append(show.to_markdown(index=False))
            except Exception:
                lines.append(show.to_csv(index=False))
        lines.append("")
    (outdir / "synthetic_abclean_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-root", default="synthetic_abclean_factorial_results")
    ap.add_argument("--off-root", action="append", default=["spiral_price_off_all_results_v2"], help="Can be repeated. Existing Synthetic off baseline root.")
    ap.add_argument("--output-dir", default="synthetic_abclean_factorial_analysis")
    ap.add_argument("--iso-t", type=float, default=60.0)
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    args = ap.parse_args()

    ab_root = Path(args.ab_root)
    off_roots = [Path(x) for x in args.off_root]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = discover_manifest(ab_root)
    manifest.to_csv(outdir / "pricecal_treatment_manifest_synth.csv", index=False)

    treatment_records = []
    for p in discover_treatment_traces(ab_root):
        rec = load_trace_record(p, args.iso_t)
        if rec.get("arm") in {"", "off"}:
            # Existing/off baselines are handled via --off-root; do not mix them into treatments.
            continue
        treatment_records.append(rec)
    records = pd.DataFrame(treatment_records)
    if not records.empty:
        records.to_csv(outdir / "all_treatment_trace_records_iso60_synth.csv", index=False)

    # Build off index for the requested grid.
    off_records = []
    for n in args.sizes:
        for sc in args.scenarios:
            for seed in args.seeds:
                p = find_off_trace(off_roots, sc, n, seed)
                if p is None:
                    off_records.append({"dataset": "Synthetic", "n": n, "scenario": sc, "seed": seed, "arm": "off", "status": "off_trace_not_found"})
                else:
                    off_records.append(load_trace_record(p, args.iso_t, arm_override="off"))
    off_df = pd.DataFrame(off_records)
    off_df.to_csv(outdir / "off_trace_records_iso60_synth.csv", index=False)
    off_index = {
        (int(r["n"]), str(r["scenario"]), int(r["seed"])): dict(r)
        for _, r in off_df.iterrows()
        if pd.notna(r.get("n")) and pd.notna(r.get("seed"))
    }

    if records.empty:
        raise SystemExit(f"No treatment traces found under {ab_root}")
    pairwise = make_pairwise(records, off_index)
    pairwise.to_csv(outdir / "final_pairwise_iso60_detailed_synth.csv", index=False)

    level0 = fit_level0_slope(pairwise, records)
    level0.to_csv(outdir / "level0_pressure_regression_synth.csv", index=False)

    # Direct arm-vs-arm comparisons from trace records.
    clean = compare_arms(pd.concat([records, off_df], ignore_index=True), "off", "price_off_clean", "clean_gap_off_minus_clean")
    clean.to_csv(outdir / "clean_gate_detailed_synth.csv", index=False)
    clean_sum = concat_summaries(clean, "clean_gap_off_minus_clean")
    clean_sum.to_csv(outdir / "clean_gate_summary_synth.csv", index=False)

    A = compare_arms(records, "price_off_clean", "A_only", "A_gap_clean_minus_Aonly")
    A.to_csv(outdir / "A_effect_w0_vs_clean_detailed_synth.csv", index=False)
    A_sum = concat_summaries(A, "A_gap_clean_minus_Aonly")
    A_sum.to_csv(outdir / "A_effect_w0_vs_clean_summary_synth.csv", index=False)

    Bonly = compare_arms(records, "price_off_clean", "B_only", "Bonly_gap_clean_minus_Bonly")
    Bonly.to_csv(outdir / "B_only_effect_vs_clean_detailed_synth.csv", index=False)
    Bonly_sum = concat_summaries(Bonly, "Bonly_gap_clean_minus_Bonly")
    Bonly_sum.to_csv(outdir / "B_only_effect_vs_clean_summary_synth.csv", index=False)

    check2 = compare_arms(pd.concat([records, off_df], ignore_index=True), "off", "A_only", "check2_gap_off_minus_Aonly")
    check2.to_csv(outdir / "check2_w0_vs_off_detailed_synth.csv", index=False)
    check2_sum = concat_summaries(check2, "check2_gap_off_minus_Aonly")
    check2_sum.to_csv(outdir / "check2_w0_vs_off_summary_synth.csv", index=False)

    dose_parts = []
    for arm in ["AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"]:
        d = compare_arms(records, "A_only", arm, "B_dose_gap_Aonly_minus_AB")
        if not d.empty:
            d["dose_arm"] = arm
            dose_parts.append(d)
    dose = pd.concat(dose_parts, ignore_index=True) if dose_parts else pd.DataFrame()
    dose.to_csv(outdir / "B_dose_effect_vs_w0_detailed_synth.csv", index=False)
    dose_sum = concat_summaries(dose, "B_dose_gap_Aonly_minus_AB") if not dose.empty else pd.DataFrame()
    if not dose_sum.empty and "dose_arm" not in dose_sum.columns:
        # Main concat did not group by dose. Add a dose-specific summary too.
        dose_sum = pd.concat([
            dose_sum,
            summarize_gap(dose, "B_dose_gap_Aonly_minus_AB", ["dose_arm", "n"]),
            summarize_gap(dose, "B_dose_gap_Aonly_minus_AB", ["dose_arm"]),
        ], ignore_index=True)
    dose_sum.to_csv(outdir / "B_dose_effect_vs_w0_summary_synth.csv", index=False)

    coverage = []
    for name, df in [("treatment", records), ("off", off_df), ("pairwise", pairwise)]:
        if df is None or df.empty:
            coverage.append({"table": name, "rows": 0})
        else:
            row = {"table": name, "rows": len(df)}
            if "status" in df.columns:
                for st, cnt in df["status"].value_counts(dropna=False).items():
                    row[f"status_{st}"] = int(cnt)
            coverage.append(row)
    pd.DataFrame(coverage).to_csv(outdir / "coverage_by_table_synth.csv", index=False)

    write_report(outdir, {
        "coverage": pd.DataFrame(coverage),
        "level0 pressure regression": level0,
        "clean gate summary": clean_sum,
        "A effect summary": A_sum,
        "B-only effect summary": Bonly_sum,
        "B dose effect summary": dose_sum,
        "A_only vs off check": check2_sum,
    })

    print(f"Saved outputs under: {outdir}")
    print("\n=== Coverage ===")
    print(pd.DataFrame(coverage).to_string(index=False))
    if not level0.empty:
        print("\n=== Synthetic Level-0 slope ===")
        print(level0.to_string(index=False))
    if not clean_sum.empty:
        print("\n=== Clean gate overall ===")
        print(clean_sum[clean_sum["grouping"].eq("overall")].to_string(index=False))
    if not A_sum.empty:
        print("\n=== A effect by n ===")
        print(A_sum[A_sum["grouping"].eq("dataset+n")].to_string(index=False))


if __name__ == "__main__":
    main()

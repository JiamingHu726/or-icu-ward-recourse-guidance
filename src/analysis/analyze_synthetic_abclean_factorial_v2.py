#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_synthetic_abclean_factorial_v2.py

Corrected analyzer for Synthetic A/B/clean factorial experiments.

Why v2 exists
-------------
The original analyzer used the same iso-time comparison for the clean gate:
    clean_gap = old_off_at_60s - price_off_clean_at_60s
That is not a valid identity check when old Synthetic off and the new patched
clean runner have different wall-clock overhead. The search trajectory can be
identical while the 60-second row occurs at very different iterations.

v2 separates three roles:
  1) Clean-gate identity check:
       official: final-row and common-iteration comparisons against old off.
       diagnostic: iso-time comparison against old off.
  2) Treatment effects:
       compare A_only/B_only/AB arms against the matched new price_off_clean
       baseline under strict iso-time.
  3) Synthetic Level-0 pressure slope:
       regress treatment score delta against matched price_off_clean, not old off.

Gap convention:
  gap = base_score - treatment_score
  gap > 0 means the treatment is better than the base.

Main outputs:
  pricecal_treatment_manifest_synth.csv
  all_treatment_trace_records_iso60_synth.csv
  final_pairwise_iso60_vs_clean_detailed_synth.csv
  final_pairwise_iso60_detailed_synth.csv                  # alias: vs clean, official
  clean_gate_final_summary_synth.csv                       # official clean gate
  clean_gate_common_iteration_summary_synth.csv            # official/robustness clean gate
  clean_gate_iso60_diagnostic_summary_synth.csv            # diagnostic only
  clean_gate_summary_synth.csv                             # alias: final official
  A_effect_w0_vs_clean_summary_synth.csv
  B_only_effect_vs_clean_summary_synth.csv
  B_dose_effect_vs_w0_summary_synth.csv
  level0_pressure_regression_synth.csv                     # treatment - clean baseline
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
    out: Dict[str, Any] = {}
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
    candidates: List[Path] = []
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


def read_trace_df(trace_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    if df.empty:
        return df
    time_col = choose_col(df, ["elapsed_sec", "wallclock_elapsed_s", "elapsed_s", "time_s", "elapsed_seconds", "runtime_sec"])
    if time_col is not None:
        df = df.copy()
        df["_elapsed_sec"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df = df.copy()
        df["_elapsed_sec"] = np.nan
    return df


def select_iso_row(df: pd.DataFrame, iso_t: float) -> Tuple[Optional[pd.Series], str, float]:
    if "_elapsed_sec" not in df.columns or df["_elapsed_sec"].isna().all():
        return None, "missing_elapsed_column", np.nan
    tmp = df[np.isfinite(df["_elapsed_sec"])].copy()
    if tmp.empty:
        return None, "no_finite_elapsed", np.nan
    max_elapsed = float(tmp["_elapsed_sec"].max())
    if max_elapsed + 1e-9 < iso_t:
        return None, "insufficient_horizon", max_elapsed
    before = tmp[tmp["_elapsed_sec"] <= iso_t + 1e-9]
    if before.empty:
        return None, "no_row_at_or_before_iso", max_elapsed
    return before.sort_values("_elapsed_sec").iloc[-1], "ok", max_elapsed


def select_final_row(df: pd.DataFrame) -> Tuple[Optional[pd.Series], str, float]:
    if df.empty:
        return None, "empty_trace", np.nan
    max_elapsed = as_float(df["_elapsed_sec"].max()) if "_elapsed_sec" in df.columns else np.nan
    return df.iloc[-1], "ok", max_elapsed


def max_iteration(df: pd.DataFrame) -> float:
    if "iteration" not in df.columns:
        return np.nan
    vals = pd.to_numeric(df["iteration"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if len(vals) else np.nan


def select_iteration_row(df: pd.DataFrame, iteration: float) -> Tuple[Optional[pd.Series], str, float]:
    if df.empty:
        return None, "empty_trace", np.nan
    if "iteration" not in df.columns:
        return None, "missing_iteration_column", as_float(df.get("_elapsed_sec", pd.Series([np.nan])).max())
    tmp = df.copy()
    tmp["_iteration_num"] = pd.to_numeric(tmp["iteration"], errors="coerce")
    tmp = tmp[np.isfinite(tmp["_iteration_num"])]
    if tmp.empty:
        return None, "no_finite_iteration", np.nan
    before = tmp[tmp["_iteration_num"] <= float(iteration) + 1e-9]
    if before.empty:
        return None, "no_row_at_or_before_iteration", as_float(tmp["_elapsed_sec"].max())
    max_elapsed = as_float(tmp["_elapsed_sec"].max()) if "_elapsed_sec" in tmp.columns else np.nan
    return before.sort_values("_iteration_num").iloc[-1], "ok", max_elapsed


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
    pressure_cost = value_row_or_meta(row, meta, ["best_pressure_cost", "pressure_cost", "spiral_pressure_cost"], ["best_evaluator_summary.pressure_cost", "pressure_cost"])

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
        "trace_best_exact_score": value_from_row(row, ["best_exact_score"]),
        "trace_best_fast_obj": value_from_row(row, ["best_fast_obj", "best_fast_objective"]),
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


def load_trace_record(trace_path: Path, iso_t: float, arm_override: Optional[str] = None,
                      selection: str = "iso_time", iteration: Optional[float] = None) -> Dict[str, Any]:
    info = infer_case_fields(trace_path)
    if arm_override:
        info["arm"] = arm_override
    rec: Dict[str, Any] = {**info, "trace_path": str(trace_path), "status": "", "error": "", "selection": selection}
    try:
        df = read_trace_df(trace_path)
    except Exception as e:
        rec.update({"status": "read_error", "error": repr(e)})
        return rec
    if df.empty:
        rec.update({"status": "empty_trace"})
        return rec
    meta = read_metadata_near_trace(trace_path)
    if selection == "iso_time":
        row, st, max_elapsed = select_iso_row(df, iso_t)
    elif selection == "final_row":
        row, st, max_elapsed = select_final_row(df)
    elif selection == "iteration":
        if iteration is None:
            rec.update({"status": "missing_requested_iteration"})
            return rec
        row, st, max_elapsed = select_iteration_row(df, float(iteration))
        rec["requested_iteration"] = float(iteration)
    else:
        rec.update({"status": f"unknown_selection:{selection}"})
        return rec

    rec["max_elapsed_sec"] = max_elapsed
    rec["iso_t_sec"] = iso_t
    rec["max_iteration"] = max_iteration(df)
    if row is None:
        rec.update({"status": st})
        return rec
    comps, missing = exact_components(row, meta)
    score = score_from_components(comps)
    rec.update({
        "status": "ok" if np.isfinite(score) else "missing_components",
        "missing_components": missing,
        "elapsed_sec_at_selection": as_float(row.get("_elapsed_sec")),
        "iteration_at_selection": as_float(row.get("iteration")),
        "exact_nopressure_score": score,
    })
    for c in COMPONENTS:
        rec[c] = comps.get(c, np.nan)
        rec[f"weighted_{c}"] = as_float(comps.get(c)) * W_EXACT[c] if np.isfinite(as_float(comps.get(c))) else np.nan
    for k in ["pressure_cost", "target_volume", "high_target", "n_scheduled", "n_high_priority_scheduled", "trace_best_exact_score", "trace_best_fast_obj"]:
        rec[k] = comps.get(k, np.nan)
    # Manifest/meta fields.
    for k in [
        "actual_pressure_weight", "pressure_weight_unit", "b_multiplier", "operator_on", "guidance_mode",
        "lns_seed", "cycles", "proposals_per_cycle", "exact_every", "exact_top_k", "max_or_overtime",
        "recourse_pressure_weight_fast", "abclean_manifest.actual_pressure_weight", "abclean_manifest.operator_on",
        "abclean_manifest.lns_seed", "abclean_manifest.seed",
    ]:
        if k in meta:
            rec[k.replace(".", "_")] = meta[k]
    return rec


def discover_treatment_traces(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("spiral_trace.csv"))


def discover_manifest(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for p in sorted(root.rglob("abclean_run_manifest.json")):
        d = read_json(p)
        if d:
            d["manifest_path"] = str(p)
            rows.append(d)
    for p in sorted(root.rglob("synthetic_abclean_run_log.csv")):
        try:
            df = pd.read_csv(p)
            df["run_log_path"] = str(p)
            rows += df.to_dict("records")
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    subset = [c for c in ["dataset", "n", "scenario", "seed", "arm", "output_dir"] if c in df.columns]
    return df.drop_duplicates(subset=subset, keep="last") if subset else df


def find_off_trace(off_roots: List[Path], scenario: str, n: int, seed: int) -> Optional[Path]:
    cname = f"case_{int(n)}_seed{int(seed)}"
    direct_patterns: List[Path] = []
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
    rows: List[Dict[str, Any]] = []
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


def concat_summaries(detailed: pd.DataFrame, gap_col: str) -> pd.DataFrame:
    if detailed is None or detailed.empty:
        return pd.DataFrame()
    parts = [
        summarize_gap(detailed, gap_col, ["dataset", "n"]),
        summarize_gap(detailed, gap_col, ["dataset", "scenario", "n"]),
        summarize_gap(detailed, gap_col, ["n"]),
        summarize_gap(detailed, gap_col, []),
    ]
    return pd.concat(parts, ignore_index=True)


def compare_arms(records: pd.DataFrame, base_arm: str, treat_arm: str, out_gap: str) -> pd.DataFrame:
    key = ["dataset", "n", "scenario", "seed"]
    b = records[(records["arm"] == base_arm) & (records["status"] == "ok")].copy()
    t = records[(records["arm"] == treat_arm) & (records["status"] == "ok")].copy()
    if b.empty or t.empty:
        return pd.DataFrame()
    keep = key + ["exact_nopressure_score"] + COMPONENTS + ["pressure_cost", "trace_path", "actual_pressure_weight", "operator_on", "guidance_mode", "iteration_at_selection", "elapsed_sec_at_selection"]
    b = b[[c for c in keep if c in b.columns]].rename(columns={
        "exact_nopressure_score": "base_score", "pressure_cost": "base_pressure_cost", "trace_path": "base_trace_path",
        "iteration_at_selection": "base_iteration_at_selection", "elapsed_sec_at_selection": "base_elapsed_sec_at_selection",
    })
    t = t[[c for c in keep if c in t.columns]].rename(columns={
        "exact_nopressure_score": "treatment_score", "pressure_cost": "treatment_pressure_cost", "trace_path": "treatment_trace_path",
        "iteration_at_selection": "treatment_iteration_at_selection", "elapsed_sec_at_selection": "treatment_elapsed_sec_at_selection",
    })
    for comp in COMPONENTS:
        if comp in b.columns:
            b = b.rename(columns={comp: f"base_{comp}"})
        if comp in t.columns:
            t = t.rename(columns={comp: f"treatment_{comp}"})
    # Preserve treatment metadata with explicit names.
    for col in ["actual_pressure_weight", "operator_on", "guidance_mode"]:
        if col in t.columns:
            t = t.rename(columns={col: f"treatment_{col}"})
        if col in b.columns:
            b = b.rename(columns={col: f"base_{col}"})
    m = b.merge(t, on=key, how="inner")
    m["base_arm"] = base_arm
    m["treatment_arm"] = treat_arm
    m[out_gap] = m["base_score"] - m["treatment_score"]
    for comp in COMPONENTS:
        cb, ct = f"base_{comp}", f"treatment_{comp}"
        if cb in m.columns and ct in m.columns:
            m[f"{out_gap}_component__{comp}"] = W_EXACT[comp] * (m[cb] - m[ct])
    return m


def build_pairwise_vs_clean(records: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for arm in ["A_only", "B_only", "AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"]:
        d = compare_arms(records, "price_off_clean", arm, "gap_clean_minus_treatment")
        if not d.empty:
            d["arm"] = arm
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def clean_gate_pair(clean_trace: Path, off_trace: Path, iso_t: float, mode: str) -> Dict[str, Any]:
    base_info = infer_case_fields(clean_trace)
    n, scenario, seed = int(base_info["n"]), str(base_info["scenario"]), int(base_info["seed"])
    row: Dict[str, Any] = {"dataset": "Synthetic", "n": n, "scenario": scenario, "seed": seed, "mode": mode,
                           "clean_trace_path": str(clean_trace), "off_trace_path": str(off_trace), "status": ""}
    if mode == "iso_time":
        clean = load_trace_record(clean_trace, iso_t, arm_override="price_off_clean", selection="iso_time")
        off = load_trace_record(off_trace, iso_t, arm_override="off", selection="iso_time")
    elif mode == "final_row":
        clean = load_trace_record(clean_trace, iso_t, arm_override="price_off_clean", selection="final_row")
        off = load_trace_record(off_trace, iso_t, arm_override="off", selection="final_row")
    elif mode == "common_iteration":
        try:
            cdf = read_trace_df(clean_trace)
            odf = read_trace_df(off_trace)
            common_it = min(max_iteration(cdf), max_iteration(odf))
        except Exception as e:
            row.update({"status": "read_error", "error": repr(e)})
            return row
        if not np.isfinite(common_it):
            row.update({"status": "missing_common_iteration"})
            return row
        row["common_iteration"] = common_it
        clean = load_trace_record(clean_trace, iso_t, arm_override="price_off_clean", selection="iteration", iteration=common_it)
        off = load_trace_record(off_trace, iso_t, arm_override="off", selection="iteration", iteration=common_it)
    else:
        row.update({"status": f"unknown_mode:{mode}"})
        return row

    row["clean_status"] = clean.get("status")
    row["off_status"] = off.get("status")
    if clean.get("status") != "ok" or off.get("status") != "ok":
        row["status"] = f"bad_records:clean={clean.get('status')};off={off.get('status')}"
        return row
    row["status"] = "ok"
    row["clean_score"] = clean["exact_nopressure_score"]
    row["off_score"] = off["exact_nopressure_score"]
    row["clean_gap_off_minus_clean"] = off["exact_nopressure_score"] - clean["exact_nopressure_score"]
    row["clean_elapsed_sec"] = clean.get("elapsed_sec_at_selection")
    row["off_elapsed_sec"] = off.get("elapsed_sec_at_selection")
    row["clean_iteration"] = clean.get("iteration_at_selection")
    row["off_iteration"] = off.get("iteration_at_selection")
    row["clean_max_elapsed_sec"] = clean.get("max_elapsed_sec")
    row["off_max_elapsed_sec"] = off.get("max_elapsed_sec")
    row["runtime_ratio_clean_over_off"] = (as_float(clean.get("max_elapsed_sec")) / as_float(off.get("max_elapsed_sec"))
                                           if np.isfinite(as_float(clean.get("max_elapsed_sec"))) and np.isfinite(as_float(off.get("max_elapsed_sec"))) and as_float(off.get("max_elapsed_sec")) != 0 else np.nan)
    for comp in COMPONENTS:
        row[f"off_{comp}"] = off.get(comp, np.nan)
        row[f"clean_{comp}"] = clean.get(comp, np.nan)
        row[f"clean_gap_off_minus_clean_component__{comp}"] = W_EXACT[comp] * (off.get(comp, np.nan) - clean.get(comp, np.nan))
    # A stricter trace identity diagnostic that ignores elapsed-time and extra diagnostic columns.
    try:
        cdf = read_trace_df(clean_trace).drop(columns=["_elapsed_sec"], errors="ignore")
        odf = read_trace_df(off_trace).drop(columns=["_elapsed_sec"], errors="ignore")
        common_cols = [c for c in odf.columns if c in cdf.columns and c != "elapsed_sec"]
        same_shape_common = len(cdf) == len(odf)
        same_common_cols = bool(common_cols) and cdf[common_cols].reset_index(drop=True).equals(odf[common_cols].reset_index(drop=True))
        row["common_trace_columns_identical"] = int(same_shape_common and same_common_cols)
        row["num_common_trace_columns_checked"] = len(common_cols)
        row["clean_trace_rows"] = len(cdf)
        row["off_trace_rows"] = len(odf)
    except Exception as e:
        row["common_trace_columns_identical"] = np.nan
        row["trace_identity_error"] = repr(e)
    return row


def build_clean_gate(records: pd.DataFrame, off_roots: List[Path], iso_t: float, sizes: List[int], scenarios: List[str], seeds: List[int], mode: str) -> pd.DataFrame:
    clean_map: Dict[Tuple[int, str, int], str] = {}
    for _, r in records[(records["arm"] == "price_off_clean") & (records["status"] == "ok")].iterrows():
        clean_map[(int(r["n"]), str(r["scenario"]), int(r["seed"]))] = str(r["trace_path"])
    rows: List[Dict[str, Any]] = []
    for n in sizes:
        for sc in scenarios:
            for seed in seeds:
                key = (int(n), str(sc), int(seed))
                clean_path = clean_map.get(key)
                off_path = find_off_trace(off_roots, sc, n, seed)
                if clean_path is None or off_path is None:
                    rows.append({"dataset": "Synthetic", "n": n, "scenario": sc, "seed": seed, "mode": mode,
                                 "status": "missing_clean_or_off", "clean_trace_path": clean_path or "", "off_trace_path": str(off_path or "")})
                    continue
                rows.append(clean_gate_pair(Path(clean_path), Path(off_path), iso_t, mode))
    return pd.DataFrame(rows)


def fit_level0_slope_vs_clean(pairwise_vs_clean: pd.DataFrame) -> pd.DataFrame:
    df = pairwise_vs_clean.copy()
    if df.empty:
        return pd.DataFrame([{"dataset": "Synthetic", "status": "insufficient", "n_points": 0, "baseline": "price_off_clean"}])
    # Prefer operator-on arms where pressure_cost exists. Include A_only and AB arms; if absent, fall back to all treatments.
    preferred = df[df["treatment_arm"].astype(str).isin(["A_only", "AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"])].copy()
    use = preferred if len(preferred) >= 3 else df.copy()
    use["x_pressure_cost"] = pd.to_numeric(use.get("treatment_pressure_cost"), errors="coerce")
    use["y_exact_delta_vs_clean"] = pd.to_numeric(use["treatment_score"], errors="coerce") - pd.to_numeric(use["base_score"], errors="coerce")
    use = use[np.isfinite(use["x_pressure_cost"]) & np.isfinite(use["y_exact_delta_vs_clean"])]
    if len(use) < 3 or float(use["x_pressure_cost"].var()) <= 1e-12:
        return pd.DataFrame([{
            "dataset": "Synthetic", "status": "insufficient", "n_points": int(len(use)),
            "baseline": "price_off_clean", "reason": "need >=3 points with nonzero pressure-cost variance",
        }])
    x = use["x_pressure_cost"].to_numpy(dtype=float)
    y = use["y_exact_delta_vs_clean"].to_numpy(dtype=float)
    xm, ym = x.mean(), y.mean()
    slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    intercept = float(ym - slope * xm)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return pd.DataFrame([{
        "dataset": "Synthetic", "status": "ok", "slope_syn": slope, "intercept_syn": intercept,
        "R2_syn": r2, "n_points": int(len(use)), "baseline": "price_off_clean",
        "x": "treatment_pressure_cost",
        "y": "treatment_exact_nopressure_score - price_off_clean_exact_nopressure_score",
        "note": "If R2_syn < threshold, treat B as weak/uninformative and prioritize A_only interpretation.",
    }])


def write_report(outdir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Synthetic A/B/clean factorial report (v2)", "",
        "Gap convention: `gap = base_score - treatment_score`; positive means treatment is better.", "",
        "Clean gate is judged by final-row and common-iteration comparisons, not by iso-60 wall-clock rows.", "",
        "Treatment effects are compared against the matched new `price_off_clean` baseline at strict iso-time.", "",
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
    ap.add_argument("--off-root", action="append", default=["spiral_price_off_all_results_v2"], help="Can be repeated. Existing Synthetic off baseline root. Used only for clean-gate diagnostics.")
    ap.add_argument("--output-dir", default="synthetic_abclean_factorial_analysis_v2")
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

    treatment_records: List[Dict[str, Any]] = []
    for p in discover_treatment_traces(ab_root):
        rec = load_trace_record(p, args.iso_t, selection="iso_time")
        if rec.get("arm") in {"", "off"}:
            continue
        treatment_records.append(rec)
    records = pd.DataFrame(treatment_records)
    if records.empty:
        raise SystemExit(f"No treatment traces found under {ab_root}")
    records.to_csv(outdir / "all_treatment_trace_records_iso60_synth.csv", index=False)

    # Clean gate: official final/common-iteration + diagnostic iso-time against old off.
    clean_final = build_clean_gate(records, off_roots, args.iso_t, args.sizes, args.scenarios, args.seeds, "final_row")
    clean_common = build_clean_gate(records, off_roots, args.iso_t, args.sizes, args.scenarios, args.seeds, "common_iteration")
    clean_iso = build_clean_gate(records, off_roots, args.iso_t, args.sizes, args.scenarios, args.seeds, "iso_time")
    for name, df in [
        ("clean_gate_final_detailed_synth.csv", clean_final),
        ("clean_gate_common_iteration_detailed_synth.csv", clean_common),
        ("clean_gate_iso60_diagnostic_detailed_synth.csv", clean_iso),
    ]:
        df.to_csv(outdir / name, index=False)
    clean_final_sum = concat_summaries(clean_final[clean_final["status"].eq("ok")], "clean_gap_off_minus_clean")
    clean_common_sum = concat_summaries(clean_common[clean_common["status"].eq("ok")], "clean_gap_off_minus_clean")
    clean_iso_sum = concat_summaries(clean_iso[clean_iso["status"].eq("ok")], "clean_gap_off_minus_clean")
    clean_final_sum.to_csv(outdir / "clean_gate_final_summary_synth.csv", index=False)
    clean_common_sum.to_csv(outdir / "clean_gate_common_iteration_summary_synth.csv", index=False)
    clean_iso_sum.to_csv(outdir / "clean_gate_iso60_diagnostic_summary_synth.csv", index=False)
    # Backward-compatible alias: clean_gate_summary now means official final-row clean gate, not iso-60.
    clean_final_sum.to_csv(outdir / "clean_gate_summary_synth.csv", index=False)

    # Official treatment pairwise: all treatment arms vs matched new clean baseline under iso-time.
    pairwise_clean = build_pairwise_vs_clean(records)
    pairwise_clean.to_csv(outdir / "final_pairwise_iso60_vs_clean_detailed_synth.csv", index=False)
    # Backward-compatible alias; document clearly in report and stdout.
    pairwise_clean.to_csv(outdir / "final_pairwise_iso60_detailed_synth.csv", index=False)

    level0 = fit_level0_slope_vs_clean(pairwise_clean)
    level0.to_csv(outdir / "level0_pressure_regression_synth.csv", index=False)

    # Arm-vs-arm summaries from new runner only.
    A = compare_arms(records, "price_off_clean", "A_only", "A_gap_clean_minus_Aonly")
    A.to_csv(outdir / "A_effect_w0_vs_clean_detailed_synth.csv", index=False)
    A_sum = concat_summaries(A, "A_gap_clean_minus_Aonly")
    A_sum.to_csv(outdir / "A_effect_w0_vs_clean_summary_synth.csv", index=False)

    Bonly = compare_arms(records, "price_off_clean", "B_only", "Bonly_gap_clean_minus_Bonly")
    Bonly.to_csv(outdir / "B_only_effect_vs_clean_detailed_synth.csv", index=False)
    Bonly_sum = concat_summaries(Bonly, "Bonly_gap_clean_minus_Bonly")
    Bonly_sum.to_csv(outdir / "B_only_effect_vs_clean_summary_synth.csv", index=False)

    dose_parts: List[pd.DataFrame] = []
    for arm in ["AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"]:
        d = compare_arms(records, "A_only", arm, "B_dose_gap_Aonly_minus_AB")
        if not d.empty:
            d["dose_arm"] = arm
            dose_parts.append(d)
    dose = pd.concat(dose_parts, ignore_index=True) if dose_parts else pd.DataFrame()
    dose.to_csv(outdir / "B_dose_effect_vs_w0_detailed_synth.csv", index=False)
    dose_sum = pd.DataFrame()
    if not dose.empty:
        dose_sum = pd.concat([
            concat_summaries(dose, "B_dose_gap_Aonly_minus_AB"),
            summarize_gap(dose, "B_dose_gap_Aonly_minus_AB", ["dose_arm", "n"]),
            summarize_gap(dose, "B_dose_gap_Aonly_minus_AB", ["dose_arm"]),
        ], ignore_index=True)
    dose_sum.to_csv(outdir / "B_dose_effect_vs_w0_summary_synth.csv", index=False)

    # Optional diagnostic only: old-off vs A_only at iso-time, known to be confounded by runtime overhead.
    # We write it with an explicit diagnostic filename to avoid accidental use as a main result.
    # It can still be useful to show why old off should not be used as the matched iso-time baseline.
    # Build old-off iso records for the requested grid.
    off_records: List[Dict[str, Any]] = []
    for n in args.sizes:
        for sc in args.scenarios:
            for seed in args.seeds:
                p = find_off_trace(off_roots, sc, n, seed)
                if p is None:
                    off_records.append({"dataset": "Synthetic", "n": n, "scenario": sc, "seed": seed, "arm": "off", "status": "off_trace_not_found"})
                else:
                    off_records.append(load_trace_record(p, args.iso_t, arm_override="off", selection="iso_time"))
    off_df = pd.DataFrame(off_records)
    off_df.to_csv(outdir / "off_trace_records_iso60_diagnostic_synth.csv", index=False)
    check2_diag = compare_arms(pd.concat([records, off_df], ignore_index=True), "off", "A_only", "check2_diag_gap_oldoff_minus_Aonly")
    check2_diag.to_csv(outdir / "check2_w0_vs_oldoff_iso60_diagnostic_detailed_synth.csv", index=False)
    check2_diag_sum = concat_summaries(check2_diag, "check2_diag_gap_oldoff_minus_Aonly")
    check2_diag_sum.to_csv(outdir / "check2_w0_vs_oldoff_iso60_diagnostic_summary_synth.csv", index=False)

    coverage = []
    for name, df in [
        ("treatment_iso60", records),
        ("clean_gate_final", clean_final),
        ("clean_gate_common_iteration", clean_common),
        ("clean_gate_iso60_diagnostic", clean_iso),
        ("pairwise_vs_clean", pairwise_clean),
        ("oldoff_iso60_diagnostic", off_df),
    ]:
        row: Dict[str, Any] = {"table": name, "rows": 0 if df is None else len(df)}
        if df is not None and not df.empty and "status" in df.columns:
            for st, cnt in df["status"].value_counts(dropna=False).items():
                row[f"status_{st}"] = int(cnt)
        coverage.append(row)
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(outdir / "coverage_by_table_synth.csv", index=False)

    write_report(outdir, {
        "coverage": coverage_df,
        "level0 pressure regression vs clean": level0,
        "clean gate final summary (official)": clean_final_sum,
        "clean gate common-iteration summary (official robustness)": clean_common_sum,
        "clean gate iso-60 summary (diagnostic only)": clean_iso_sum,
        "A effect summary vs clean": A_sum,
        "B-only effect summary vs clean": Bonly_sum,
        "B dose effect summary vs A_only": dose_sum,
        "old off vs A_only iso-60 diagnostic (do not use as main result)": check2_diag_sum,
    })

    print(f"Saved outputs under: {outdir}")
    print("\n=== Coverage ===")
    print(coverage_df.to_string(index=False))
    print("\n=== Clean gate final overall (official) ===")
    if not clean_final_sum.empty:
        print(clean_final_sum[clean_final_sum["grouping"].eq("overall")].to_string(index=False))
    else:
        print("No final clean-gate summary rows.")
    print("\n=== Clean gate iso-60 diagnostic overall ===")
    if not clean_iso_sum.empty:
        print(clean_iso_sum[clean_iso_sum["grouping"].eq("overall")].to_string(index=False))
    else:
        print("No iso-60 clean-gate diagnostic rows.")
    print("\n=== Synthetic Level-0 slope vs clean ===")
    print(level0.to_string(index=False))
    print("\n=== A effect by n vs clean ===")
    if not A_sum.empty:
        print(A_sum[A_sum["grouping"].eq("dataset+n")].to_string(index=False))
    else:
        print("No A-effect summary rows.")


if __name__ == "__main__":
    main()

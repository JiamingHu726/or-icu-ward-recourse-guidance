#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_clean_gate_alignment.py

Purpose
-------
Diagnose why Synthetic price_off_clean != old off under iso-60.
It compares the same off/clean trace pair under three alignments:
  1) iso-time row: last row with elapsed <= --iso-t
  2) common-iteration row: last row with iteration <= min(max_iter_off, max_iter_clean)
  3) final row: last row in each trace

This is a diagnostic script. It does not replace the main analyzer.
It imports helper functions and exact-score weights from analyze_synthetic_abclean_factorial.py,
so put both files in the same repository directory.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import analyze_synthetic_abclean_factorial as ana


def _finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _sort_by_elapsed_or_index(df: pd.DataFrame) -> pd.DataFrame:
    time_col = ana.choose_col(df, ["elapsed_sec", "wallclock_elapsed_s", "elapsed_s", "time_s", "elapsed_seconds", "runtime_sec"])
    out = df.copy()
    if time_col is not None:
        out["_elapsed_sec"] = pd.to_numeric(out[time_col], errors="coerce")
        out = out[np.isfinite(out["_elapsed_sec"])]
        return out.sort_values("_elapsed_sec")
    out["_elapsed_sec"] = np.nan
    return out.reset_index(drop=True)


def row_iso(df: pd.DataFrame, iso_t: float) -> Tuple[Optional[pd.Series], str]:
    row, status, _ = ana.select_iso_row(df, iso_t)
    return row, status


def row_final(df: pd.DataFrame) -> Tuple[Optional[pd.Series], str]:
    tmp = _sort_by_elapsed_or_index(df)
    if tmp.empty:
        return None, "empty_or_no_elapsed"
    return tmp.iloc[-1], "ok"


def row_at_or_before_iteration(df: pd.DataFrame, target_iter: float) -> Tuple[Optional[pd.Series], str]:
    if "iteration" not in df.columns:
        return None, "missing_iteration"
    tmp = df.copy()
    tmp["_iteration"] = pd.to_numeric(tmp["iteration"], errors="coerce")
    tmp = tmp[np.isfinite(tmp["_iteration"])]
    tmp = tmp[tmp["_iteration"] <= target_iter]
    if tmp.empty:
        return None, "no_row_at_or_before_common_iteration"
    return tmp.sort_values("_iteration").iloc[-1], "ok"


def max_iteration(df: pd.DataFrame) -> float:
    if "iteration" not in df.columns:
        return np.nan
    vals = pd.to_numeric(df["iteration"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if len(vals) else np.nan


def score_row(row: pd.Series, meta: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    comps, missing = ana.exact_components(row, meta)
    score = ana.score_from_components(comps)
    return score, comps, missing


def make_record(mode: str, n: int, scenario: str, seed: int, off_trace: Path, clean_trace: Path,
                off_row: Optional[pd.Series], clean_row: Optional[pd.Series], off_status: str, clean_status: str,
                off_meta: Dict[str, Any], clean_meta: Dict[str, Any]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "mode": mode,
        "dataset": "Synthetic",
        "n": int(n),
        "scenario": scenario,
        "seed": int(seed),
        "off_trace_path": str(off_trace),
        "clean_trace_path": str(clean_trace),
        "off_row_status": off_status,
        "clean_row_status": clean_status,
    }
    if off_row is None or clean_row is None:
        rec["status"] = "bad_row"
        return rec
    off_score, off_comps, off_missing = score_row(off_row, off_meta)
    clean_score, clean_comps, clean_missing = score_row(clean_row, clean_meta)
    rec.update({
        "status": "ok" if np.isfinite(off_score) and np.isfinite(clean_score) else "missing_components",
        "off_missing": off_missing,
        "clean_missing": clean_missing,
        "off_score": off_score,
        "clean_score": clean_score,
        "gap_off_minus_clean": off_score - clean_score,
        "off_elapsed_sec": ana.as_float(off_row.get("_elapsed_sec", off_row.get("elapsed_sec", np.nan))),
        "clean_elapsed_sec": ana.as_float(clean_row.get("_elapsed_sec", clean_row.get("elapsed_sec", np.nan))),
        "off_iteration": ana.as_float(off_row.get("iteration", np.nan)),
        "clean_iteration": ana.as_float(clean_row.get("iteration", np.nan)),
    })
    for comp in ana.COMPONENTS:
        ov = ana.as_float(off_comps.get(comp))
        cv = ana.as_float(clean_comps.get(comp))
        rec[f"off_{comp}"] = ov
        rec[f"clean_{comp}"] = cv
        rec[f"gap_raw__{comp}"] = ov - cv
        rec[f"gap_weighted__{comp}"] = ana.W_EXACT[comp] * (ov - cv) if np.isfinite(ov) and np.isfinite(cv) else np.nan
    return rec


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, g in df[df["status"].eq("ok")].groupby("mode"):
        vals = pd.to_numeric(g["gap_off_minus_clean"], errors="coerce").dropna()
        rows.append({
            "mode": mode,
            "num_cases": int(len(vals)),
            "wins": int((vals > 1e-9).sum()),
            "losses": int((vals < -1e-9).sum()),
            "ties": int((vals.abs() <= 1e-9).sum()),
            "mean_gap": float(vals.mean()) if len(vals) else np.nan,
            "median_gap": float(vals.median()) if len(vals) else np.nan,
            "min_gap": float(vals.min()) if len(vals) else np.nan,
            "max_gap": float(vals.max()) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-root", required=True)
    ap.add_argument("--off-root", action="append", required=True)
    ap.add_argument("--output-dir", default="synthetic_clean_gate_alignment_diagnosis")
    ap.add_argument("--iso-t", type=float, default=60.0)
    ap.add_argument("--sizes", nargs="+", type=int, default=[100,150])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7,11,19,23,29])
    ap.add_argument("--scenarios", nargs="+", default=["nominal", "transfer_bottleneck"])
    ap.add_argument("--clean-arm", default="price_off_clean")
    args = ap.parse_args()

    ab_root = Path(args.ab_root)
    off_roots = [Path(x) for x in args.off_root]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.sizes:
        for scenario in args.scenarios:
            for seed in args.seeds:
                clean_trace = ab_root / "Synthetic" / f"n{int(n)}" / args.clean_arm / scenario / f"case_{int(n)}_seed{int(seed)}" / args.clean_arm / "spiral_trace.csv"
                if not clean_trace.exists():
                    rows.append({"mode":"all", "dataset":"Synthetic", "n":n, "scenario":scenario, "seed":seed, "status":"clean_trace_not_found", "clean_trace_path":str(clean_trace)})
                    continue
                off_trace = ana.find_off_trace(off_roots, scenario, n, seed)
                if off_trace is None:
                    rows.append({"mode":"all", "dataset":"Synthetic", "n":n, "scenario":scenario, "seed":seed, "status":"off_trace_not_found"})
                    continue
                off_df = pd.read_csv(off_trace)
                clean_df = pd.read_csv(clean_trace)
                off_meta = ana.read_metadata_near_trace(off_trace)
                clean_meta = ana.read_metadata_near_trace(clean_trace)

                # iso-time
                off_r, off_st = row_iso(off_df, args.iso_t)
                clean_r, clean_st = row_iso(clean_df, args.iso_t)
                rows.append(make_record("iso_time", n, scenario, seed, off_trace, clean_trace, off_r, clean_r, off_st, clean_st, off_meta, clean_meta))

                # common iteration
                oi = max_iteration(off_df)
                ci = max_iteration(clean_df)
                if np.isfinite(oi) and np.isfinite(ci):
                    target = min(oi, ci)
                    off_r, off_st = row_at_or_before_iteration(off_df, target)
                    clean_r, clean_st = row_at_or_before_iteration(clean_df, target)
                    rec = make_record("common_iteration", n, scenario, seed, off_trace, clean_trace, off_r, clean_r, off_st, clean_st, off_meta, clean_meta)
                    rec["common_iteration_target"] = target
                    rows.append(rec)
                else:
                    rows.append({"mode":"common_iteration", "dataset":"Synthetic", "n":n, "scenario":scenario, "seed":seed, "status":"missing_iteration"})

                # final row
                off_r, off_st = row_final(off_df)
                clean_r, clean_st = row_final(clean_df)
                rows.append(make_record("final_row", n, scenario, seed, off_trace, clean_trace, off_r, clean_r, off_st, clean_st, off_meta, clean_meta))

    detailed = pd.DataFrame(rows)
    detailed.to_csv(outdir / "clean_gate_alignment_detailed.csv", index=False)
    summ = summarize(detailed)
    summ.to_csv(outdir / "clean_gate_alignment_summary.csv", index=False)
    print(f"Saved: {outdir}")
    if not summ.empty:
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()

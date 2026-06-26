#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
collect_level1_v3_from_inventory.py

Robust collector for Level-1 v3 calibrated-price experiments.

Why this script exists
----------------------
Your audit shows all 800 price traces exist, but the older collector may still
fail because it searches paths/method folders heuristically. This collector
uses v3_trace_inventory.csv directly, so it does not need to rediscover price
trace paths.

It compares:
    price arms from v3_trace_inventory.csv
against:
    off traces under spiral_price_off_all_results_v2

Strict iso-time rule:
    At t-main, both price and off must have trace rows covering at least t-main.
    No hold-last beyond a method's actual trace horizon.

Main outputs
------------
final_pairwise_iso60_detailed.csv
final_verdict_iso60.csv
final_component_decomposition_iso60.csv
coverage_by_t.csv
collector_errors.csv
collector_column_diagnostics.csv

Gap convention
--------------
    gap = off - price
    gap > 0 means price arm is better.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


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

COMPONENT_ORDER = list(W_EXACT.keys())

DEFAULT_DATASETS = ["GermanOR", "Mannino"]
DEFAULT_SIZES = [50, 70, 100, 150]
DEFAULT_SCENARIOS = ["nominal", "transfer_bottleneck"]
DEFAULT_SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
DEFAULT_ARMS = ["price_cal_w0", "price_cal_w0p25", "price_cal_w0p5", "price_cal_w1", "price_cal_w2"]


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


def norm_path_str(s: str) -> str:
    return str(s).replace("\\", os.sep).replace("/", os.sep)


def resolve_existing_path(path_text: str, root_hint: Optional[Path] = None) -> Optional[Path]:
    if path_text is None or str(path_text).strip() == "":
        return None

    raw = Path(norm_path_str(str(path_text)))
    candidates = [raw]

    if root_hint is not None and not raw.is_absolute():
        candidates.append(root_hint / raw)

    for p in candidates:
        if p.exists():
            return p

    return None


def choose_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower.get(c.lower())
        if hit is not None:
            return hit
    for c in df.columns:
        cl = str(c).lower()
        for key in candidates:
            if key.lower() in cl:
                return c
    return None


def row_value(row: pd.Series, keys: List[str]) -> float:
    lower = {str(c).lower(): c for c in row.index}
    for k in keys:
        c = lower.get(k.lower())
        if c is not None:
            v = as_float(row.get(c))
            if np.isfinite(v):
                return v
    for c in row.index:
        cl = str(c).lower()
        for k in keys:
            if k.lower() in cl:
                v = as_float(row.get(c))
                if np.isfinite(v):
                    return v
    return np.nan


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


def read_metadata_near_trace(trace_path: Path) -> Dict[str, Any]:
    """
    Look around the trace folder and its parents for metadata JSON.
    """
    candidates = []
    p = trace_path.parent
    for _ in range(4):
        candidates += [
            p / "spiral_pr_glns_metadata.json",
            p / "metadata.json",
            p / "run_metadata.json",
        ]
        p = p.parent

    for c in candidates:
        if c.exists():
            try:
                import json
                return flatten(json.loads(c.read_text(encoding="utf-8")))
            except Exception:
                pass
    return {}


def meta_value(meta: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        if k in meta:
            v = as_float(meta.get(k))
            if np.isfinite(v):
                return v
    for mk, mv in meta.items():
        ml = str(mk).lower()
        for k in keys:
            if ml.endswith(k.lower()):
                v = as_float(mv)
                if np.isfinite(v):
                    return v
    return np.nan


def value_from_row_or_meta(row: pd.Series, meta: Dict[str, Any], row_keys: List[str], meta_keys: Optional[List[str]] = None) -> float:
    v = row_value(row, row_keys)
    if np.isfinite(v):
        return v
    return meta_value(meta, meta_keys if meta_keys is not None else row_keys)


def exact_components(row: pd.Series, meta: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
    target_volume = value_from_row_or_meta(row, meta, ["target_volume", "best_target_volume"])
    high_target = value_from_row_or_meta(row, meta, ["high_target", "best_high_target"])

    n_scheduled = value_from_row_or_meta(row, meta, ["best_n_scheduled", "n_scheduled"])
    high_scheduled = value_from_row_or_meta(row, meta, [
        "best_high_scheduled",
        "best_n_high_priority_scheduled",
        "n_high_priority_scheduled",
    ])

    violation = value_from_row_or_meta(row, meta, ["best_violation", "violation_count"])
    overtime = value_from_row_or_meta(row, meta, ["best_overtime", "or_overtime_min"])

    blocked = value_from_row_or_meta(row, meta, [
        "best_stage3_blocked",
        "blocked_transfer_patient_days_stage3",
        "best_eval_blocked",
        "blocked_transfer_patient_days",
    ])
    icu_excess = value_from_row_or_meta(row, meta, [
        "best_stage3_icu_excess",
        "icu_excess_bed_days_stage3",
        "best_eval_icu_excess",
        "icu_excess_bed_days_blocking",
    ])
    ward_excess = value_from_row_or_meta(row, meta, [
        "best_stage3_ward_excess",
        "ward_excess_bed_days_stage3",
        "best_eval_ward_excess",
        "ward_excess_bed_days_blocking",
    ])
    peak = value_from_row_or_meta(row, meta, [
        "best_stage3_peak_blocked",
        "peak_icu_ready_blocked_stage3",
        "best_eval_peak_blocked",
        "peak_icu_ready_blocked",
    ])

    high_deficit = (
        max(0.0, high_target - high_scheduled)
        if np.isfinite(high_target) and np.isfinite(high_scheduled)
        else np.nan
    )
    volume_deficit = (
        max(0.0, target_volume - n_scheduled)
        if np.isfinite(target_volume) and np.isfinite(n_scheduled)
        else np.nan
    )
    volume_excess = (
        max(0.0, n_scheduled - target_volume)
        if np.isfinite(target_volume) and np.isfinite(n_scheduled)
        else np.nan
    )

    comps = {
        "exact_high_deficit": high_deficit,
        "exact_volume_deficit": volume_deficit,
        "exact_volume_excess": volume_excess,
        "exact_violation": violation,
        "exact_blocked": blocked,
        "exact_ward_excess": ward_excess,
        "exact_peak": peak,
        "exact_icu_excess": icu_excess,
        "exact_overtime": overtime,
    }
    missing = [k for k, v in comps.items() if not np.isfinite(v)]
    return comps, "|".join(missing)


def score_from_components(comps: Dict[str, float]) -> float:
    score = 0.0
    for c, w in W_EXACT.items():
        v = as_float(comps.get(c))
        if not np.isfinite(v):
            return np.nan
        score += w * v
    return score


def enrich_trace(trace_path: Path) -> Tuple[Optional[pd.DataFrame], str, Dict[str, Any]]:
    try:
        df = pd.read_csv(trace_path)
    except Exception as e:
        return None, f"read_error:{repr(e)}", {}

    if df.empty:
        return None, "empty_trace", {}

    time_col = choose_col(df, ["elapsed_sec", "wallclock_elapsed_s", "elapsed_s", "time_s", "elapsed_seconds", "runtime_sec"])
    if time_col is None:
        return None, "missing_elapsed_column:" + "|".join(map(str, df.columns)), {}

    meta = read_metadata_near_trace(trace_path)

    out_rows = []
    for _, r in df.iterrows():
        rr = dict(r)
        rr["_elapsed_sec"] = as_float(r.get(time_col))
        comps, missing = exact_components(r, meta)

        for c, v in comps.items():
            rr[f"raw__{c}"] = v
            rr[f"contrib__{c}"] = v * W_EXACT[c] if np.isfinite(v) else np.nan

        reconstructed = score_from_components(comps)
        logged_exact = row_value(r, ["best_exact_score", "exact_score"])

        # Prefer reconstructed exact-no-pressure if complete. Fall back to logged exact only when reconstruction fails.
        if np.isfinite(reconstructed):
            rr["exact_nopressure_score"] = reconstructed
            rr["score_source"] = "reconstructed_components"
        elif np.isfinite(logged_exact):
            rr["exact_nopressure_score"] = logged_exact
            rr["score_source"] = "FALLBACK_logged_exact_score"
        else:
            rr["exact_nopressure_score"] = np.nan
            rr["score_source"] = "missing_score"

        rr["missing_exact_components"] = missing
        out_rows.append(rr)

    edf = pd.DataFrame(out_rows)
    edf = edf[pd.to_numeric(edf["_elapsed_sec"], errors="coerce").notna()].copy()
    edf = edf.sort_values("_elapsed_sec").reset_index(drop=True)
    if edf.empty:
        return None, "no_valid_elapsed", meta

    return edf, "ok", meta


def best_so_far(df: pd.DataFrame, t: float, strict: bool = True) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if strict and float(df["_elapsed_sec"].max()) < float(t):
        return None
    use = df[df["_elapsed_sec"] <= float(t)].copy()
    use = use[pd.to_numeric(use["exact_nopressure_score"], errors="coerce").notna()]
    if use.empty:
        return None
    idx = pd.to_numeric(use["exact_nopressure_score"], errors="coerce").idxmin()
    return use.loc[idx]


def case_name(n: int, seed: int) -> str:
    return f"case_{int(n)}_seed{int(seed)}"


def find_off_trace(off_root: Path, dataset: str, n: int, scenario: str, seed: int) -> Optional[Path]:
    cname = case_name(n, seed)

    direct_patterns = [
        off_root / dataset / f"n{n}" / scenario / cname,
        off_root / dataset / scenario / cname,
        off_root / f"n{n}" / scenario / cname,
        off_root / scenario / cname,
        off_root / cname,
    ]

    case_dirs = []
    for p in direct_patterns:
        if p.exists():
            case_dirs.append(p)

    if not case_dirs and off_root.exists():
        hits = [p for p in off_root.rglob(cname) if p.is_dir()]
        scored = []
        for h in hits:
            parts = [str(x).lower() for x in h.parts]
            score = 0
            if dataset.lower() in parts:
                score += 4
            if scenario.lower() in parts:
                score += 4
            if f"n{n}".lower() in parts:
                score += 2
            scored.append((score, len(str(h)), h))
        scored.sort(key=lambda x: (-x[0], x[1]))
        case_dirs = [x[2] for x in scored]

    for cd in case_dirs:
        traces = list(cd.rglob("spiral_trace.csv"))
        # Prefer spiral_off / off traces, otherwise any trace.
        traces_sorted = sorted(
            traces,
            key=lambda p: (
                0 if ("spiral_off" in str(p).lower() or "price_off" in str(p).lower() or "off" in str(p).lower()) else 1,
                len(str(p))
            )
        )
        if traces_sorted:
            return traces_sorted[0]

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=r"v3_trace_audit\v3_trace_inventory.csv")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--off-root", default="spiral_price_off_all_results_v2")
    ap.add_argument("--output-dir", default="recourse_recalibration_collected_v3_inventory")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--t-main", type=float, default=60.0)
    ap.add_argument("--t-grid", nargs="+", type=float, default=[15, 30, 60, 90, 120])
    args = ap.parse_args()

    repo = Path(args.repo_root)
    inv_path = Path(args.inventory)
    if not inv_path.exists():
        raise FileNotFoundError(inv_path)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(inv_path)
    inv = inv[
        inv["dataset"].isin(args.datasets)
        & inv["n"].astype(int).isin(args.sizes)
        & inv["scenario"].isin(args.scenarios)
        & inv["seed"].astype(int).isin(args.seeds)
        & inv["arm"].isin(args.arms)
    ].copy()

    inv["present"] = inv["present"].astype(str).str.lower().isin(["true", "1", "yes"])
    inv = inv[inv["present"]].copy()

    off_root = Path(args.off_root)

    detailed_rows = []
    coverage_rows = []
    error_rows = []
    coldiag_rows = []

    # Cache off traces because each off run is reused for 5 price arms.
    off_cache: Dict[Tuple[str, int, str, int], Tuple[Optional[pd.DataFrame], str, Optional[Path]]] = {}

    for _, pr in inv.iterrows():
        dataset = str(pr["dataset"])
        n = int(pr["n"])
        arm = str(pr["arm"])
        scenario = str(pr["scenario"])
        seed = int(pr["seed"])

        base = {
            "dataset": dataset,
            "n": n,
            "arm": arm,
            "scenario": scenario,
            "seed": seed,
        }

        price_trace = resolve_existing_path(str(pr["trace_path"]), repo)
        if price_trace is None:
            error_rows.append({**base, "error": "price_trace_path_not_found", "price_trace_path": str(pr["trace_path"])})
            continue

        price_df, price_status, _ = enrich_trace(price_trace)
        coldiag_rows.append({
            **base,
            "method": "price",
            "trace_path": str(price_trace),
            "status": price_status,
            "columns": "" if price_df is None else "|".join(map(str, price_df.columns)),
            "tmax": np.nan if price_df is None else float(price_df["_elapsed_sec"].max()),
        })

        if price_df is None:
            error_rows.append({**base, "error": f"price_trace_{price_status}", "price_trace_path": str(price_trace)})
            continue

        off_key = (dataset, n, scenario, seed)
        if off_key not in off_cache:
            off_trace = find_off_trace(off_root, dataset, n, scenario, seed)
            if off_trace is None:
                off_cache[off_key] = (None, "off_trace_not_found", None)
            else:
                off_df, off_status, _ = enrich_trace(off_trace)
                off_cache[off_key] = (off_df, off_status, off_trace)

        off_df, off_status, off_trace = off_cache[off_key]
        coldiag_rows.append({
            **base,
            "method": "off",
            "trace_path": "" if off_trace is None else str(off_trace),
            "status": off_status,
            "columns": "" if off_df is None else "|".join(map(str, off_df.columns)),
            "tmax": np.nan if off_df is None else float(off_df["_elapsed_sec"].max()),
        })

        if off_df is None:
            error_rows.append({**base, "error": f"off_trace_{off_status}", "off_trace_path": "" if off_trace is None else str(off_trace)})
            continue

        for t in args.t_grid:
            coverage_rows.append({
                **base,
                "t_seconds": t,
                "price_tmax": float(price_df["_elapsed_sec"].max()),
                "off_tmax": float(off_df["_elapsed_sec"].max()),
                "valid_strict": bool(float(price_df["_elapsed_sec"].max()) >= t and float(off_df["_elapsed_sec"].max()) >= t),
            })

        price_best = best_so_far(price_df, args.t_main, strict=True)
        off_best = best_so_far(off_df, args.t_main, strict=True)

        if price_best is None or off_best is None:
            error_rows.append({
                **base,
                "error": "not_covered_at_t_main_or_missing_score",
                "price_tmax": float(price_df["_elapsed_sec"].max()),
                "off_tmax": float(off_df["_elapsed_sec"].max()),
                "price_trace_path": str(price_trace),
                "off_trace_path": "" if off_trace is None else str(off_trace),
            })
            continue

        pscore = float(price_best["exact_nopressure_score"])
        oscore = float(off_best["exact_nopressure_score"])

        row = {
            **base,
            "t_seconds": args.t_main,
            "price_exact_nopressure_score": pscore,
            "off_exact_nopressure_score": oscore,
            "gap_exact_nopressure": oscore - pscore,
            "price_better": oscore - pscore > 1e-9,
            "off_better": oscore - pscore < -1e-9,
            "tie": abs(oscore - pscore) <= 1e-9,
            "price_score_source": price_best.get("score_source", ""),
            "off_score_source": off_best.get("score_source", ""),
            "price_missing_exact_components": price_best.get("missing_exact_components", ""),
            "off_missing_exact_components": off_best.get("missing_exact_components", ""),
            "price_elapsed_selected": float(price_best["_elapsed_sec"]),
            "off_elapsed_selected": float(off_best["_elapsed_sec"]),
            "price_tmax": float(price_df["_elapsed_sec"].max()),
            "off_tmax": float(off_df["_elapsed_sec"].max()),
            "price_trace_path": str(price_trace),
            "off_trace_path": "" if off_trace is None else str(off_trace),
        }

        for c in COMPONENT_ORDER:
            praw = as_float(price_best.get(f"raw__{c}"))
            oraw = as_float(off_best.get(f"raw__{c}"))
            row[f"price_raw__{c}"] = praw
            row[f"off_raw__{c}"] = oraw
            row[f"gap_raw__{c}_off_minus_price"] = oraw - praw if np.isfinite(oraw) and np.isfinite(praw) else np.nan
            row[f"gap_contrib__{c}_off_minus_price"] = (oraw - praw) * W_EXACT[c] if np.isfinite(oraw) and np.isfinite(praw) else np.nan

        row["sum_contrib"] = sum(as_float(row.get(f"gap_contrib__{c}_off_minus_price"), 0.0) for c in COMPONENT_ORDER)
        row["sum_contrib_minus_gap"] = row["sum_contrib"] - row["gap_exact_nopressure"]
        detailed_rows.append(row)

    detailed = pd.DataFrame(detailed_rows)
    coverage = pd.DataFrame(coverage_rows)
    errors = pd.DataFrame(error_rows)
    coldiag = pd.DataFrame(coldiag_rows)

    detailed.to_csv(outdir / "final_pairwise_iso60_detailed.csv", index=False)
    coverage.to_csv(outdir / "coverage_by_t.csv", index=False)
    errors.to_csv(outdir / "collector_errors.csv", index=False)
    coldiag.to_csv(outdir / "collector_column_diagnostics.csv", index=False)

    verdict_rows = []
    comp_rows = []

    if len(detailed):
        for (dataset, n, arm), g in detailed.groupby(["dataset", "n", "arm"], dropna=False):
            gaps = pd.to_numeric(g["gap_exact_nopressure"], errors="coerce").dropna()
            if len(gaps) == 0:
                continue

            pval = np.nan
            if wilcoxon is not None and len(gaps) >= 2 and np.any(np.abs(gaps.to_numpy()) > 1e-9):
                try:
                    pval = float(wilcoxon(gaps, alternative="greater").pvalue)
                except Exception:
                    pval = np.nan

            verdict_rows.append({
                "dataset": dataset,
                "n": n,
                "arm": arm,
                "num_cases": int(len(gaps)),
                "price_exact_wins": int((gaps > 1e-9).sum()),
                "off_exact_wins": int((gaps < -1e-9).sum()),
                "ties": int((gaps.abs() <= 1e-9).sum()),
                "price_win_rate": float((gaps > 1e-9).mean()),
                "mean_gap_exact_nopressure_iso60": float(gaps.mean()),
                "median_gap_exact_nopressure_iso60": float(gaps.median()),
                "wilcoxon_p_greater": pval,
            })

            crow = {"dataset": dataset, "n": n, "arm": arm}
            total = 0.0
            for c in COMPONENT_ORDER:
                col = f"gap_contrib__{c}_off_minus_price"
                val = float(pd.to_numeric(g[col], errors="coerce").mean()) if col in g.columns else np.nan
                crow[f"contrib_{c}"] = val
                if np.isfinite(val):
                    total += val
            crow["sum_contrib"] = total
            crow["mean_gap_exact_nopressure_iso60"] = float(gaps.mean())
            crow["sum_contrib_minus_mean_gap"] = total - float(gaps.mean())
            comp_rows.append(crow)

    verdict = pd.DataFrame(verdict_rows)
    comps = pd.DataFrame(comp_rows)

    verdict.to_csv(outdir / "final_verdict_iso60.csv", index=False)
    comps.to_csv(outdir / "final_component_decomposition_iso60.csv", index=False)

    print("\nSaved outputs under:", outdir)
    print("Detailed cases:", len(detailed))
    print("Errors:", len(errors))
    if len(verdict):
        print("\n=== final_verdict_iso60 ===")
        print(verdict.to_string(index=False))
    if len(errors):
        print("\n=== first collector errors ===")
        print(errors.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

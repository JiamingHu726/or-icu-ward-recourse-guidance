#!/usr/bin/env python3
from __future__ import annotations

"""
collect_publication_results.py

Collect publication-level results from scenario folders and map internal method
folders to manuscript-ready method names.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from publication_experiment_config import (
    PUBLICATION_SCENARIOS,
    PUBLICATION_RESULT_ROOT,
    PUBLICATION_TABLE_ROOT,
    PUBLIC_METHOD_LABELS,
    PUBLIC_METHOD_SHORT_LABELS,
    METHOD_CATEGORIES,
)


def _parse_case_name(case_name: str):
    m = re.match(r"case_(\d+)_seed(\d+)", case_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _read_csv_row(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception:
        return {}


def _read_json(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _method_specs(case_dir: Path):
    return [
        {
            "internal": "01_ba_hla_v41",
            "stage2_summary": case_dir / "01_ba_hla_v41" / "evaluation" / "summary.csv",
            "stage3_summary": None,
            "status_dir": case_dir / "01_ba_hla_v41",
        },
        {
            "internal": "02_stage2_v3_volume_first",
            "stage2_summary": case_dir / "02_stage2_v3_volume_first" / "evaluation" / "summary.csv",
            "stage3_summary": case_dir / "03_stage3_on_stage2_v3" / "stage3_summary.csv",
            "status_dir": case_dir / "02_stage2_v3_volume_first",
        },
        {
            "internal": "04_hp_forced_ot_feedback",
            "stage2_summary": case_dir / "04_hp_forced_ot_feedback" / "evaluation" / "summary.csv",
            "stage3_summary": case_dir / "05_stage3_on_hp_forced_ot_feedback" / "stage3_summary.csv",
            "status_dir": case_dir / "04_hp_forced_ot_feedback",
        },
        {
            "internal": "07_v41_weak_pr_glns",
            "stage2_summary": case_dir / "07_v41_weak_pr_glns" / "evaluation" / "summary.csv",
            "stage3_summary": case_dir / "07_v41_weak_pr_glns" / "stage3_results" / "stage3_summary.csv",
            "status_dir": case_dir / "07_v41_weak_pr_glns",
        },
        {
            "internal": "06_downstream_aggressive_spiral_pr_glns",
            "stage2_summary": case_dir / "06_downstream_aggressive_spiral_pr_glns" / "evaluation" / "summary.csv",
            "stage3_summary": case_dir / "06_downstream_aggressive_spiral_pr_glns" / "stage3_results" / "stage3_summary.csv",
            "status_dir": case_dir / "06_downstream_aggressive_spiral_pr_glns",
        },
        {
            "internal": "10_shehadeh_adaptive_access",
            "stage2_summary": case_dir / "10_shehadeh_adaptive_access" / "evaluation" / "summary.csv",
            "stage3_summary": case_dir / "11_stage3_on_shehadeh_adaptive_access" / "stage3_summary.csv",
            "status_dir": case_dir / "10_shehadeh_adaptive_access",
        },
        {
            "internal": "10_shehadeh_adaptive_free",
            "stage2_summary": case_dir / "10_shehadeh_adaptive_free" / "evaluation" / "summary.csv",
            "stage3_summary": case_dir / "11_stage3_on_shehadeh_adaptive_free" / "stage3_summary.csv",
            "status_dir": case_dir / "10_shehadeh_adaptive_free",
        },
    ]


def collect_one_case(scenario: str, case_dir: Path) -> list[dict]:
    n, seed = _parse_case_name(case_dir.name)
    rows = []
    for spec in _method_specs(case_dir):
        st2 = _read_csv_row(spec["stage2_summary"])
        st3 = _read_csv_row(spec["stage3_summary"])
        status = _read_json(spec["status_dir"] / "run_status.json")
        has_any_summary = bool(st2) or bool(st3)
        internal = spec["internal"]
        row = {
            "scenario": scenario,
            "case": case_dir.name,
            "n": n,
            "seed": seed,
            "method_internal": internal,
            "method": PUBLIC_METHOD_LABELS.get(internal, internal),
            "method_short": PUBLIC_METHOD_SHORT_LABELS.get(internal, internal),
            "category": METHOD_CATEGORIES.get(internal, ""),
            "status": status.get("status", "success" if has_any_summary else "missing"),
            "error": status.get("error", ""),
            "fallback_used": status.get("fallback_used", ""),
            "chosen_config": status.get("chosen_config", ""),
            "chosen_warm_variant": status.get("chosen_warm_variant", ""),
            "n_scheduled": st2.get("n_scheduled"),
            "n_high_priority_scheduled": st2.get("n_high_priority_scheduled"),
            "n_high_priority_postponed": st2.get("n_high_priority_postponed"),
            "execution_violations": st2.get("violation_count"),
            "or_overtime_min": st2.get("or_overtime_min"),
            "eval_blocked_transfer_days": st2.get("blocked_transfer_patient_days"),
            "eval_icu_excess": st2.get("icu_excess_bed_days_blocking"),
            "eval_ward_excess": st2.get("ward_excess_bed_days_blocking"),
            "eval_peak_blocked": st2.get("peak_icu_ready_blocked"),
            "blocked_transfer_patient_days": st3.get("blocked_transfer_patient_days_stage3"),
            "icu_excess_bed_days": st3.get("icu_excess_bed_days_stage3"),
            "ward_excess_bed_days": st3.get("ward_excess_bed_days_stage3"),
            "peak_icu_ready_blocked": st3.get("peak_icu_ready_blocked_stage3"),
            "n_transfer_candidates": st3.get("n_transfer_candidates"),
        }
        rows.append(row)
    return rows


def make_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "n_scheduled",
        "n_high_priority_scheduled",
        "n_high_priority_postponed",
        "execution_violations",
        "or_overtime_min",
        "blocked_transfer_patient_days",
        "icu_excess_bed_days",
        "ward_excess_bed_days",
        "peak_icu_ready_blocked",
    ]
    work = df.copy()
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    rows = []
    group_cols = ["scenario", "n", "method", "method_short", "category"]
    for keys, g in work.groupby(group_cols, dropna=False):
        scenario, n, method, method_short, category = keys
        row = {
            "scenario": scenario,
            "n": n,
            "method": method,
            "method_short": method_short,
            "category": category,
            "num_cases": int(g["case"].nunique()),
            "num_success_rows": int((g["status"].fillna("success") == "success").sum()),
            "num_missing_rows": int((g["status"].fillna("") == "missing").sum()),
            "num_failed_rows": int((g["status"].fillna("").str.contains("failed", case=False)).sum()),
            "num_fallback": int((g["fallback_used"].astype(str).str.lower() == "true").sum()) if "fallback_used" in g else 0,
        }
        for c in numeric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            row[f"{c}_count"] = int(vals.count())
            row[f"{c}_mean"] = float(vals.mean()) if not vals.empty else None
            row[f"{c}_std"] = float(vals.std()) if vals.count() >= 2 else None
            row[f"{c}_min"] = float(vals.min()) if not vals.empty else None
            row[f"{c}_max"] = float(vals.max()) if not vals.empty else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "n", "category", "method"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Collect publication-level batch results.")
    parser.add_argument("--result-root", default=str(PUBLICATION_RESULT_ROOT))
    parser.add_argument("--scenarios", nargs="+", default=PUBLICATION_SCENARIOS)
    parser.add_argument("--output-dir", default=str(PUBLICATION_TABLE_ROOT))
    parser.add_argument("--detailed-output", default="publication_results_detailed.csv")
    parser.add_argument("--aggregate-output", default="publication_results_aggregate.csv")
    args = parser.parse_args()

    root = Path(args.result_root)
    rows = []
    for scenario in args.scenarios:
        scen_dir = root / scenario
        for case_dir in sorted(scen_dir.glob("case_*_seed*")):
            if case_dir.is_dir():
                rows.extend(collect_one_case(scenario, case_dir))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(rows)
    detailed_path = out_dir / args.detailed_output
    detailed.to_csv(detailed_path, index=False)

    aggregate = make_aggregate(detailed) if not detailed.empty else pd.DataFrame()
    aggregate_path = out_dir / args.aggregate_output
    aggregate.to_csv(aggregate_path, index=False)

    print(f"Saved detailed results: {detailed_path}")
    print(f"Saved aggregate results: {aggregate_path}")
    if not aggregate.empty:
        show_cols = [
            "scenario", "n", "method", "num_cases", "num_success_rows", "num_fallback",
            "n_scheduled_mean", "n_high_priority_scheduled_mean",
            "execution_violations_mean", "or_overtime_min_mean",
            "blocked_transfer_patient_days_mean", "icu_excess_bed_days_mean",
        ]
        show_cols = [c for c in show_cols if c in aggregate.columns]
        print(aggregate[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

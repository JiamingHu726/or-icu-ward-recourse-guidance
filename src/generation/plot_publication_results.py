#!/usr/bin/env python3
from __future__ import annotations

"""
plot_publication_results.py

Generate manuscript-ready diagnostic figures from publication_results_detailed.csv.
The script uses matplotlib only and writes both PNG and PDF files.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from publication_experiment_config import PUBLICATION_FIGURE_ROOT, PUBLIC_METHOD_ORDER, SCENARIO_LABELS


def _savefig(out_root: Path, name: str):
    out_root.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_root / f"{name}.png", dpi=300)
    plt.savefig(out_root / f"{name}.pdf")
    plt.close()


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "n", "seed", "n_scheduled", "n_high_priority_scheduled", "execution_violations",
        "or_overtime_min", "blocked_transfer_patient_days", "icu_excess_bed_days",
        "ward_excess_bed_days", "peak_icu_ready_blocked",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    present = list(df["method"].dropna().unique())
    out = [m for m in PUBLIC_METHOD_ORDER if m in present]
    out += [m for m in present if m not in out]
    return out


def plot_stress_blocking(df: pd.DataFrame, out_root: Path):
    methods = [
        "Execution-repaired baseline",
        "Recourse-feedback repair",
        "Recourse-guided LNS",
        "Adaptive integrated MIP",
    ]
    d = df[df["method"].isin(methods)].dropna(subset=["blocked_transfer_patient_days"]).copy()
    if d.empty:
        return
    agg = d.groupby(["scenario", "method"], as_index=False)["blocked_transfer_patient_days"].mean()
    scenarios = [s for s in ["nominal", "ward_pressure", "transfer_bottleneck"] if s in agg["scenario"].unique()]
    x = range(len(scenarios))
    plt.figure(figsize=(7.2, 4.6))
    for method in methods:
        g = agg[agg["method"] == method]
        if g.empty:
            continue
        vals = []
        for s in scenarios:
            row = g[g["scenario"] == s]
            vals.append(float(row["blocked_transfer_patient_days"].iloc[0]) if not row.empty else None)
        plt.plot(list(x), vals, marker="o", label=method)
    plt.xticks(list(x), [SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=15, ha="right")
    plt.ylabel("Mean blocked transfer patient-days")
    plt.xlabel("Scenario")
    plt.title("Downstream blocking under ward-transfer stress")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    _savefig(out_root, "fig_stress_blocked_transfer")


def plot_nominal_ablation(df: pd.DataFrame, out_root: Path):
    methods = [
        "Execution-repaired baseline",
        "Recourse-feedback repair",
        "Unguided LNS control",
        "Recourse-guided LNS",
    ]
    d = df[(df["scenario"] == "nominal") & df["method"].isin(methods)].dropna(subset=["blocked_transfer_patient_days"]).copy()
    if d.empty:
        return
    agg = d.groupby("method", as_index=False)[["blocked_transfer_patient_days", "n_high_priority_scheduled"]].mean()
    order = [m for m in methods if m in agg["method"].values]
    vals = [float(agg.loc[agg["method"] == m, "blocked_transfer_patient_days"].iloc[0]) for m in order]
    plt.figure(figsize=(7.4, 4.6))
    plt.bar(range(len(order)), vals)
    plt.xticks(range(len(order)), order, rotation=20, ha="right")
    plt.ylabel("Mean blocked transfer patient-days")
    plt.title("Ablation of recourse feedback and LNS guidance")
    plt.grid(True, axis="y", alpha=0.3)
    _savefig(out_root, "fig_nominal_ablation_blocked_transfer")


def plot_tradeoff(df: pd.DataFrame, out_root: Path, scenario: str = "nominal"):
    d = df[(df["scenario"] == scenario)].dropna(subset=["execution_violations", "blocked_transfer_patient_days"]).copy()
    if d.empty:
        return
    methods = _ordered_methods(d)
    plt.figure(figsize=(7.2, 4.8))
    for method in methods:
        g = d[d["method"] == method]
        if g.empty:
            continue
        plt.scatter(
            g["execution_violations"],
            g["blocked_transfer_patient_days"],
            s=45,
            alpha=0.75,
            label=method,
        )
    plt.xlabel("Execution violations")
    plt.ylabel("Blocked transfer patient-days")
    plt.title(f"Execution feasibility and downstream blocking ({SCENARIO_LABELS.get(scenario, scenario)})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    _savefig(out_root, f"fig_tradeoff_violation_blocking_{scenario}")


def plot_overtime_tradeoff(df: pd.DataFrame, out_root: Path, scenario: str = "nominal"):
    d = df[(df["scenario"] == scenario)].dropna(subset=["or_overtime_min", "blocked_transfer_patient_days"]).copy()
    if d.empty:
        return
    methods = _ordered_methods(d)
    plt.figure(figsize=(7.2, 4.8))
    for method in methods:
        g = d[d["method"] == method]
        if g.empty:
            continue
        plt.scatter(g["or_overtime_min"], g["blocked_transfer_patient_days"], s=45, alpha=0.75, label=method)
    plt.xlabel("OR overtime minutes")
    plt.ylabel("Blocked transfer patient-days")
    plt.title(f"OR overtime and downstream blocking ({SCENARIO_LABELS.get(scenario, scenario)})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    _savefig(out_root, f"fig_tradeoff_overtime_blocking_{scenario}")


def main():
    parser = argparse.ArgumentParser(description="Plot publication-level OR-ICU-Ward results.")
    parser.add_argument("--input", default="publication_tables/publication_results_detailed.csv")
    parser.add_argument("--output-root", default=str(PUBLICATION_FIGURE_ROOT))
    args = parser.parse_args()

    df = _load(Path(args.input))
    out = Path(args.output_root)
    plot_stress_blocking(df, out)
    plot_nominal_ablation(df, out)
    for scenario in sorted(df["scenario"].dropna().unique()):
        plot_tradeoff(df, out, scenario=scenario)
        plot_overtime_tradeoff(df, out, scenario=scenario)
    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()

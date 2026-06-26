#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild Figure M1 with seed-clustered inference.

Why this version exists
-----------------------
The original M1 treated nominal and transfer-bottleneck outcomes from the
same base seed as two independent observations. This script first averages
those two matched scenarios within each (dataset, n, seed) cluster. It then
plots ten seed-level paired effects per cell.

Input
-----
Either:
  * A_effect_inferential_stats.zip containing A_effect_input_standardized.csv, or
  * A CSV with columns: dataset, n, scenario, seed, gap.

Gap convention
--------------
  gap = F(no-price guidance) - F(recourse-guided LNS)
  gap > 0 means recourse guidance is better.

Outputs
-------
  M1_seed_clustered_load_response.pdf
  M1_seed_clustered_load_response.png
  M1_seed_clustered_summary.csv
  M1_seed_clustered_trend_tests.csv

The figure intentionally:
  * removes red/green regime shading;
  * removes per-cell significance stars;
  * removes scenario-level win-rate labels;
  * labels the two duration-derived benchmark families as confirmatory;
  * retains Synthetic as an exploratory stress probe, not confirmatory trend evidence.
"""

from __future__ import annotations

import argparse
import io
import math
import zipfile
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import rankdata
except ImportError as exc:
    raise SystemExit("This script requires scipy. Install it with: pip install scipy") from exc


REQUIRED_COLUMNS = {"dataset", "n", "scenario", "seed", "gap"}
DATASET_ORDER = ["GermanOR", "Mannino", "Synthetic"]
PANEL_TITLES = {
    "GermanOR": "GermanOR (duration-derived)",
    "Mannino": "Mannino (duration-derived)",
    "Synthetic": "Synthetic stress probe (exploratory)",
}
PANEL_LABELS = {"GermanOR": "(a)", "Mannino": "(b)", "Synthetic": "(c)"}


def read_input(path: Path) -> pd.DataFrame:
    """Read the standardized A-effect table from a CSV or ZIP archive."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [
                name for name in zf.namelist()
                if Path(name).name == "A_effect_input_standardized.csv"
            ]
            if not candidates:
                raise FileNotFoundError(
                    "The ZIP does not contain A_effect_input_standardized.csv."
                )
            with zf.open(candidates[0]) as fh:
                df = pd.read_csv(io.BytesIO(fh.read()))
    else:
        df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(int)
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["gap"] = pd.to_numeric(df["gap"], errors="raise").astype(float)
    return df


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of independent seed clusters."""
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def exact_page_test(block_table: pd.DataFrame) -> tuple[float, float, str]:
    """
    Exact one-sided Page ordered-trend test.

    Rows are independent base seeds and columns are ordered n levels. If a seed
    has tied within-seed values, this falls back to a deterministic Monte Carlo
    permutation test because the no-tie exact convolution is no longer valid.
    """
    arr = block_table.to_numpy(dtype=float)
    n_blocks, k = arr.shape
    ranks = np.vstack([rankdata(row, method="average") for row in arr])
    weights = np.arange(1, k + 1, dtype=float)
    observed = float(np.sum(ranks.sum(axis=0) * weights))

    tied = any(len(np.unique(row)) < k for row in ranks)
    if tied:
        rng = np.random.default_rng(20260625)
        n_perm = 200_000
        ge = 0
        for _ in range(n_perm):
            permuted = np.vstack([rng.permutation(row) for row in ranks])
            stat = float(np.sum(permuted.sum(axis=0) * weights))
            ge += int(stat >= observed - 1e-12)
        p = (ge + 1) / (n_perm + 1)
        return observed, p, f"Monte Carlo permutation ({n_perm:,} draws)"

    # With no within-seed ties, each seed has the same 4! permutation support.
    # Build the exact null distribution by dynamic-programming convolution.
    per_block = Counter()
    for perm in permutations(range(1, k + 1)):
        stat = int(sum(w * r for w, r in zip(weights, perm)))
        per_block[stat] += 1

    dist: Counter[int] = Counter({0: 1})
    for _ in range(n_blocks):
        next_dist: Counter[int] = Counter()
        for a, count_a in dist.items():
            for b, count_b in per_block.items():
                next_dist[a + b] += count_a * count_b
        dist = next_dist

    total = sum(dist.values())
    p = sum(count for stat, count in dist.items() if stat >= observed - 1e-12) / total
    return observed, float(p), "exact permutation"


def p_label(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild M1 using seed-clustered inference.")
    parser.add_argument("--input", required=True, help="A-effect ZIP or standardized CSV input.")
    parser.add_argument("--output-dir", default="M1_seed_clustered", help="Output directory.")
    parser.add_argument("--bootstrap-reps", type=int, default=20_000, help="Bootstrap draws per cell.")
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_input(source)
    raw = raw[raw["dataset"].isin(DATASET_ORDER)].copy()

    # The primary inferential unit is the base seed. Each seed contributes one
    # paired effect per n after averaging its nominal and transfer-bottleneck
    # scenario outcomes.
    expected_scenarios = {"nominal", "transfer_bottleneck"}
    primary = raw[raw["scenario"].isin(expected_scenarios)].copy()
    coverage = primary.groupby(["dataset", "n", "seed"])["scenario"].nunique()
    incomplete = coverage[coverage != 2]
    if not incomplete.empty:
        raise ValueError(
            "Each seed cluster must contain exactly nominal and transfer_bottleneck. "
            f"Incomplete clusters: {incomplete.to_dict()}"
        )

    clustered = (
        primary.groupby(["dataset", "n", "seed"], as_index=False)["gap"]
        .mean()
        .rename(columns={"gap": "seed_cluster_gap"})
    )

    rng = np.random.default_rng(20260625)
    summary_rows = []
    trend_rows = []
    n_levels_global = sorted(clustered["n"].unique())

    for dataset in DATASET_ORDER:
        data_ds = clustered[clustered["dataset"] == dataset].copy()
        if data_ds.empty:
            continue
        n_levels = sorted(data_ds["n"].unique())
        if n_levels != n_levels_global:
            raise ValueError(f"Unexpected n-level coverage for {dataset}: {n_levels}")

        pivot = data_ds.pivot(index="seed", columns="n", values="seed_cluster_gap").sort_index(axis=1)
        if pivot.isna().any().any():
            raise ValueError(f"Missing seed/n value in {dataset}.")
        page_stat, page_p, page_method = exact_page_test(pivot)
        positive_slopes = 0
        for _, row in pivot.iterrows():
            slope = np.polyfit(np.asarray(n_levels, dtype=float), row.to_numpy(dtype=float), 1)[0]
            positive_slopes += int(slope > 0)

        trend_rows.append({
            "dataset": dataset,
            "n_seed_clusters": len(pivot),
            "n_levels": "|".join(map(str, n_levels)),
            "page_statistic": page_stat,
            "page_p_one_sided": page_p,
            "page_method": page_method,
            "positive_seed_level_linear_slopes": positive_slopes,
            "negative_or_zero_seed_level_linear_slopes": len(pivot) - positive_slopes,
        })

        for n in n_levels:
            values = data_ds.loc[data_ds["n"] == n, "seed_cluster_gap"].to_numpy(dtype=float)
            ci_lo, ci_hi = bootstrap_median_ci(values, rng, args.bootstrap_reps)
            summary_rows.append({
                "dataset": dataset,
                "n": n,
                "n_seed_clusters": len(values),
                "median_seed_cluster_gap": float(np.median(values)),
                "mean_seed_cluster_gap": float(np.mean(values)),
                "bootstrap_ci_low": float(ci_lo),
                "bootstrap_ci_high": float(ci_hi),
                "positive_seed_clusters": int(np.sum(values > 0)),
                "zero_seed_clusters": int(np.sum(values == 0)),
                "negative_seed_clusters": int(np.sum(values < 0)),
            })

    summary = pd.DataFrame(summary_rows)
    trends = pd.DataFrame(trend_rows)
    summary.to_csv(out_dir / "M1_seed_clustered_summary.csv", index=False)
    trends.to_csv(out_dir / "M1_seed_clustered_trend_tests.csv", index=False)
    clustered.to_csv(out_dir / "M1_seed_clustered_effects.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), constrained_layout=True)
    fig.supxlabel(r"Candidate-pool size $n$")
    fig.supylabel(r"Paired exact-score difference $\Delta F$ (↑ guidance better)")

    for ax, dataset in zip(axes, DATASET_ORDER):
        data_ds = clustered[clustered["dataset"] == dataset].copy()
        stats_ds = summary[summary["dataset"] == dataset].sort_values("n")
        trend = trends[trends["dataset"] == dataset].iloc[0]
        n_levels = stats_ds["n"].to_numpy(dtype=float)

        med = stats_ds["median_seed_cluster_gap"].to_numpy(dtype=float)
        lo = stats_ds["bootstrap_ci_low"].to_numpy(dtype=float)
        hi = stats_ds["bootstrap_ci_high"].to_numpy(dtype=float)
        # Draw the seed-cluster summary first, then reuse its automatically
        # assigned Matplotlib color for the underlying dots. This keeps each
        # panel visually coherent without hard-coding a palette.
        err = ax.errorbar(
            n_levels, med, yerr=np.vstack([med - lo, hi - med]),
            fmt="o-", capsize=3, linewidth=1.5
        )
        panel_color = err[0].get_color()

        # Underlying independent seed clusters: one dot per seed after the two
        # paired scenarios have been averaged. A fixed jitter keeps dots visible.
        for n in n_levels:
            vals = data_ds.loc[data_ds["n"] == n, "seed_cluster_gap"].to_numpy(dtype=float)
            jitter = np.linspace(-2.1, 2.1, len(vals))
            ax.scatter(
                np.full(len(vals), n) + jitter, vals, s=24, alpha=0.50,
                color=panel_color
            )

        ax.axhline(0.0, linestyle="--", linewidth=0.9)
        ax.set_xticks(n_levels)
        ax.set_title(PANEL_TITLES[dataset])
        ax.text(0.02, 1.02, PANEL_LABELS[dataset], transform=ax.transAxes, fontweight="bold")

        if dataset == "Synthetic":
            abs_vals = np.abs(data_ds["seed_cluster_gap"].to_numpy(dtype=float))
            nonzero = abs_vals[abs_vals > 0]
            linthresh = 10.0 ** math.floor(math.log10(max(1.0, float(np.quantile(nonzero, 0.25)))))
            ax.set_yscale("symlog", linthresh=linthresh)
            annotation = (
                "Exploratory stress probe\n"
                f"No stable ordered trend ({p_label(float(trend['page_p_one_sided']))})"
            )
        else:
            annotation = (
                "Seed-clustered ordered trend\n"
                f"Page test: {p_label(float(trend['page_p_one_sided']))}"
            )
        ax.text(0.04, 0.96, annotation, transform=ax.transAxes, va="top")

    pdf_path = out_dir / "M1_seed_clustered_load_response.pdf"
    png_path = out_dir / "M1_seed_clustered_load_response.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    for path in [pdf_path, png_path, out_dir / "M1_seed_clustered_summary.csv", out_dir / "M1_seed_clustered_trend_tests.csv"]:
        print(" ", path)
    print("\nSeed-clustered summary:")
    print(summary.to_string(index=False))
    print("\nOrdered-trend tests:")
    print(trends.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

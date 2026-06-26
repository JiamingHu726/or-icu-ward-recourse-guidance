#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_M2_seed_clustered_mechanism.py

Generate the revised M2 figure for the EJOR OR--ICU--ward manuscript.

This revision fixes three issues in the earlier M2:
  1) nominal and transfer-bottleneck scenarios are averaged within base seed
     before any cross-seed summary is computed;
  2) component-wise medians are displayed as a heatmap, not stacked bars;
  3) the n=150 coupling panel reports blocked-transfer relief versus the
     high-priority-access deficit, rather than blocked transfers versus
     execution violations.

Required source
---------------
The script needs ``all_standardized_effect_rows_v21.csv``. It can be supplied
as a CSV file, or located recursively inside a folder or ZIP archive.

The expected records contain all three data families and the primary scenarios
``nominal`` and ``transfer_bottleneck``. Their component columns must use the
prefix ``gap_component__exact_``.

Outputs
-------
M2_mechanism_decomposition_and_access_tradeoff.pdf
M2_mechanism_decomposition_and_access_tradeoff.png
M2_seed_clustered_component_summary.csv
M2_seed_clustered_scatter_points_n150.csv
M2_seed_cluster_validation.csv
M2_figure_notes.txt

Usage (PowerShell)
------------------
$PY = "E:\\anaconda3\\python.exe"
& $PY make_M2_seed_clustered_mechanism.py `
  --input-root "synthetic_tail_mechanism_diagnostics_v22 (2)(1).zip" `
  --output-dir "M2_seed_clustered"

The script is read-only with respect to experiment inputs.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------
# Figure constants
# ---------------------------------------------------------------------

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATASET_ORDER = ["GermanOR", "Mannino", "Synthetic"]
N_ORDER = [50, 70, 100, 150]
PRIMARY_SCENARIOS = ["nominal", "transfer_bottleneck"]

DATASET_COLOR = {
    "GermanOR": "#0072B2",
    "Mannino": "#D55E00",
    "Synthetic": "#7A7A7A",
}
ZERO_COLOR = "#444444"

# These are score-contribution components. Positive means control minus
# guidance is positive, i.e., guidance reduces the corresponding penalty.
COMPONENT_SPECS = [
    ("High-priority\naccess deficit", ["exact_high_deficit"]),
    ("Volume\nmismatch", ["exact_volume_deficit", "exact_volume_excess"]),
    ("Blocked\ntransfers", ["exact_blocked"]),
    ("Peak transfer\nblocking", ["exact_peak"]),
    ("ICU excess", ["exact_icu_excess"]),
    ("Ward excess", ["exact_ward_excess"]),
    ("Execution\nviolations", ["exact_violation"]),
    ("OR overtime", ["exact_overtime"]),
]

HIGH_DEFICIT_WEIGHT = 1_500_000.0
BLOCKED_TRANSFER_WEIGHT = 1_000.0


# ---------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _load_csv_from_zip(zip_path: Path, wanted_names: Sequence[str]) -> Optional[pd.DataFrame]:
    wanted = {_norm(n) for n in wanted_names}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if _norm(Path(member).name) in wanted:
                with zf.open(member) as fh:
                    return pd.read_csv(fh)
    return None


def find_effect_rows(explicit: Optional[str], roots: Sequence[str]) -> pd.DataFrame:
    wanted = ["all_standardized_effect_rows_v21.csv", "all_standardized_effect_rows.csv"]
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"--effects does not exist: {path}")
        if path.suffix.lower() == ".zip":
            df = _load_csv_from_zip(path, wanted)
            if df is None:
                raise FileNotFoundError(f"Could not find any accepted standardized-effect file inside {path}")
            print(f"Loaded {wanted[0]} from zip: {path}")
            return df
        print(f"Loaded effects CSV: {path}")
        return pd.read_csv(path)

    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() == ".zip":
            df = _load_csv_from_zip(path, wanted)
            if df is not None:
                print(f"Loaded {wanted[0]} from zip: {path}")
                return df
        elif path.is_file() and _norm(path.name) == _norm(wanted[0]):
            print(f"Loaded effects CSV: {path}")
            return pd.read_csv(path)
        elif path.is_dir():
            found = sorted(path.rglob(wanted[0]), key=lambda p: len(str(p)))
            if found:
                print(f"Loaded {wanted[0]} from file: {found[0]}")
                return pd.read_csv(found[0])

    raise FileNotFoundError(
        "Could not locate all_standardized_effect_rows_v21.csv. "
        "Pass --effects explicitly or provide its folder/ZIP through --input-root."
    )


def fmt_score(x: float) -> str:
    """Compact signed number formatter for score contributions."""
    if not np.isfinite(x) or abs(x) < 0.5:
        return "0"
    sign = "+" if x > 0 else "−"
    ax = abs(float(x))
    if ax >= 1_000_000:
        text = f"{ax / 1_000_000:.1f}m"
    elif ax >= 1_000:
        text = f"{ax / 1_000:.1f}k"
    else:
        text = f"{ax:.0f}"
    return sign + text


def fmt_axis_score(x: float, _pos: int = 0) -> str:
    if abs(x) < 0.5:
        return "0"
    ax = abs(float(x))
    sign = "−" if x < 0 else ""
    if ax >= 1_000_000:
        return f"{sign}{ax / 1_000_000:g}m"
    if ax >= 1_000:
        return f"{sign}{ax / 1_000:g}k"
    return f"{sign}{ax:g}"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13, 1.06, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


# ---------------------------------------------------------------------
# Seed-clustered summaries
# ---------------------------------------------------------------------

def validate_and_prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"dataset", "n", "scenario", "seed"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Effect rows are missing required columns: {missing}")

    df = raw.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(int)
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)

    required_component_columns = []
    for _label, pieces in COMPONENT_SPECS:
        for piece in pieces:
            required_component_columns.append(f"gap_component__{piece}")
    missing_components = [c for c in required_component_columns if c not in df.columns]
    if missing_components:
        raise ValueError(
            "Effect rows are missing exact-component contribution columns: "
            + ", ".join(missing_components)
        )

    df = df[
        df["dataset"].isin(DATASET_ORDER)
        & df["n"].isin(N_ORDER)
        & df["scenario"].isin(PRIMARY_SCENARIOS)
    ].copy()

    # Exactly two primary scenario rows per dataset/n/seed are required before
    # averaging. This makes the cluster definition auditable.
    validation = (
        df.groupby(["dataset", "n", "seed"], as_index=False)
        .agg(
            n_primary_scenarios=("scenario", "nunique"),
            scenarios=("scenario", lambda x: "|".join(sorted(set(map(str, x))))),
            n_rows=("scenario", "size"),
        )
        .sort_values(["dataset", "n", "seed"])
        .reset_index(drop=True)
    )
    bad = validation[
        (validation["n_primary_scenarios"] != 2)
        | (validation["n_rows"] != 2)
        | (validation["scenarios"] != "nominal|transfer_bottleneck")
    ]
    if not bad.empty:
        raise ValueError(
            "Primary-scenario clustering failed: each dataset/n/seed must have exactly "
            "one nominal and one transfer_bottleneck record. Problem rows:\n"
            + bad.to_string(index=False)
        )

    # Build the eight grouped component contributions for every raw scenario row.
    out = df[["dataset", "n", "scenario", "seed"]].copy()
    for label, pieces in COMPONENT_SPECS:
        clean = label.replace("\n", " ")
        cols = [f"gap_component__{piece}" for piece in pieces]
        out[clean] = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Main M2 unit: average the two primary scenario effects within base seed.
    seed_effects = (
        out.groupby(["dataset", "n", "seed"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["dataset", "n", "seed"])
        .reset_index(drop=True)
    )

    expected = len(DATASET_ORDER) * len(N_ORDER) * 10
    if len(seed_effects) != expected:
        counts = seed_effects.groupby(["dataset", "n"], as_index=False).size()
        raise ValueError(
            f"Expected {expected} seed-cluster rows (3 families × 4 n × 10 seeds), "
            f"found {len(seed_effects)}. Counts:\n{counts.to_string(index=False)}"
        )

    return seed_effects, validation


def component_summary(seed_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    component_columns = [label.replace("\n", " ") for label, _ in COMPONENT_SPECS]
    for dataset in DATASET_ORDER:
        for n in N_ORDER:
            sub = seed_effects[(seed_effects["dataset"] == dataset) & (seed_effects["n"] == n)]
            for component in component_columns:
                values = pd.to_numeric(sub[component], errors="coerce").dropna()
                rows.append({
                    "dataset": dataset,
                    "n": n,
                    "component": component,
                    "seed_clusters": int(len(values)),
                    "median_score_contribution": float(values.median()),
                    "mean_score_contribution": float(values.mean()),
                    "q25_score_contribution": float(values.quantile(0.25)),
                    "q75_score_contribution": float(values.quantile(0.75)),
                    "min_score_contribution": float(values.min()),
                    "max_score_contribution": float(values.max()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_m2(seed_effects: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    component_labels = [label.replace("\n", " ") for label, _ in COMPONENT_SPECS]

    # Use one symmetric-log scale across all heatmaps. It preserves large
    # high-priority tails without erasing downstream components.
    values = summary["median_score_contribution"].to_numpy(dtype=float)
    max_abs = max(1.0, float(np.nanmax(np.abs(values))))
    linthresh = max(5_000.0, min(100_000.0, max_abs / 100.0))
    norm = SymLogNorm(linthresh=linthresh, linscale=0.8, vmin=-max_abs, vmax=max_abs, base=10)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(7.35, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.42, 1.0],
        left=0.16,
        right=0.975,
        top=0.94,
        bottom=0.09,
        wspace=0.22,
        hspace=0.45,
    )
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    ax_scatter = fig.add_subplot(gs[1, 0:3])

    image = None
    for idx, dataset in enumerate(DATASET_ORDER):
        ax = top_axes[idx]
        mat = np.empty((len(component_labels), len(N_ORDER)), dtype=float)
        for ri, component in enumerate(component_labels):
            vals = (
                summary[
                    (summary["dataset"] == dataset)
                    & (summary["component"] == component)
                ]
                .set_index("n")
                .reindex(N_ORDER)["median_score_contribution"]
                .to_numpy(dtype=float)
            )
            mat[ri, :] = vals

        image = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="upper")

        # Cell grid.
        ax.set_xticks(np.arange(-0.5, len(N_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(component_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.85)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(len(N_ORDER)))
        ax.set_xticklabels([str(n) for n in N_ORDER])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", length=0, pad=3)

        if idx == 0:
            ax.set_yticks(np.arange(len(component_labels)))
            ax.set_yticklabels([label.replace("\n", " ") for label, _ in COMPONENT_SPECS], fontsize=6.8)
        else:
            ax.set_yticks(np.arange(len(component_labels)))
            ax.set_yticklabels([])

        title = dataset if dataset != "Synthetic" else "Synthetic stress probe"
        ax.set_title(title, pad=5)
        ax.set_xlabel("Candidate pool size $n$", labelpad=2)
        if idx == 0:
            ax.set_ylabel("Exact-score component")
        add_panel_label(ax, f"({chr(97 + idx)})")

        for ri in range(len(component_labels)):
            for ci in range(len(N_ORDER)):
                val = mat[ri, ci]
                rgba = cmap(norm(val))
                # Luminance-based text contrast.
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if lum < 0.50 else "black"
                ax.text(ci, ri, fmt_score(val), ha="center", va="center", fontsize=5.6, color=color)

    cb = fig.colorbar(image, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.55)
    cb.ax.tick_params(labelsize=6.3, length=2, width=0.6, pad=1)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(fmt_axis_score))
    cb.set_label(
        "Seed-clustered median\nscore contribution\n(control − guidance)",
        fontsize=6.7,
        labelpad=5,
    )

    # Panel (d): n=150 seed-clustered relief/access relation.
    d = seed_effects[seed_effects["n"] == 150].copy()
    d["delta_blocked_transfer_days"] = d["Blocked transfers"] / BLOCKED_TRANSFER_WEIGHT
    d["delta_high_priority_deficit"] = d["High-priority access deficit"] / HIGH_DEFICIT_WEIGHT

    xvals = d["delta_blocked_transfer_days"].to_numpy(dtype=float)
    yvals = d["delta_high_priority_deficit"].to_numpy(dtype=float)
    xmax = max(1.0, float(np.nanmax(np.abs(xvals))) * 1.16)
    ymax = max(1.0, float(np.nanmax(np.abs(yvals))) * 1.18)

    # Desired operational quadrants. Only the right-hand quadrants receive
    # semantic shading because x>0 means less downstream blocking.
    ax_scatter.add_patch(Rectangle((0, 0), xmax, ymax, facecolor="#009E73", alpha=0.08, edgecolor="none", zorder=0))
    ax_scatter.add_patch(Rectangle((0, -ymax), xmax, ymax, facecolor="#C44E52", alpha=0.08, edgecolor="none", zorder=0))
    ax_scatter.axhline(0, color=ZERO_COLOR, linewidth=0.8, zorder=1)
    ax_scatter.axvline(0, color=ZERO_COLOR, linewidth=0.8, zorder=1)

    centroid_offsets = {
        "GermanOR": (-34, -22),
        "Mannino": (28, 17),
        "Synthetic": (20, 18),
    }
    for dataset in DATASET_ORDER:
        sub = d[d["dataset"] == dataset]
        xs = sub["delta_blocked_transfer_days"].to_numpy(dtype=float)
        ys = sub["delta_high_priority_deficit"].to_numpy(dtype=float)
        color = DATASET_COLOR[dataset]
        ax_scatter.scatter(
            xs, ys,
            s=24,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.84,
            zorder=2,
            label=dataset if dataset != "Synthetic" else "Synthetic stress probe",
        )
        mx = float(np.nanmedian(xs))
        my = float(np.nanmedian(ys))
        ax_scatter.scatter(
            [mx], [my],
            s=66,
            marker="D",
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
        dx, dy = centroid_offsets[dataset]
        label = dataset if dataset != "Synthetic" else "Synthetic"
        ax_scatter.annotate(
            label,
            xy=(mx, my),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=color, shrinkA=0, shrinkB=3),
            zorder=5,
        )

    ax_scatter.text(
        0.985, 0.94, "coupled relief",
        transform=ax_scatter.transAxes,
        ha="right", va="top",
        fontsize=7.8,
        color="#006837",
    )
    ax_scatter.text(
        0.985, 0.06, "access trade-off",
        transform=ax_scatter.transAxes,
        ha="right", va="bottom",
        fontsize=7.8,
        color="#A33A46",
    )
    ax_scatter.set_xlim(-0.08 * xmax, xmax)
    ax_scatter.set_ylim(-ymax, ymax)
    ax_scatter.set_xlabel(r"$\Delta$ blocked-transfer days (positive = guidance reduces blocking)")
    ax_scatter.set_ylabel(r"$\Delta$ high-priority deficit (positive = guidance improves access)")
    ax_scatter.set_title(r"$n=150$ downstream-relief / high-priority-access coupling", pad=5)
    ax_scatter.grid(True, color="0.90", linewidth=0.5, zorder=0)
    add_panel_label(ax_scatter, "(d)")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "M2_mechanism_decomposition_and_access_tradeoff"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    # Export exact values used by the visual.
    scatter_export = d[[
        "dataset", "n", "seed",
        "delta_blocked_transfer_days", "delta_high_priority_deficit",
        "Blocked transfers", "High-priority access deficit",
    ]].copy()
    scatter_export.rename(columns={
        "Blocked transfers": "blocked_score_contribution",
        "High-priority access deficit": "high_priority_score_contribution",
    }, inplace=True)
    scatter_export.to_csv(out_dir / "M2_seed_clustered_scatter_points_n150.csv", index=False)


def write_notes(out_dir: Path) -> None:
    text = "M2 figure notes\n\n"
    text += "Unit of analysis\n"
    text += "- For every family, candidate-pool size, and base seed, nominal and transfer-bottleneck component contributions are averaged first.\n"
    text += "- Each top-panel cell is the median across the ten resulting seed clusters.\n"
    text += "- Component-wise medians are intentionally displayed separately and are not additive.\n\n"
    text += "Sign convention\n"
    text += "- All top-panel values are exact-score contributions defined as control minus guidance.\n"
    text += "- Positive values mean that guidance reduces the corresponding penalty.\n"
    text += "- Panel (d) uses raw changes: delta blocked-transfer days = exact blocked contribution / 1000; delta high-priority deficit = exact high-deficit contribution / 1,500,000.\n"
    text += "- In panel (d), positive y means guidance improves high-priority access.\n"
    (out_dir / "M2_figure_notes.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the revised seed-clustered M2 mechanism figure.")
    parser.add_argument("--input-root", action="append", default=[], help="Folder, ZIP archive, or CSV search root. Repeatable.")
    parser.add_argument("--effects", default=None, help="Explicit all_standardized_effect_rows_v21.csv path or ZIP archive.")
    parser.add_argument("--output-dir", default="M2_seed_clustered", help="Output folder.")
    args = parser.parse_args()

    roots = args.input_root if args.input_root else ["."]
    raw = find_effect_rows(args.effects, roots)
    seed_effects, validation = validate_and_prepare(raw)
    summary = component_summary(seed_effects)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "M2_seed_clustered_component_summary.csv", index=False)
    validation.to_csv(out_dir / "M2_seed_cluster_validation.csv", index=False)
    seed_effects.to_csv(out_dir / "M2_seed_clustered_effects.csv", index=False)
    write_notes(out_dir)
    plot_m2(seed_effects, summary, out_dir)

    print("[DONE]")
    print(f"Figure PDF: {out_dir / 'M2_mechanism_decomposition_and_access_tradeoff.pdf'}")
    print(f"Figure PNG: {out_dir / 'M2_mechanism_decomposition_and_access_tradeoff.png'}")
    print(f"Summary:    {out_dir / 'M2_seed_clustered_component_summary.csv'}")
    print(f"Scatter:    {out_dir / 'M2_seed_clustered_scatter_points_n150.csv'}")


if __name__ == "__main__":
    main()

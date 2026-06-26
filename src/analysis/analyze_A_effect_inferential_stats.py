#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_A_effect_inferential_stats.py

Inferential statistics for A-effect = clean - treatment (>0 means A is better).

Outputs:
  A_effect_input_standardized.csv
  A_effect_inferential_stats.csv
  A_effect_trend_by_dataset.csv
  A_effect_median_gap_bootstrap_CI.png
  A_effect_inferential_stats_report.md
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
    from scipy.special import ndtri, ndtr
except Exception as e:
    raise SystemExit("This script requires scipy. Use an Anaconda environment with scipy.\n" + repr(e))

DEFAULT_ORDERED_N = [50, 70, 100, 150]


def norm_col(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def choose_col(df: pd.DataFrame, exact: Iterable[str] = (), contains_all: Iterable[str] = (), contains_any: Iterable[str] = ()) -> Optional[str]:
    lower = {norm_col(c): c for c in df.columns}
    for e in exact:
        hit = lower.get(norm_col(e))
        if hit is not None:
            return hit
    all_keys = [norm_col(k) for k in contains_all]
    any_keys = [norm_col(k) for k in contains_any]
    for c in df.columns:
        nc = norm_col(c)
        if all(k in nc for k in all_keys) and (not any_keys or any(k in nc for k in any_keys)):
            return c
    return None


def infer_dataset_from_path(path: Path, default: str = "Unknown") -> str:
    s = str(path).lower()
    if "german" in s:
        return "GermanOR"
    if "mannino" in s:
        return "Mannino"
    if "synth" in s or "synthetic" in s:
        return "Synthetic"
    return default


def infer_gap_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"--gap-col={override!r} not found. Available columns: {list(df.columns)}")
        return override
    candidates = [
        "gap_clean_minus_treatment", "A_gap_clean_minus_w0", "A_gap_clean_minus_Aonly",
        "gap_clean_minus_Aonly", "gap_exact_nopressure", "gap_exact_nopressure_vs_clean",
        "gap_off_minus_w0", "gap",
    ]
    hit = choose_col(df, exact=candidates)
    if hit:
        return hit
    for c in df.columns:
        nc = norm_col(c)
        if "component" in nc or "contrib" in nc:
            continue
        if "gap" in nc and ("clean" in nc or "treatment" in nc or "w0" in nc or "aonly" in nc or "a_only" in nc):
            return c
    gap_like = [c for c in df.columns if "gap" in norm_col(c) and "component" not in norm_col(c)]
    if len(gap_like) == 1:
        return gap_like[0]
    raise ValueError("Cannot infer A-effect gap column. Provide --gap-col.")


def standardize_input(paths: List[Path], gap_col: Optional[str] = None, arm_filter: Optional[str] = None) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            warnings.warn(f"Empty input skipped: {p}")
            continue
        if arm_filter:
            arm_col = choose_col(df, exact=["arm", "treatment", "treatment_arm", "method", "scenario_arm"])
            if arm_col:
                before = len(df)
                df = df[df[arm_col].astype(str).str.lower().eq(arm_filter.lower())].copy()
                print(f"Filtered {p} by {arm_col}={arm_filter}: {before} -> {len(df)} rows")
        gcol = infer_gap_col(df, gap_col)
        out = pd.DataFrame()
        out["dataset"] = df["dataset"].astype(str) if "dataset" in df.columns else infer_dataset_from_path(p)
        n_col = choose_col(df, exact=["n", "size", "num_patients", "instance_size"])
        if n_col is None:
            raise ValueError(f"Cannot find n/size column in {p}")
        out["n"] = pd.to_numeric(df[n_col], errors="coerce")
        scenario_col = choose_col(df, exact=["scenario", "scenario_name", "setting"])
        out["scenario"] = df[scenario_col].astype(str) if scenario_col else ""
        seed_col = choose_col(df, exact=["seed", "case_seed", "instance_seed"])
        out["seed"] = pd.to_numeric(df[seed_col], errors="coerce") if seed_col else np.nan
        out["gap"] = pd.to_numeric(df[gcol], errors="coerce")
        out["source_file"] = str(p)
        out["gap_col_used"] = gcol
        frames.append(out)
    if not frames:
        raise ValueError("No input rows loaded.")
    data = pd.concat(frames, ignore_index=True).dropna(subset=["dataset", "n", "gap"]).copy()
    data["n"] = data["n"].astype(int)
    if data["seed"].notna().any():
        before = len(data)
        data = data.drop_duplicates(subset=["dataset", "n", "scenario", "seed"], keep="first").copy()
        if len(data) != before:
            print(f"Deduplicated rows: {before} -> {len(data)}")
    return data.reset_index(drop=True)


def win_tie_loss(x: np.ndarray, eps: float = 1e-9) -> Tuple[int, int, int]:
    return int(np.sum(x > eps)), int(np.sum(np.abs(x) <= eps)), int(np.sum(x < -eps))


def sign_test_one_sided_greater(x: np.ndarray, eps: float = 1e-9) -> Tuple[float, float, int]:
    nz = np.asarray(x, dtype=float)
    nz = nz[np.abs(nz) > eps]
    n = int(len(nz))
    if n == 0:
        return 0.0, 1.0, 0
    wins = int(np.sum(nz > 0))
    return float(wins), float(stats.binomtest(wins, n, p=0.5, alternative="greater").pvalue), n


def wilcoxon_one_sided_greater(x: np.ndarray, eps: float = 1e-9) -> Tuple[float, float, int]:
    nz = np.asarray(x, dtype=float)
    nz = nz[np.abs(nz) > eps]
    n = int(len(nz))
    if n == 0:
        return 0.0, 1.0, 0
    try:
        res = stats.wilcoxon(nz, zero_method="wilcox", alternative="greater", method="auto")
    except TypeError:
        res = stats.wilcoxon(nz, zero_method="wilcox", alternative="greater")
    return float(res.statistic), float(res.pvalue), n


def rank_biserial_signed_rank(x: np.ndarray, eps: float = 1e-9) -> float:
    nz = np.asarray(x, dtype=float)
    nz = nz[np.abs(nz) > eps]
    if len(nz) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(nz), method="average")
    r_pos = float(np.sum(ranks[nz > 0]))
    r_neg = float(np.sum(ranks[nz < 0]))
    denom = r_pos + r_neg
    return (r_pos - r_neg) / denom if denom else np.nan


def hodges_lehmann_one_sample(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    vals = []
    for i in range(len(x)):
        vals.extend((x[i] + x[i:]) / 2.0)
    return float(np.median(np.asarray(vals, dtype=float)))


def bootstrap_stat_values(x: np.ndarray, stat_func, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        vals[b] = stat_func(x[rng.integers(0, n, size=n)])
    return vals


def bca_ci(x: np.ndarray, stat_func, n_boot: int, alpha: float, rng: np.random.Generator) -> Tuple[float, float, str]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, "insufficient"
    theta_hat = float(stat_func(x))
    boot = bootstrap_stat_values(x, stat_func, n_boot, rng)
    boot = boot[np.isfinite(boot)]
    if len(boot) < max(100, n_boot // 2):
        return np.nan, np.nan, "bootstrap_failed"
    if np.allclose(boot, boot[0]):
        return float(boot[0]), float(boot[0]), "degenerate"
    prop_less = np.mean(boot < theta_hat)
    prop_less = min(max(prop_less, 1.0 / (2 * len(boot))), 1 - 1.0 / (2 * len(boot)))
    z0 = ndtri(prop_less)
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = stat_func(np.delete(x, i))
    jm = np.mean(jack)
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    acc = num / den if den != 0 else 0.0
    if not np.isfinite(acc):
        acc = 0.0
    def adj_q(z_alpha: float) -> float:
        denom = 1 - acc * (z0 + z_alpha)
        if denom == 0:
            return np.nan
        return ndtr(z0 + (z0 + z_alpha) / denom)
    ql = adj_q(ndtri(alpha / 2))
    qh = adj_q(ndtri(1 - alpha / 2))
    if not np.isfinite(ql) or not np.isfinite(qh):
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi), "percentile_fallback"
    ql, qh = sorted([min(max(ql, 0.0), 1.0), min(max(qh, 0.0), 1.0)])
    lo, hi = np.quantile(boot, [ql, qh])
    return float(lo), float(hi), "bca"


def percentile_bootstrap_ci(x: np.ndarray, stat_func, n_boot: int, alpha: float, rng: np.random.Generator) -> Tuple[float, float, str]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, "insufficient"
    boot = bootstrap_stat_values(x, stat_func, n_boot, rng)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), "percentile"


def holm_adjust(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def bh_adjust(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m, dtype=float)
    running = 1.0
    for rank_from_high, idx in enumerate(order):
        rank = m - rank_from_high
        val = p[idx] * m / rank
        running = min(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def distribution_note(dataset: str, n: int, x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "empty"
    if str(dataset).lower().startswith("synthetic") and int(n) == 150:
        return "bimodal_high_deficit"
    med = float(np.median(x))
    q10, q90 = np.quantile(x, [0.10, 0.90])
    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    if iqr > 0 and (abs(float(np.min(x)) - med) / iqr > 10 or abs(float(np.max(x)) - med) / iqr > 10):
        return "heavy_tail"
    if q10 < 0 < q90:
        return "mixed_sign"
    return "regular"


def robustness_label(dataset: str, p_adj: float, hl: float, hl_low: float, win_rate: float, note: str) -> str:
    ds = str(dataset).lower()
    if "bimodal" in note or "heavy_tail" in note:
        return "fragile"
    if ds.startswith("synthetic"):
        return "fragile"
    if p_adj < 0.05 and hl > 0 and hl_low > 0 and win_rate >= 0.80:
        return "robust"
    return "fragile"


def analyze_cells(data: pd.DataFrame, n_boot: int, alpha: float, seed: int, p_adjust: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, n), g in data.groupby(["dataset", "n"], dropna=False):
        x = pd.to_numeric(g["gap"], errors="coerce").dropna().to_numpy(dtype=float)
        num = int(len(x))
        win, tie, loss = win_tie_loss(x)
        win_rate = win / num if num else np.nan
        if str(dataset).lower().startswith("synthetic"):
            test_used = "sign"
            stat, p_raw, n_nonzero = sign_test_one_sided_greater(x)
            stat_name = "wins_nonzero"
        else:
            test_used = "wilcoxon"
            stat, p_raw, n_nonzero = wilcoxon_one_sided_greater(x)
            stat_name = "W_plus"
        median_gap = float(np.median(x)) if num else np.nan
        hl = hodges_lehmann_one_sample(x)
        med_low, med_high, med_ci_method = bca_ci(x, np.median, n_boot, alpha, rng)
        hl_low, hl_high, hl_ci_method = bca_ci(x, hodges_lehmann_one_sample, n_boot, alpha, rng)
        win_indicator = (x > 1e-9).astype(float)
        wr_low, wr_high, wr_ci_method = percentile_bootstrap_ci(win_indicator, lambda z: float(np.mean(z)), n_boot, alpha, rng)
        rbc = rank_biserial_signed_rank(x)
        note = distribution_note(str(dataset), int(n), x)
        rows.append({
            "dataset": dataset, "n": int(n), "num_cases": num,
            "win": win, "tie": tie, "loss": loss,
            "test_used": test_used, "stat_name": stat_name, "stat": stat,
            "p_raw": p_raw, "n_nonzero": n_nonzero,
            "median_gap": median_gap, "median_ci_low": med_low, "median_ci_high": med_high, "median_ci_method": med_ci_method,
            "hl_shift": hl, "hl_ci_low": hl_low, "hl_ci_high": hl_high, "hl_ci_method": hl_ci_method,
            "rank_biserial": rbc,
            "win_rate": win_rate, "win_rate_ci_low": wr_low, "win_rate_ci_high": wr_high, "win_rate_ci_method": wr_ci_method,
            "distribution_note": note,
        })
    out = pd.DataFrame(rows).sort_values(["dataset", "n"]).reset_index(drop=True)
    pvals = out["p_raw"].fillna(1.0).tolist()
    if p_adjust.lower() == "bh":
        out["p_adj"] = bh_adjust(pvals); out["p_adjust_method"] = "BH"
    else:
        out["p_adj"] = holm_adjust(pvals); out["p_adjust_method"] = "Holm"
    out["robustness"] = [
        robustness_label(r["dataset"], r["p_adj"], r["hl_shift"], r["hl_ci_low"], r["win_rate"], r["distribution_note"])
        for _, r in out.iterrows()
    ]
    first = ["dataset", "n", "num_cases", "win", "tie", "loss", "test_used", "stat", "p_raw", "p_adj", "hl_shift", "hl_ci_low", "hl_ci_high", "rank_biserial", "win_rate", "win_rate_ci_low", "win_rate_ci_high", "robustness", "distribution_note"]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def jt_statistic(values: np.ndarray, groups: np.ndarray, ordered_levels: List[int]) -> float:
    stat = 0.0
    vals = {lev: values[groups == lev] for lev in ordered_levels}
    for i, li in enumerate(ordered_levels[:-1]):
        xi = vals[li]
        for lj in ordered_levels[i+1:]:
            xj = vals[lj]
            if len(xi) == 0 or len(xj) == 0:
                continue
            diff = xj[:, None] - xi[None, :]
            stat += float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
    return stat


def jt_trend_test(data: pd.DataFrame, n_permutations: int, seed: int, ordered_n: List[int]) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 12345)
    rows = []
    for dataset, g in data.groupby("dataset", dropna=False):
        gg = g[g["n"].isin(ordered_n)].copy()
        levels = [n for n in ordered_n if n in set(gg["n"])]
        values = pd.to_numeric(gg["gap"], errors="coerce").to_numpy(dtype=float)
        groups = pd.to_numeric(gg["n"], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(values)
        values, groups = values[mask], groups[mask]
        med = gg.groupby("n")["gap"].median().to_dict()
        if len(levels) < 2 or len(values) < 4:
            rows.append({"dataset": dataset, "JT_stat": np.nan, "p": np.nan, "direction": "insufficient", "num_cases": len(values), "n_levels": "|".join(map(str, levels)), "medians_by_n": ";".join(f"{k}:{med.get(k, np.nan):.6g}" for k in levels)})
            continue
        obs = jt_statistic(values, groups, levels)
        ge = 0
        for _ in range(n_permutations):
            if jt_statistic(values, rng.permutation(groups), levels) >= obs - 1e-12:
                ge += 1
        p_perm = (ge + 1) / (n_permutations + 1)
        med_list = [float(med.get(k, np.nan)) for k in levels]
        if med_list[-1] > med_list[0] and p_perm < 0.05:
            direction = "increasing_significant"
        elif med_list[-1] > med_list[0]:
            direction = "increasing_not_significant"
        elif med_list[-1] < med_list[0] and p_perm < 0.05:
            direction = "decreasing_significant_against_expected"
        else:
            direction = "no_increasing_trend"
        counts = {lev: int(np.sum(groups == lev)) for lev in levels}
        rows.append({"dataset": dataset, "JT_stat": obs, "p": p_perm, "direction": direction, "num_cases": int(len(values)), "n_levels": "|".join(map(str, levels)), "group_counts": ";".join(f"{k}:{counts[k]}" for k in levels), "medians_by_n": ";".join(f"{k}:{med.get(k, np.nan):.6g}" for k in levels), "jt_p_method": f"permutation_{n_permutations}"})
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def plot_median_ci(cell_stats: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if cell_stats.empty:
        return None
    datasets = list(cell_stats["dataset"].dropna().astype(str).unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        g = cell_stats[cell_stats["dataset"].astype(str).eq(dataset)].sort_values("n")
        xs = np.arange(len(g))
        y = pd.to_numeric(g["median_gap"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(g["median_ci_low"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(g["median_ci_high"], errors="coerce").to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        ax.errorbar(xs, y, yerr=yerr, marker="o", capsize=4)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xticks(xs); ax.set_xticklabels(g["n"].astype(str).tolist())
        ax.set_title(str(dataset)); ax.set_xlabel("n"); ax.set_ylabel("median gap with bootstrap CI")
        for xi, yi, row in zip(xs, y, g.to_dict("records")):
            sig = "*" if row.get("p_adj", 1.0) < 0.05 else ""
            tag = "R" if row.get("robustness") == "robust" else "F"
            ax.annotate(f"{sig}{tag}", xy=(xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    path = outdir / "A_effect_median_gap_bootstrap_CI.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_report(outdir: Path, cell_stats: pd.DataFrame, trend: pd.DataFrame, data: pd.DataFrame) -> None:
    lines = []
    lines.append("# A-effect inferential statistics report")
    lines.append("")
    lines.append("Gap convention: `gap = clean - treatment`; positive values mean A is better.")
    lines.append("")
    lines.append("## Input coverage")
    cov = data.groupby(["dataset", "n"], dropna=False).size().reset_index(name="num_rows")
    lines.append(cov.to_markdown(index=False))
    lines.append("")
    lines.append("## Cell-level inferential statistics")
    show_cols = ["dataset", "n", "num_cases", "win", "tie", "loss", "test_used", "stat", "p_raw", "p_adj", "median_gap", "hl_shift", "hl_ci_low", "hl_ci_high", "rank_biserial", "win_rate", "win_rate_ci_low", "win_rate_ci_high", "robustness", "distribution_note"]
    lines.append(cell_stats[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Trend tests by dataset")
    lines.append(trend.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("- GermanOR/Mannino use one-sided Wilcoxon signed-rank tests.")
    lines.append("- Synthetic uses one-sided sign tests because heavy-tailed/bimodal A effects violate the symmetry assumption behind Wilcoxon.")
    lines.append("- Synthetic cells are labelled fragile when the distribution is heavy-tailed/bimodal, even if the sign test is significant.")
    lines.append("- The JT trend p-value is permutation based.")
    (outdir / "A_effect_inferential_stats_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="A-effect detailed CSV. Repeat for real and synthetic files.")
    ap.add_argument("--output-dir", default="A_effect_inferential_stats")
    ap.add_argument("--gap-col", default=None)
    ap.add_argument("--arm-filter", default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--jt-permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=202706)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--p-adjust", choices=["holm", "bh"], default="holm")
    ap.add_argument("--ordered-n", nargs="+", type=int, default=DEFAULT_ORDERED_N)
    args = ap.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    data = standardize_input([Path(p) for p in args.input], gap_col=args.gap_col, arm_filter=args.arm_filter)
    data.to_csv(outdir / "A_effect_input_standardized.csv", index=False)
    cell_stats = analyze_cells(data, args.bootstrap, args.alpha, args.seed, args.p_adjust)
    cell_stats.to_csv(outdir / "A_effect_inferential_stats.csv", index=False)
    trend = jt_trend_test(data, args.jt_permutations, args.seed, args.ordered_n)
    trend.to_csv(outdir / "A_effect_trend_by_dataset.csv", index=False)
    plot_median_ci(cell_stats, outdir)
    write_report(outdir, cell_stats, trend, data)
    print(f"Saved A-effect inferential statistics under: {outdir}")
    for name in ["A_effect_input_standardized.csv", "A_effect_inferential_stats.csv", "A_effect_trend_by_dataset.csv", "A_effect_median_gap_bootstrap_CI.png", "A_effect_inferential_stats_report.md"]:
        print(" -", outdir / name)


if __name__ == "__main__":
    main()

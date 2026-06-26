#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic tail decomposition + cross-dataset mechanism reconciliation.

Conventions:
  gap = clean score - A_only score; gap > 0 means A_only is better.
  gap_component__exact_X > 0 means A improves component X.

Inputs:
  --synthetic-detailed: detailed Synthetic A-only-vs-clean CSV, e.g.
      A_effect_w0_vs_clean_detailed_synth.csv
      or final_pairwise_iso60_detailed_synth.csv
  --real-detailed: optional, repeatable. Needed for full cross-dataset Part 2.
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr, gaussian_kde
except Exception:
    pearsonr = spearmanr = gaussian_kde = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
except Exception:
    DecisionTreeClassifier = None
    export_text = None

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
DOWNSTREAM = ["exact_blocked", "exact_peak", "exact_icu_excess"]


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


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def choose_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        hit = lower.get(norm_col(cand))
        if hit is not None:
            return hit
    return None


def infer_dataset_name(path: Path, default: str) -> str:
    s = str(path).lower()
    if "german" in s:
        return "GermanOR"
    if "mannino" in s:
        return "Mannino"
    if "synthetic" in s or "synth" in s:
        return "Synthetic"
    return default


def infer_gap_col(df: pd.DataFrame) -> str:
    hit = choose_first_existing(df, [
        "gap_clean_minus_treatment",
        "A_gap_clean_minus_Aonly",
        "gap_clean_minus_Aonly",
        "gap_exact_nopressure",
        "gap_exact_nopressure_vs_clean",
        "gap",
    ])
    if hit:
        return hit
    for c in df.columns:
        nc = norm_col(c)
        if "gap" in nc and "component" not in nc and ("clean" in nc or "aonly" in nc or "a_only" in nc):
            return c
    for c in df.columns:
        nc = norm_col(c)
        if "gap" in nc and "component" not in nc:
            return c
    raise ValueError("Cannot infer A-effect gap column. Provide --gap-col.")


def infer_component_gap_col(df: pd.DataFrame, gap_col: str, comp: str) -> Optional[str]:
    hit = choose_first_existing(df, [
        f"{gap_col}_component__{comp}",
        f"{gap_col}_component_{comp}",
        f"gap_clean_minus_treatment_component__{comp}",
        f"A_gap_clean_minus_Aonly_component__{comp}",
        f"gap_exact_nopressure_component__{comp}",
        f"gap_component__{comp}",
    ])
    if hit:
        return hit
    comp_norm = norm_col(comp)
    for c in df.columns:
        nc = norm_col(c)
        if "gap" in nc and "component" in nc and comp_norm in nc:
            return c
    return None


def infer_base_col(df: pd.DataFrame, comp: str) -> Optional[str]:
    tail = comp.replace("exact_", "")
    hit = choose_first_existing(df, [
        f"base_{comp}", f"clean_{comp}", f"off_{comp}", f"baseline_{comp}",
        f"base_exact_{tail}", f"clean_exact_{tail}", f"off_exact_{tail}",
    ])
    if hit:
        return hit
    tnorm = norm_col(tail)
    for c in df.columns:
        nc = norm_col(c)
        if tnorm in nc and ("base" in nc or "clean" in nc or "off" in nc) and "component" not in nc and "gap" not in nc and "weighted" not in nc:
            return c
    return None


def infer_treatment_col(df: pd.DataFrame, comp: str) -> Optional[str]:
    tail = comp.replace("exact_", "")
    hit = choose_first_existing(df, [
        f"treatment_{comp}", f"A_{comp}", f"Aonly_{comp}", f"A_only_{comp}",
        f"treatment_exact_{tail}",
    ])
    if hit:
        return hit
    tnorm = norm_col(tail)
    for c in df.columns:
        nc = norm_col(c)
        if tnorm in nc and ("treatment" in nc or "aonly" in nc or "a_only" in nc) and "component" not in nc and "gap" not in nc and "weighted" not in nc:
            return c
    return None


def ensure_basic_columns(df: pd.DataFrame, source_path: Path, default_dataset: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = infer_dataset_name(source_path, default_dataset)
    if "n" not in out.columns:
        hit = choose_first_existing(out, ["size", "num_patients", "instance_size"])
        if hit:
            out["n"] = out[hit]
    if "seed" not in out.columns:
        hit = choose_first_existing(out, ["case_seed", "instance_seed", "random_seed"])
        if hit:
            out["seed"] = out[hit]
    if "scenario" not in out.columns:
        hit = choose_first_existing(out, ["scenario_name", "setting"])
        out["scenario"] = out[hit] if hit else ""
    for c in ["n", "seed"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def standardize_effect_table(path: Path, default_dataset: str, gap_col_arg: Optional[str] = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Empty input: {path}")
    raw = ensure_basic_columns(raw, path, default_dataset)
    gap_col = gap_col_arg if gap_col_arg else infer_gap_col(raw)
    if gap_col not in raw.columns:
        raise ValueError(f"Gap column not found: {gap_col}")

    out = pd.DataFrame()
    for c in ["dataset", "n", "scenario", "seed"]:
        out[c] = raw[c] if c in raw.columns else np.nan
    out["source_path"] = str(path)
    out["gap"] = pd.to_numeric(raw[gap_col], errors="coerce")

    for out_col, keys in {
        "treatment_trace_path": ["treatment_trace_path", "trace_path", "A_trace_path", "aonly_trace_path"],
        "base_trace_path": ["base_trace_path", "clean_trace_path", "off_trace_path"],
    }.items():
        hit = choose_first_existing(raw, keys)
        out[out_col] = raw[hit] if hit else ""

    for comp in COMPONENTS:
        gcol = infer_component_gap_col(raw, gap_col, comp)
        bcol = infer_base_col(raw, comp)
        tcol = infer_treatment_col(raw, comp)
        if gcol:
            out[f"gap_component__{comp}"] = pd.to_numeric(raw[gcol], errors="coerce")
        elif bcol and tcol:
            b = pd.to_numeric(raw[bcol], errors="coerce")
            t = pd.to_numeric(raw[tcol], errors="coerce")
            out[f"gap_component__{comp}"] = W_EXACT[comp] * (b - t)
        else:
            out[f"gap_component__{comp}"] = np.nan

        if bcol:
            out[f"base_{comp}"] = pd.to_numeric(raw[bcol], errors="coerce")
        elif gcol and tcol:
            out[f"base_{comp}"] = pd.to_numeric(raw[tcol], errors="coerce") + pd.to_numeric(raw[gcol], errors="coerce") / W_EXACT[comp]
        else:
            out[f"base_{comp}"] = np.nan

        if tcol:
            out[f"treatment_{comp}"] = pd.to_numeric(raw[tcol], errors="coerce")
        elif gcol and bcol:
            out[f"treatment_{comp}"] = pd.to_numeric(raw[bcol], errors="coerce") - pd.to_numeric(raw[gcol], errors="coerce") / W_EXACT[comp]
        else:
            out[f"treatment_{comp}"] = np.nan

    out["delta_blocked"] = out["base_exact_blocked"] - out["treatment_exact_blocked"]
    out["delta_viol"] = out["treatment_exact_violation"] - out["base_exact_violation"]
    out["extra_high_missed_by_A"] = out["treatment_exact_high_deficit"] - out["base_exact_high_deficit"]
    return out


def dominant_component(row: pd.Series) -> Tuple[str, float]:
    vals = {comp: as_float(row.get(f"gap_component__{comp}")) for comp in COMPONENTS}
    vals = {k: v for k, v in vals.items() if np.isfinite(v)}
    if not vals:
        return "", np.nan
    k = max(vals, key=lambda c: abs(vals[c]))
    return k, vals[k]


def classify_tails(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dominant_component"] = ""
    out["dominant_component_value"] = np.nan
    for i, r in out.iterrows():
        k, v = dominant_component(r)
        out.at[i, "dominant_component"] = k
        out.at[i, "dominant_component_value"] = v
    out["tail_class"] = "middle"
    for n, g in out.groupby("n", dropna=False):
        vals = pd.to_numeric(g["gap"], errors="coerce").dropna()
        if vals.empty:
            continue
        q10 = vals.quantile(0.10)
        q90 = vals.quantile(0.90)
        idx = g.index
        loser = (out.loc[idx, "gap"] < 0) & ((out.loc[idx, "gap"] <= q10) | (out.loc[idx, "gap"] < -1_000_000.0))
        winner = (out.loc[idx, "gap"] > 0) & ((out.loc[idx, "gap"] >= q90) | (out.loc[idx, "gap"] > 1_000_000.0))
        out.loc[idx[loser], "tail_class"] = "catastrophic_loser"
        out.loc[idx[winner], "tail_class"] = "big_winner"
    return out.sort_values(["n", "gap"], ascending=[True, True]).reset_index(drop=True)


def summarize_tail_drivers(tail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(f"n={n}", g) for n, g in tail.groupby("n", dropna=False)] + [("overall", tail)]
    for label, g in groups:
        losers = g[g["tail_class"].eq("catastrophic_loser")].copy()
        if losers.empty:
            rows.append({"group": label, "num_losers": 0})
            continue
        high_comp = pd.to_numeric(losers["gap_component__exact_high_deficit"], errors="coerce")
        gap = pd.to_numeric(losers["gap"], errors="coerce")
        ratio = high_comp.abs() / gap.abs().replace(0, np.nan)
        extra = pd.to_numeric(losers["extra_high_missed_by_A"], errors="coerce")
        rows.append({
            "group": label,
            "num_losers": int(len(losers)),
            "high_deficit_dominant_share": float(losers["dominant_component"].eq("exact_high_deficit").mean()),
            "high_deficit_near_total_share_abs_ratio_ge_0p7": float((ratio >= 0.7).mean()),
            "mean_extra_high_missed_by_A": float(extra.mean()) if extra.notna().any() else np.nan,
            "median_extra_high_missed_by_A": float(extra.median()) if extra.notna().any() else np.nan,
            "mean_high_deficit_component": float(high_comp.mean()) if high_comp.notna().any() else np.nan,
            "median_high_deficit_component": float(high_comp.median()) if high_comp.notna().any() else np.nan,
            "mean_gap": float(gap.mean()) if gap.notna().any() else np.nan,
            "median_gap": float(gap.median()) if gap.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def histogram_peaks(vals: np.ndarray) -> Tuple[int, str]:
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return 0, "insufficient"
    counts, edges = np.histogram(vals, bins=min(12, max(5, len(vals)//2)))
    peaks = []
    for i, cnt in enumerate(counts):
        left = counts[i-1] if i > 0 else -1
        right = counts[i+1] if i < len(counts)-1 else -1
        if cnt > left and cnt > right and cnt > 0:
            peaks.append(i)
    centers = [(edges[i] + edges[i+1]) / 2 for i in peaks]
    return len(peaks), "|".join(f"{c:.6g}" for c in centers)


def distribution_stats(tail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in tail.groupby("n", dropna=False):
        vals = pd.to_numeric(g["gap"], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        npeaks, centers = histogram_peaks(vals)
        rows.append({
            "dataset": "Synthetic", "n": n, "num_cases": int(len(vals)),
            "mean_gap": float(np.mean(vals)), "median_gap": float(np.median(vals)),
            "std_gap": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min_gap": float(np.min(vals)), "max_gap": float(np.max(vals)),
            "q10_gap": float(np.quantile(vals, 0.10)), "q90_gap": float(np.quantile(vals, 0.90)),
            "histogram_local_peak_count": int(npeaks), "histogram_peak_centers": centers,
            "interpretation_hint": "split/heavy-tail if min/max far from median or peak_count>=2",
        })
    return pd.DataFrame(rows)


def plot_synthetic_gap_distribution(tail: pd.DataFrame, outdir: Path, n_target: int = 150) -> None:
    g = tail[pd.to_numeric(tail["n"], errors="coerce").eq(n_target)].copy()
    if g.empty:
        return
    vals = pd.to_numeric(g["gap"], errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=min(12, max(5, len(vals)//2)), alpha=0.75)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axvline(np.median(vals), linestyle=":", linewidth=1)
    if gaussian_kde is not None and len(vals) >= 5 and np.std(vals) > 0:
        try:
            xs = np.linspace(np.min(vals), np.max(vals), 300)
            kde = gaussian_kde(vals)
            scale = len(vals) * (np.max(vals) - np.min(vals)) / min(12, max(5, len(vals)//2))
            ax.plot(xs, kde(xs) * scale, linewidth=1)
        except Exception:
            pass
    for _, r in g.iterrows():
        if r.get("tail_class") in {"catastrophic_loser", "big_winner"}:
            ax.annotate(f"{r.get('scenario','')}\nseed{int(r.get('seed')) if pd.notna(r.get('seed')) else ''}",
                        xy=(as_float(r.get("gap")), 0), xytext=(0, 10), textcoords="offset points",
                        rotation=75, fontsize=7, ha="center")
    ax.set_title(f"Synthetic n={n_target} A-effect gap distribution")
    ax.set_xlabel("gap = clean score - A_only score (>0 means A better)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(outdir / f"synth_gap_distribution_n{n_target}.png", dpi=200)
    plt.close(fig)


def best_threshold_rule(df: pd.DataFrame, feature: str, target_col: str) -> dict:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = df[target_col].astype(int)
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy(float)
    y = y[mask].to_numpy(int)
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {"feature": feature, "target": target_col, "status": "insufficient_variation"}
    candidates = np.unique(np.quantile(x, np.linspace(0.1, 0.9, 17)))
    best = None
    for th in candidates:
        for direction in [">=", "<="]:
            pred = (x >= th).astype(int) if direction == ">=" else (x <= th).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            sens = tp / (tp + fn) if (tp + fn) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            bal = float(np.nanmean([sens, spec]))
            row = {"feature": feature, "target": target_col, "direction": direction, "threshold": float(th),
                   "balanced_accuracy": bal, "sensitivity": float(sens), "specificity": float(spec),
                   "tp": tp, "tn": tn, "fp": fp, "fn": fn, "status": "ok"}
            if roc_auc_score is not None:
                try:
                    row["auc_raw_feature"] = float(roc_auc_score(y, x))
                except Exception:
                    row["auc_raw_feature"] = np.nan
            if best is None or bal > best["balanced_accuracy"]:
                best = row
    return best or {"feature": feature, "target": target_col, "status": "failed"}


def predictor_diagnostics(tail: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = tail.copy()
    d["is_catastrophic_loser"] = d["tail_class"].eq("catastrophic_loser").astype(int)
    d["is_big_winner"] = d["tail_class"].eq("big_winner").astype(int)
    d["scenario_transfer_bottleneck"] = d["scenario"].astype(str).str.contains("transfer", case=False, na=False).astype(int)
    features = ["n", "seed", "scenario_transfer_bottleneck", "base_exact_violation", "base_exact_blocked",
                "base_exact_high_deficit", "base_exact_icu_excess", "base_exact_peak", "base_exact_overtime"]
    rows = []
    for target in ["is_catastrophic_loser", "is_big_winner"]:
        for f in features:
            if f in d.columns:
                rows.append(best_threshold_rule(d, f, target))
    rules = pd.DataFrame(rows).sort_values(["target", "balanced_accuracy"], ascending=[True, False], na_position="last")

    tree_rows = []
    if DecisionTreeClassifier is not None and export_text is not None:
        use_features = [f for f in features if f in d.columns]
        X = d[use_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        for target in ["is_catastrophic_loser", "is_big_winner"]:
            y = d[target].astype(int)
            if y.nunique() < 2 or len(X) < 6:
                tree_rows.append({"target": target, "status": "insufficient"})
                continue
            try:
                clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=max(2, len(X)//12), random_state=0)
                clf.fit(X, y)
                tree_rows.append({"target": target, "status": "ok", "training_accuracy": float(clf.score(X, y)),
                                  "tree_text": export_text(clf, feature_names=use_features)})
            except Exception as e:
                tree_rows.append({"target": target, "status": f"failed:{repr(e)}"})
    else:
        tree_rows.append({"target": "all", "status": "sklearn_not_available"})
    return rules, pd.DataFrame(tree_rows)


def choose_time_col(df: pd.DataFrame) -> Optional[str]:
    return choose_first_existing(df, ["elapsed_sec", "wallclock_elapsed_s", "elapsed_s", "time_s", "elapsed_seconds", "runtime_sec"])


def choose_iteration_col(df: pd.DataFrame) -> Optional[str]:
    return choose_first_existing(df, ["iteration", "iter", "candidate_id", "step", "proposal_index"])


def choose_score_col(df: pd.DataFrame) -> Optional[str]:
    hit = choose_first_existing(df, ["exact_nopressure_score", "best_exact_nopressure_score", "best_no_pressure_exact_score",
                                     "best_exact_score", "exact_score", "best_score", "incumbent_score"])
    if hit:
        return hit
    for c in df.columns:
        nc = norm_col(c)
        if "score" in nc and "pressure" not in nc and "fast" not in nc:
            return c
    return None


def selection_artifact_check(tail: pd.DataFrame, iso_t: float) -> pd.DataFrame:
    rows = []
    losers = tail[tail["tail_class"].eq("catastrophic_loser")].copy()
    if losers.empty:
        return pd.DataFrame([{"status": "no_catastrophic_losers"}])
    for _, r in losers.iterrows():
        row = {"dataset": r.get("dataset"), "n": r.get("n"), "scenario": r.get("scenario"),
               "seed": r.get("seed"), "gap": r.get("gap"), "treatment_trace_path": r.get("treatment_trace_path", "")}
        p = Path(str(row["treatment_trace_path"] or ""))
        if not str(p) or not p.exists():
            row["status"] = "trace_path_missing_or_not_found"
            rows.append(row); continue
        try:
            tr = pd.read_csv(p)
        except Exception as e:
            row["status"] = f"trace_read_failed:{repr(e)}"
            rows.append(row); continue
        tcol, icol, scol = choose_time_col(tr), choose_iteration_col(tr), choose_score_col(tr)
        row.update({"time_col": tcol or "", "iteration_col": icol or "", "score_col": scol or ""})
        if scol is None:
            row["status"] = "score_column_not_found"
            rows.append(row); continue
        work = tr.copy()
        if tcol:
            work["_time"] = pd.to_numeric(work[tcol], errors="coerce")
            work = work[work["_time"].notna() & (work["_time"] <= iso_t + 1e-9)]
        work["_score"] = pd.to_numeric(work[scol], errors="coerce")
        work = work[work["_score"].notna()]
        if work.empty:
            row["status"] = "no_finite_score_before_iso"
            rows.append(row); continue
        selected = work.iloc[-1]
        best = work.loc[work["_score"].idxmin()]
        row["status"] = "ok"
        row["selected_score_at_iso_last_row"] = float(selected["_score"])
        row["best_score_before_iso"] = float(best["_score"])
        row["selected_minus_best_score"] = float(selected["_score"] - best["_score"])
        if icol:
            row["selected_iteration"] = as_float(selected.get(icol))
            row["best_iteration_before_iso"] = as_float(best.get(icol))
            row["max_iteration_before_iso"] = as_float(work[icol].max())
            row["selected_is_last_iteration"] = int(as_float(selected.get(icol)) == as_float(work[icol].max()))
        if tcol:
            row["selected_elapsed_sec"] = as_float(selected.get(tcol))
            row["best_elapsed_sec"] = as_float(best.get(tcol))
        rows.append(row)
    return pd.DataFrame(rows)


def c1_downstream_signs(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, n), g in all_df.groupby(["dataset", "n"], dropna=False):
        row = {"dataset": dataset, "n": n, "num_cases": int(len(g))}
        for comp in DOWNSTREAM + ["exact_violation", "exact_high_deficit"]:
            vals = pd.to_numeric(g[f"gap_component__{comp}"], errors="coerce")
            med = float(vals.median()) if vals.notna().any() else np.nan
            row[f"median_component_{comp}"] = med
            row[f"mean_component_{comp}"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"positive_median_{comp}"] = bool(med > 0) if np.isfinite(med) else None
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "n"]).reset_index(drop=True)
    n150 = out[pd.to_numeric(out["n"], errors="coerce").eq(150)]
    if not n150.empty:
        out["C1_blocked_positive_all_datasets_at_n150"] = bool((pd.to_numeric(n150["median_component_exact_blocked"], errors="coerce") > 0).all())
    else:
        out["C1_blocked_positive_all_datasets_at_n150"] = np.nan
    return out


def coupling_scatter_data(all_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["dataset", "n", "scenario", "seed", "gap", "tail_class", "delta_blocked", "delta_viol", "extra_high_missed_by_A",
            "base_exact_violation", "base_exact_blocked", "base_exact_high_deficit", "base_exact_icu_excess",
            "treatment_exact_violation", "treatment_exact_blocked", "treatment_exact_high_deficit", "treatment_exact_icu_excess",
            "gap_component__exact_blocked", "gap_component__exact_violation", "gap_component__exact_high_deficit"]
    out = all_df[[c for c in cols if c in all_df.columns]].copy()
    corr_rows = []
    for (dataset, n), g in out.groupby(["dataset", "n"], dropna=False):
        x = pd.to_numeric(g["delta_blocked"], errors="coerce")
        y = pd.to_numeric(g["delta_viol"], errors="coerce")
        mask = x.notna() & y.notna()
        row = {"dataset": dataset, "n": n, "num_cases": int(mask.sum())}
        if mask.sum() >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
            xv, yv = x[mask].to_numpy(float), y[mask].to_numpy(float)
            row["pearson_r"] = float(pearsonr(xv, yv)[0]) if pearsonr else float(np.corrcoef(xv, yv)[0, 1])
            row["spearman_r"] = float(spearmanr(xv, yv)[0]) if spearmanr else np.nan
            row["median_delta_blocked"] = float(np.median(xv))
            row["median_delta_viol"] = float(np.median(yv))
            if row["median_delta_blocked"] > 0 and row["median_delta_viol"] <= 0:
                row["channel_label"] = "coupled_downstream_and_violation_relief"
            elif row["median_delta_blocked"] > 0 and row["median_delta_viol"] > 0:
                row["channel_label"] = "tradeoff_downstream_relief_for_more_violation"
            elif row["median_delta_blocked"] <= 0:
                row["channel_label"] = "no_median_downstream_relief"
            else:
                row["channel_label"] = "unclear"
        else:
            row["status"] = "insufficient_variation"
        corr_rows.append(row)
    return out, pd.DataFrame(corr_rows).sort_values(["dataset", "n"]).reset_index(drop=True)


def plot_coupling_scatter(scatter: pd.DataFrame, outdir: Path) -> None:
    if scatter.empty:
        return
    datasets = list(scatter["dataset"].dropna().astype(str).unique())
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    axes = axes[0]
    for ax, dataset in zip(axes, datasets):
        g = scatter[scatter["dataset"].astype(str).eq(dataset)]
        ax.scatter(pd.to_numeric(g["delta_blocked"], errors="coerce"), pd.to_numeric(g["delta_viol"], errors="coerce"), alpha=0.75)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(str(dataset))
        ax.set_xlabel("Δblocked = base blocked - A blocked")
        ax.set_ylabel("Δviol = A violation - base violation")
    fig.tight_layout()
    fig.savefig(outdir / "mechanism_delta_blocked_delta_viol_scatter.png", dpi=200)
    plt.close(fig)


def binding_profile(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, n), g in all_df.groupby(["dataset", "n"], dropna=False):
        row = {"dataset": dataset, "n": n, "num_cases": int(len(g))}
        for comp in ["exact_violation", "exact_blocked", "exact_high_deficit", "exact_icu_excess", "exact_peak"]:
            vals = pd.to_numeric(g[f"base_{comp}"], errors="coerce")
            row[f"median_base_{comp}"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"mean_base_{comp}"] = float(vals.mean()) if vals.notna().any() else np.nan
        row["normalization_status"] = "raw_only_no_capacity_metadata"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "n"]).reset_index(drop=True)


def tail_mechanism_link(tail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for klass, g in tail.groupby("tail_class", dropna=False):
        row = {"tail_class": klass, "num_cases": int(len(g))}
        for col in ["gap", "delta_blocked", "delta_viol", "extra_high_missed_by_A", "gap_component__exact_high_deficit", "gap_component__exact_blocked", "gap_component__exact_violation"]:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"median_{col}"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(outdir: Path, tail_driver: pd.DataFrame, dist: pd.DataFrame, rules: pd.DataFrame, c1: pd.DataFrame, corr: pd.DataFrame, real_present: bool) -> None:
    lines = [
        "# Synthetic tail and mechanism diagnostics report", "",
        "Gap convention: `gap = clean score - A_only score`; positive means A_only is better.", "",
        "## Part 1: tail driver summary", "",
        tail_driver.to_markdown(index=False) if not tail_driver.empty else "No tail rows.", "",
        "## Part 1: distribution stats", "",
        dist.to_markdown(index=False) if not dist.empty else "No distribution rows.", "",
        "## Part 1: top simple rules for catastrophic losers", "",
        rules[rules.get("target", pd.Series(dtype=str)).eq("is_catastrophic_loser")].head(8).to_markdown(index=False) if not rules.empty else "No rule rows.", "",
        "## Part 2: C1 downstream signs", "",
        c1.to_markdown(index=False) if not c1.empty else "No C1 rows.", "",
    ]
    if not real_present:
        lines.append("C1/unified-framework gates are not fully evaluable because no real detailed file was provided.")
        lines.append("")
    lines += [
        "## Part 2: coupling vs tradeoff", "",
        corr.to_markdown(index=False) if not corr.empty else "No coupling rows.", "",
        "## Hard caveat", "",
        "This report is diagnostic. If real detailed files are missing or schema-incompatible, cross-dataset conclusions are not guessed.",
    ]
    (outdir / "diagnostic_decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-detailed", required=True)
    ap.add_argument("--real-detailed", action="append", default=[])
    ap.add_argument("--output-dir", default="synthetic_tail_mechanism_diagnostics")
    ap.add_argument("--gap-col", default=None)
    ap.add_argument("--synthetic-name", default="Synthetic")
    ap.add_argument("--iso-t", type=float, default=60.0)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    syn = standardize_effect_table(Path(args.synthetic_detailed), default_dataset=args.synthetic_name, gap_col_arg=args.gap_col)
    syn["dataset"] = args.synthetic_name
    tail = classify_tails(syn)

    tail_cols = ["dataset", "n", "scenario", "seed", "gap", "dominant_component", "dominant_component_value", "tail_class", "extra_high_missed_by_A", "delta_blocked", "delta_viol"]
    for comp in COMPONENTS:
        tail_cols += [f"gap_component__{comp}", f"base_{comp}", f"treatment_{comp}"]
    tail[[c for c in tail_cols if c in tail.columns]].to_csv(outdir / "synth_tail_per_instance.csv", index=False)

    tail_driver = summarize_tail_drivers(tail)
    tail_driver.to_csv(outdir / "synth_tail_driver_summary.csv", index=False)

    dist = distribution_stats(tail)
    dist.to_csv(outdir / "synth_gap_distribution_stats.csv", index=False)
    plot_synthetic_gap_distribution(tail, outdir, n_target=150)

    rules, tree = predictor_diagnostics(tail)
    rules.to_csv(outdir / "synth_tail_predictor_rules.csv", index=False)
    tree.to_csv(outdir / "synth_tail_decision_tree.csv", index=False)

    selection_artifact_check(tail, iso_t=args.iso_t).to_csv(outdir / "synth_selection_artifact_check.csv", index=False)

    dfs = [tail]
    real_present = False
    for p_text in args.real_detailed:
        p = Path(p_text)
        if not p.exists():
            warnings.warn(f"Real detailed file missing, skipped: {p}")
            continue
        d = standardize_effect_table(p, default_dataset=infer_dataset_name(p, "Real"), gap_col_arg=args.gap_col)
        dfs.append(classify_tails(d))
        real_present = True

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(outdir / "all_standardized_effect_rows.csv", index=False)

    c1 = c1_downstream_signs(all_df)
    c1.to_csv(outdir / "mechanism_C1_downstream_signs.csv", index=False)
    c1[[c for c in ["dataset", "n", "num_cases", "median_component_exact_violation", "mean_component_exact_violation"] if c in c1.columns]].to_csv(outdir / "mechanism_C2_violation_signs.csv", index=False)

    scatter, corr = coupling_scatter_data(all_df)
    scatter.to_csv(outdir / "mechanism_coupling_scatter_data.csv", index=False)
    corr.to_csv(outdir / "mechanism_coupling_correlations.csv", index=False)
    plot_coupling_scatter(scatter, outdir)

    binding_profile(all_df).to_csv(outdir / "mechanism_binding_profile.csv", index=False)
    tail_mechanism_link(tail).to_csv(outdir / "mechanism_tail_link_summary.csv", index=False)

    write_report(outdir, tail_driver, dist, rules, c1, corr, real_present=real_present)

    print(f"Saved diagnostics under: {outdir}")
    print("Key outputs:")
    for name in [
        "synth_tail_per_instance.csv",
        "synth_tail_driver_summary.csv",
        "synth_gap_distribution_stats.csv",
        "synth_tail_predictor_rules.csv",
        "synth_selection_artifact_check.csv",
        "mechanism_C1_downstream_signs.csv",
        "mechanism_coupling_scatter_data.csv",
        "mechanism_coupling_correlations.csv",
        "mechanism_binding_profile.csv",
        "mechanism_tail_link_summary.csv",
        "diagnostic_decision_report.md",
        "synth_gap_distribution_n150.png",
        "mechanism_delta_blocked_delta_viol_scatter.png",
    ]:
        print(" -", outdir / name)


if __name__ == "__main__":
    main()

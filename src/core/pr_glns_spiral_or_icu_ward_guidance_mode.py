
#!/usr/bin/env python3
from __future__ import annotations
import os
"""
pr_glns_spiral_or_icu_ward.py

Stage-3-guided multi-start spiral PR-GLNS for OR-ICU-Ward scheduling.

This is a stronger successor to pr_glns_or_icu_ward.py.

Core ideas
----------
1. Use the three known Pareto points as initial anchors:
   - access-oriented;
   - overtime-controlled access-preserving;
   - aggressive overtime/downstream-controlled.

2. Build an elite pool instead of keeping a single incumbent.

3. Accumulate experience:
   - aggregate day-pressure signals from available Stage-3 results;
   - record which patients/days contribute to downstream pressure;
   - bias neighborhoods toward moving ICU-heavy/ward-heavy patients away from high-pressure days.

4. Spiral exploration:
   - start near each anchor with conservative moves;
   - gradually increase perturbation radius;
   - periodically run exact Stage-3 recourse on elite candidates;
   - accept candidates by Pareto dominance and scalarized exact score.

5. Final output:
   - best schedule by exact Stage-3 score;
   - elite pool summaries;
   - Stage-3 evaluation of the final best schedule.

This is still a heuristic. It is not an exact LBBD/decomposition method.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from surgery_schedule_evaluator import load_instance, evaluate_schedule, save_evaluation_results
from stage2_priority_soft_gurobi_repair_v3_fixed import generate_soft_candidate_placements
from shehadeh_style_integrated_mip_baseline_v2_fixed import _priority_table, build_pool_warm_schedule
from stage3_icu_ward_blocking_flow_mip_fixed import solve_stage3_blocking_flow_mip



# ---------------------------------------------------------------------
# Guidance and wall-clock helpers
# ---------------------------------------------------------------------

def _normalize_guidance_mode(guidance_mode: str) -> str:
    mode = str(guidance_mode or "price").strip().lower().replace("-", "_")
    aliases = {
        "priced": "price",
        "recourse": "price",
        "recourse_price": "price",
        "recourse_guided": "price",
        "none": "off",
        "zero": "off",
        "unguided": "off",
        "no_price": "off",
        "graph_guided": "graph",
        "graph_enhanced": "graph",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"price", "graph", "off"}:
        raise ValueError(f"Unknown guidance_mode={guidance_mode!r}. Use 'price', 'graph', or 'off'.")
    return mode

def remaining_time_limit(start_time, wallclock_limit_s, cap=300.0):
    if wallclock_limit_s and wallclock_limit_s > 0:
        rem = float(wallclock_limit_s) - (time.time() - start_time)
        return max(0.0, min(float(cap), rem))
    return float(cap)

def _resolve_wallclock_limit(wallclock_limit_s=None, time_limit_seconds=None):
    if wallclock_limit_s is not None and float(wallclock_limit_s) > 0:
        return float(wallclock_limit_s)
    raw = os.environ.get("ORSCHE_LNS_WALLCLOCK_LIMIT_S", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except Exception:
            pass
    if time_limit_seconds is not None and float(time_limit_seconds) > 0:
        return float(time_limit_seconds)
    return 0.0

# --- graph-guided LNS patch ---
try:
    from graph_guided_destroy import graph_guided_destroy_set, graph_guided_enabled
except Exception:
    def graph_guided_enabled():
        return False
    def graph_guided_destroy_set(*args, **kwargs):
        return []
# --- end graph-guided LNS patch ---



# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def _horizon(instance: Dict[str, Any]) -> int:
    return int(instance.get("metadata", {}).get("T", int(instance["capacities"]["day_index"].max())))


def _priority_maps(instance: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[int, float], set[int]]:
    pr = _priority_table(instance)
    pclass = dict(zip(pr["patient_id"].astype(int), pr["priority_class"].astype(str)))
    pscore = dict(zip(pr["patient_id"].astype(int), pr["priority_score"].astype(float)))
    high = set(pr[pr["priority_class"].astype(str) == "high"]["patient_id"].astype(int))
    return pclass, pscore, high


def _priority_scale(priority_class: str) -> float:
    p = str(priority_class).lower()
    if p == "high":
        return 0.50
    if p == "medium":
        return 0.80
    return 1.00


def load_day_pressure(path: Optional[str | Path]) -> Dict[int, float]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "day_index" not in df.columns:
        return {}
    if "stage3_pressure_score" in df.columns:
        col = "stage3_pressure_score"
    else:
        needed = ["icu_ready_blocked_stage3", "ward_excess_stage3", "icu_excess_stage3"]
        if all(c in df.columns for c in needed):
            df["stage3_pressure_score"] = (
                df["icu_ready_blocked_stage3"].astype(float)
                + 5.0 * df["ward_excess_stage3"].astype(float)
                + 2.0 * df["icu_excess_stage3"].astype(float)
            )
            col = "stage3_pressure_score"
        else:
            return {}
    out = df[["day_index", col]].copy()
    out["day_index"] = out["day_index"].astype(int)
    out[col] = out[col].fillna(0.0).astype(float)
    maxv = float(out[col].max())
    if maxv > 0:
        out[col] = out[col] / maxv
    return dict(zip(out["day_index"], out[col]))


def aggregate_day_pressures(paths: List[str | Path]) -> Dict[int, float]:
    maps = []
    for p in paths:
        m = load_day_pressure(p)
        if m:
            maps.append(m)
    if not maps:
        return {}

    days = sorted(set().union(*[set(m.keys()) for m in maps]))
    agg = {}
    for d in days:
        vals = [float(m.get(d, 0.0)) for m in maps]
        # Use max + mean blend: stable but still highlights bottleneck days.
        agg[d] = 0.6 * max(vals) + 0.4 * (sum(vals) / len(vals))
    maxv = max(agg.values()) if agg else 0.0
    if maxv > 0:
        agg = {d: v / maxv for d, v in agg.items()}
    return agg


def add_pressure_costs(placements: pd.DataFrame,
                       pressure: Dict[int, float],
                       horizon: int,
                       lambda_ward: float = 1.0,
                       lambda_ready: float = 2.0) -> pd.DataFrame:
    p = placements.copy()
    costs, ready_days, ward_days_str = [], [], []
    for _, r in p.iterrows():
        day = int(r["day_index"])
        requires_icu = int(r.get("requires_icu", 0)) == 1
        icu_len = int(max(0, r.get("icu_treatment_days", 0)))
        ward_los = int(max(0, r.get("ward_los_days", 0)))

        if requires_icu and icu_len > 0:
            ready = day + icu_len
            ward_start = ready
        else:
            ready = day
            ward_start = day

        ward_days = [d for d in range(ward_start, ward_start + ward_los) if 1 <= d <= horizon]
        ward_pressure = sum(float(pressure.get(d, 0.0)) for d in ward_days)
        ready_pressure = float(pressure.get(ready, 0.0)) if 1 <= ready <= horizon else 0.0
        raw = lambda_ward * ward_pressure + lambda_ready * ready_pressure
        scaled = _priority_scale(str(r.get("priority_class", "medium"))) * raw

        costs.append(float(scaled))
        ready_days.append(int(ready))
        ward_days_str.append(",".join(map(str, ward_days)))

    p["spiral_pressure_cost"] = costs
    p["spiral_ready_day"] = ready_days
    p["spiral_ward_days"] = ward_days_str
    return p


def _str_days_to_set(x: Any) -> set[int]:
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    return {int(v) for v in s.split(",") if v.strip()}


def enrich_placements_with_priority(placements: pd.DataFrame, instance: Dict[str, Any]) -> pd.DataFrame:
    pclass, pscore, _ = _priority_maps(instance)
    p = placements.copy()
    p["priority_class"] = p["patient_id"].astype(int).map(pclass).fillna("medium")
    p["priority_score"] = p["patient_id"].astype(int).map(pscore).fillna(2.0)
    return p


def selected_to_schedule(placements: pd.DataFrame, selected: Dict[int, int]) -> pd.DataFrame:
    rows = []
    for pid, j in selected.items():
        r = placements.loc[int(j)]
        start = _safe_float(r.get("start_min", r.get("planned_start_min", 0.0)))
        duration = _safe_float(r.get("duration_min", 0.0))
        end = _safe_float(r.get("end_min", start + duration))
        rows.append({
            "patient_id": int(r["patient_id"]),
            "patient_uid": str(r.get("patient_uid", f"E_{int(r['patient_id']):04d}")),
            "surgery_id": r.get("surgery_id"),
            "specialty": r.get("specialty"),
            "block_id": int(r["block_id"]),
            "or_id": int(r["or_id"]),
            "day": r.get("day"),
            "day_index": int(r["day_index"]),
            "position": 0,
            "planned_start_min": round(start, 3),
            "duration_min": round(duration, 3),
            "planned_end_min": round(end, 3),
            "requires_icu": int(r.get("requires_icu", 0)),
            "icu_treatment_days": int(max(0, r.get("icu_treatment_days", 0))),
            "ward_los_days": int(max(0, r.get("ward_los_days", 0))),
            "surgeon_id": str(r.get("surgeon_id", "")),
        })
    sched = pd.DataFrame(rows)
    if not sched.empty:
        sched = sched.sort_values(["day_index", "or_id", "block_id", "planned_start_min"]).reset_index(drop=True)
        sched["position"] = sched.groupby("block_id")["planned_start_min"].rank(method="first").astype(int)
    return sched


def schedule_to_selected_map(placements: pd.DataFrame, schedule: pd.DataFrame) -> Dict[int, int]:
    selected = {}
    for _, r in schedule.iterrows():
        pid = int(r["patient_id"])
        cand = placements[placements["patient_id"].astype(int) == pid].copy()
        if cand.empty:
            continue
        target_day = int(r.get("day_index", 1))
        target_block = int(r.get("block_id", -1))
        target_start = _safe_float(r.get("planned_start_min", 0.0))
        target_surgeon = str(r.get("surgeon_id", ""))
        score = (
            1000.0 * (cand["day_index"].astype(int) != target_day).astype(float)
            + 500.0 * (cand["block_id"].astype(int) != target_block).astype(float)
            + (cand["start_min"].astype(float) - target_start).abs()
            + 20.0 * (cand["surgeon_id"].astype(str) != target_surgeon).astype(float)
        )
        selected[pid] = int(score.idxmin())
    return selected


def build_augmented_warm(instance: Dict[str, Any],
                         anchor_schedules: List[pd.DataFrame],
                         include_pool: str = "all") -> pd.DataFrame:
    base = pd.concat(anchor_schedules, ignore_index=True).drop_duplicates(subset=["patient_id"], keep="first")
    if include_pool == "none":
        return base

    _, _, high = _priority_maps(instance)
    existing = set(base["patient_id"].astype(int))
    pool = build_pool_warm_schedule(instance)

    if include_pool == "high_only":
        add = pool[(~pool["patient_id"].astype(int).isin(existing)) & (pool["patient_id"].astype(int).isin(high))].copy()
    else:
        add = pool[~pool["patient_id"].astype(int).isin(existing)].copy()

    if add.empty:
        return base
    return pd.concat([base, add], ignore_index=True)


def ensure_anchor_exact_candidates(placements: pd.DataFrame,
                                   anchor_schedules: List[pd.DataFrame]) -> pd.DataFrame:
    p = placements.copy()
    template_cols = list(p.columns)
    rows = []
    for sched in anchor_schedules:
        for _, r in sched.iterrows():
            row = {c: np.nan for c in template_cols}
            for c in [
                "patient_id", "patient_uid", "surgery_id", "specialty", "block_id", "or_id",
                "day", "day_index", "duration_min", "requires_icu", "icu_treatment_days",
                "ward_los_days", "surgeon_id",
            ]:
                if c in r.index and c in row:
                    row[c] = r[c]
            start = _safe_float(r.get("planned_start_min", 0.0))
            duration = _safe_float(r.get("duration_min", 0.0))
            end = _safe_float(r.get("planned_end_min", start + duration))
            row["start_min"] = start
            row["end_min"] = end
            row["duration_min"] = duration
            row["or_overtime_min"] = max(0.0, end - 480.0)
            row["calendar_violation"] = _safe_int(row.get("calendar_violation", 0), 0)
            row["calendar_outside_min"] = _safe_float(row.get("calendar_outside_min", 0.0), 0.0)
            row["same_day"] = 1
            row["same_block"] = 1
            row["same_surgeon"] = 1
            row["start_deviation"] = 0.0
            row["lateness_days"] = 0.0
            rows.append(row)
    if rows:
        p = pd.concat([p, pd.DataFrame(rows)], ignore_index=True)
    return p.reset_index(drop=True)


# ---------------------------------------------------------------------
# Evaluation and elite pool
# ---------------------------------------------------------------------

def fast_evaluate(selected: Dict[int, int],
                  placements: pd.DataFrame,
                  instance: Dict[str, Any],
                  high_ids: set[int],
                  target_volume: int,
                  high_target: int,
                  weights: Dict[str, float]) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    sched = selected_to_schedule(placements, selected)
    eval_res = evaluate_schedule(sched, instance, fill_preferred_surgeon=False)
    summ = eval_res["summary"].iloc[0].to_dict()

    scheduled = _safe_float(summ.get("n_scheduled", len(selected)))
    high_sched = _safe_float(summ.get("n_high_priority_scheduled", 0))
    violation = _safe_float(summ.get("violation_count", 0))
    overtime = _safe_float(summ.get("or_overtime_min", 0))
    blocked = _safe_float(summ.get("blocked_transfer_patient_days", 0))
    icu_excess = _safe_float(summ.get("icu_excess_bed_days_blocking", 0))
    ward_excess = _safe_float(summ.get("ward_excess_bed_days_blocking", 0))
    peak_blocked = _safe_float(summ.get("peak_icu_ready_blocked", 0))

    pressure_cost = 0.0
    for j in selected.values():
        pressure_cost += _safe_float(placements.loc[int(j)].get("spiral_pressure_cost", 0.0), 0.0)

    high_deficit = max(0.0, high_target - high_sched)
    volume_deficit = max(0.0, target_volume - scheduled)
    volume_excess = max(0.0, scheduled - target_volume)

    obj = (
        weights.get("high_deficit", 1_200_000.0) * high_deficit
        + weights.get("volume_deficit", 160_000.0) * volume_deficit
        + weights.get("volume_excess", 15_000.0) * volume_excess
        + weights.get("violation", 4_000.0) * violation
        + weights.get("overtime", 8.0) * overtime
        + weights.get("blocked", 500.0) * blocked
        + weights.get("icu_excess", 120.0) * icu_excess
        + weights.get("ward_excess", 1_000.0) * ward_excess
        + weights.get("peak_blocked", 550.0) * peak_blocked
        + weights.get("pressure", 1_500.0) * pressure_cost
    )

    metrics = dict(summ)
    metrics.update({
        "fast_objective": float(obj),
        "pressure_cost": float(pressure_cost),
        "target_volume": int(target_volume),
        "high_target": int(high_target),
        "high_deficit": float(high_deficit),
        "volume_deficit": float(volume_deficit),
        "volume_excess": float(volume_excess),
    })
    return float(obj), metrics, sched


def exact_stage3_score(stage2_metrics: Dict[str, Any],
                       stage3_metrics: Dict[str, Any],
                       high_target: int,
                       target_volume: int,
                       weights: Dict[str, float]) -> float:
    scheduled = _safe_float(stage2_metrics.get("n_scheduled", 0))
    high_sched = _safe_float(stage2_metrics.get("n_high_priority_scheduled", 0))
    violation = _safe_float(stage2_metrics.get("violation_count", 9999))
    overtime = _safe_float(stage2_metrics.get("or_overtime_min", 9999))
    blocked = _safe_float(stage3_metrics.get("blocked_transfer_patient_days_stage3", 9999))
    icu_excess = _safe_float(stage3_metrics.get("icu_excess_bed_days_stage3", 9999))
    ward_excess = _safe_float(stage3_metrics.get("ward_excess_bed_days_stage3", 9999))
    peak = _safe_float(stage3_metrics.get("peak_icu_ready_blocked_stage3", 9999))

    high_deficit = max(0.0, high_target - high_sched)
    volume_deficit = max(0.0, target_volume - scheduled)
    volume_excess = max(0.0, scheduled - target_volume)

    return (
        weights.get("exact_high_deficit", 1_500_000.0) * high_deficit
        + weights.get("exact_volume_deficit", 200_000.0) * volume_deficit
        + weights.get("exact_volume_excess", 20_000.0) * volume_excess
        + weights.get("exact_blocked", 1_000.0) * blocked
        + weights.get("exact_icu_excess", 250.0) * icu_excess
        + weights.get("exact_ward_excess", 2_500.0) * ward_excess
        + weights.get("exact_peak", 900.0) * peak
        + weights.get("exact_violation", 4_500.0) * violation
        + weights.get("exact_overtime", 8.0) * overtime
    )


def signature(selected: Dict[int, int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(pid), int(j)) for pid, j in selected.items()))


def pareto_dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Dominance over practical metrics. Lower is better except scheduled/high."""
    keys_lower = ["violation_count", "or_overtime_min", "stage3_blocked", "stage3_icu_excess", "stage3_ward_excess", "stage3_peak_blocked"]
    keys_higher = ["n_scheduled", "n_high_priority_scheduled"]

    no_worse = True
    strictly_better = False

    for k in keys_lower:
        av = _safe_float(a.get(k, 999999), 999999)
        bv = _safe_float(b.get(k, 999999), 999999)
        if av > bv + 1e-9:
            no_worse = False
        if av < bv - 1e-9:
            strictly_better = True

    for k in keys_higher:
        av = _safe_float(a.get(k, -999999), -999999)
        bv = _safe_float(b.get(k, -999999), -999999)
        if av < bv - 1e-9:
            no_worse = False
        if av > bv + 1e-9:
            strictly_better = True

    return no_worse and strictly_better


class ElitePool:
    def __init__(self, max_size: int = 12):
        self.max_size = max_size
        self.items: List[Dict[str, Any]] = []
        self.seen = set()

    def add_fast(self, selected, fast_obj, fast_metrics, schedule, source: str):
        sig = signature(selected)
        if sig in self.seen:
            return False
        self.seen.add(sig)
        item = {
            "selected": dict(selected),
            "fast_obj": float(fast_obj),
            "fast_metrics": dict(fast_metrics),
            "schedule": schedule.copy(),
            "source": source,
            "exact_evaluated": False,
            "exact_score": None,
            "stage3_metrics": {},
        }
        self.items.append(item)
        self._trim()
        return True

    def add_exact_result(self, idx: int, exact_score: float, stage3_metrics: Dict[str, Any]):
        self.items[idx]["exact_evaluated"] = True
        self.items[idx]["exact_score"] = float(exact_score)
        self.items[idx]["stage3_metrics"] = dict(stage3_metrics)

    def _combined_sort_key(self, item):
        if item["exact_evaluated"]:
            return (0, item["exact_score"])
        return (1, item["fast_obj"])

    def _trim(self):
        self.items.sort(key=self._combined_sort_key)
        if len(self.items) > self.max_size:
            self.items = self.items[:self.max_size]

    def best(self):
        self.items.sort(key=self._combined_sort_key)
        return self.items[0] if self.items else None

    def sample(self, rng: random.Random):
        if not self.items:
            return None
        weights = [1.0 / (1.0 + i) for i in range(len(self.items))]
        return rng.choices(self.items, weights=weights, k=1)[0]

    def unevaluated_indices(self, top_k: int = 3) -> List[int]:
        self.items.sort(key=self._combined_sort_key)
        idxs = []
        for idx, item in enumerate(self.items):
            if not item["exact_evaluated"]:
                idxs.append(idx)
            if len(idxs) >= top_k:
                break
        return idxs

    def summary_rows(self):
        rows = []
        for rank, item in enumerate(sorted(self.items, key=self._combined_sort_key), start=1):
            m = dict(item["fast_metrics"])
            st3 = dict(item.get("stage3_metrics", {}))
            rows.append({
                "rank": rank,
                "source": item["source"],
                "fast_obj": item["fast_obj"],
                "exact_evaluated": item["exact_evaluated"],
                "exact_score": item["exact_score"],
                "n_scheduled": m.get("n_scheduled"),
                "n_high_priority_scheduled": m.get("n_high_priority_scheduled"),
                "violation_count": m.get("violation_count"),
                "or_overtime_min": m.get("or_overtime_min"),
                "eval_blocked": m.get("blocked_transfer_patient_days"),
                "eval_icu_excess": m.get("icu_excess_bed_days_blocking"),
                "eval_peak_blocked": m.get("peak_icu_ready_blocked"),
                "stage3_blocked": st3.get("blocked_transfer_patient_days_stage3"),
                "stage3_icu_excess": st3.get("icu_excess_bed_days_stage3"),
                "stage3_ward_excess": st3.get("ward_excess_bed_days_stage3"),
                "stage3_peak_blocked": st3.get("peak_icu_ready_blocked_stage3"),
            })
        return rows


# ---------------------------------------------------------------------
# Neighborhoods
# ---------------------------------------------------------------------

def candidate_score_for_insertion(row: pd.Series) -> float:
    return (
        2000.0 * _safe_float(row.get("spiral_pressure_cost", 0.0))
        + 5.0 * _safe_float(row.get("or_overtime_min", 0.0))
        + 2.0 * _safe_float(row.get("calendar_outside_min", 0.0))
        + 0.01 * _safe_float(row.get("duration_min", 0.0))
    )


def choose_best_candidate(placements: pd.DataFrame, pid: int, rng: random.Random, top_k: int = 8) -> Optional[int]:
    cand = placements[placements["patient_id"].astype(int) == int(pid)].copy()
    if cand.empty:
        return None
    score = cand.apply(candidate_score_for_insertion, axis=1)
    idxs = score.nsmallest(min(top_k, len(score))).index.tolist()
    return int(rng.choice(idxs))


def choose_pressure_relocation_candidate(placements: pd.DataFrame,
                                         pid: int,
                                         current_j: int,
                                         rng: random.Random,
                                         radius: int) -> Optional[int]:
    cur = placements.loc[int(current_j)]
    cur_day = int(cur["day_index"])
    cand = placements[placements["patient_id"].astype(int) == int(pid)].copy()
    if cand.empty:
        return None

    # Spiral radius: early cycles search near current day, later cycles wider.
    cand = cand[(cand["day_index"].astype(int) - cur_day).abs() <= max(1, radius)]
    if cand.empty:
        cand = placements[placements["patient_id"].astype(int) == int(pid)].copy()

    score = (
        3000.0 * cand.get("spiral_pressure_cost", pd.Series(0.0, index=cand.index)).astype(float)
        + 6.0 * cand.get("or_overtime_min", pd.Series(0.0, index=cand.index)).astype(float)
        + 2.0 * cand.get("calendar_outside_min", pd.Series(0.0, index=cand.index)).astype(float)
        + 3.0 * (cand["day_index"].astype(int) - cur_day).abs()
    )
    idxs = score.nsmallest(min(8, len(score))).index.tolist()
    return int(rng.choice(idxs))


def removal_score(pid: int, j: int, placements: pd.DataFrame, pclass: Dict[int, str], high_ids: set[int], rng: random.Random) -> float:
    r = placements.loc[int(j)]
    cls = pclass.get(pid, "medium")
    rank = {"low": 0, "medium": 1, "high": 3}.get(cls, 1)
    pressure = _safe_float(r.get("spiral_pressure_cost", 0.0), 0.0)
    duration = _safe_float(r.get("duration_min", 0.0), 0.0)
    overtime = _safe_float(r.get("or_overtime_min", 0.0), 0.0)
    return -5000.0 * rank + 3000.0 * pressure + 0.03 * duration + 0.05 * overtime + rng.random()


def choose_removal(selected: Dict[int, int], placements: pd.DataFrame, pclass: Dict[int, str], high_ids: set[int],
                   rng: random.Random, allow_high: bool = False) -> Optional[int]:
    cand = []
    for pid, j in selected.items():
        if pid in high_ids and not allow_high:
            continue
        cand.append((removal_score(pid, j, placements, pclass, high_ids, rng), pid))
    if not cand:
        return None
    cand.sort(reverse=True)
    return int(rng.choice(cand[:min(5, len(cand))])[1])



# --- graph-guided LNS helper functions ---
def _gg_current_patient_table(selected_map, placements):
    # Build patient-level current-placement table for graph-guided perturbation.
    try:
        import pandas as _pd
    except Exception:
        return None

    if placements is None or not hasattr(placements, "columns"):
        return None

    rows = []
    for _pid, _j in selected_map.items():
        _row = None
        try:
            _jj = int(_j)
            if _jj in placements.index:
                _row = placements.loc[_jj].copy()
        except Exception:
            _row = None

        if _row is None:
            try:
                _cand = placements[placements["patient_id"].astype(int) == int(_pid)]
                if len(_cand) > 0:
                    _row = _cand.iloc[0].copy()
            except Exception:
                _row = None

        if _row is None:
            rows.append({"patient_id": int(_pid)})
            continue

        try:
            _row["patient_id"] = int(_pid)
        except Exception:
            _row["patient_id"] = _pid

        # Aliases recognized by graph_guided_destroy.py.
        if "day_index" in _row.index and "surgery_day" not in _row.index:
            _row["surgery_day"] = _row["day_index"]
        if "pr_glns_pressure_cost" in _row.index and "pressure" not in _row.index:
            _row["pressure"] = _row["pr_glns_pressure_cost"]
        if "block_id" in _row.index and "block" not in _row.index:
            _row["block"] = _row["block_id"]
        if "or_id" in _row.index and "or" not in _row.index:
            _row["or"] = _row["or_id"]

        rows.append(dict(_row))

    try:
        return _pd.DataFrame(rows)
    except Exception:
        return None


def _gg_node_scores_from_placements(placements):
    # Patient-level pressure score used as node weight.
    scores = {}
    try:
        if placements is None or "patient_id" not in placements.columns:
            return scores
        pressure_cols = [
            "pr_glns_pressure_cost",
            "pressure",
            "recourse_score",
            "congestion_score",
            "blocking_score",
            "downstream_score",
        ]
        col = None
        for c in pressure_cols:
            if c in placements.columns:
                col = c
                break
        if col is None:
            return scores
        tmp = placements[["patient_id", col]].copy()
        tmp["patient_id"] = tmp["patient_id"].astype(int)
        tmp[col] = tmp[col].astype(float)
        scores = tmp.groupby("patient_id")[col].max().to_dict()
    except Exception:
        return {}
    return scores


def _gg_patient_order(selected_map, placements, rng, k):
    # Return graph-guided patient ids; fallback to random sample if unavailable.
    try:
        ids = list(selected_map.keys())
        if not ids:
            return []
        k = max(1, min(int(k), len(ids)))
        scores = _gg_node_scores_from_placements(placements)
        table = _gg_current_patient_table(selected_map, placements)
        out = graph_guided_destroy_set(
            candidate_ids=ids,
            node_scores=scores,
            patient_table=table,
            destroy_size=k,
            rng=rng,
        )
        valid = set(int(x) for x in ids)
        out = [int(x) for x in out if int(x) in valid]
        if out:
            return out[:k]
    except Exception:
        pass

    try:
        return rng.sample(list(selected_map.keys()), k=k)
    except Exception:
        return list(selected_map.keys())[:k]
# --- end graph-guided LNS helper functions ---


def propose_neighbor(selected: Dict[int, int],
                     placements: pd.DataFrame,
                     pclass: Dict[int, str],
                     high_ids: set[int],
                     target_volume: int,
                     rng: random.Random,
                     radius: int,
                     phase: str,
                     guidance_mode: str = "price") -> Tuple[Dict[int, int], str]:
    new = dict(selected)
    all_pids = set(placements["patient_id"].astype(int).unique())
    scheduled = set(new.keys())
    missing_high = list(high_ids - scheduled)

    if phase == "pressure":
        probs = {
            "pressure_relocate": 0.40,
            "icu_heavy_relocate": 0.20,
            "insert_high": 0.20,
            "replace": 0.15,
            "multi_shift": 0.05,
        }
    elif phase == "overtime":
        probs = {
            "overtime_relocate": 0.40,
            "pressure_relocate": 0.20,
            "insert_high": 0.15,
            "replace": 0.15,
            "multi_shift": 0.10,
        }
    else:
        probs = {
            "pressure_relocate": 0.25,
            "overtime_relocate": 0.20,
            "insert_high": 0.20,
            "replace": 0.20,
            "multi_shift": 0.15,
        }

    moves = list(probs.keys())
    weights = [probs[m] for m in moves]
    move = rng.choices(moves, weights=weights, k=1)[0]

    if move == "pressure_relocate" and new:
        # Pick a currently high pressure placement.
        scored = []
        for pid, j in new.items():
            r = placements.loc[int(j)]
            pressure = _safe_float(r.get("spiral_pressure_cost", 0.0), 0.0)
            scored.append((pressure + 0.02 * _safe_float(r.get("duration_min", 0.0), 0.0), pid))
        scored.sort(reverse=True)
        pid = int(rng.choice(scored[:max(1, min(6, len(scored)))])[1])
        jj = choose_pressure_relocation_candidate(placements, pid, new[pid], rng, radius)
        if jj is not None:
            new[pid] = jj

    elif move == "icu_heavy_relocate" and new:
        icu_pids = []
        for pid, j in new.items():
            r = placements.loc[int(j)]
            score = int(r.get("requires_icu", 0)) * (1 + _safe_float(r.get("icu_treatment_days", 0)) + _safe_float(r.get("ward_los_days", 0)))
            if score > 0:
                icu_pids.append((score + rng.random(), pid))
        if icu_pids:
            icu_pids.sort(reverse=True)
            pid = int(rng.choice(icu_pids[:min(8, len(icu_pids))])[1])
            jj = choose_pressure_relocation_candidate(placements, pid, new[pid], rng, radius)
            if jj is not None:
                new[pid] = jj

    elif move == "overtime_relocate" and new:
        overtime_pids = []
        for pid, j in new.items():
            ot = _safe_float(placements.loc[int(j)].get("or_overtime_min", 0.0), 0.0)
            if ot > 0:
                overtime_pids.append((ot + rng.random(), pid))
        if overtime_pids:
            overtime_pids.sort(reverse=True)
            pid = int(rng.choice(overtime_pids[:min(8, len(overtime_pids))])[1])
            cand = placements[placements["patient_id"].astype(int) == pid].copy()
            if not cand.empty:
                score = (
                    10.0 * cand.get("or_overtime_min", pd.Series(0.0, index=cand.index)).astype(float)
                    + 1500.0 * cand.get("spiral_pressure_cost", pd.Series(0.0, index=cand.index)).astype(float)
                    + 2.0 * cand.get("calendar_outside_min", pd.Series(0.0, index=cand.index)).astype(float)
                )
                jj = int(rng.choice(score.nsmallest(min(8, len(score))).index.tolist()))
                new[pid] = jj

    elif move == "insert_high":
        if missing_high:
            pid_add = int(rng.choice(missing_high))
            j_add = choose_best_candidate(placements, pid_add, rng)
            if j_add is not None:
                if len(new) >= target_volume:
                    pid_rm = choose_removal(new, placements, pclass, high_ids, rng, allow_high=False)
                    if pid_rm is not None:
                        new.pop(pid_rm, None)
                new[pid_add] = j_add

    elif move == "replace":
        unscheduled = list(all_pids - scheduled)
        if unscheduled:
            w = []
            for pid in unscheduled:
                cls = pclass.get(int(pid), "medium")
                w.append({"high": 10.0, "medium": 3.0, "low": 1.0}.get(cls, 2.0))
            pid_add = int(rng.choices(unscheduled, weights=w, k=1)[0])
            j_add = choose_best_candidate(placements, pid_add, rng)
            if j_add is not None:
                if len(new) >= target_volume:
                    pid_rm = choose_removal(new, placements, pclass, high_ids, rng, allow_high=False)
                    if pid_rm is not None:
                        new.pop(pid_rm, None)
                new[pid_add] = j_add

    elif move == "multi_shift" and new:
        k = rng.randint(2, min(max(2, radius + 1), len(new)))
        if guidance_mode == "graph" and graph_guided_enabled():
            pids = _gg_patient_order(new, placements, rng, k)
        else:
            pids = rng.sample(list(new.keys()), k=k)
        for pid in pids:
            jj = choose_pressure_relocation_candidate(placements, pid, new[pid], rng, radius)
            if jj is not None:
                new[pid] = jj

    return new, move


def spiral_perturb(selected: Dict[int, int],
                   placements: pd.DataFrame,
                   pclass: Dict[int, str],
                   high_ids: set[int],
                   target_volume: int,
                   rng: random.Random,
                   radius: int) -> Dict[int, int]:
    new = dict(selected)
    remove_frac = min(0.25, 0.06 + 0.03 * radius)
    k_remove = max(1, int(round(remove_frac * len(new))))
    removable = [pid for pid in new if pid not in high_ids]
    removable.sort(key=lambda pid: removal_score(pid, new[pid], placements, pclass, high_ids, rng), reverse=True)
    for pid in removable[:k_remove]:
        new.pop(pid, None)

    all_pids = set(placements["patient_id"].astype(int).unique())
    missing_high = list(high_ids - set(new.keys()))
    rng.shuffle(missing_high)
    fill = missing_high + list(all_pids - set(new.keys()))
    for pid in fill:
        if len(new) >= target_volume:
            break
        jj = choose_best_candidate(placements, int(pid), rng)
        if jj is not None:
            new[int(pid)] = jj
    return new


# ---------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------

def run_spiral_pr_glns(instance: Dict[str, Any],
                       anchor_schedules: Dict[str, pd.DataFrame],
                       output_dir: str | Path,
                       pressure_day_paths: Optional[List[str | Path]] = None,
                       include_pool: str = "all",
                       target_volume: Optional[int] = None,
                       high_target: Optional[int] = None,
                       cycles: int = 5,
                       proposals_per_cycle: int = 80,
                       exact_every: int = 25,
                       exact_top_k: int = 3,
                       seed: int = 202706,
                       max_or_overtime: int = 180,
                       time_limit_seconds: Optional[int] = None,
                       wallclock_limit_s: Optional[float] = None,
                       guidance_mode: str = "price",
                       verbose: bool = True) -> Dict[str, Any]:

    rng = random.Random(seed)
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    guidance_mode = _normalize_guidance_mode(guidance_mode)
    wallclock_limit_s = _resolve_wallclock_limit(wallclock_limit_s, time_limit_seconds)
    if wallclock_limit_s > 0:
        # The spiral search is cycle/proposal controlled, not iteration controlled.
        # Let wall-clock checks terminate the loop.
        cycles = 10**9

    pclass, pscore, high_ids = _priority_maps(instance)
    T = _horizon(instance)

    anchor_list = list(anchor_schedules.values())
    if not anchor_list:
        raise ValueError("At least one anchor schedule is required.")

    if target_volume is None:
        # Use the largest anchor volume as target.
        target_volume = max(int(s["patient_id"].nunique()) for s in anchor_list)
    if high_target is None:
        high_target = len(high_ids)

    pressure = aggregate_day_pressures(pressure_day_paths or [])
    warm = build_augmented_warm(instance, anchor_list, include_pool=include_pool)
    placements = generate_soft_candidate_placements(
        instance=instance,
        warm_schedule=warm,
        slot_minutes=30,
        max_or_overtime=max_or_overtime,
    )
    placements = ensure_anchor_exact_candidates(placements, anchor_list)
    placements = enrich_placements_with_priority(placements, instance)
    placements = add_pressure_costs(placements, pressure, horizon=T)
    if guidance_mode == "off":
        # True price-off ablation: the spiral framework is unchanged, but the
        # downstream recourse-price signal is removed at its source.
        placements["spiral_pressure_cost"] = 0.0
    placements.to_csv(out / "spiral_candidate_placements.csv", index=False)

    weights_fast = {
        "high_deficit": 1_200_000.0,
        "volume_deficit": 160_000.0,
        "volume_excess": 15_000.0,
        "violation": 4_000.0,
        "overtime": 8.0,
        "blocked": 500.0,
        "icu_excess": 120.0,
        "ward_excess": 1_000.0,
        "peak_blocked": 550.0,
        "pressure": float(os.environ.get("RECOURSE_PRESSURE_WEIGHT", "0.0")) if guidance_mode == "off" else 1_500.0,
    }
    weights_exact = {
        "exact_high_deficit": 1_500_000.0,
        "exact_volume_deficit": 200_000.0,
        "exact_volume_excess": 20_000.0,
        "exact_blocked": 1_000.0,
        "exact_icu_excess": 250.0,
        "exact_ward_excess": 2_500.0,
        "exact_peak": 900.0,
        "exact_violation": 4_500.0,
        "exact_overtime": 8.0,
    }

    pool = ElitePool(max_size=16)

    # Seed elite pool with all anchors.
    for name, sched in anchor_schedules.items():
        sel = schedule_to_selected_map(placements, sched)
        obj, metrics, ss = fast_evaluate(sel, placements, instance, high_ids, target_volume, high_target, weights_fast)
        pool.add_fast(sel, obj, metrics, ss, source=f"anchor:{name}")

    # Exact-evaluate anchors first. This is the "experience accumulation" step.
    trace = []
    for idx in pool.unevaluated_indices(top_k=min(len(pool.items), exact_top_k + 5)):
        tl = remaining_time_limit(start_time, wallclock_limit_s, cap=300.0)
        if wallclock_limit_s > 0 and tl <= 0:
            break
        item = pool.items[idx]
        st3_dir = out / "stage3_elite" / f"init_{idx}"
        res3 = solve_stage3_blocking_flow_mip(
            schedule=item["schedule"],
            instance=instance,
            output_dir=st3_dir,
            time_limit=(tl if wallclock_limit_s > 0 else 300),
            mip_gap=0.01,
            objective_mode="capacity_first",
            allow_ward_excess=True,
            verbose=False,
        )
        st3 = res3["stage3_summary"].iloc[0].to_dict()
        score = exact_stage3_score(item["fast_metrics"], st3, high_target, target_volume, weights_exact)
        pool.add_exact_result(idx, score, st3)

    phases = ["balanced", "pressure", "overtime", "pressure", "balanced"]
    iteration = 0

    for cyc in range(1, cycles + 1):
        radius = min(T, 1 + cyc)
        phase = phases[(cyc - 1) % len(phases)]
        if verbose:
            print(f"\n=== Spiral cycle {cyc}/{cycles}: phase={phase}, radius={radius} ===")

        for k in range(1, proposals_per_cycle + 1):
            iteration += 1
            if (
                (wallclock_limit_s > 0 and time.time() - start_time >= wallclock_limit_s)
                or (time_limit_seconds is not None and time.time() - start_time > time_limit_seconds)
            ):
                break

            parent = pool.sample(rng)
            if parent is None:
                break
            base_selected = parent["selected"]

            # Every few proposals, use a larger perturbation restart.
            if k % max(10, exact_every // 2) == 0:
                cand_selected = spiral_perturb(base_selected, placements, pclass, high_ids, target_volume, rng, radius)
                move = "spiral_perturb"
            else:
                cand_selected, move = propose_neighbor(
                    base_selected, placements, pclass, high_ids, target_volume, rng, radius=radius, phase=phase,
                    guidance_mode=guidance_mode
                )

            obj, metrics, sched = fast_evaluate(
                cand_selected, placements, instance, high_ids, target_volume, high_target, weights_fast
            )

            # Keep candidates that improve scalar fast score or add diversity.
            added = False
            best = pool.best()
            if best is None or obj < best["fast_obj"] * 1.05 or rng.random() < 0.08:
                added = pool.add_fast(cand_selected, obj, metrics, sched, source=f"cycle{cyc}:{move}")

            if iteration % exact_every == 0:
                for idx in pool.unevaluated_indices(top_k=exact_top_k):
                    tl = remaining_time_limit(start_time, wallclock_limit_s, cap=300.0)
                    if wallclock_limit_s > 0 and tl <= 0:
                        break
                    item = pool.items[idx]
                    st3_dir = out / "stage3_elite" / f"it{iteration}_rank{idx}"
                    try:
                        res3 = solve_stage3_blocking_flow_mip(
                            schedule=item["schedule"],
                            instance=instance,
                            output_dir=st3_dir,
                            time_limit=(tl if wallclock_limit_s > 0 else 300),
                            mip_gap=0.01,
                            objective_mode="capacity_first",
                            allow_ward_excess=True,
                            verbose=False,
                        )
                        st3 = res3["stage3_summary"].iloc[0].to_dict()
                        score = exact_stage3_score(item["fast_metrics"], st3, high_target, target_volume, weights_exact)
                        pool.add_exact_result(idx, score, st3)
                    except Exception as e:
                        item["exact_evaluated"] = True
                        item["exact_score"] = float("inf")
                        item["stage3_metrics"] = {"stage3_error": f"{type(e).__name__}: {e}"}

            if iteration % 10 == 0 or added:
                best_now = pool.best()
                bm = best_now["fast_metrics"]
                bs = best_now.get("stage3_metrics", {})
                trace.append({
                    "iteration": iteration,
                    "cycle": cyc,
                    "phase": phase,
                    "radius": radius,
                    "move": move,
                    "candidate_added": int(added),
                    "candidate_fast_obj": obj,
                    "best_fast_obj": best_now["fast_obj"],
                    "best_exact_evaluated": int(best_now["exact_evaluated"]),
                    "best_exact_score": best_now["exact_score"],
                    "best_n_scheduled": bm.get("n_scheduled"),
                    "best_high": bm.get("n_high_priority_scheduled"),
                    "best_violation": bm.get("violation_count"),
                    "best_overtime": bm.get("or_overtime_min"),
                    "best_eval_blocked": bm.get("blocked_transfer_patient_days"),
                    "best_stage3_blocked": bs.get("blocked_transfer_patient_days_stage3"),
                    "best_stage3_icu_excess": bs.get("icu_excess_bed_days_stage3"),
                    "best_stage3_peak": bs.get("peak_icu_ready_blocked_stage3"),
                    "elapsed_sec": time.time() - start_time,
                })
                pd.DataFrame(trace).to_csv(out / "spiral_trace.csv", index=False)
                pd.DataFrame(pool.summary_rows()).to_csv(out / "spiral_elite_pool.csv", index=False)

        if (
            (wallclock_limit_s > 0 and time.time() - start_time >= wallclock_limit_s)
            or (time_limit_seconds is not None and time.time() - start_time > time_limit_seconds)
        ):
            break

    # Exact-evaluate remaining top elite items.
    for idx in pool.unevaluated_indices(top_k=8):
        tl = remaining_time_limit(start_time, wallclock_limit_s, cap=300.0)
        if wallclock_limit_s > 0 and tl <= 0:
            break
        item = pool.items[idx]
        st3_dir = out / "stage3_elite" / f"final_rank{idx}"
        try:
            res3 = solve_stage3_blocking_flow_mip(
                schedule=item["schedule"],
                instance=instance,
                output_dir=st3_dir,
                time_limit=(tl if wallclock_limit_s > 0 else 300),
                mip_gap=0.01,
                objective_mode="capacity_first",
                allow_ward_excess=True,
                verbose=False,
            )
            st3 = res3["stage3_summary"].iloc[0].to_dict()
            score = exact_stage3_score(item["fast_metrics"], st3, high_target, target_volume, weights_exact)
            pool.add_exact_result(idx, score, st3)
        except Exception as e:
            item["exact_evaluated"] = True
            item["exact_score"] = float("inf")
            item["stage3_metrics"] = {"stage3_error": f"{type(e).__name__}: {e}"}

    pool.items.sort(key=pool._combined_sort_key)
    best = pool.best()
    if best is None:
        raise RuntimeError("Spiral PR-GLNS failed to keep any elite solution.")

    best_schedule = best["schedule"].copy()
    best_schedule.to_csv(out / "spiral_pr_glns_schedule.csv", index=False)
    eval_res = evaluate_schedule(best_schedule, instance, fill_preferred_surgeon=False)
    save_evaluation_results(eval_res, out / "evaluation")

    # Final Stage-3 result for the final best. In wall-clock spot-check mode,
    # cap the final MIP by the remaining budget so a post-processing MIP cannot
    # consume far more time than the search budget.
    final_stage3_dir = out / "stage3_results"
    tl = remaining_time_limit(start_time, wallclock_limit_s, cap=300.0)
    if wallclock_limit_s > 0 and tl <= 0:
        final_stage3 = {"stage3_status": "skipped_no_remaining_wallclock"}
    else:
        final_res3 = solve_stage3_blocking_flow_mip(
            schedule=best_schedule,
            instance=instance,
            output_dir=final_stage3_dir,
            time_limit=(tl if wallclock_limit_s > 0 else 300),
            mip_gap=0.01,
            objective_mode="capacity_first",
            allow_ward_excess=True,
            verbose=verbose,
        )
        final_stage3 = final_res3["stage3_summary"].iloc[0].to_dict()

    pd.DataFrame(pool.summary_rows()).to_csv(out / "spiral_elite_pool.csv", index=False)

    meta = {
        "seed": seed,
        "guidance_mode": guidance_mode,
        "wallclock_limit_s": wallclock_limit_s,
        "cycles": cycles,
        "proposals_per_cycle": proposals_per_cycle,
        "exact_every": exact_every,
        "exact_top_k": exact_top_k,
        "target_volume": int(target_volume),
        "high_target": int(high_target),
        "anchors": list(anchor_schedules.keys()),
        "runtime_sec": time.time() - start_time,
        "best_source": best["source"],
        "best_fast_objective": best["fast_obj"],
        "best_exact_score": best["exact_score"],
        "best_evaluator_summary": eval_res["summary"].iloc[0].to_dict(),
        "best_stage3_summary": final_stage3,
    }
    (out / "spiral_pr_glns_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    if verbose:
        print("\n=== Spiral PR-GLNS final evaluator summary ===")
        print(eval_res["summary"].to_string(index=False))
        print("\n=== Spiral PR-GLNS final Stage-3 summary ===")
        print(pd.DataFrame([final_stage3]).to_string(index=False))
        print(f"\nBest source: {best['source']}")

    return meta


def load_existing_anchors(anchor_paths: Dict[str, str | Path]) -> Dict[str, pd.DataFrame]:
    anchors = {}
    for name, p in anchor_paths.items():
        p = Path(p)
        if p.exists():
            anchors[name] = pd.read_csv(p)
    if not anchors:
        raise FileNotFoundError("No anchor schedules found.")
    return anchors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage-3-guided spiral PR-GLNS.")
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--output-dir", default="spiral_pr_glns_results")
    parser.add_argument("--anchor", action="append", nargs=2, metavar=("NAME", "PATH"),
                        help="Anchor schedule as NAME PATH. Can be repeated.")
    parser.add_argument("--pressure-day", action="append", default=[], help="Stage-3 feedback day pressure CSV. Can be repeated.")
    parser.add_argument("--target-volume", type=int, default=-1)
    parser.add_argument("--high-target", type=int, default=-1)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--proposals-per-cycle", type=int, default=80)
    parser.add_argument("--exact-every", type=int, default=25)
    parser.add_argument("--exact-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202706)
    parser.add_argument("--guidance-mode", choices=["price", "graph", "off"], default="price",
                        help="price: recourse-priced spiral LNS; graph: price + graph-guided destroy; off: same spiral framework with recourse price zeroed.")
    parser.add_argument("--wallclock-limit-s", type=float, default=None,
                        help="Optional wall-clock limit in seconds. Overrides ORSCHE_LNS_WALLCLOCK_LIMIT_S when positive.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    instance = load_instance(args.instance_dir)
    anchors = load_existing_anchors({name: path for name, path in (args.anchor or [])})

    run_spiral_pr_glns(
        instance=instance,
        anchor_schedules=anchors,
        output_dir=args.output_dir,
        pressure_day_paths=args.pressure_day,
        target_volume=None if args.target_volume < 0 else args.target_volume,
        high_target=None if args.high_target < 0 else args.high_target,
        cycles=args.cycles,
        proposals_per_cycle=args.proposals_per_cycle,
        exact_every=args.exact_every,
        exact_top_k=args.exact_top_k,
        seed=args.seed,
        wallclock_limit_s=args.wallclock_limit_s,
        guidance_mode=args.guidance_mode,
        verbose=not args.quiet,
    )
# [recourse recalibration patch] pressure weight can be overridden by RECOURSE_PRESSURE_WEIGHT.

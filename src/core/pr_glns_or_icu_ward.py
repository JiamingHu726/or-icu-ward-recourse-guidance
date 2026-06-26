
#!/usr/bin/env python3
from __future__ import annotations
import os
"""
pr_glns_or_icu_ward.py

Perturbation-Restart Generalized Large-Neighborhood Search (PR-GLNS)
for OR-ICU-Ward scheduling.

Purpose
-------
This is a strong heuristic search layer that starts from an existing schedule
(e.g., PP-HPR tuned result, HP-forced feedback result, or Stage-2 v3 result)
and explores large neighborhoods by:

    - shifting a patient's placement;
    - replacing low/medium-priority cases with unscheduled high-priority cases;
    - replacing one scheduled case with another candidate patient;
    - multi-patient reassignment;
    - perturbation restart after search stagnation.

The search uses the normal evaluator as a fast surrogate and optionally uses
Stage-3 day-pressure as a placement-level downstream penalty. The final best
schedule is evaluated by the Stage-3 ICU-to-Ward blocking-flow MIP.

Important positioning
---------------------
This is not an exact decomposition algorithm. It is a strong heuristic baseline
or an optional intensification layer on top of PP-HPR.

Dependencies
------------
    surgery_schedule_evaluator.py
    stage2_priority_soft_gurobi_repair_v3_fixed.py
    shehadeh_style_integrated_mip_baseline_v2_fixed.py
    stage3_icu_ward_blocking_flow_mip_fixed.py
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
# Utilities
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


def _priority_maps(instance: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[int, float], set[int]]:
    pr = _priority_table(instance)
    pclass = dict(zip(pr["patient_id"].astype(int), pr["priority_class"].astype(str)))
    pscore = dict(zip(pr["patient_id"].astype(int), pr["priority_score"].astype(float)))
    high = set(pr[pr["priority_class"].astype(str) == "high"]["patient_id"].astype(int))
    return pclass, pscore, high


def load_day_pressure(path: Optional[str | Path]) -> Dict[int, float]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "day_index" not in df.columns:
        return {}
    col = "stage3_pressure_score" if "stage3_pressure_score" in df.columns else None
    if col is None:
        candidates = ["icu_ready_blocked_stage3", "ward_excess_stage3", "icu_excess_stage3"]
        if all(c in df.columns for c in candidates):
            df["stage3_pressure_score"] = (
                df["icu_ready_blocked_stage3"].astype(float)
                + 5.0 * df["ward_excess_stage3"].astype(float)
                + 2.0 * df["icu_excess_stage3"].astype(float)
            )
            col = "stage3_pressure_score"
        else:
            return {}
    scores = df[["day_index", col]].copy()
    scores["day_index"] = scores["day_index"].astype(int)
    scores[col] = scores[col].fillna(0.0).astype(float)
    maxv = float(scores[col].max())
    if maxv > 0:
        scores[col] = scores[col] / maxv
    return dict(zip(scores["day_index"], scores[col]))


def _priority_scale(priority_class: str) -> float:
    p = str(priority_class).lower()
    if p == "high":
        return 0.50
    if p == "medium":
        return 0.80
    return 1.00


def add_pressure_costs(placements: pd.DataFrame,
                       pressure: Dict[int, float],
                       horizon: int,
                       lambda_ward: float = 1.0,
                       lambda_ready: float = 2.0) -> pd.DataFrame:
    p = placements.copy()
    costs = []
    ward_days_str = []
    ready_days = []

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

        days = [d for d in range(ward_start, ward_start + ward_los) if 1 <= d <= horizon]
        ward_pressure = sum(float(pressure.get(d, 0.0)) for d in days)
        ready_pressure = float(pressure.get(ready, 0.0)) if 1 <= ready <= horizon else 0.0
        raw = lambda_ward * ward_pressure + lambda_ready * ready_pressure
        scaled = _priority_scale(str(r.get("priority_class", "medium"))) * raw

        costs.append(float(scaled))
        ward_days_str.append(",".join(map(str, days)))
        ready_days.append(int(ready))

    p["pr_glns_pressure_cost"] = costs
    p["pr_glns_ward_days"] = ward_days_str
    p["pr_glns_ready_day"] = ready_days
    return p


def build_augmented_warm_schedule(instance: Dict[str, Any],
                                  initial_schedule: pd.DataFrame,
                                  include_pool: str = "all") -> pd.DataFrame:
    """Build warm schedule for candidate generation.

    include_pool:
        all        -> add all patients not in the initial schedule
        high_only  -> add missing high-priority patients only
        none       -> only initial scheduled patients
    """
    init = initial_schedule.copy()
    existing = set(init["patient_id"].astype(int))

    if include_pool == "none":
        return init

    pclass, _, high = _priority_maps(instance)
    pool = build_pool_warm_schedule(instance)

    if include_pool == "high_only":
        add = pool[(~pool["patient_id"].astype(int).isin(existing)) & (pool["patient_id"].astype(int).isin(high))].copy()
    else:
        add = pool[~pool["patient_id"].astype(int).isin(existing)].copy()

    if add.empty:
        return init

    # Keep generated placeholders after current schedule rows.
    out = pd.concat([init, add], ignore_index=True)
    return out


def ensure_initial_exact_candidates(placements: pd.DataFrame,
                                    initial_schedule: pd.DataFrame) -> pd.DataFrame:
    """Append exact initial schedule placements to guarantee a valid starting point."""
    p = placements.copy()
    template_cols = list(p.columns)
    rows = []

    for _, r in initial_schedule.iterrows():
        row = {c: np.nan for c in template_cols}
        pid = int(r["patient_id"])
        for c in [
            "patient_id", "patient_uid", "surgery_id", "specialty", "block_id", "or_id",
            "day", "day_index", "duration_min", "requires_icu", "icu_treatment_days",
            "ward_los_days", "surgeon_id",
        ]:
            if c in r.index and c in row:
                row[c] = r[c]

        if "start_min" in row:
            row["start_min"] = float(r.get("planned_start_min", 0.0))
        if "end_min" in row:
            row["end_min"] = float(r.get("planned_end_min", row.get("start_min", 0.0) + r.get("duration_min", 0.0)))
        if "planned_start_min" in row:
            row["planned_start_min"] = float(r.get("planned_start_min", 0.0))
        if "planned_end_min" in row:
            row["planned_end_min"] = float(r.get("planned_end_min", row.get("start_min", 0.0) + r.get("duration_min", 0.0)))

        start = _safe_float(row.get("start_min", 0.0))
        end = _safe_float(row.get("end_min", start + _safe_float(row.get("duration_min", 0.0))))
        duration = _safe_float(row.get("duration_min", end - start))
        row["duration_min"] = duration
        row["start_min"] = start
        row["end_min"] = end
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


def initial_selected_map(placements: pd.DataFrame, initial_schedule: pd.DataFrame) -> Dict[int, int]:
    selected = {}
    for _, r in initial_schedule.iterrows():
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


# ---------------------------------------------------------------------
# Objective and search
# ---------------------------------------------------------------------

def evaluate_solution(selected: Dict[int, int],
                      placements: pd.DataFrame,
                      instance: Dict[str, Any],
                      pclass: Dict[int, str],
                      high_ids: set[int],
                      target_volume: int,
                      high_target: Optional[int] = None,
                      weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    if weights is None:
        weights = {}

    high_target = len(high_ids) if high_target is None else int(high_target)

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

    selected_pressure = 0.0
    if "pr_glns_pressure_cost" in placements.columns:
        for j in selected.values():
            selected_pressure += _safe_float(placements.loc[int(j), "pr_glns_pressure_cost"], 0.0)

    high_deficit = max(0.0, float(high_target) - high_sched)
    volume_deficit = max(0.0, float(target_volume) - scheduled)
    volume_excess = max(0.0, scheduled - float(target_volume))

    obj = (
        weights.get("high_deficit", 1_000_000.0) * high_deficit
        + weights.get("volume_deficit", 120_000.0) * volume_deficit
        + weights.get("volume_excess", 15_000.0) * volume_excess
        + weights.get("violation", 5_000.0) * violation
        + weights.get("overtime", 12.0) * overtime
        + weights.get("blocked", 650.0) * blocked
        + weights.get("icu_excess", 180.0) * icu_excess
        + weights.get("ward_excess", 800.0) * ward_excess
        + weights.get("peak_blocked", 500.0) * peak_blocked
        + weights.get("pressure", 300.0) * selected_pressure
    )

    metrics = dict(summ)
    metrics.update({
        "objective": obj,
        "selected_pressure_cost": selected_pressure,
        "target_volume": target_volume,
        "high_target": high_target,
        "high_deficit": high_deficit,
        "volume_deficit": volume_deficit,
        "volume_excess": volume_excess,
    })
    return float(obj), metrics, sched


def random_candidate_for_patient(placements: pd.DataFrame, pid: int, rng: random.Random) -> Optional[int]:
    idxs = placements.index[placements["patient_id"].astype(int) == int(pid)].tolist()
    if not idxs:
        return None
    return int(rng.choice(idxs))


def best_pressure_candidate_for_patient(placements: pd.DataFrame, pid: int, rng: random.Random) -> Optional[int]:
    cand = placements[placements["patient_id"].astype(int) == int(pid)].copy()
    if cand.empty:
        return None
    # Use low pressure + low overtime as an insertion heuristic.
    score = (
        cand.get("pr_glns_pressure_cost", pd.Series(0.0, index=cand.index)).astype(float)
        + 0.01 * cand.get("or_overtime_min", pd.Series(0.0, index=cand.index)).astype(float)
        + 0.001 * cand.get("calendar_outside_min", pd.Series(0.0, index=cand.index)).astype(float)
    )
    best = score.nsmallest(min(5, len(score))).index.tolist()
    return int(rng.choice(best))


def choose_removal_patient(selected: Dict[int, int],
                           placements: pd.DataFrame,
                           pclass: Dict[int, str],
                           high_ids: set[int],
                           rng: random.Random,
                           allow_high: bool = False) -> Optional[int]:
    candidates = []
    for pid, j in selected.items():
        if (pid in high_ids) and not allow_high:
            continue
        cls = pclass.get(pid, "medium")
        rank = {"low": 0, "medium": 1, "high": 2}.get(cls, 1)
        pressure = _safe_float(placements.loc[int(j)].get("pr_glns_pressure_cost", 0.0), 0.0)
        duration = _safe_float(placements.loc[int(j)].get("duration_min", 0.0), 0.0)
        # Lower rank is more removable; higher pressure/duration is more removable.
        score = -1000.0 * rank + 10.0 * pressure + 0.01 * duration + rng.random()
        candidates.append((score, pid))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top = candidates[:max(1, min(5, len(candidates)))]
    return int(rng.choice(top)[1])


def propose_neighbor(selected: Dict[int, int],
                     placements: pd.DataFrame,
                     pclass: Dict[int, str],
                     high_ids: set[int],
                     target_volume: int,
                     rng: random.Random,
                     move_probs: Optional[Dict[str, float]] = None) -> Dict[int, int]:
    if move_probs is None:
        move_probs = {
            "shift": 0.35,
            "multi_shift": 0.20,
            "insert_missing_high": 0.20,
            "replace": 0.20,
            "drop_low": 0.05,
        }

    moves = list(move_probs.keys())
    probs = np.array([move_probs[m] for m in moves], dtype=float)
    probs = probs / probs.sum()
    move = rng.choices(moves, weights=probs, k=1)[0]

    new = dict(selected)
    selected_pids = set(new.keys())
    all_pids = set(placements["patient_id"].astype(int).unique())

    if move == "shift" and new:
        pid = int(rng.choice(list(new.keys())))
        j = random_candidate_for_patient(placements, pid, rng)
        if j is not None:
            new[pid] = j

    elif move == "multi_shift" and new:
        k = rng.randint(2, min(5, max(2, len(new))))
        for pid in rng.sample(list(new.keys()), k=k):
            j = random_candidate_for_patient(placements, int(pid), rng)
            if j is not None:
                new[int(pid)] = j

    elif move == "insert_missing_high":
        missing = list(high_ids - selected_pids)
        if missing:
            pid_add = int(rng.choice(missing))
            j_add = best_pressure_candidate_for_patient(placements, pid_add, rng)
            if j_add is not None:
                if len(new) >= target_volume:
                    pid_rm = choose_removal_patient(new, placements, pclass, high_ids, rng, allow_high=False)
                    if pid_rm is not None:
                        new.pop(pid_rm, None)
                new[pid_add] = j_add
        else:
            # No missing high: fall back to shift.
            if new:
                pid = int(rng.choice(list(new.keys())))
                j = random_candidate_for_patient(placements, pid, rng)
                if j is not None:
                    new[pid] = j

    elif move == "replace":
        unscheduled = list(all_pids - selected_pids)
        if unscheduled:
            # Priority-biased insertion.
            weights = []
            for pid in unscheduled:
                cls = pclass.get(int(pid), "medium")
                weights.append({"high": 10.0, "medium": 3.0, "low": 1.0}.get(cls, 2.0))
            pid_add = int(rng.choices(unscheduled, weights=weights, k=1)[0])
            j_add = best_pressure_candidate_for_patient(placements, pid_add, rng)
            if j_add is not None:
                if len(new) >= target_volume:
                    pid_rm = choose_removal_patient(new, placements, pclass, high_ids, rng, allow_high=False)
                    if pid_rm is not None:
                        new.pop(pid_rm, None)
                new[pid_add] = j_add

    elif move == "drop_low" and len(new) > max(1, target_volume - 2):
        pid_rm = choose_removal_patient(new, placements, pclass, high_ids, rng, allow_high=False)
        if pid_rm is not None:
            new.pop(pid_rm, None)

    return new


def perturb_solution(selected: Dict[int, int],
                     placements: pd.DataFrame,
                     pclass: Dict[int, str],
                     high_ids: set[int],
                     target_volume: int,
                     rng: random.Random,
                     remove_frac: float = 0.12) -> Dict[int, int]:
    new = dict(selected)
    removable = [pid for pid in new if pid not in high_ids]
    rng.shuffle(removable)
    k_remove = max(1, int(round(remove_frac * max(1, len(new)))))
    for pid in removable[:k_remove]:
        new.pop(pid, None)

    all_pids = list(set(placements["patient_id"].astype(int).unique()) - set(new.keys()))
    # Refill: missing high first, then priority-biased low-pressure candidates.
    missing_high = list(high_ids - set(new.keys()))
    rng.shuffle(missing_high)
    fill_order = missing_high + all_pids
    seen = set()

    for pid in fill_order:
        pid = int(pid)
        if pid in seen or pid in new:
            continue
        seen.add(pid)
        if len(new) >= target_volume:
            break
        j = best_pressure_candidate_for_patient(placements, pid, rng)
        if j is not None:
            new[pid] = j
    return new


def run_pr_glns_search(instance: Dict[str, Any],
                       initial_schedule: pd.DataFrame,
                       output_dir: str | Path,
                       pressure_day_path: Optional[str | Path] = None,
                       include_pool: str = "all",
                       slot_minutes: int = 30,
                       max_or_overtime: int = 150,
                       iterations: int = 500,
                       restart_after: int = 80,
                       perturb_frac: float = 0.12,
                       seed: int = 123,
                       target_volume: Optional[int] = None,
                       high_target: Optional[int] = None,
                       time_limit_seconds: Optional[int] = None,
                       weights: Optional[Dict[str, float]] = None,
                       final_stage3: bool = True,
                       verbose: bool = True) -> Dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    wallclock_limit_s = float(os.environ.get("ORSCHE_LNS_WALLCLOCK_LIMIT_S", "0") or 0)
    if wallclock_limit_s > 0:
        iterations = 10**9  # Just a large number.
        print(f"Wallclock limit: {wallclock_limit_s} seconds")
        print(f"Setting iterations to {iterations}")

    pclass, pscore, high_ids = _priority_maps(instance)
    T = int(instance.get("metadata", {}).get("T", int(instance["capacities"]["day_index"].max())))

    if target_volume is None:
        target_volume = int(initial_schedule["patient_id"].nunique())
    if high_target is None:
        high_target = len(high_ids)

    warm = build_augmented_warm_schedule(instance, initial_schedule, include_pool=include_pool)
    placements = generate_soft_candidate_placements(
        instance=instance,
        warm_schedule=warm,
        slot_minutes=slot_minutes,
        max_or_overtime=max_or_overtime,
    )
    placements = ensure_initial_exact_candidates(placements, initial_schedule)
    placements = enrich_placements_with_priority(placements, instance)
    pressure = load_day_pressure(pressure_day_path)
    placements = add_pressure_costs(placements, pressure, horizon=T)

    placements.to_csv(out / "pr_glns_candidate_placements.csv", index=False)

    current = initial_selected_map(placements, initial_schedule)
    cur_obj, cur_metrics, cur_sched = evaluate_solution(
        current, placements, instance, pclass, high_ids,
        target_volume=target_volume, high_target=high_target, weights=weights,
    )
    best = dict(current)
    best_obj = cur_obj
    best_metrics = dict(cur_metrics)
    best_sched = cur_sched.copy()

    trace = []
    no_improve = 0
    temperature0 = max(1.0, 0.05 * abs(cur_obj))

    if verbose:
        print(f"Initial objective: {cur_obj:.3f}")
        print(cur_metrics)

    for it in range(1, iterations + 1):
        if wallclock_limit_s > 0 and time.time() - start_time >= wallclock_limit_s:
            break
        
        if no_improve >= restart_after:
            candidate = perturb_solution(
                best, placements, pclass, high_ids, target_volume, rng, remove_frac=perturb_frac
            )
            no_improve = 0
            move_type = "perturb_restart"
        else:
            candidate = propose_neighbor(current, placements, pclass, high_ids, target_volume, rng)
            move_type = "neighbor"

        cand_obj, cand_metrics, cand_sched = evaluate_solution(
            candidate, placements, instance, pclass, high_ids,
            target_volume=target_volume, high_target=high_target, weights=weights,
        )

        # Simulated-annealing style acceptance for non-improving moves.
        delta = cand_obj - cur_obj
        temp = max(1e-9, temperature0 * (1.0 - it / max(1, iterations)))
        accept = delta <= 0 or rng.random() < math.exp(-delta / temp)

        if accept:
            current = candidate
            cur_obj = cand_obj
            cur_metrics = cand_metrics
            cur_sched = cand_sched

        improved = cand_obj < best_obj
        if improved:
            best = dict(candidate)
            best_obj = cand_obj
            best_metrics = dict(cand_metrics)
            best_sched = cand_sched.copy()
            no_improve = 0
        else:
            no_improve += 1

        if it == 1 or it % 10 == 0 or improved:
            trace.append({
                "iteration": it,
                "move_type": move_type,
                "accepted": int(accept),
                "improved_best": int(improved),
                "current_objective": cur_obj,
                "candidate_objective": cand_obj,
                "best_objective": best_obj,
                "best_n_scheduled": best_metrics.get("n_scheduled"),
                "best_high_priority": best_metrics.get("n_high_priority_scheduled"),
                "best_violation": best_metrics.get("violation_count"),
                "best_overtime": best_metrics.get("or_overtime_min"),
                "best_eval_blocked": best_metrics.get("blocked_transfer_patient_days"),
                "best_eval_icu_excess": best_metrics.get("icu_excess_bed_days_blocking"),
                "best_eval_peak_blocked": best_metrics.get("peak_icu_ready_blocked"),
                "elapsed_sec": time.time() - start_time,
            })
            pd.DataFrame(trace).to_csv(out / "pr_glns_trace.csv", index=False)

        if verbose and (it % 50 == 0 or improved):
            print(
                f"it={it:4d}, best={best_obj:.2f}, "
                f"n={best_metrics.get('n_scheduled')}, "
                f"high={best_metrics.get('n_high_priority_scheduled')}, "
                f"vio={best_metrics.get('violation_count')}, "
                f"ot={best_metrics.get('or_overtime_min')}, "
                f"blocked={best_metrics.get('blocked_transfer_patient_days')}"
            )

    # Save final best.
    best_sched.to_csv(out / "pr_glns_schedule.csv", index=False)
    eval_res = evaluate_schedule(best_sched, instance, fill_preferred_surgeon=False)
    save_evaluation_results(eval_res, out / "evaluation")

    final_stage3_summary = {}
    if final_stage3:
        stage3_dir = out / "stage3_results"
        res3 = solve_stage3_blocking_flow_mip(
            schedule=best_sched,
            instance=instance,
            output_dir=stage3_dir,
            time_limit=300,
            mip_gap=0.01,
            objective_mode="capacity_first",
            allow_ward_excess=True,
            verbose=verbose,
        )
        final_stage3_summary = res3["stage3_summary"].iloc[0].to_dict()

    metadata = {
        "seed": seed,
        "iterations_requested": iterations,
        "iterations_completed": trace[-1]["iteration"] if trace else 0,
        "target_volume": int(target_volume),
        "high_target": int(high_target),
        "include_pool": include_pool,
        "slot_minutes": slot_minutes,
        "max_or_overtime": max_or_overtime,
        "restart_after": restart_after,
        "perturb_frac": perturb_frac,
        "best_objective": best_obj,
        "best_eval_metrics": best_metrics,
        "final_stage3_summary": final_stage3_summary,
        "runtime_sec": time.time() - start_time,
    }
    (out / "pr_glns_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    if verbose:
        print("\n=== PR-GLNS final evaluator summary ===")
        print(eval_res["summary"].to_string(index=False))
        if final_stage3_summary:
            print("\n=== PR-GLNS final Stage-3 summary ===")
            print(pd.DataFrame([final_stage3_summary]).to_string(index=False))

    return metadata


def run_pr_glns_from_paths(instance_dir: str | Path,
                           initial_schedule_path: str | Path,
                           output_dir: str | Path,
                           pressure_day_path: Optional[str | Path] = None,
                           iterations: int = 500,
                           seed: int = 123,
                           include_pool: str = "all",
                           target_volume: Optional[int] = None,
                           high_target: Optional[int] = None,
                           verbose: bool = True) -> Dict[str, Any]:
    instance = load_instance(instance_dir)
    initial_schedule = pd.read_csv(initial_schedule_path)
    return run_pr_glns_search(
        instance=instance,
        initial_schedule=initial_schedule,
        output_dir=output_dir,
        pressure_day_path=pressure_day_path,
        include_pool=include_pool,
        iterations=iterations,
        seed=seed,
        target_volume=target_volume,
        high_target=high_target,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PR-GLNS for OR-ICU-Ward scheduling.")
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--initial-schedule", required=True)
    parser.add_argument("--output-dir", default="pr_glns_results")
    parser.add_argument("--pressure-day", default=None)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-pool", choices=["all", "high_only", "none"], default="all")
    parser.add_argument("--target-volume", type=int, default=-1)
    parser.add_argument("--high-target", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_pr_glns_from_paths(
        instance_dir=args.instance_dir,
        initial_schedule_path=args.initial_schedule,
        output_dir=args.output_dir,
        pressure_day_path=args.pressure_day,
        iterations=args.iterations,
        seed=args.seed,
        include_pool=args.include_pool,
        target_volume=None if args.target_volume < 0 else args.target_volume,
        high_target=None if args.high_target < 0 else args.high_target,
        verbose=not args.quiet,
    )

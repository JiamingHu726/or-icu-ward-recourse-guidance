
#!/usr/bin/env python3
from __future__ import annotations

"""
stage3_icu_ward_blocking_flow_mip_fixed.py

Stage-3 ICU-Ward blocking-flow MIP for the OR-ICU-Ward scheduling project.

Purpose
-------
Given a fixed surgery schedule from BA-HLA / Stage-2 Gurobi repair, optimize
ICU-to-Ward transfer timing under ward capacity.

This layer does NOT change OR assignments. It answers:

    Given the surgery days and ICU treatment completion days,
    how should ICU-ready patients be transferred to ward beds so that
    downstream blocking is minimized?

Outputs
-------
    stage3_summary.csv
    stage3_daily_flow.csv
    stage3_patient_transfer_plan.csv
    stage3_patient_day_states.csv
    stage3_feedback_day_pressure.csv
    stage3_feedback_patient_pressure.csv
    stage3_solver_metadata.json
    greedy_evaluation/...

The feedback CSVs are designed for the next step:
    Stage-3 pressure -> feedback penalty/cuts -> Stage-1/Stage-2 re-optimization.

Dependencies
------------
    gurobipy
    surgery_schedule_evaluator.py
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

from surgery_schedule_evaluator import load_instance, evaluate_schedule, save_evaluation_results


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated column names created by repeated merge/enrichment.

    Pandas returns a DataFrame rather than a Series when selecting a duplicated
    column name. That is exactly what triggers:
        NotImplementedError: cannot align with a higher dimensional NDFrame
    inside surgery_schedule_evaluator._ensure_schedule_fields().
    """
    return df.loc[:, ~df.columns.duplicated()].copy()


def _canonical_schedule_for_evaluator(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return a clean schedule table for evaluate_schedule().

    Stage-3 internally enriches schedules with priority/surgery metadata. The
    evaluator should only receive the canonical schedule columns, otherwise
    repeated merges can create duplicate/suffixed columns.
    """
    s = _drop_duplicate_columns(schedule)

    # Fill canonical columns from *_from_surgery fields if needed.
    for c in ["surgery_id", "specialty", "duration_min", "requires_icu", "icu_treatment_days", "ward_los_days"]:
        alt = f"{c}_from_surgery"
        if c not in s.columns and alt in s.columns:
            s[c] = s[alt]
        elif c in s.columns and alt in s.columns:
            s[c] = s[c].where(s[c].notna(), s[alt])

    cols = [
        "patient_id", "patient_uid", "surgery_id", "specialty",
        "block_id", "or_id", "day", "day_index", "position",
        "planned_start_min", "duration_min", "planned_end_min",
        "requires_icu", "icu_treatment_days", "ward_los_days",
        "surgeon_id",
    ]
    cols = [c for c in cols if c in s.columns]
    return s[cols].copy()


# ---------------------------------------------------------------------
# Safe Gurobi attributes
# ---------------------------------------------------------------------

def _safe_model_float_attr(model, attr_name: str, default=None):
    try:
        val = getattr(model, attr_name)
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _safe_solution_count(model) -> int:
    try:
        return int(model.SolCount)
    except Exception:
        return 0


# ---------------------------------------------------------------------
# Loading and enrichment helpers
# ---------------------------------------------------------------------

def _priority_table(instance: Dict[str, Any]) -> pd.DataFrame:
    surgeries = instance["surgeries"][["patient_id", "surgery_id", "specialty"]].copy()
    pr = instance.get("patient_priority")
    if isinstance(pr, pd.DataFrame) and not pr.empty:
        return surgeries.merge(pr, on=["patient_id", "surgery_id", "specialty"], how="left")

    out = surgeries.copy()
    out["priority_class"] = "medium"
    out["priority_score"] = 2.0
    out["release_day"] = 1
    out["due_day"] = 7
    return out


def _ensure_schedule_fields(schedule: pd.DataFrame, instance: Dict[str, Any]) -> pd.DataFrame:
    """Minimal schedule normalization for Stage-3."""
    s = _drop_duplicate_columns(schedule.copy())
    surgeries = _drop_duplicate_columns(instance["surgeries"].copy())

    base_cols = [
        "patient_id", "surgery_id", "specialty", "duration_min",
        "requires_icu", "icu_treatment_days", "icu_los_days", "ward_los_days",
    ]
    base_cols = [c for c in base_cols if c in surgeries.columns]
    base = surgeries[base_cols].copy()
    s = s.merge(base, on="patient_id", how="left", suffixes=("", "_from_surgery"))
    s = _drop_duplicate_columns(s)

    for c in ["surgery_id", "specialty", "duration_min", "requires_icu", "icu_treatment_days", "ward_los_days"]:
        alt = f"{c}_from_surgery"
        if c not in s.columns and alt in s.columns:
            s[c] = s[alt]
        elif c in s.columns and alt in s.columns:
            s[c] = s[c].where(s[c].notna(), s[alt])

    if "requires_icu" not in s.columns:
        if "icu_treatment_days" in s.columns:
            s["requires_icu"] = (s["icu_treatment_days"].fillna(0).astype(float) > 0).astype(int)
        elif "icu_los_days" in s.columns:
            s["requires_icu"] = (s["icu_los_days"].fillna(0).astype(float) > 0).astype(int)
        else:
            s["requires_icu"] = 0

    if "icu_treatment_days" not in s.columns:
        if "icu_los_days" in s.columns:
            s["icu_treatment_days"] = s["icu_los_days"]
        else:
            s["icu_treatment_days"] = 0

    if "ward_los_days" not in s.columns:
        s["ward_los_days"] = 0

    if "patient_uid" not in s.columns:
        s["patient_uid"] = s["patient_id"].astype(int).map(lambda x: f"E_{x:04d}")

    if "day_index" not in s.columns:
        raise ValueError("schedule must contain day_index for Stage-3")

    pr = _priority_table(instance)
    keep = ["patient_id", "priority_class", "priority_score", "due_day", "release_day"]
    keep = [c for c in keep if c in pr.columns]
    s = s.merge(pr[keep], on="patient_id", how="left", suffixes=("", "_from_priority"))
    s = _drop_duplicate_columns(s)
    if "priority_class" not in s.columns:
        s["priority_class"] = "medium"
    if "priority_score" not in s.columns:
        s["priority_score"] = 2.0

    return s


def _horizon(instance: Dict[str, Any]) -> int:
    caps = instance["capacities"]
    return int(instance.get("metadata", {}).get("T", int(caps["day_index"].max())))


def _capacity_maps(instance: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    caps = instance["capacities"].copy()
    icu = dict(zip(caps["day_index"].astype(int), caps["icu_capacity"].astype(int)))
    ward = dict(zip(caps["day_index"].astype(int), caps["ward_capacity"].astype(int)))
    day_name = dict(zip(caps["day_index"].astype(int), caps["day"].astype(str)))
    return icu, ward, day_name


# ---------------------------------------------------------------------
# Stage-3 flow construction
# ---------------------------------------------------------------------

def build_stage3_flow_inputs(schedule: pd.DataFrame, instance: Dict[str, Any]) -> Dict[str, Any]:
    """Build fixed loads and transfer candidates for the ICU-Ward flow MIP."""
    T = _horizon(instance)
    icu_cap, ward_cap, day_name = _capacity_maps(instance)

    schedule = _ensure_schedule_fields(schedule, instance)

    fixed_ward = {d: 0 for d in range(1, T + 1)}
    fixed_icu_treating = {d: 0 for d in range(1, T + 1)}
    candidates = []

    # Existing ward patients: fixed ward occupancy until planned discharge day - 1.
    current_ward = instance.get("current_ward", pd.DataFrame())
    if isinstance(current_ward, pd.DataFrame) and not current_ward.empty:
        for _, r in current_ward.iterrows():
            discharge_day = int(r.get("planned_discharge_day", 1))
            for d in range(1, min(discharge_day, T + 1)):
                fixed_ward[d] += 1

    # Existing ICU patients.
    current_icu = instance.get("current_icu", pd.DataFrame())
    if isinstance(current_icu, pd.DataFrame) and not current_icu.empty:
        for _, r in current_icu.iterrows():
            uid = str(r["patient_uid"])
            ready_day = int(r.get("ready_for_ward_day", 1))
            ward_los = int(r.get("ward_los_days", 0))

            # ICU treating before ready day.
            for d in range(1, min(ready_day, T + 1)):
                fixed_icu_treating[d] += 1

            if ward_los > 0 and ready_day <= T:
                candidates.append({
                    "flow_id": f"CICU::{uid}",
                    "patient_uid": uid,
                    "patient_id": None,
                    "source": "current_icu",
                    "surgery_id": None,
                    "specialty": None,
                    "surgery_day": None,
                    "ready_day": ready_day,
                    "ward_los_days": ward_los,
                    "priority_class": "current",
                    "priority_score": 10.0,
                })

    # Scheduled elective patients.
    for _, r in schedule.iterrows():
        uid = str(r.get("patient_uid", f"E_{int(r['patient_id']):04d}"))
        pid = int(r["patient_id"])
        surgery_day = int(r["day_index"])
        ward_los = int(max(0, r.get("ward_los_days", 0)))
        requires_icu = int(r.get("requires_icu", 0)) == 1
        icu_len = int(max(0, r.get("icu_treatment_days", r.get("icu_los_days", 0))))

        if requires_icu and icu_len > 0:
            ready_day = surgery_day + icu_len

            # ICU treating before ready day.
            for d in range(surgery_day, min(ready_day, T + 1)):
                if 1 <= d <= T:
                    fixed_icu_treating[d] += 1

            if ward_los > 0 and ready_day <= T:
                candidates.append({
                    "flow_id": f"ELECTIVE::{uid}",
                    "patient_uid": uid,
                    "patient_id": pid,
                    "source": "elective_icu",
                    "surgery_id": r.get("surgery_id"),
                    "specialty": r.get("specialty"),
                    "surgery_day": surgery_day,
                    "ready_day": ready_day,
                    "ward_los_days": ward_los,
                    "priority_class": str(r.get("priority_class", "medium")),
                    "priority_score": float(r.get("priority_score", 2.0)),
                })

        else:
            # Direct ward admissions are fixed at surgery day.
            if ward_los > 0:
                for d in range(surgery_day, min(surgery_day + ward_los, T + 1)):
                    if 1 <= d <= T:
                        fixed_ward[d] += 1

    candidates_df = pd.DataFrame(candidates)
    return {
        "T": T,
        "icu_cap": icu_cap,
        "ward_cap": ward_cap,
        "day_name": day_name,
        "fixed_ward": fixed_ward,
        "fixed_icu_treating": fixed_icu_treating,
        "transfer_candidates": candidates_df,
        "schedule": schedule,
    }


def _priority_delay_weight(row: pd.Series, current_weight: float = 1.25, high_weight: float = 1.15) -> float:
    source = str(row.get("source", ""))
    pclass = str(row.get("priority_class", ""))
    if source == "current_icu":
        return current_weight
    if pclass == "high":
        return high_weight
    return 1.0


# ---------------------------------------------------------------------
# MIP solver
# ---------------------------------------------------------------------

def solve_stage3_blocking_flow_mip(schedule: pd.DataFrame,
                                   instance: Dict[str, Any],
                                   output_dir: Optional[str | Path] = None,
                                   time_limit: int = 300,
                                   mip_gap: float = 0.01,
                                   objective_mode: str = "capacity_first",
                                   allow_ward_excess: bool = True,
                                   verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Solve the ICU-to-Ward transfer-flow MIP for a fixed surgery schedule."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("gurobipy is required for Stage-3 blocking-flow MIP.") from e

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    flow = build_stage3_flow_inputs(schedule, instance)
    T = int(flow["T"])
    day_name = flow["day_name"]
    ward_cap = flow["ward_cap"]
    icu_cap = flow["icu_cap"]
    fixed_ward = flow["fixed_ward"]
    fixed_icu_treating = flow["fixed_icu_treating"]
    cand = flow["transfer_candidates"].copy()
    sch = flow["schedule"].copy()

    # Also save the greedy evaluator for comparison.
    # Use a canonical schedule table to avoid duplicated/suffixed columns from
    # Stage-3 enrichment leaking into surgery_schedule_evaluator.
    greedy_eval = evaluate_schedule(_canonical_schedule_for_evaluator(sch), instance, fill_preferred_surgeon=False)
    if out is not None:
        save_evaluation_results(greedy_eval, out / "greedy_evaluation")

    model = gp.Model("stage3_icu_ward_blocking_flow")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1 if verbose else 0

    # Empty candidate case: only fixed occupancy matters.
    if cand.empty:
        daily = []
        for d in range(1, T + 1):
            ward_occ = fixed_ward[d]
            icu_occ = fixed_icu_treating[d]
            daily.append({
                "day_index": d,
                "day": day_name.get(d, str(d)),
                "ward_capacity": ward_cap[d],
                "fixed_ward_occupancy": fixed_ward[d],
                "optimized_transfer_ward_occupancy": 0,
                "ward_occupancy_stage3": ward_occ,
                "ward_excess_stage3": max(0, ward_occ - ward_cap[d]),
                "icu_capacity": icu_cap[d],
                "icu_treating_stage3": fixed_icu_treating[d],
                "icu_ready_blocked_stage3": 0,
                "icu_occupancy_stage3": icu_occ,
                "icu_excess_stage3": max(0, icu_occ - icu_cap[d]),
                "transfer_count": 0,
            })
        daily_df = pd.DataFrame(daily)
        summary = pd.DataFrame([{
            "n_transfer_candidates": 0,
            "n_transferred_within_horizon": 0,
            "blocked_transfer_patient_days_stage3": 0,
            "ward_excess_bed_days_stage3": int(daily_df["ward_excess_stage3"].sum()),
            "icu_excess_bed_days_stage3": int(daily_df["icu_excess_stage3"].sum()),
            "peak_icu_ready_blocked_stage3": 0,
        }])
        results = {
            "stage3_summary": summary,
            "stage3_daily_flow": daily_df,
            "stage3_patient_transfer_plan": pd.DataFrame(),
            "stage3_patient_day_states": pd.DataFrame(),
            "stage3_feedback_day_pressure": daily_df.copy(),
            "stage3_feedback_patient_pressure": pd.DataFrame(),
        }
        if out is not None:
            for name, df in results.items():
                df.to_csv(out / f"{name}.csv", index=False)
        return results

    # Variable sets.
    patient_ids = cand.index.tolist()
    transfer_days: Dict[int, List[int]] = {}
    for i, r in cand.iterrows():
        ready = int(r["ready_day"])
        transfer_days[i] = list(range(ready, T + 1))

    q = {}
    for i in patient_ids:
        for d in transfer_days[i]:
            q[i, d] = model.addVar(vtype=GRB.BINARY, name=f"q[{i},{d}]")

    untransferred = model.addVars(patient_ids, vtype=GRB.BINARY, name="untransferred")
    ward_excess = model.addVars(range(1, T + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="ward_excess")

    # Each candidate transfers once within horizon or remains blocked through horizon.
    for i in patient_ids:
        model.addConstr(
            gp.quicksum(q[i, d] for d in transfer_days[i]) + untransferred[i] == 1,
            name=f"transfer_or_wait[{i}]"
        )

    # Ward capacity constraints.
    for d in range(1, T + 1):
        occ_terms = []
        for i, r in cand.iterrows():
            los = int(r["ward_los_days"])
            for tau in transfer_days[i]:
                if tau <= d < tau + los:
                    occ_terms.append(q[i, tau])

        if allow_ward_excess:
            model.addConstr(
                fixed_ward[d] + gp.quicksum(occ_terms) <= ward_cap[d] + ward_excess[d],
                name=f"ward_capacity_soft[{d}]"
            )
        else:
            model.addConstr(
                fixed_ward[d] + gp.quicksum(occ_terms) <= ward_cap[d],
                name=f"ward_capacity_hard[{d}]"
            )
            model.addConstr(ward_excess[d] == 0.0, name=f"ward_excess_zero[{d}]")

    # Objective expressions.
    ward_excess_expr = gp.quicksum(ward_excess[d] for d in range(1, T + 1))

    raw_block_terms = []
    weighted_block_terms = []
    transfer_tiebreak_terms = []

    for i, r in cand.iterrows():
        ready = int(r["ready_day"])
        horizon_block = max(0, T - ready + 1)
        weight = _priority_delay_weight(r)

        raw_block_terms.append(horizon_block * untransferred[i])
        weighted_block_terms.append(weight * horizon_block * untransferred[i])

        for d in transfer_days[i]:
            delay = max(0, d - ready)
            raw_block_terms.append(delay * q[i, d])
            weighted_block_terms.append(weight * delay * q[i, d])
            transfer_tiebreak_terms.append(0.001 * d * q[i, d])

    raw_block_expr = gp.quicksum(raw_block_terms)
    weighted_block_expr = gp.quicksum(weighted_block_terms)
    tiebreak_expr = gp.quicksum(transfer_tiebreak_terms)

    mode = str(objective_mode).lower().strip()
    if mode not in {"capacity_first", "blocking_first", "weighted"}:
        raise ValueError("objective_mode must be one of: capacity_first, blocking_first, weighted")

    model.ModelSense = GRB.MINIMIZE

    if mode == "capacity_first":
        # Closest to hard-capacity transfer planning:
        # first avoid ward overflow, then minimize blocked ICU-ready patient-days.
        model.setObjectiveN(ward_excess_expr, index=0, priority=3, weight=1.0, name="min_ward_excess")
        model.setObjectiveN(raw_block_expr, index=1, priority=2, weight=1.0, name="min_raw_blocked_days")
        model.setObjectiveN(weighted_block_expr + tiebreak_expr, index=2, priority=1, weight=1.0, name="priority_tiebreak")
    elif mode == "blocking_first":
        # Allows ward overflow if it substantially reduces ICU blocking.
        model.setObjectiveN(raw_block_expr, index=0, priority=3, weight=1.0, name="min_raw_blocked_days")
        model.setObjectiveN(ward_excess_expr, index=1, priority=2, weight=1.0, name="min_ward_excess")
        model.setObjectiveN(weighted_block_expr + tiebreak_expr, index=2, priority=1, weight=1.0, name="priority_tiebreak")
    else:
        # Single weighted objective: useful for sensitivity analysis.
        model.setObjective(10000.0 * ward_excess_expr + weighted_block_expr + tiebreak_expr, GRB.MINIMIZE)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        if out is not None:
            model.computeIIS()
            model.write(str(out / "stage3_blocking_flow_infeasible.ilp"))
        raise RuntimeError("Stage-3 MIP infeasible. IIS written if output_dir is provided.")

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi did not find a usable Stage-3 solution. Status={model.Status}")

    # Extract transfer plan.
    transfer_rows = []
    transfer_day_by_i: Dict[int, Optional[int]] = {}
    for i, r in cand.iterrows():
        chosen_day = None
        for d in transfer_days[i]:
            if q[i, d].X > 0.5:
                chosen_day = d
                break
        transfer_day_by_i[i] = chosen_day
        ready = int(r["ready_day"])
        if chosen_day is None:
            blocked_days = max(0, T - ready + 1)
        else:
            blocked_days = max(0, chosen_day - ready)

        transfer_rows.append({
            "flow_index": int(i),
            "flow_id": r["flow_id"],
            "patient_uid": r["patient_uid"],
            "patient_id": r.get("patient_id"),
            "source": r["source"],
            "surgery_id": r.get("surgery_id"),
            "specialty": r.get("specialty"),
            "surgery_day": r.get("surgery_day"),
            "ready_day": ready,
            "transfer_day": chosen_day,
            "transferred_within_horizon": int(chosen_day is not None),
            "ward_los_days": int(r["ward_los_days"]),
            "blocked_days_stage3": int(blocked_days),
            "priority_class": r.get("priority_class"),
            "priority_score": r.get("priority_score"),
        })

    transfer_plan = pd.DataFrame(transfer_rows)

    # Build daily flow and patient-day states.
    daily_rows = []
    state_rows = []

    for d in range(1, T + 1):
        transfer_count = 0
        transfer_ward_occ = 0
        icu_ready_blocked = 0

        for i, r in cand.iterrows():
            ready = int(r["ready_day"])
            los = int(r["ward_los_days"])
            td = transfer_day_by_i[i]

            # ICU ready but not yet transferred.
            if ready <= d and (td is None or td > d):
                icu_ready_blocked += 1
                state_rows.append({
                    "day_index": d,
                    "day": day_name.get(d, str(d)),
                    "flow_id": r["flow_id"],
                    "patient_uid": r["patient_uid"],
                    "patient_id": r.get("patient_id"),
                    "source": r["source"],
                    "state": "ICU_READY_BLOCKED",
                    "occupies_icu": 1,
                    "occupies_ward": 0,
                    "ready_for_ward": 1,
                    "blocked_in_icu": 1,
                    "surgery_id": r.get("surgery_id"),
                    "specialty": r.get("specialty"),
                })

            if td is not None and td == d:
                transfer_count += 1

            if td is not None and td <= d < td + los:
                transfer_ward_occ += 1
                state_rows.append({
                    "day_index": d,
                    "day": day_name.get(d, str(d)),
                    "flow_id": r["flow_id"],
                    "patient_uid": r["patient_uid"],
                    "patient_id": r.get("patient_id"),
                    "source": r["source"],
                    "state": "WARD",
                    "occupies_icu": 0,
                    "occupies_ward": 1,
                    "ready_for_ward": 0,
                    "blocked_in_icu": 0,
                    "surgery_id": r.get("surgery_id"),
                    "specialty": r.get("specialty"),
                })

        ward_occ = fixed_ward[d] + transfer_ward_occ
        icu_occ = fixed_icu_treating[d] + icu_ready_blocked

        daily_rows.append({
            "day_index": d,
            "day": day_name.get(d, str(d)),
            "ward_capacity": ward_cap[d],
            "fixed_ward_occupancy": fixed_ward[d],
            "optimized_transfer_ward_occupancy": transfer_ward_occ,
            "ward_occupancy_stage3": ward_occ,
            "ward_excess_stage3": max(0, ward_occ - ward_cap[d]),
            "icu_capacity": icu_cap[d],
            "icu_treating_stage3": fixed_icu_treating[d],
            "icu_ready_blocked_stage3": icu_ready_blocked,
            "icu_occupancy_stage3": icu_occ,
            "icu_excess_stage3": max(0, icu_occ - icu_cap[d]),
            "transfer_count": transfer_count,
        })

    daily_flow = pd.DataFrame(daily_rows)
    patient_day_states = pd.DataFrame(state_rows)

    # Feedback day pressure.
    feedback_day = daily_flow.copy()
    feedback_day["ward_load_ratio"] = feedback_day["ward_occupancy_stage3"] / feedback_day["ward_capacity"].clip(lower=1)
    feedback_day["icu_load_ratio"] = feedback_day["icu_occupancy_stage3"] / feedback_day["icu_capacity"].clip(lower=1)
    feedback_day["stage3_pressure_score"] = (
        feedback_day["icu_ready_blocked_stage3"].astype(float)
        + 5.0 * feedback_day["ward_excess_stage3"].astype(float)
        + 2.0 * feedback_day["icu_excess_stage3"].astype(float)
        + feedback_day["ward_load_ratio"].clip(lower=0.8).sub(0.8).fillna(0.0)
    )

    # Patient feedback pressure for scheduled elective surgeries.
    pressure_by_day = dict(zip(feedback_day["day_index"].astype(int), feedback_day["stage3_pressure_score"].astype(float)))
    transfer_plan_by_uid = transfer_plan.set_index("patient_uid").to_dict("index") if not transfer_plan.empty else {}

    feedback_patient_rows = []
    for _, r in sch.iterrows():
        uid = str(r.get("patient_uid", f"E_{int(r['patient_id']):04d}"))
        pid = int(r["patient_id"])
        surgery_day = int(r["day_index"])
        ward_los = int(max(0, r.get("ward_los_days", 0)))
        requires_icu = int(r.get("requires_icu", 0)) == 1
        icu_len = int(max(0, r.get("icu_treatment_days", r.get("icu_los_days", 0))))

        actual_blocked = 0
        ward_days = []

        if requires_icu and icu_len > 0:
            plan = transfer_plan_by_uid.get(uid)
            if plan is not None and plan.get("transfer_day") is not None and not pd.isna(plan.get("transfer_day")):
                td = int(plan["transfer_day"])
                actual_blocked = int(plan.get("blocked_days_stage3", 0))
                ward_days = list(range(td, min(td + ward_los, T + 1)))
            else:
                ready_day = surgery_day + icu_len
                actual_blocked = max(0, T - ready_day + 1) if ready_day <= T else 0
                ward_days = []
        else:
            ward_days = list(range(surgery_day, min(surgery_day + ward_los, T + 1)))

        ward_overlap_pressure = sum(float(pressure_by_day.get(d, 0.0)) for d in ward_days)
        feedback_penalty = ward_overlap_pressure + 2.0 * actual_blocked

        feedback_patient_rows.append({
            "patient_id": pid,
            "patient_uid": uid,
            "surgery_id": r.get("surgery_id"),
            "specialty": r.get("specialty"),
            "surgery_day": surgery_day,
            "requires_icu": int(requires_icu),
            "icu_treatment_days": icu_len,
            "ward_los_days": ward_los,
            "priority_class": r.get("priority_class", "medium"),
            "priority_score": r.get("priority_score", 2.0),
            "actual_blocked_days_stage3": actual_blocked,
            "ward_overlap_pressure": round(ward_overlap_pressure, 6),
            "stage3_feedback_penalty": round(feedback_penalty, 6),
            "ward_days_under_plan": ",".join(map(str, ward_days)),
        })

    feedback_patient = pd.DataFrame(feedback_patient_rows).sort_values(
        ["stage3_feedback_penalty", "actual_blocked_days_stage3", "ward_los_days"],
        ascending=[False, False, False]
    )

    blocked_total = int(transfer_plan["blocked_days_stage3"].sum()) if not transfer_plan.empty else 0
    summary = pd.DataFrame([{
        "n_transfer_candidates": int(len(cand)),
        "n_transferred_within_horizon": int(transfer_plan["transferred_within_horizon"].sum()) if not transfer_plan.empty else 0,
        "n_untransferred_within_horizon": int((1 - transfer_plan["transferred_within_horizon"]).sum()) if not transfer_plan.empty else 0,
        "blocked_transfer_patient_days_stage3": blocked_total,
        "ward_excess_bed_days_stage3": int(daily_flow["ward_excess_stage3"].sum()),
        "icu_excess_bed_days_stage3": int(daily_flow["icu_excess_stage3"].sum()),
        "peak_icu_ready_blocked_stage3": int(daily_flow["icu_ready_blocked_stage3"].max()),
        "peak_ward_occupancy_stage3": int(daily_flow["ward_occupancy_stage3"].max()),
        "mean_transfer_delay_stage3": round(float(transfer_plan["blocked_days_stage3"].mean()), 6) if not transfer_plan.empty else 0.0,
        "objective_mode": mode,
        "solver_status": int(model.Status),
        "solver_solution_count": _safe_solution_count(model),
        "solver_runtime": _safe_model_float_attr(model, "Runtime", None),
        "solver_mip_gap": _safe_model_float_attr(model, "MIPGap", None),
        "solver_obj_val": _safe_model_float_attr(model, "ObjVal", None),
        "solver_obj_bound": _safe_model_float_attr(model, "ObjBound", None),
    }])

    # Compare to greedy evaluator.
    greedy_summary = greedy_eval["summary"].iloc[0].to_dict()
    comparison = pd.DataFrame([{
        "method": "greedy_evaluator_current_first",
        "blocked_transfer_patient_days": greedy_summary.get("blocked_transfer_patient_days"),
        "icu_excess_bed_days": greedy_summary.get("icu_excess_bed_days_blocking"),
        "ward_excess_bed_days": greedy_summary.get("ward_excess_bed_days_blocking"),
        "peak_icu_ready_blocked": greedy_summary.get("peak_icu_ready_blocked"),
    }, {
        "method": f"stage3_mip_{mode}",
        "blocked_transfer_patient_days": blocked_total,
        "icu_excess_bed_days": int(daily_flow["icu_excess_stage3"].sum()),
        "ward_excess_bed_days": int(daily_flow["ward_excess_stage3"].sum()),
        "peak_icu_ready_blocked": int(daily_flow["icu_ready_blocked_stage3"].max()),
    }])

    results = {
        "stage3_summary": summary,
        "stage3_daily_flow": daily_flow,
        "stage3_patient_transfer_plan": transfer_plan,
        "stage3_patient_day_states": patient_day_states,
        "stage3_feedback_day_pressure": feedback_day,
        "stage3_feedback_patient_pressure": feedback_patient,
        "stage3_greedy_vs_mip_comparison": comparison,
    }

    if out is not None:
        for name, df in results.items():
            df.to_csv(out / f"{name}.csv", index=False)

        metadata = {
            "status": int(model.Status),
            "solution_count": _safe_solution_count(model),
            "runtime": _safe_model_float_attr(model, "Runtime", None),
            "mip_gap": _safe_model_float_attr(model, "MIPGap", None),
            "obj_val": _safe_model_float_attr(model, "ObjVal", None),
            "obj_bound": _safe_model_float_attr(model, "ObjBound", None),
            "time_limit": time_limit,
            "objective_mode": mode,
            "allow_ward_excess": bool(allow_ward_excess),
            "T": T,
            "n_transfer_candidates": int(len(cand)),
        }
        (out / "stage3_solver_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return results


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_stage3_experiment(instance_dir: str | Path,
                          schedule_path: str | Path,
                          output_dir: str | Path,
                          time_limit: int = 300,
                          mip_gap: float = 0.01,
                          objective_mode: str = "capacity_first",
                          allow_ward_excess: bool = True,
                          verbose: bool = True) -> None:
    instance = load_instance(instance_dir)
    schedule = pd.read_csv(schedule_path)

    results = solve_stage3_blocking_flow_mip(
        schedule=schedule,
        instance=instance,
        output_dir=output_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        objective_mode=objective_mode,
        allow_ward_excess=allow_ward_excess,
        verbose=verbose,
    )

    print("\n=== Stage-3 ICU-Ward blocking-flow summary ===")
    print(results["stage3_summary"].to_string(index=False))
    print("\n=== Greedy evaluator vs Stage-3 MIP ===")
    print(results["stage3_greedy_vs_mip_comparison"].to_string(index=False))
    print(f"\nSaved Stage-3 results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-3 ICU-Ward blocking-flow MIP.")
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output-dir", default="stage3_blocking_flow_results")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--objective-mode", choices=["capacity_first", "blocking_first", "weighted"], default="capacity_first")
    parser.add_argument("--hard-ward-capacity", action="store_true", help="Disallow ward-capacity slack unless already forced by fixed loads.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_stage3_experiment(
        instance_dir=args.instance_dir,
        schedule_path=args.schedule,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        objective_mode=args.objective_mode,
        allow_ward_excess=not args.hard_ward_capacity,
        verbose=not args.quiet,
    )


#!/usr/bin/env python3
from __future__ import annotations

"""
shehadeh_style_integrated_mip_baseline_v2_fixed.py

Shehadeh-style deterministic integrated MIP baseline v2 with optional execution-violation cap.

Important naming
----------------
This is NOT claimed to be an exact reproduction of Shehadeh et al.'s code or
private dataset. It is a strong literature-style baseline:

    - single integrated deterministic MIP;
    - candidate surgery placements;
    - OR/surgeon/equipment execution constraints;
    - expected ICU/Ward occupancy constraints;
    - postponement/drop penalties;
    - downstream bed-overflow slack.

It deliberately does NOT use our Stage-3 downstream-pressure feedback loop.

Two recommended experimental settings:
    1. access_constrained:
       enforce min_scheduled and min_high_priority_scheduled, then minimize
       expected downstream bed risk and execution violations.

    2. free_admission:
       allow admission flexibility with high-priority protection and drop
       penalties, showing the access-risk Pareto trade-off.

Dependencies
------------
    gurobipy
    surgery_schedule_evaluator.py
    stage2_priority_soft_gurobi_repair_v3_fixed.py
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from surgery_schedule_evaluator import load_instance, evaluate_schedule, save_evaluation_results

from stage2_priority_soft_gurobi_repair_v3_fixed import (
    generate_soft_candidate_placements,
    _patient_table,
    _patient_equipment,
    _equipment_caps,
    _surgeon_daily_max,
    _placement_occupies_slot,
    _safe_model_float_attr,
    _safe_solution_count,
)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _parse_block_ids(x: Any) -> List[int]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def _priority_table(instance: Dict[str, Any]) -> pd.DataFrame:
    surgeries = instance["surgeries"][["patient_id", "surgery_id", "specialty"]].copy()
    pr = instance.get("patient_priority")
    if isinstance(pr, pd.DataFrame) and not pr.empty:
        return surgeries.merge(pr, on=["patient_id", "surgery_id", "specialty"], how="left")

    out = surgeries.copy()
    out["priority_class"] = "medium"
    out["priority_score"] = 2.0
    out["postpone_penalty"] = 1000.0
    out["delay_penalty"] = 300.0
    out["release_day"] = 1
    out["due_day"] = 7
    return out


def _horizon(instance: Dict[str, Any]) -> int:
    caps = instance["capacities"]
    return int(instance.get("metadata", {}).get("T", int(caps["day_index"].max())))


def _capacity_maps(instance: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    caps = instance["capacities"].copy()
    icu = dict(zip(caps["day_index"].astype(int), caps["icu_capacity"].astype(int)))
    ward = dict(zip(caps["day_index"].astype(int), caps["ward_capacity"].astype(int)))
    day_name = dict(zip(caps["day_index"].astype(int), caps["day"].astype(str)))
    return icu, ward, day_name


def _eligible_surgeons(instance: Dict[str, Any], patient_id: int) -> List[str]:
    elig = instance.get("patient_surgeon_eligibility")
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        rows = elig[elig["patient_id"].astype(int) == int(patient_id)].copy()
        if rows.empty:
            return []
        if "preferred" in rows.columns:
            rows = rows.sort_values("preferred", ascending=False)
        return rows["surgeon_id"].astype(str).tolist()

    surgeons = instance.get("surgeons")
    if isinstance(surgeons, pd.DataFrame) and not surgeons.empty:
        return surgeons["surgeon_id"].astype(str).tolist()
    return []


def build_pool_warm_schedule(instance: Dict[str, Any],
                             patient_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Build a minimal warm schedule for candidate generation over a patient pool.

    The Stage-2 placement generator uses warm_schedule only to define the patient
    set and to rank candidate starts/blocks. This warm schedule need not be a
    feasible final schedule.
    """
    surgeries = instance["surgeries"].copy()
    pr = _priority_table(instance)
    add_cols = ["patient_id", "release_day", "due_day", "priority_class", "priority_score"]
    add_cols = [c for c in add_cols if c in pr.columns]
    surgeries = surgeries.merge(pr[add_cols], on="patient_id", how="left")

    if patient_ids is not None:
        patient_ids = [int(x) for x in patient_ids]
        surgeries = surgeries[surgeries["patient_id"].astype(int).isin(patient_ids)]

    blocks = instance["blocks"].copy()
    rows = []

    for _, s in surgeries.iterrows():
        pid = int(s["patient_id"])
        specialty = str(s["specialty"])
        release = int(s.get("release_day", 1) if not pd.isna(s.get("release_day", 1)) else 1)
        compatible = _parse_block_ids(s.get("compatible_block_ids", ""))

        cand = blocks[blocks["specialty"].astype(str) == specialty].copy()
        if compatible:
            cand = cand[cand["block_id"].astype(int).isin(compatible)]
        cand = cand[cand["day_index"].astype(int) >= release]
        cand = cand.sort_values(["day_index", "or_id", "block_id"])

        if cand.empty:
            continue

        b = cand.iloc[0]
        surgeons = _eligible_surgeons(instance, pid)
        surgeon_id = surgeons[0] if surgeons else ""

        rows.append({
            "patient_id": pid,
            "patient_uid": f"E_{pid:04d}",
            "surgery_id": s["surgery_id"],
            "specialty": specialty,
            "block_id": int(b["block_id"]),
            "or_id": int(b["or_id"]),
            "day": b["day"],
            "day_index": int(b["day_index"]),
            "position": 1,
            "planned_start_min": 0.0,
            "duration_min": float(s["duration_min"]),
            "planned_end_min": float(s["duration_min"]),
            "requires_icu": int(s.get("requires_icu", int(float(s.get("icu_los_days", 0)) > 0))),
            "icu_treatment_days": int(s.get("icu_treatment_days", s.get("icu_los_days", 0))),
            "ward_los_days": int(s.get("ward_los_days", 0)),
            "surgeon_id": surgeon_id,
        })

    return pd.DataFrame(rows)


def build_fixed_expected_loads(instance: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Fixed ICU/Ward expected occupancy from current patients.

    For current ICU patients, the deterministic baseline assumes immediate ward
    transfer on ready_day if ward_los_days > 0. This is an expected-load proxy,
    not the Stage-3 recourse model.
    """
    T = _horizon(instance)
    fixed_icu = {d: 0 for d in range(1, T + 1)}
    fixed_ward = {d: 0 for d in range(1, T + 1)}

    current_ward = instance.get("current_ward", pd.DataFrame())
    if isinstance(current_ward, pd.DataFrame) and not current_ward.empty:
        for _, r in current_ward.iterrows():
            discharge_day = int(r.get("planned_discharge_day", 1))
            for d in range(1, min(discharge_day, T + 1)):
                fixed_ward[d] += 1

    current_icu = instance.get("current_icu", pd.DataFrame())
    if isinstance(current_icu, pd.DataFrame) and not current_icu.empty:
        for _, r in current_icu.iterrows():
            ready_day = int(r.get("ready_for_ward_day", 1))
            ward_los = int(r.get("ward_los_days", 0))

            for d in range(1, min(ready_day, T + 1)):
                fixed_icu[d] += 1

            if ward_los > 0:
                for d in range(max(1, ready_day), min(ready_day + ward_los, T + 1)):
                    fixed_ward[d] += 1

    return fixed_icu, fixed_ward



def _write_no_incumbent_metadata(out: Optional[Path],
                                 model,
                                 objective_mode: str,
                                 min_scheduled: int,
                                 min_high_priority_scheduled: int,
                                 high_drop_limit: int,
                                 execution_violation_budget: Optional[int],
                                 slot_minutes: int,
                                 max_or_overtime: int,
                                 num_pool_patients: int) -> None:
    """Write useful metadata when Gurobi stops without an incumbent."""
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": int(model.Status),
        "solution_count": _safe_solution_count(model),
        "runtime": _safe_model_float_attr(model, "Runtime", None),
        "mip_gap": _safe_model_float_attr(model, "MIPGap", None),
        "obj_val": _safe_model_float_attr(model, "ObjVal", None),
        "obj_bound": _safe_model_float_attr(model, "ObjBound", None),
        "objective_mode": objective_mode,
        "min_scheduled": int(min_scheduled),
        "min_high_priority_scheduled": int(min_high_priority_scheduled),
        "high_drop_limit": int(high_drop_limit),
        "execution_violation_budget": None if execution_violation_budget is None else int(execution_violation_budget),
        "slot_minutes": int(slot_minutes),
        "max_or_overtime": int(max_or_overtime),
        "num_pool_patients": int(num_pool_patients),
        "note": "Gurobi ended without a feasible incumbent; no schedule can be extracted.",
    }
    (out / "solver_no_incumbent_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def add_expected_bed_load_columns(placements: pd.DataFrame,
                                  horizon: int) -> pd.DataFrame:
    """Add compact string columns that record expected ICU/Ward occupancy days."""
    p = placements.copy()
    icu_days = []
    ward_days = []

    for _, r in p.iterrows():
        day = int(r["day_index"])
        requires_icu = int(r.get("requires_icu", 0)) == 1
        icu_len = int(max(0, r.get("icu_treatment_days", 0)))
        ward_len = int(max(0, r.get("ward_los_days", 0)))

        if requires_icu and icu_len > 0:
            ids = [d for d in range(day, day + icu_len) if 1 <= d <= horizon]
            ready = day + icu_len
            wds = [d for d in range(ready, ready + ward_len) if 1 <= d <= horizon]
        else:
            ids = []
            wds = [d for d in range(day, day + ward_len) if 1 <= d <= horizon]

        icu_days.append(",".join(map(str, ids)))
        ward_days.append(",".join(map(str, wds)))

    p["expected_icu_days"] = icu_days
    p["expected_ward_days"] = ward_days
    return p


def _str_days_to_set(x: Any) -> set[int]:
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    return {int(v) for v in s.split(",") if v.strip()}


# ---------------------------------------------------------------------
# Integrated baseline MIP
# ---------------------------------------------------------------------

def solve_shehadeh_style_integrated_mip(instance: Dict[str, Any],
                                        output_dir: Optional[str | Path] = None,
                                        warm_schedule: Optional[pd.DataFrame] = None,
                                        objective_mode: str = "access_constrained",
                                        min_scheduled: int = 49,
                                        min_high_priority_scheduled: int = 22,
                                        high_drop_limit: int = 2,
                                        execution_violation_budget: Optional[int] = None,
                                        slot_minutes: int = 30,
                                        max_or_overtime: int = 90,
                                        time_limit: int = 600,
                                        mip_gap: float = 0.03,
                                        verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Solve Shehadeh-style integrated deterministic MIP baseline."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("gurobipy is required for Shehadeh-style integrated MIP baseline.") from e

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    T = _horizon(instance)
    icu_cap, ward_cap, day_name = _capacity_maps(instance)
    fixed_icu, fixed_ward = build_fixed_expected_loads(instance)

    if warm_schedule is None:
        warm_schedule = build_pool_warm_schedule(instance)

    admitted_ids = sorted(warm_schedule["patient_id"].astype(int).unique().tolist())
    patients = _patient_table(instance, admitted_ids)

    placements = generate_soft_candidate_placements(
        instance=instance,
        warm_schedule=warm_schedule,
        slot_minutes=slot_minutes,
        max_or_overtime=max_or_overtime,
    )

    if placements.empty:
        raise RuntimeError("No candidate placements generated for Shehadeh-style integrated baseline.")

    placements = add_expected_bed_load_columns(placements, horizon=T)

    if out is not None:
        warm_schedule.to_csv(out / "baseline_pool_warm_schedule.csv", index=False)
        placements.to_csv(out / "baseline_candidate_placements.csv", index=False)

    pclass = dict(zip(patients["patient_id"].astype(int), patients["priority_class"].astype(str)))
    pscore = dict(zip(patients["patient_id"].astype(int), patients["priority_score"].astype(float)))
    postpone = dict(zip(patients["patient_id"].astype(int), patients["postpone_penalty"].astype(float)))

    high_ids = [pid for pid in admitted_ids if pclass.get(pid, "medium") == "high"]

    equipment_caps = _equipment_caps(instance)
    surgeon_daily_max = _surgeon_daily_max(instance)

    P_by_patient = {
        int(pid): placements.index[placements["patient_id"].astype(int) == int(pid)].tolist()
        for pid in admitted_ids
    }

    max_time = int(480 + max_or_overtime)
    block_slot_to_placements: Dict[Tuple[int, int], List[int]] = {}
    surgeon_slot_to_placements: Dict[Tuple[str, int, int], List[int]] = {}
    equip_slot_to_items: Dict[Tuple[str, int, int], List[Tuple[int, int]]] = {}
    surgeon_day_to_placements: Dict[Tuple[str, int], List[int]] = {}
    icu_day_to_placements: Dict[int, List[int]] = {d: [] for d in range(1, T + 1)}
    ward_day_to_placements: Dict[int, List[int]] = {d: [] for d in range(1, T + 1)}

    for j, r in placements.iterrows():
        block_id = int(r["block_id"])
        day = int(r["day_index"])
        surgeon_id = str(r["surgeon_id"])
        start = float(r["start_min"])
        end = float(r["end_min"])

        for slot in range(0, max_time + slot_minutes, slot_minutes):
            if _placement_occupies_slot(start, end, slot, slot_minutes):
                block_slot_to_placements.setdefault((block_id, slot), []).append(j)
                surgeon_slot_to_placements.setdefault((surgeon_id, day, slot), []).append(j)

        surgeon_day_to_placements.setdefault((surgeon_id, day), []).append(j)

        for etype, q in _patient_equipment(instance, int(r["patient_id"])):
            for slot in range(0, max_time + slot_minutes, slot_minutes):
                if _placement_occupies_slot(start, end, slot, slot_minutes):
                    equip_slot_to_items.setdefault((etype, day, slot), []).append((j, q))

        for d in _str_days_to_set(r.get("expected_icu_days", "")):
            if 1 <= d <= T:
                icu_day_to_placements[d].append(j)

        for d in _str_days_to_set(r.get("expected_ward_days", "")):
            if 1 <= d <= T:
                ward_day_to_placements[d].append(j)

    model = gp.Model("shehadeh_style_integrated_mip_baseline")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1 if verbose else 0

    x = model.addVars(list(placements.index), vtype=GRB.BINARY, name="x")
    drop = model.addVars(admitted_ids, vtype=GRB.BINARY, name="drop")

    # Execution soft slacks.
    surgeon_over = model.addVars(list(surgeon_day_to_placements.keys()), lb=0.0, vtype=GRB.CONTINUOUS, name="surgeon_overload")
    equip_excess = model.addVars(list(equip_slot_to_items.keys()), lb=0.0, vtype=GRB.CONTINUOUS, name="equip_excess")
    surgeon_over_flag = model.addVars(list(surgeon_day_to_placements.keys()), vtype=GRB.BINARY, name="surgeon_overload_flag")
    equip_excess_flag = model.addVars(list(equip_slot_to_items.keys()), vtype=GRB.BINARY, name="equip_excess_flag")

    # Bed expected overflow slacks and flags.
    icu_excess = model.addVars(range(1, T + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="expected_icu_excess")
    ward_excess = model.addVars(range(1, T + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="expected_ward_excess")
    icu_excess_flag = model.addVars(range(1, T + 1), vtype=GRB.BINARY, name="expected_icu_excess_flag")
    ward_excess_flag = model.addVars(range(1, T + 1), vtype=GRB.BINARY, name="expected_ward_excess_flag")

    # Assignment/drop.
    for pid in admitted_ids:
        cand = P_by_patient.get(pid, [])
        model.addConstr(gp.quicksum(x[j] for j in cand) + drop[pid] == 1, name=f"assign_or_drop[{pid}]")

    # Access constraints.
    scheduled_expr = gp.quicksum(x[j] for j in placements.index)
    high_scheduled_expr = gp.quicksum(x[j] for j in placements.index if int(placements.loc[j, "patient_id"]) in high_ids)
    high_drop_expr = gp.quicksum(drop[pid] for pid in high_ids) if high_ids else gp.LinExpr(0.0)
    total_drop_expr = gp.quicksum(drop[pid] for pid in admitted_ids)

    if high_ids:
        model.addConstr(high_drop_expr <= int(high_drop_limit), name="high_drop_limit")

    if objective_mode == "access_constrained":
        model.addConstr(scheduled_expr >= int(min_scheduled), name="min_scheduled_access")
        model.addConstr(high_scheduled_expr >= int(min_high_priority_scheduled), name="min_high_access")

    # OR hard non-overlap.
    for (block_id, slot), idxs in block_slot_to_placements.items():
        model.addConstr(gp.quicksum(x[j] for j in idxs) <= 1, name=f"or_slot[{block_id},{slot}]")

    # Surgeon hard non-overlap.
    for (sid, day, slot), idxs in surgeon_slot_to_placements.items():
        model.addConstr(gp.quicksum(x[j] for j in idxs) <= 1, name=f"surgeon_slot[{sid},{day},{slot}]")

    # Surgeon daily workload soft.
    for key, idxs in surgeon_day_to_placements.items():
        sid, day = key
        cap = float(surgeon_daily_max.get(str(sid), 999999.0))
        model.addConstr(
            gp.quicksum(float(placements.loc[j, "duration_min"]) * x[j] for j in idxs)
            <= cap + surgeon_over[key],
            name=f"surgeon_day_soft_cap[{sid},{day}]"
        )
        m_over = max(1.0, sum(float(placements.loc[j, "duration_min"]) for j in idxs))
        model.addConstr(surgeon_over[key] <= m_over * surgeon_over_flag[key], name=f"surgeon_over_flag[{sid},{day}]")

    # Equipment soft slot capacity.
    for key, items in equip_slot_to_items.items():
        etype, day, slot = key
        cap = int(equipment_caps.get(str(etype), 999999))
        model.addConstr(
            gp.quicksum(int(q) * x[j] for j, q in items) <= cap + equip_excess[key],
            name=f"equipment_soft_slot[{etype},{day},{slot}]"
        )
        m_excess = max(1.0, sum(int(q) for _, q in items))
        model.addConstr(equip_excess[key] <= m_excess * equip_excess_flag[key], name=f"equipment_excess_flag[{etype},{day},{slot}]")

    # Expected ICU/Ward load constraints.
    for d in range(1, T + 1):
        model.addConstr(
            fixed_icu[d] + gp.quicksum(x[j] for j in icu_day_to_placements[d])
            <= icu_cap[d] + icu_excess[d],
            name=f"expected_icu_capacity[{d}]"
        )
        model.addConstr(
            fixed_ward[d] + gp.quicksum(x[j] for j in ward_day_to_placements[d])
            <= ward_cap[d] + ward_excess[d],
            name=f"expected_ward_capacity[{d}]"
        )
        # Big-M link flags. Candidate load cannot exceed number of placements.
        model.addConstr(icu_excess[d] <= max(1.0, fixed_icu[d] + len(icu_day_to_placements[d])) * icu_excess_flag[d],
                        name=f"expected_icu_flag_link[{d}]")
        model.addConstr(ward_excess[d] <= max(1.0, fixed_ward[d] + len(ward_day_to_placements[d])) * ward_excess_flag[d],
                        name=f"expected_ward_flag_link[{d}]")

    # Objective components.
    calendar_violation_expr = gp.quicksum(
        int(placements.loc[j, "calendar_violation"]) * x[j]
        for j in placements.index
    )
    surgeon_over_flag_expr = gp.quicksum(surgeon_over_flag[key] for key in surgeon_over_flag.keys())
    equip_excess_flag_expr = gp.quicksum(equip_excess_flag[key] for key in equip_excess_flag.keys())
    execution_violation_event_expr = calendar_violation_expr + surgeon_over_flag_expr + equip_excess_flag_expr

    # Optional fairness/stress-test cap:
    # Shehadeh-style integrated MIP can otherwise buy lower downstream blocking
    # by accepting many execution-layer violations. This cap tests whether its
    # downstream advantage survives under comparable execution feasibility.
    if execution_violation_budget is not None and int(execution_violation_budget) >= 0:
        model.addConstr(
            execution_violation_event_expr <= int(execution_violation_budget),
            name="execution_violation_budget"
        )

    bed_excess_event_expr = gp.quicksum(icu_excess_flag[d] + ward_excess_flag[d] for d in range(1, T + 1))
    bed_excess_magnitude_expr = gp.quicksum(5.0 * icu_excess[d] + 8.0 * ward_excess[d] for d in range(1, T + 1))
    bed_risk_expr = 10000.0 * bed_excess_event_expr + 1000.0 * bed_excess_magnitude_expr

    execution_magnitude_terms = []
    stability_terms = []
    weighted_drop_terms = []

    for j, r in placements.iterrows():
        outside_minutes = float(r["calendar_outside_min"])
        overtime_minutes = float(r["or_overtime_min"])
        execution_magnitude_terms.append((100.0 * outside_minutes + 200.0 * overtime_minutes) * x[j])

        delay = float(r.get("delay_penalty", 300.0)) * float(r.get("lateness_days", 0))
        switch_day = 30.0 * (1 - int(r["same_day"]))
        switch_block = 15.0 * (1 - int(r["same_block"]))
        switch_surgeon = 8.0 * (1 - int(r["same_surgeon"]))
        start_dev = 0.05 * float(r["start_deviation"])
        stability_terms.append((delay + switch_day + switch_block + switch_surgeon + start_dev) * x[j])

    for key in surgeon_over.keys():
        execution_magnitude_terms.append(1500.0 * surgeon_over[key])
    for key in equip_excess.keys():
        execution_magnitude_terms.append(8000.0 * equip_excess[key])

    for pid in admitted_ids:
        cls = pclass.get(pid, "medium")
        base = float(postpone.get(pid, 1000.0))
        score = float(pscore.get(pid, 2.0))
        if cls == "high":
            penalty = 10_000_000.0 + 1000.0 * base + 100_000.0 * score
        elif cls == "medium":
            penalty = 800_000.0 + 200.0 * base + 30_000.0 * score
        else:
            penalty = 300_000.0 + 80.0 * base + 10_000.0 * score
        weighted_drop_terms.append(penalty * drop[pid])

    execution_magnitude_expr = gp.quicksum(execution_magnitude_terms)
    stability_expr = gp.quicksum(stability_terms)
    weighted_drop_expr = gp.quicksum(weighted_drop_terms)

    model.ModelSense = GRB.MINIMIZE

    mode = str(objective_mode).lower().strip()
    if mode == "access_constrained":
        # Strong integrated deterministic baseline under comparable surgical access.
        model.setObjectiveN(high_drop_expr, index=0, priority=6, weight=1.0, name="min_high_drops")
        model.setObjectiveN(bed_risk_expr, index=1, priority=5, weight=1.0, name="min_expected_bed_risk")
        model.setObjectiveN(execution_violation_event_expr, index=2, priority=4, weight=1.0, name="min_execution_events")
        model.setObjectiveN(total_drop_expr, index=3, priority=3, weight=1.0, name="min_total_drops")
        model.setObjectiveN(execution_magnitude_expr, index=4, priority=2, weight=1.0, name="min_execution_magnitude")
        model.setObjectiveN(stability_expr, index=5, priority=1, weight=1.0, name="min_stability")
    elif mode == "free_admission":
        # Free-admission literature-style baseline: jointly trade off postponement
        # and expected downstream risk.
        obj = (
            weighted_drop_expr
            + 10000.0 * bed_excess_event_expr
            + 1000.0 * bed_excess_magnitude_expr
            + 50000.0 * execution_violation_event_expr
            + execution_magnitude_expr
            + stability_expr
        )
        model.setObjective(obj, GRB.MINIMIZE)
    else:
        raise ValueError("objective_mode must be 'access_constrained' or 'free_admission'.")

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        if out is not None:
            model.computeIIS()
            model.write(str(out / "shehadeh_style_integrated_infeasible.ilp"))
        raise RuntimeError("Shehadeh-style integrated MIP infeasible. IIS written if output_dir is provided.")

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi did not find usable solution. Status={model.Status}")

    # Important: TIME_LIMIT does not imply that an incumbent exists.
    # If SolCount == 0, variable attributes .X are unavailable and Gurobi raises
    # AttributeError: Unable to retrieve attribute 'X'.
    if _safe_solution_count(model) <= 0:
        _write_no_incumbent_metadata(
            out=out,
            model=model,
            objective_mode=objective_mode,
            min_scheduled=min_scheduled,
            min_high_priority_scheduled=min_high_priority_scheduled,
            high_drop_limit=high_drop_limit,
            execution_violation_budget=execution_violation_budget,
            slot_minutes=slot_minutes,
            max_or_overtime=max_or_overtime,
            num_pool_patients=len(admitted_ids),
        )
        raise RuntimeError(
            "Gurobi stopped without a feasible incumbent. "
            "No schedule can be extracted. Try relaxing execution_violation_budget, "
            "min_high_priority_scheduled, max_or_overtime, or increasing time_limit."
        )

    selected_idx = [j for j in placements.index if x[j].X > 0.5]
    selected = placements.loc[selected_idx].copy()
    dropped = [pid for pid in admitted_ids if drop[pid].X > 0.5]

    rows = []
    for _, r in selected.sort_values(["day_index", "or_id", "block_id", "start_min"]).iterrows():
        rows.append({
            "patient_id": int(r["patient_id"]),
            "patient_uid": f"E_{int(r['patient_id']):04d}",
            "surgery_id": r["surgery_id"],
            "specialty": r["specialty"],
            "block_id": int(r["block_id"]),
            "or_id": int(r["or_id"]),
            "day": r["day"],
            "day_index": int(r["day_index"]),
            "position": 0,
            "planned_start_min": round(float(r["start_min"]), 3),
            "duration_min": round(float(r["duration_min"]), 3),
            "planned_end_min": round(float(r["end_min"]), 3),
            "requires_icu": int(r["requires_icu"]),
            "icu_treatment_days": int(r["icu_treatment_days"]),
            "ward_los_days": int(r["ward_los_days"]),
            "surgeon_id": str(r["surgeon_id"]),
        })

    schedule = pd.DataFrame(rows)
    if not schedule.empty:
        schedule["position"] = schedule.groupby("block_id")["planned_start_min"].rank(method="first").astype(int)
        schedule = schedule.sort_values(["day_index", "or_id", "block_id", "planned_start_min"]).reset_index(drop=True)

    results = evaluate_schedule(schedule, instance, fill_preferred_surgeon=False)

    # Expected bed-load table from the integrated model.
    bed_rows = []
    for d in range(1, T + 1):
        selected_icu = sum(1 for j in selected_idx if d in _str_days_to_set(placements.loc[j, "expected_icu_days"]))
        selected_ward = sum(1 for j in selected_idx if d in _str_days_to_set(placements.loc[j, "expected_ward_days"]))
        bed_rows.append({
            "day_index": d,
            "day": day_name.get(d, str(d)),
            "icu_capacity": icu_cap[d],
            "fixed_icu_expected": fixed_icu[d],
            "selected_icu_expected": selected_icu,
            "icu_expected_occupancy": fixed_icu[d] + selected_icu,
            "icu_expected_excess": max(0, fixed_icu[d] + selected_icu - icu_cap[d]),
            "ward_capacity": ward_cap[d],
            "fixed_ward_expected": fixed_ward[d],
            "selected_ward_expected": selected_ward,
            "ward_expected_occupancy": fixed_ward[d] + selected_ward,
            "ward_expected_excess": max(0, fixed_ward[d] + selected_ward - ward_cap[d]),
            "mip_icu_excess": float(icu_excess[d].X),
            "mip_ward_excess": float(ward_excess[d].X),
        })
    expected_bed_load = pd.DataFrame(bed_rows)

    if out is not None:
        schedule.to_csv(out / "schedule.csv", index=False)
        selected.to_csv(out / "selected_placements.csv", index=False)
        placements.to_csv(out / "candidate_placements.csv", index=False)
        expected_bed_load.to_csv(out / "expected_bed_load.csv", index=False)

        drop_rows = []
        for pid in dropped:
            info = patients[patients["patient_id"].astype(int) == int(pid)]
            drop_rows.append(info.iloc[0].to_dict() if not info.empty else {"patient_id": pid})
        pd.DataFrame(drop_rows).to_csv(out / "dropped_patients.csv", index=False)

        save_evaluation_results(results, out / "evaluation")

        sol_count = _safe_solution_count(model)
        meta = {
            "status": int(model.Status),
            "solution_count": sol_count,
            "runtime": _safe_model_float_attr(model, "Runtime", None),
            "mip_gap": _safe_model_float_attr(model, "MIPGap", None),
            "obj_val": _safe_model_float_attr(model, "ObjVal", None),
            "obj_bound": _safe_model_float_attr(model, "ObjBound", None),
            "objective_mode": objective_mode,
            "min_scheduled": int(min_scheduled),
            "min_high_priority_scheduled": int(min_high_priority_scheduled),
            "high_drop_limit": int(high_drop_limit),
            "execution_violation_budget": None if execution_violation_budget is None else int(execution_violation_budget),
            "slot_minutes": int(slot_minutes),
            "max_or_overtime": int(max_or_overtime),
            "num_pool_patients": len(admitted_ids),
            "num_scheduled": int(schedule["patient_id"].nunique()) if not schedule.empty else 0,
            "num_dropped": len(dropped),
            "num_high_pool": int(len(high_ids)),
            "num_high_scheduled": int(schedule[schedule["patient_id"].astype(int).isin(high_ids)]["patient_id"].nunique()) if not schedule.empty else 0,
            "num_high_dropped": int(sum(1 for pid in dropped if pid in high_ids)),
            "dropped_patients": [int(pid) for pid in dropped],
            "calendar_violation_selected_count": int(sum(int(placements.loc[j, "calendar_violation"]) for j in selected_idx)),
            "surgeon_overload_flag_count": int(sum(1 for var in surgeon_over_flag.values() if var.X > 0.5)),
            "equipment_excess_flag_count": int(sum(1 for var in equip_excess_flag.values() if var.X > 0.5)),
            "mip_expected_icu_excess_total": float(sum(icu_excess[d].X for d in range(1, T + 1))),
            "mip_expected_ward_excess_total": float(sum(ward_excess[d].X for d in range(1, T + 1))),
        }
        (out / "solver_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return schedule, placements, results


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_shehadeh_style_baseline_experiment(instance_dir: str | Path,
                                           output_dir: str | Path,
                                           objective_mode: str = "access_constrained",
                                           min_scheduled: int = 49,
                                           min_high_priority_scheduled: int = 22,
                                           high_drop_limit: int = 2,
                                           execution_violation_budget: Optional[int] = None,
                                           slot_minutes: int = 30,
                                           max_or_overtime: int = 90,
                                           time_limit: int = 600,
                                           mip_gap: float = 0.03,
                                           verbose: bool = True) -> None:
    instance = load_instance(instance_dir)
    warm_schedule = build_pool_warm_schedule(instance)

    schedule, placements, results = solve_shehadeh_style_integrated_mip(
        instance=instance,
        output_dir=output_dir,
        warm_schedule=warm_schedule,
        objective_mode=objective_mode,
        min_scheduled=min_scheduled,
        min_high_priority_scheduled=min_high_priority_scheduled,
        high_drop_limit=high_drop_limit,
        execution_violation_budget=execution_violation_budget,
        slot_minutes=slot_minutes,
        max_or_overtime=max_or_overtime,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )

    print("\n=== Shehadeh-style integrated MIP baseline summary ===")
    print(results["summary"].to_string(index=False))
    print(f"\nSaved baseline results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Shehadeh-style integrated deterministic MIP baseline.")
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--output-dir", default="case_70_shehadeh_style_results")
    parser.add_argument("--objective-mode", choices=["access_constrained", "free_admission"], default="access_constrained")
    parser.add_argument("--min-scheduled", type=int, default=49)
    parser.add_argument("--min-high-priority-scheduled", type=int, default=22)
    parser.add_argument("--high-drop-limit", type=int, default=2)
    parser.add_argument("--execution-violation-budget", type=int, default=-1, help="If >=0, cap execution violation events.")
    parser.add_argument("--slot-minutes", type=int, default=30)
    parser.add_argument("--max-or-overtime", type=int, default=90)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--mip-gap", type=float, default=0.03)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_shehadeh_style_baseline_experiment(
        instance_dir=args.instance_dir,
        output_dir=args.output_dir,
        objective_mode=args.objective_mode,
        min_scheduled=args.min_scheduled,
        min_high_priority_scheduled=args.min_high_priority_scheduled,
        high_drop_limit=args.high_drop_limit,
        execution_violation_budget=(None if args.execution_violation_budget < 0 else args.execution_violation_budget),
        slot_minutes=args.slot_minutes,
        max_or_overtime=args.max_or_overtime,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=not args.quiet,
    )

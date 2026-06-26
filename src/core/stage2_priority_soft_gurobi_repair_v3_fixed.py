
#!/usr/bin/env python3
from __future__ import annotations

"""
stage2_priority_soft_gurobi_repair_v3_fixed.py

Priority-protected multi-objective Stage-2 Gurobi repair v3.

This replaces the strict Stage-2 repair.

Core idea
---------
Given a BA-HLA-v4.1 warm-start schedule, keep its admitted patient set as much
as possible, especially high-priority patients, and use Gurobi to repair the
execution layer with soft constraints:

Hard:
    - one selected placement or drop for each admitted patient;
    - high-priority drop is forbidden by default;
    - OR non-overlap;
    - surgeon non-overlap;
    - specialty/block compatibility is enforced by candidate generation;
    - surgeon eligibility is enforced by candidate generation.

Soft, but violation-count aligned:
    - surgeon calendar violation with stronger binary-style placement penalty;
    - surgeon daily workload overload with overload flag penalty;
    - equipment capacity excess with excess flag penalty;
    - OR overtime with reduced allowance and stronger penalty;
    - medium/low patient drop.

This is designed to avoid the previous failure mode:
    strict repair => violation_count = 0, but drops too many surgeries and
    high-priority patients.

Dependencies:
    gurobipy
    surgery_schedule_evaluator.py
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from surgery_schedule_evaluator import load_instance, evaluate_schedule, save_evaluation_results


def _safe_model_float_attr(model, attr_name: str, default=None):
    """Safely read a numeric Gurobi model attribute."""
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
# Data helpers
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
    out["postpone_penalty"] = instance["surgeries"].get("postpone_cost", 1000)
    out["delay_penalty"] = 300.0
    out["release_day"] = 1
    out["due_day"] = 7
    return out


def _patient_table(instance: Dict[str, Any], admitted_patient_ids: List[int]) -> pd.DataFrame:
    surgeries = instance["surgeries"].copy()
    pr = _priority_table(instance)

    add_cols = [
        "patient_id", "priority_class", "priority_score", "postpone_penalty",
        "delay_penalty", "release_day", "due_day", "case_category",
        "complexity_class", "cancer_flag", "open_flag", "surgery_approach",
    ]
    add_cols = [c for c in add_cols if c in pr.columns]
    p = surgeries.merge(pr[add_cols], on="patient_id", how="left")
    p = p[p["patient_id"].astype(int).isin([int(x) for x in admitted_patient_ids])].copy()

    if "requires_icu" not in p.columns:
        p["requires_icu"] = (p.get("icu_los_days", 0).fillna(0).astype(float) > 0).astype(int)
    if "icu_treatment_days" not in p.columns:
        p["icu_treatment_days"] = p.get("icu_los_days", 0)

    defaults = {
        "priority_class": "medium",
        "priority_score": 2.0,
        "postpone_penalty": 1000.0,
        "delay_penalty": 300.0,
        "release_day": 1,
        "due_day": 7,
        "ward_los_days": 0,
        "can_be_blocked_in_icu": 0,
    }
    for c, v in defaults.items():
        if c not in p.columns:
            p[c] = v
        p[c] = p[c].fillna(v)

    return p.reset_index(drop=True)


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


def _surgeon_daily_max(instance: Dict[str, Any]) -> Dict[str, float]:
    surgeons = instance.get("surgeons")
    if isinstance(surgeons, pd.DataFrame) and not surgeons.empty and "daily_max_minutes" in surgeons.columns:
        return dict(zip(surgeons["surgeon_id"].astype(str), surgeons["daily_max_minutes"].astype(float)))
    return {}


def _surgeon_windows(instance: Dict[str, Any], surgeon_id: str, day_index: int) -> List[Tuple[float, float]]:
    cal = instance.get("surgeon_calendar")
    if not isinstance(cal, pd.DataFrame) or cal.empty:
        return [(0.0, 480.0)]

    rows = cal[
        (cal["surgeon_id"].astype(str) == str(surgeon_id)) &
        (cal["day_index"].astype(int) == int(day_index))
    ].sort_values("available_start_min")

    return [(float(r["available_start_min"]), float(r["available_end_min"])) for _, r in rows.iterrows()]


def _inside_any_window(start: float, end: float, windows: List[Tuple[float, float]]) -> bool:
    return any(start >= a - 1e-9 and end <= b + 1e-9 for a, b in windows)


def _outside_window_minutes(start: float, end: float, windows: List[Tuple[float, float]]) -> float:
    """Minimum minutes outside one available window.

    If the interval is inside a window, return 0.
    Otherwise return duration - max overlap with any single window.
    This is a soft proxy. It avoids needing interval-union linearization.
    """
    dur = max(0.0, end - start)
    if not windows:
        return dur

    best_overlap = 0.0
    for a, b in windows:
        best_overlap = max(best_overlap, max(0.0, min(end, b) - max(start, a)))
    return max(0.0, dur - best_overlap)


def _patient_equipment(instance: Dict[str, Any], patient_id: int) -> List[Tuple[str, int]]:
    peq = instance.get("patient_equipment")
    if not isinstance(peq, pd.DataFrame) or peq.empty:
        return []
    rows = peq[peq["patient_id"].astype(int) == int(patient_id)]
    return [(str(r["equipment_type"]), int(r.get("quantity_required", 1))) for _, r in rows.iterrows()]


def _equipment_caps(instance: Dict[str, Any]) -> Dict[str, int]:
    eq = instance.get("equipment")
    if not isinstance(eq, pd.DataFrame) or eq.empty:
        return {}
    return dict(zip(eq["equipment_type"].astype(str), eq["quantity"].astype(int)))


def _placement_occupies_slot(start: float, end: float, slot_start: float, slot_len: int) -> bool:
    return max(start, slot_start) < min(end, slot_start + slot_len) - 1e-9


def _warm_start_map(warm_schedule: pd.DataFrame) -> Dict[int, dict]:
    out = {}
    if warm_schedule is None or warm_schedule.empty:
        return out
    for _, r in warm_schedule.iterrows():
        pid = int(r["patient_id"])
        out[pid] = {
            "day_index": int(r["day_index"]),
            "block_id": int(r["block_id"]),
            "surgeon_id": str(r["surgeon_id"]) if "surgeon_id" in warm_schedule.columns and not pd.isna(r.get("surgeon_id")) else None,
            "planned_start_min": float(r.get("planned_start_min", 0.0)),
        }
    return out


# ---------------------------------------------------------------------
# Candidate placement generation
# ---------------------------------------------------------------------

def generate_soft_candidate_placements(instance: Dict[str, Any],
                                       warm_schedule: pd.DataFrame,
                                       slot_minutes: int = 30,
                                       max_or_overtime: int = 60,
                                       max_starts_per_patient_block_surgeon: int = 12) -> pd.DataFrame:
    """Generate broad placement candidates.

    Unlike the strict Stage-2 model, this generator does NOT require the surgery
    interval to be inside a surgeon calendar window. Calendar violations are
    measured and penalized softly in the objective.

    It also allows OR overtime up to max_or_overtime minutes.
    """
    admitted_ids = sorted(warm_schedule["patient_id"].astype(int).unique().tolist())
    patients = _patient_table(instance, admitted_ids)
    blocks = instance["blocks"].copy()
    warm = _warm_start_map(warm_schedule)

    rows = []
    pid_to_row = {int(r["patient_id"]): r for _, r in patients.iterrows()}
    placement_id = 0

    for pid in admitted_ids:
        p = pid_to_row[int(pid)]
        specialty = str(p["specialty"])
        duration = float(p["duration_min"])
        release_day = int(p.get("release_day", 1))
        due_day = int(p.get("due_day", 7))
        compatible = _parse_block_ids(p.get("compatible_block_ids", ""))

        cand_blocks = blocks[blocks["specialty"].astype(str) == specialty].copy()
        if compatible:
            cand_blocks = cand_blocks[cand_blocks["block_id"].astype(int).isin(compatible)]
        cand_blocks = cand_blocks[cand_blocks["day_index"].astype(int) >= release_day]
        cand_blocks = cand_blocks.sort_values(["day_index", "or_id", "block_id"])

        surgeons = _eligible_surgeons(instance, pid)
        if not surgeons:
            continue

        for _, b in cand_blocks.iterrows():
            block_id = int(b["block_id"])
            day = int(b["day_index"])
            block_len = float(b["block_length_min"])
            latest_start = block_len + max_or_overtime - duration
            if latest_start < 0:
                continue

            all_starts = [float(t) for t in range(0, int(math.floor(latest_start)) + 1, slot_minutes)]
            if not all_starts:
                continue

            for surgeon_id in surgeons:
                windows = _surgeon_windows(instance, surgeon_id, day)

                starts = all_starts
                # Moderate candidate size. Prefer warm-start neighborhood first.
                if len(starts) > max_starts_per_patient_block_surgeon:
                    wm = warm.get(pid, {})
                    if wm.get("block_id") == block_id:
                        ws = float(wm.get("planned_start_min", 0.0))
                        starts = sorted(starts, key=lambda x: abs(x - ws))[:max_starts_per_patient_block_surgeon]
                        starts = sorted(starts)
                    else:
                        idx = np.linspace(0, len(starts) - 1, max_starts_per_patient_block_surgeon).round().astype(int)
                        starts = [starts[int(j)] for j in idx]

                for start in starts:
                    end = start + duration
                    overtime_min = max(0.0, end - block_len)
                    cal_bad = int(not _inside_any_window(start, end, windows))
                    cal_out_min = _outside_window_minutes(start, end, windows)

                    wm = warm.get(pid, {})
                    same_block = int(wm.get("block_id") == block_id)
                    same_day = int(wm.get("day_index") == day)
                    same_surgeon = int(str(wm.get("surgeon_id")) == str(surgeon_id)) if wm.get("surgeon_id") is not None else 0
                    start_dev = abs(start - float(wm.get("planned_start_min", start))) if same_block else 0.0

                    placement_id += 1
                    rows.append({
                        "placement_id": placement_id,
                        "patient_id": int(pid),
                        "surgery_id": p["surgery_id"],
                        "specialty": specialty,
                        "block_id": block_id,
                        "or_id": int(b["or_id"]),
                        "day": b["day"],
                        "day_index": day,
                        "surgeon_id": str(surgeon_id),
                        "start_min": round(start, 3),
                        "end_min": round(end, 3),
                        "duration_min": round(duration, 3),
                        "release_day": release_day,
                        "due_day": due_day,
                        "lateness_days": max(0, day - due_day),
                        "same_block": same_block,
                        "same_day": same_day,
                        "same_surgeon": same_surgeon,
                        "start_deviation": round(start_dev, 3),
                        "or_overtime_min": round(overtime_min, 3),
                        "calendar_violation": cal_bad,
                        "calendar_outside_min": round(cal_out_min, 3),
                        "requires_icu": int(p.get("requires_icu", int(float(p.get("icu_los_days", 0)) > 0))),
                        "icu_treatment_days": int(p.get("icu_treatment_days", p.get("icu_los_days", 0))),
                        "ward_los_days": int(p.get("ward_los_days", 0)),
                        "priority_score": float(p.get("priority_score", 2.0)),
                        "priority_class": str(p.get("priority_class", "medium")),
                        "postpone_penalty": float(p.get("postpone_penalty", 1000.0)),
                        "delay_penalty": float(p.get("delay_penalty", 300.0)),
                    })

    return pd.DataFrame(rows)


def write_placement_diagnostic(placements: pd.DataFrame,
                               admitted_ids: List[int],
                               patient_info: pd.DataFrame,
                               out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    if placements.empty:
        counts = pd.DataFrame({"patient_id": admitted_ids, "n_placements": 0})
    else:
        counts = placements.groupby("patient_id").size().reset_index(name="n_placements")
        counts["patient_id"] = counts["patient_id"].astype(int)
        counts = pd.DataFrame({"patient_id": admitted_ids}).merge(counts, on="patient_id", how="left")
        counts["n_placements"] = counts["n_placements"].fillna(0).astype(int)

    keep = ["patient_id", "surgery_id", "specialty", "duration_min", "priority_class",
            "priority_score", "release_day", "due_day", "requires_icu",
            "icu_treatment_days", "ward_los_days"]
    keep = [c for c in keep if c in patient_info.columns]
    diag = counts.merge(patient_info[keep], on="patient_id", how="left")
    diag = diag.sort_values(["n_placements", "priority_class", "duration_min"], ascending=[True, True, False])
    diag.to_csv(out_dir / "stage2_soft_v3_candidate_count_by_patient.csv", index=False)
    return diag


# ---------------------------------------------------------------------
# Soft Gurobi repair
# ---------------------------------------------------------------------

def solve_priority_soft_stage2_repair(instance: Dict[str, Any],
                                      warm_schedule: pd.DataFrame,
                                      output_dir: Optional[str | Path] = None,
                                      slot_minutes: int = 30,
                                      max_or_overtime: int = 60,
                                      time_limit: int = 300,
                                      mip_gap: float = 0.01,
                                      high_drop_limit: int = 0,
                                      max_total_drop: int = 2,
                                      objective_mode: str = "violation_first",
                                      verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Priority-protected multi-objective Stage-2 repair.

    Parameters
    ----------
    high_drop_limit:
        Default 0. High-priority patients are protected:
            sum_{i in high} drop_i <= high_drop_limit.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("gurobipy is required. Please install/use a valid Gurobi environment.") from e

    warm_schedule = warm_schedule.copy()
    admitted_ids = sorted(warm_schedule["patient_id"].astype(int).unique().tolist())
    patients = _patient_table(instance, admitted_ids)
    patient_info = patients.copy()

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    placements = generate_soft_candidate_placements(
        instance=instance,
        warm_schedule=warm_schedule,
        slot_minutes=slot_minutes,
        max_or_overtime=max_or_overtime,
    )

    if out is not None:
        diag = write_placement_diagnostic(placements, admitted_ids, patient_info, out)
        print("\n[diagnostic] lowest candidate counts:")
        print(diag.head(20).to_string(index=False))

    if placements.empty:
        raise RuntimeError("No candidate placements generated at all. Check blocks/surgeons/calendar data.")

    # Basic dictionaries.
    pclass = dict(zip(patients["patient_id"].astype(int), patients["priority_class"].astype(str)))
    pscore = dict(zip(patients["patient_id"].astype(int), patients["priority_score"].astype(float)))
    postpone = dict(zip(patients["patient_id"].astype(int), patients["postpone_penalty"].astype(float)))

    equipment_caps = _equipment_caps(instance)
    surgeon_daily_max = _surgeon_daily_max(instance)

    P_by_patient = {
        int(pid): placements.index[placements["patient_id"].astype(int) == int(pid)].tolist()
        for pid in admitted_ids
    }

    # Slot mappings.
    max_time = int(480 + max_or_overtime)
    block_slot_to_placements: Dict[Tuple[int, int], List[int]] = {}
    surgeon_slot_to_placements: Dict[Tuple[str, int, int], List[int]] = {}
    equip_slot_to_items: Dict[Tuple[str, int, int], List[Tuple[int, int]]] = {}
    surgeon_day_to_placements: Dict[Tuple[str, int], List[int]] = {}

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

    model = gp.Model("priority_soft_stage2_repair")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1 if verbose else 0

    x = model.addVars(list(placements.index), vtype=GRB.BINARY, name="x")
    drop = model.addVars(admitted_ids, vtype=GRB.BINARY, name="drop")

    # Soft continuous/integer slacks.
    surgeon_over = model.addVars(list(surgeon_day_to_placements.keys()), lb=0.0, vtype=GRB.CONTINUOUS, name="surgeon_overload")
    equip_excess = model.addVars(list(equip_slot_to_items.keys()), lb=0.0, vtype=GRB.CONTINUOUS, name="equip_excess")

    # v2: binary violation flags to align the MIP objective with evaluate_schedule(),
    # which counts violation events, not only violation minutes/units.
    surgeon_over_flag = model.addVars(list(surgeon_day_to_placements.keys()), vtype=GRB.BINARY, name="surgeon_overload_flag")
    equip_excess_flag = model.addVars(list(equip_slot_to_items.keys()), vtype=GRB.BINARY, name="equip_excess_flag")

    # Assignment/drop.
    for pid in admitted_ids:
        cand = P_by_patient.get(pid, [])
        model.addConstr(gp.quicksum(x[j] for j in cand) + drop[pid] == 1, name=f"assign_or_drop[{pid}]")

    # High priority protection.
    high_ids = [pid for pid in admitted_ids if pclass.get(pid, "medium") == "high"]
    if high_ids:
        model.addConstr(gp.quicksum(drop[pid] for pid in high_ids) <= int(high_drop_limit), name="high_priority_drop_limit")

    # v3: keep admission volume protected, but allow a small controlled drop
    # budget so that one or two low-priority cases can be removed if this
    # dramatically improves execution feasibility.
    if max_total_drop is not None and int(max_total_drop) >= 0:
        model.addConstr(gp.quicksum(drop[pid] for pid in admitted_ids) <= int(max_total_drop), name="total_drop_budget")

    # OR block non-overlap remains hard.
    for (block_id, slot), idxs in block_slot_to_placements.items():
        model.addConstr(gp.quicksum(x[j] for j in idxs) <= 1, name=f"or_slot[{block_id},{slot}]")

    # Surgeon non-overlap remains hard.
    for (sid, day, slot), idxs in surgeon_slot_to_placements.items():
        model.addConstr(gp.quicksum(x[j] for j in idxs) <= 1, name=f"surgeon_slot[{sid},{day},{slot}]")

    # Surgeon daily workload is soft, but v2 adds a binary violation flag.
    for key, idxs in surgeon_day_to_placements.items():
        sid, day = key
        cap = float(surgeon_daily_max.get(str(sid), 999999.0))
        model.addConstr(
            gp.quicksum(float(placements.loc[j, "duration_min"]) * x[j] for j in idxs)
            <= cap + surgeon_over[key],
            name=f"surgeon_day_soft_cap[{sid},{day}]"
        )
        # Big-M can be safely bounded by total candidate duration for this surgeon-day.
        m_over = max(1.0, sum(float(placements.loc[j, "duration_min"]) for j in idxs))
        model.addConstr(
            surgeon_over[key] <= m_over * surgeon_over_flag[key],
            name=f"surgeon_overload_flag_link[{sid},{day}]"
        )

    # Equipment capacity is soft, but v2 adds a binary excess flag.
    for key, items in equip_slot_to_items.items():
        etype, day, slot = key
        cap = int(equipment_caps.get(str(etype), 999999))
        model.addConstr(
            gp.quicksum(int(q) * x[j] for j, q in items) <= cap + equip_excess[key],
            name=f"equipment_soft_slot[{etype},{day},{slot}]"
        )
        m_excess = max(1.0, sum(int(q) for _, q in items))
        model.addConstr(
            equip_excess[key] <= m_excess * equip_excess_flag[key],
            name=f"equipment_excess_flag_link[{etype},{day},{slot}]"
        )

    # ------------------------------------------------------------------
    # v3 multi-objective structure
    # ------------------------------------------------------------------
    #
    # The v2 weighted-sum objective could still trade many small calendar
    # violations against volume/overtime in a way that did not align with
    # evaluate_schedule(). v3 uses Gurobi's hierarchical multi-objective API.
    #
    # Default objective_mode = "violation_first":
    #   hard: high-priority drops <= high_drop_limit; total drops <= max_total_drop
    #   P5: minimize high-priority drops
    #   P4: minimize execution violation event count
    #   P3: minimize total drops
    #   P2: minimize violation magnitude / overtime
    #   P1: minimize delay and deviation from warm start
    #
    # Optional objective_mode = "volume_first":
    #   P5 high drops, P4 total drops, P3 violations, P2 magnitudes, P1 stability.
    #
    # This avoids pure penalty calibration and makes the intended trade-off explicit.

    high_drop_expr = gp.quicksum(drop[pid] for pid in high_ids) if high_ids else 0
    total_drop_expr = gp.quicksum(drop[pid] for pid in admitted_ids)

    calendar_violation_expr = gp.quicksum(
        int(placements.loc[j, "calendar_violation"]) * x[j]
        for j in placements.index
    )
    surgeon_over_flag_expr = gp.quicksum(surgeon_over_flag[key] for key in surgeon_over_flag.keys())
    equip_excess_flag_expr = gp.quicksum(equip_excess_flag[key] for key in equip_excess_flag.keys())

    violation_event_expr = calendar_violation_expr + surgeon_over_flag_expr + equip_excess_flag_expr

    magnitude_terms = []
    stability_terms = []

    for j, r in placements.iterrows():
        delay = float(r.get("delay_penalty", 300.0)) * float(r.get("lateness_days", 0))
        switch_day = 30.0 * (1 - int(r["same_day"]))
        switch_block = 15.0 * (1 - int(r["same_block"]))
        switch_surgeon = 8.0 * (1 - int(r["same_surgeon"]))
        start_dev = 0.05 * float(r["start_deviation"])

        outside_minutes = float(r["calendar_outside_min"])
        overtime_minutes = float(r["or_overtime_min"])

        # Magnitudes are optimized after event counts.
        magnitude_terms.append((100.0 * outside_minutes + 200.0 * overtime_minutes) * x[j])

        # Stability/delay is the lowest-level clean-up objective.
        stability_terms.append((delay + switch_day + switch_block + switch_surgeon + start_dev) * x[j])

    for key in surgeon_over.keys():
        magnitude_terms.append(1500.0 * surgeon_over[key])

    for key in equip_excess.keys():
        magnitude_terms.append(8000.0 * equip_excess[key])

    # Mild drop preference inside the lowest-level objective: if two schedules
    # are identical on higher levels, keep higher-priority patients.
    for pid in admitted_ids:
        cls = pclass.get(pid, "medium")
        score = float(pscore.get(pid, 2.0))
        if cls == "high":
            drop_tiebreak = 100000.0 + 10000.0 * score
        elif cls == "medium":
            drop_tiebreak = 10000.0 + 1000.0 * score
        else:
            drop_tiebreak = 1000.0 + 100.0 * score
        stability_terms.append(drop_tiebreak * drop[pid])

    magnitude_expr = gp.quicksum(magnitude_terms)
    stability_expr = gp.quicksum(stability_terms)

    model.ModelSense = GRB.MINIMIZE

    mode = str(objective_mode).lower().strip()
    if mode not in {"violation_first", "volume_first"}:
        raise ValueError("objective_mode must be 'violation_first' or 'volume_first'")

    # Priority values are relative; larger = optimized earlier.
    model.setObjectiveN(high_drop_expr, index=0, priority=5, weight=1.0, name="min_high_priority_drops")

    if mode == "volume_first":
        model.setObjectiveN(total_drop_expr, index=1, priority=4, weight=1.0, name="min_total_drops")
        model.setObjectiveN(violation_event_expr, index=2, priority=3, weight=1.0, name="min_violation_events")
    else:
        model.setObjectiveN(violation_event_expr, index=1, priority=4, weight=1.0, name="min_violation_events")
        model.setObjectiveN(total_drop_expr, index=2, priority=3, weight=1.0, name="min_total_drops")

    model.setObjectiveN(magnitude_expr, index=3, priority=2, weight=1.0, name="min_violation_magnitudes")
    model.setObjectiveN(stability_expr, index=4, priority=1, weight=1.0, name="min_delay_and_deviation")

    # Warm start.
    warm = _warm_start_map(warm_schedule)
    for j, r in placements.iterrows():
        pid = int(r["patient_id"])
        wm = warm.get(pid, {})
        if not wm:
            continue
        if int(r["block_id"]) == int(wm.get("block_id", -1)):
            # Same block and near same start is a useful start even if surgeon differs.
            if abs(float(r["start_min"]) - float(wm.get("planned_start_min", 0.0))) <= slot_minutes:
                x[j].Start = 1

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        if out is not None:
            model.computeIIS()
            model.write(str(out / "stage2_soft_v3_infeasible.ilp"))
        raise RuntimeError("Soft Stage-2 model is infeasible. IIS written if output_dir is provided.")

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi did not find a usable solution. Status={model.Status}")

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

    repaired = pd.DataFrame(rows)
    if not repaired.empty:
        repaired["position"] = repaired.groupby("block_id")["planned_start_min"].rank(method="first").astype(int)
        repaired = repaired.sort_values(["day_index", "or_id", "block_id", "planned_start_min"]).reset_index(drop=True)

    results = evaluate_schedule(repaired, instance, fill_preferred_surgeon=False)

    # Save outputs.
    if out is not None:
        repaired.to_csv(out / "stage2_soft_v3_repaired_schedule.csv", index=False)
        placements.to_csv(out / "stage2_soft_v3_candidate_placements.csv", index=False)

        drop_rows = []
        for pid in dropped:
            info = patient_info[patient_info["patient_id"].astype(int) == int(pid)].iloc[0].to_dict()
            drop_rows.append(info)
        pd.DataFrame(drop_rows).to_csv(out / "stage2_soft_v3_dropped_patients.csv", index=False)

        # Selected placement diagnostics.
        selected.to_csv(out / "stage2_soft_v3_selected_placements.csv", index=False)

        # Slack summary.
        surgeon_slack_rows = []
        for key, var in surgeon_over.items():
            val = float(var.X)
            if val > 1e-6:
                sid, day = key
                surgeon_slack_rows.append({"surgeon_id": sid, "day_index": day, "overload_min": val})
        pd.DataFrame(surgeon_slack_rows).to_csv(out / "stage2_soft_v3_surgeon_overload_slack.csv", index=False)

        equip_slack_rows = []
        for key, var in equip_excess.items():
            val = float(var.X)
            if val > 1e-6:
                etype, day, slot = key
                equip_slack_rows.append({"equipment_type": etype, "day_index": day, "slot_start_min": slot, "excess_units": val})
        pd.DataFrame(equip_slack_rows).to_csv(out / "stage2_soft_v3_equipment_excess_slack.csv", index=False)

        # Save evaluation first. Even if some solver diagnostic attribute is
        # unavailable for a multi-objective model, the schedule summary should
        # still be written.
        save_evaluation_results(results, out / "evaluation")

        sol_count = _safe_solution_count(model)
        meta = {
            "status": int(model.Status),
            "objective": _safe_model_float_attr(model, "ObjVal", None) if sol_count > 0 else None,
            "obj_bound": _safe_model_float_attr(model, "ObjBound", None),
            "mip_gap": _safe_model_float_attr(model, "MIPGap", None) if sol_count > 0 else None,
            "runtime": _safe_model_float_attr(model, "Runtime", None),
            "solution_count": sol_count,
            "slot_minutes": slot_minutes,
            "max_or_overtime": max_or_overtime,
            "time_limit": time_limit,
            "num_admitted_warm_start": len(admitted_ids),
            "num_repaired_scheduled": int(repaired["patient_id"].nunique()) if not repaired.empty else 0,
            "num_dropped": len(dropped),
            "num_high_admitted": int(sum(1 for pid in admitted_ids if pclass.get(pid, "") == "high")),
            "num_high_dropped": int(sum(1 for pid in dropped if pclass.get(pid, "") == "high")),
            "high_drop_limit": high_drop_limit,
            "max_total_drop": max_total_drop,
            "objective_mode": objective_mode,
            "dropped_patients": [int(pid) for pid in dropped],
            "surgeon_overload_slack_total": float(sum(var.X for var in surgeon_over.values())),
            "equipment_excess_slack_total": float(sum(var.X for var in equip_excess.values())),
            "surgeon_overload_flag_count": int(sum(1 for var in surgeon_over_flag.values() if var.X > 0.5)),
            "equipment_excess_flag_count": int(sum(1 for var in equip_excess_flag.values() if var.X > 0.5)),
            "calendar_violation_selected_count": int(sum(int(placements.loc[j, "calendar_violation"]) for j in selected_idx)),
            "multiobj_high_drop_value": float(sum(drop[pid].X for pid in high_ids)) if high_ids else 0.0,
            "multiobj_total_drop_value": float(sum(drop[pid].X for pid in admitted_ids)),
        }
        (out / "stage2_soft_v3_solver_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return repaired, placements, results


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_priority_soft_repair_experiment(instance_dir: str | Path,
                                        warm_schedule_path: str | Path,
                                        output_dir: str | Path,
                                        slot_minutes: int = 30,
                                        max_or_overtime: int = 60,
                                        time_limit: int = 300,
                                        mip_gap: float = 0.01,
                                        high_drop_limit: int = 0,
                                        max_total_drop: int = 2,
                                        objective_mode: str = "violation_first",
                                        verbose: bool = True) -> None:
    instance = load_instance(instance_dir)
    warm_schedule = pd.read_csv(warm_schedule_path)

    repaired, placements, results = solve_priority_soft_stage2_repair(
        instance=instance,
        warm_schedule=warm_schedule,
        output_dir=output_dir,
        slot_minutes=slot_minutes,
        max_or_overtime=max_or_overtime,
        time_limit=time_limit,
        mip_gap=mip_gap,
        high_drop_limit=high_drop_limit,
        max_total_drop=max_total_drop,
        objective_mode=objective_mode,
        verbose=verbose,
    )

    print("\n=== Priority-soft Stage-2 Gurobi repair v3 multi-objective summary ===")
    print(results["summary"].to_string(index=False))
    print(f"\nSaved priority-soft Stage-2 repair results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priority-protected multi-objective Stage-2 Gurobi repair.")
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--warm-schedule", required=True)
    parser.add_argument("--output-dir", default="stage2_priority_soft_v3_repair_results")
    parser.add_argument("--slot-minutes", type=int, default=30)
    parser.add_argument("--max-or-overtime", type=int, default=60)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--high-drop-limit", type=int, default=0)
    parser.add_argument("--max-total-drop", type=int, default=2)
    parser.add_argument("--objective-mode", choices=["violation_first", "volume_first"], default="violation_first")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_priority_soft_repair_experiment(
        instance_dir=args.instance_dir,
        warm_schedule_path=args.warm_schedule,
        output_dir=args.output_dir,
        slot_minutes=args.slot_minutes,
        max_or_overtime=args.max_or_overtime,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        high_drop_limit=args.high_drop_limit,
        max_total_drop=args.max_total_drop,
        objective_mode=args.objective_mode,
        verbose=not args.quiet,
    )

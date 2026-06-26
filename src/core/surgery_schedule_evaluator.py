
#!/usr/bin/env python3
from __future__ import annotations

"""
surgery_schedule_evaluator.py

Unified evaluator for Shehadeh-style OR-to-downstream schedules and our
ICU-to-ward transfer-blocking extension.

Main entry point
----------------
    evaluate_schedule(schedule, instance)

Inputs
------
schedule:
    pandas.DataFrame or path to schedule.csv.

instance:
    dict returned by the generator, or a directory containing CSV/JSON files:
        surgeries.csv
        blocks.csv
        capacities.csv
        costs.json
        current_icu.csv                optional but recommended
        current_ward.csv               optional but recommended
        patient_priority.csv           optional
        surgeons.csv                   optional
        patient_surgeon_eligibility.csv optional
        surgeon_calendar.csv           optional
        equipment.csv                  optional
        patient_equipment.csv          optional

Outputs
-------
A dictionary:
    {
        "summary": pandas.DataFrame with one row,
        "daily_metrics": pandas.DataFrame,
        "patient_day_states": pandas.DataFrame,
        "block_metrics": pandas.DataFrame,
        "surgeon_metrics": pandas.DataFrame,
        "equipment_metrics": pandas.DataFrame,
        "postponed_patients": pandas.DataFrame,
        "violations": pandas.DataFrame,
    }

Purpose
-------
This evaluator deliberately separates:
    1. Shehadeh-style downstream exceedance metrics:
           ICU/Ward occupancy -> capacity exceedance, PE, ME.
    2. Our blocking-state metrics:
           ICU_TREATING vs ICU_READY_BLOCKED, blocked patient-days,
           effective ICU capacity loss.
    3. Execution feasibility metrics:
           OR blocks, surgeon eligibility, surgeon calendars, equipment.

It is an evaluator, not an optimizer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


DAYS_7 = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def _read_json(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_instance(instance_dir: str | Path) -> Dict[str, Any]:
    """Load an instance directory produced by the generator."""
    root = Path(instance_dir)
    if not root.exists():
        raise FileNotFoundError(f"Instance directory not found: {root}")

    table_names = [
        "surgeries", "blocks", "capacities", "type_stats",
        "current_icu", "current_ward", "patient_priority",
        "surgeons", "patient_surgeon_eligibility", "surgeon_calendar",
        "equipment", "patient_equipment", "scenarios",
        "schedule", "patient_day_states", "daily_bed_states",
    ]
    out: Dict[str, Any] = {}
    for name in table_names:
        df = _read_csv(root / f"{name}.csv")
        if df is not None:
            out[name] = df

    out["costs"] = _read_json(root / "costs.json", default={})
    out["metadata"] = _read_json(root / "metadata.json", default={})
    out["_instance_dir"] = str(root)
    return out


def _as_dataframe(x: Any, name: str = "table") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, (str, Path)):
        p = Path(x)
        if not p.exists():
            raise FileNotFoundError(f"{name} file not found: {p}")
        return pd.read_csv(p)
    raise TypeError(f"{name} must be a pandas DataFrame or a CSV path")


def _ensure_schedule_fields(schedule: pd.DataFrame, instance: Dict[str, Any],
                            fill_preferred_surgeon: bool = True) -> pd.DataFrame:
    """Normalize schedule columns and enrich from surgeries table when needed."""
    s = schedule.copy()
    surgeries = instance["surgeries"].copy()

    if "patient_id" not in s.columns:
        raise ValueError("schedule must contain patient_id")

    # Merge missing clinical fields from surgeries.csv.
    enrich_cols = [
        "patient_id", "surgery_id", "specialty", "duration_min",
        "requires_icu", "icu_treatment_days", "ward_los_days",
        "icu_los_days",
    ]
    available = [c for c in enrich_cols if c in surgeries.columns]
    base = surgeries[available].copy()
    s = s.merge(base, on="patient_id", how="left", suffixes=("", "_from_surgery"))

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

    # Derive day_index from blocks if needed.
    if "day_index" not in s.columns and "block_id" in s.columns and "blocks" in instance:
        b = instance["blocks"][["block_id", "day", "day_index", "or_id", "block_length_min"]]
        s = s.merge(b, on="block_id", how="left", suffixes=("", "_from_block"))
        for c in ["day", "day_index", "or_id"]:
            alt = f"{c}_from_block"
            if c not in s.columns and alt in s.columns:
                s[c] = s[alt]
            elif c in s.columns and alt in s.columns:
                s[c] = s[c].where(s[c].notna(), s[alt])

    if "planned_start_min" not in s.columns:
        s["planned_start_min"] = 0.0

    if "planned_end_min" not in s.columns:
        s["planned_end_min"] = s["planned_start_min"].astype(float) + s["duration_min"].astype(float)

    if fill_preferred_surgeon and "surgeon_id" not in s.columns and "patient_surgeon_eligibility" in instance:
        elig = instance["patient_surgeon_eligibility"].copy()
        if "preferred" in elig.columns:
            pref = elig.sort_values(["patient_id", "preferred"], ascending=[True, False]).drop_duplicates("patient_id")
        else:
            pref = elig.drop_duplicates("patient_id")
        s = s.merge(pref[["patient_id", "surgeon_id"]], on="patient_id", how="left")

    return s


# ---------------------------------------------------------------------
# Shehadeh-style downstream exceedance evaluator
# ---------------------------------------------------------------------

def _init_daily_occupancy(capacities: pd.DataFrame) -> pd.DataFrame:
    daily = capacities[["day_index", "day", "icu_capacity", "ward_capacity"]].copy()
    daily["icu_demand_shehadeh"] = 0
    daily["ward_demand_shehadeh"] = 0
    return daily


def _add_occupancy(daily: pd.DataFrame, start_day: int, length: int, col: str) -> None:
    if length <= 0:
        return
    for d in range(int(start_day), int(start_day) + int(length)):
        mask = daily["day_index"] == d
        if mask.any():
            daily.loc[mask, col] += 1


def compute_shehadeh_downstream_metrics(schedule: pd.DataFrame, instance: Dict[str, Any]) -> pd.DataFrame:
    """Compute LOS-driven ICU/Ward demand and exceedance.

    This is the direct analogue of a Shehadeh-style occupancy/exceedance view:
        surgery day + ICU LOS -> ICU demand;
        ICU discharge day + Ward LOS -> ward demand.
    It does not model capacity-dependent transfer blocking.
    """
    capacities = instance["capacities"].copy()
    daily = _init_daily_occupancy(capacities)

    # Existing current ward patients.
    current_ward = instance.get("current_ward")
    if isinstance(current_ward, pd.DataFrame) and len(current_ward) > 0:
        for _, row in current_ward.iterrows():
            discharge_day = int(row.get("planned_discharge_day", 1))
            # Occupies ward from day 1 through discharge_day - 1.
            for d in range(1, discharge_day):
                if (daily["day_index"] == d).any():
                    daily.loc[daily["day_index"] == d, "ward_demand_shehadeh"] += 1

    # Existing current ICU patients.
    current_icu = instance.get("current_icu")
    if isinstance(current_icu, pd.DataFrame) and len(current_icu) > 0:
        for _, row in current_icu.iterrows():
            ready_day = int(row.get("ready_for_ward_day", 1))
            ward_los = int(row.get("ward_los_days", 0))
            for d in range(1, ready_day):
                if (daily["day_index"] == d).any():
                    daily.loc[daily["day_index"] == d, "icu_demand_shehadeh"] += 1
            _add_occupancy(daily, ready_day, ward_los, "ward_demand_shehadeh")

    # Scheduled surgeries.
    for _, row in schedule.iterrows():
        day = int(row["day_index"])
        requires_icu = int(row.get("requires_icu", 0)) == 1
        icu_len = int(max(0, row.get("icu_treatment_days", row.get("icu_los_days", 0))))
        ward_len = int(max(0, row.get("ward_los_days", 0)))

        if requires_icu and icu_len > 0:
            _add_occupancy(daily, day, icu_len, "icu_demand_shehadeh")
            _add_occupancy(daily, day + icu_len, ward_len, "ward_demand_shehadeh")
        else:
            _add_occupancy(daily, day, ward_len, "ward_demand_shehadeh")

    daily["icu_excess_shehadeh"] = (daily["icu_demand_shehadeh"] - daily["icu_capacity"]).clip(lower=0)
    daily["ward_excess_shehadeh"] = (daily["ward_demand_shehadeh"] - daily["ward_capacity"]).clip(lower=0)
    daily["icu_exceedance_indicator"] = (daily["icu_excess_shehadeh"] > 0).astype(int)
    daily["ward_exceedance_indicator"] = (daily["ward_excess_shehadeh"] > 0).astype(int)
    return daily


# ---------------------------------------------------------------------
# Blocking-state evaluator
# ---------------------------------------------------------------------

def simulate_blocking_states(schedule: pd.DataFrame, instance: Dict[str, Any],
                             transfer_priority: str = "current_first") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate ICU-to-ward transfer blocking using the generator convention."""
    capacities = instance["capacities"].copy()
    current_icu = instance.get("current_icu", pd.DataFrame())
    current_ward = instance.get("current_ward", pd.DataFrame())

    T = int(instance.get("metadata", {}).get("T", int(capacities["day_index"].max())))

    icu_active: Dict[str, Dict[str, Any]] = {}
    ward_active: Dict[str, Dict[str, Any]] = {}
    entry_counter = 0

    if isinstance(current_icu, pd.DataFrame) and len(current_icu) > 0:
        for _, row in current_icu.iterrows():
            entry_counter += 1
            icu_active[str(row["patient_uid"])] = {
                "source": "current_icu",
                "ready_day": int(row["ready_for_ward_day"]),
                "ward_los_days": int(row["ward_los_days"]),
                "entry_order": entry_counter,
                "surgery_id": None,
                "specialty": None,
                "patient_id": None,
            }

    if isinstance(current_ward, pd.DataFrame) and len(current_ward) > 0:
        for _, row in current_ward.iterrows():
            ward_active[str(row["patient_uid"])] = {
                "source": "current_ward",
                "discharge_day": int(row["planned_discharge_day"]),
                "surgery_id": None,
                "specialty": None,
                "patient_id": None,
            }

    schedule_by_day: Dict[int, List[dict]] = {}
    if len(schedule) > 0:
        for _, row in schedule.iterrows():
            schedule_by_day.setdefault(int(row["day_index"]), []).append(row.to_dict())

    patient_rows: List[dict] = []
    daily_rows: List[dict] = []

    for day in range(1, T + 1):
        cap_row = capacities.loc[capacities["day_index"] == day]
        if cap_row.empty:
            continue
        cap = cap_row.iloc[0]
        icu_cap = int(cap["icu_capacity"])
        ward_cap = int(cap["ward_capacity"])

        # 1. Ward discharge first.
        for pid in list(ward_active.keys()):
            if int(ward_active[pid]["discharge_day"]) <= day:
                del ward_active[pid]

        # 2. New surgery arrivals.
        direct_ward_arrivals = []
        for row in schedule_by_day.get(day, []):
            pid = str(row.get("patient_uid", f"E_{int(row['patient_id']):04d}"))
            requires_icu = int(row.get("requires_icu", 0)) == 1
            icu_days = int(max(0, row.get("icu_treatment_days", row.get("icu_los_days", 0))))
            ward_days = int(max(0, row.get("ward_los_days", 0)))

            if requires_icu and icu_days > 0:
                entry_counter += 1
                icu_active[pid] = {
                    "source": "elective",
                    "ready_day": int(day + icu_days),
                    "ward_los_days": int(ward_days),
                    "entry_order": entry_counter,
                    "surgery_id": row.get("surgery_id"),
                    "specialty": row.get("specialty"),
                    "patient_id": row.get("patient_id"),
                }
            elif ward_days > 0:
                direct_ward_arrivals.append({
                    "patient_uid": pid,
                    "source": "elective_direct_ward",
                    "discharge_day": int(day + ward_days),
                    "surgery_id": row.get("surgery_id"),
                    "specialty": row.get("specialty"),
                    "patient_id": row.get("patient_id"),
                })

        # 3. Ready ICU patients attempt transfer.
        ready = [(pid, data) for pid, data in icu_active.items() if int(data["ready_day"]) <= day]
        if transfer_priority == "current_first":
            ready.sort(key=lambda x: (0 if x[1]["source"] == "current_icu" else 1, x[1]["ready_day"], x[1]["entry_order"]))
        elif transfer_priority == "elective_first":
            ready.sort(key=lambda x: (0 if x[1]["source"] == "elective" else 1, x[1]["ready_day"], x[1]["entry_order"]))
        else:
            ready.sort(key=lambda x: (x[1]["ready_day"], x[1]["entry_order"]))

        for pid, data in ready:
            ward_los = int(data.get("ward_los_days", 0))
            if ward_los <= 0:
                if pid in icu_active:
                    del icu_active[pid]
                continue

            if len(ward_active) < ward_cap:
                ward_active[pid] = {
                    "source": data["source"] + "_icu_transfer",
                    "discharge_day": int(day + ward_los),
                    "surgery_id": data.get("surgery_id"),
                    "specialty": data.get("specialty"),
                    "patient_id": data.get("patient_id"),
                }
                if pid in icu_active:
                    del icu_active[pid]
            # else remain in ICU and become ICU_READY_BLOCKED.

        # 4. Direct ward arrivals after ICU transfers.
        for row in direct_ward_arrivals:
            ward_active[row["patient_uid"]] = row

        # 5. Record states.
        icu_treating = 0
        icu_blocked = 0

        for pid, data in icu_active.items():
            is_ready = int(data["ready_day"]) <= day
            state = "ICU_READY_BLOCKED" if is_ready else "ICU_TREATING"
            icu_treating += int(not is_ready)
            icu_blocked += int(is_ready)
            patient_rows.append({
                "day_index": day,
                "day": cap["day"],
                "patient_uid": pid,
                "patient_id": data.get("patient_id"),
                "source": data["source"],
                "state": state,
                "occupies_icu": 1,
                "occupies_ward": 0,
                "ready_for_ward": int(is_ready),
                "blocked_in_icu": int(is_ready),
                "surgery_id": data.get("surgery_id"),
                "specialty": data.get("specialty"),
            })

        for pid, data in ward_active.items():
            patient_rows.append({
                "day_index": day,
                "day": cap["day"],
                "patient_uid": pid,
                "patient_id": data.get("patient_id"),
                "source": data["source"],
                "state": "WARD",
                "occupies_icu": 0,
                "occupies_ward": 1,
                "ready_for_ward": 0,
                "blocked_in_icu": 0,
                "surgery_id": data.get("surgery_id"),
                "specialty": data.get("specialty"),
            })

        icu_occ = len(icu_active)
        ward_occ = len(ward_active)
        daily_rows.append({
            "day_index": day,
            "day": cap["day"],
            "icu_capacity": icu_cap,
            "ward_capacity": ward_cap,
            "icu_treating": int(icu_treating),
            "icu_ready_blocked": int(icu_blocked),
            "icu_occupancy_blocking": int(icu_occ),
            "ward_occupancy_blocking": int(ward_occ),
            "icu_excess_blocking": int(max(0, icu_occ - icu_cap)),
            "ward_excess_blocking": int(max(0, ward_occ - ward_cap)),
            "blocked_transfer_count": int(icu_blocked),
            "effective_icu_capacity_loss": int(icu_blocked),
        })

    return pd.DataFrame(patient_rows), pd.DataFrame(daily_rows)


# ---------------------------------------------------------------------
# Execution feasibility evaluators
# ---------------------------------------------------------------------

def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end) - 1e-9


def evaluate_blocks(schedule: pd.DataFrame, instance: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    blocks = instance.get("blocks", pd.DataFrame())
    violations: List[dict] = []
    rows: List[dict] = []

    if blocks is None or blocks.empty:
        return pd.DataFrame(rows), pd.DataFrame([{"type": "missing_blocks", "count": 1, "detail": "blocks table missing"}])

    bmap = blocks.set_index("block_id").to_dict("index")

    for block_id, group in schedule.groupby("block_id", dropna=False):
        if pd.isna(block_id) or int(block_id) not in bmap:
            violations.append({"type": "invalid_block_id", "count": len(group), "detail": f"block_id={block_id}"})
            continue

        info = bmap[int(block_id)]
        block_len = float(info["block_length_min"])
        planned_end = float(group["planned_end_min"].max())
        total_duration = float(group["duration_min"].sum())
        overtime = max(0.0, planned_end - block_len)
        idle = max(0.0, block_len - total_duration)

        # Specialty compatibility.
        bad_sp = group[group["specialty"].astype(str) != str(info["specialty"])]
        if len(bad_sp) > 0:
            violations.append({
                "type": "block_specialty_incompatible",
                "count": len(bad_sp),
                "detail": f"block_id={block_id}, block_specialty={info['specialty']}",
            })

        # Overlap inside block.
        overlap_count = 0
        intervals = group[["patient_id", "planned_start_min", "planned_end_min"]].sort_values("planned_start_min").values.tolist()
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                if _interval_overlap(float(intervals[i][1]), float(intervals[i][2]), float(intervals[j][1]), float(intervals[j][2])):
                    overlap_count += 1

        if overlap_count:
            violations.append({
                "type": "within_block_overlap",
                "count": overlap_count,
                "detail": f"block_id={block_id}",
            })

        rows.append({
            "block_id": int(block_id),
            "or_id": int(info["or_id"]),
            "day_index": int(info["day_index"]),
            "day": info["day"],
            "specialty": info["specialty"],
            "n_surgeries": int(len(group)),
            "total_duration_min": round(total_duration, 3),
            "planned_end_min": round(planned_end, 3),
            "block_length_min": block_len,
            "or_overtime_min": round(overtime, 3),
            "or_idle_min": round(idle, 3),
            "within_block_overlap_count": int(overlap_count),
        })

    return pd.DataFrame(rows), pd.DataFrame(violations)


def evaluate_surgeons(schedule: pd.DataFrame, instance: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    violations: List[dict] = []

    if "surgeon_id" not in schedule.columns:
        violations.append({"type": "missing_surgeon_id", "count": len(schedule), "detail": "schedule has no surgeon_id column"})
        return pd.DataFrame(rows), pd.DataFrame(violations)

    surgeons = instance.get("surgeons", pd.DataFrame())
    eligibility = instance.get("patient_surgeon_eligibility", pd.DataFrame())
    calendar = instance.get("surgeon_calendar", pd.DataFrame())

    # Eligibility.
    if isinstance(eligibility, pd.DataFrame) and not eligibility.empty:
        eligible_pairs = set(zip(eligibility["patient_id"].astype(int), eligibility["surgeon_id"].astype(str)))
        bad = []
        for _, r in schedule.iterrows():
            pair = (int(r["patient_id"]), str(r["surgeon_id"]))
            if pair not in eligible_pairs:
                bad.append(pair)
        if bad:
            violations.append({"type": "surgeon_ineligible", "count": len(bad), "detail": str(bad[:10])})

    # Calendar and daily workload.
    surgeon_daily_max = {}
    if isinstance(surgeons, pd.DataFrame) and not surgeons.empty and "daily_max_minutes" in surgeons.columns:
        surgeon_daily_max = dict(zip(surgeons["surgeon_id"].astype(str), surgeons["daily_max_minutes"].astype(float)))

    for (sid, day), group in schedule.groupby(["surgeon_id", "day_index"], dropna=False):
        sid = str(sid)
        day = int(day)
        total_duration = float(group["duration_min"].sum())
        max_minutes = float(surgeon_daily_max.get(sid, np.inf))
        overload = max(0.0, total_duration - max_minutes)

        # Non-overlap.
        intervals = group[["patient_id", "planned_start_min", "planned_end_min"]].sort_values("planned_start_min").values.tolist()
        overlap_count = 0
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                if _interval_overlap(float(intervals[i][1]), float(intervals[i][2]), float(intervals[j][1]), float(intervals[j][2])):
                    overlap_count += 1

        # Calendar containment.
        calendar_violations = 0
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            wins = calendar[(calendar["surgeon_id"].astype(str) == sid) & (calendar["day_index"].astype(int) == day)]
            for _, r in group.iterrows():
                start = float(r["planned_start_min"])
                end = float(r["planned_end_min"])
                feasible = False
                for _, w in wins.iterrows():
                    if start >= float(w["available_start_min"]) - 1e-9 and end <= float(w["available_end_min"]) + 1e-9:
                        feasible = True
                        break
                if not feasible:
                    calendar_violations += 1

        if overload > 0:
            violations.append({"type": "surgeon_daily_overload", "count": 1, "detail": f"surgeon={sid}, day={day}, overload_min={overload:.1f}"})
        if overlap_count:
            violations.append({"type": "surgeon_overlap", "count": overlap_count, "detail": f"surgeon={sid}, day={day}"})
        if calendar_violations:
            violations.append({"type": "surgeon_calendar_violation", "count": calendar_violations, "detail": f"surgeon={sid}, day={day}"})

        rows.append({
            "surgeon_id": sid,
            "day_index": day,
            "n_surgeries": int(len(group)),
            "total_duration_min": round(total_duration, 3),
            "daily_max_minutes": max_minutes if np.isfinite(max_minutes) else None,
            "surgeon_overload_min": round(overload, 3),
            "surgeon_overlap_count": int(overlap_count),
            "calendar_violation_count": int(calendar_violations),
        })

    return pd.DataFrame(rows), pd.DataFrame(violations)


def _max_simultaneous(intervals: List[Tuple[float, float]]) -> int:
    events = []
    for s, e in intervals:
        events.append((float(s), +1))
        events.append((float(e), -1))
    # End before start at same time.
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return int(peak)


def evaluate_equipment(schedule: pd.DataFrame, instance: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equipment = instance.get("equipment", pd.DataFrame())
    peq = instance.get("patient_equipment", pd.DataFrame())
    violations: List[dict] = []
    rows: List[dict] = []

    if not isinstance(equipment, pd.DataFrame) or equipment.empty or not isinstance(peq, pd.DataFrame) or peq.empty:
        return pd.DataFrame(rows), pd.DataFrame([{"type": "missing_equipment_data", "count": 1, "detail": "equipment/patient_equipment missing"}])

    qty = dict(zip(equipment["equipment_type"].astype(str), equipment["quantity"].astype(int)))
    merged = schedule.merge(peq[["patient_id", "equipment_type", "quantity_required"]], on="patient_id", how="left")
    merged = merged.dropna(subset=["equipment_type"])

    for (etype, day), group in merged.groupby(["equipment_type", "day_index"]):
        intervals = [(r["planned_start_min"], r["planned_end_min"]) for _, r in group.iterrows()]
        peak = _max_simultaneous(intervals)
        cap = int(qty.get(str(etype), 0))
        excess = max(0, peak - cap)
        if excess:
            violations.append({"type": "equipment_capacity_excess", "count": int(excess), "detail": f"equipment={etype}, day={day}, peak={peak}, cap={cap}"})
        rows.append({
            "equipment_type": str(etype),
            "day_index": int(day),
            "peak_simultaneous_use": int(peak),
            "capacity": int(cap),
            "equipment_excess": int(excess),
        })

    return pd.DataFrame(rows), pd.DataFrame(violations)


def evaluate_priority(schedule: pd.DataFrame, instance: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    surgeries = instance["surgeries"].copy()
    priority = instance.get("patient_priority", pd.DataFrame())
    if isinstance(priority, pd.DataFrame) and not priority.empty:
        patient = surgeries[["patient_id", "surgery_id", "specialty"]].merge(priority, on=["patient_id", "surgery_id", "specialty"], how="left")
    else:
        patient = surgeries[["patient_id", "surgery_id", "specialty"]].copy()
        patient["priority_class"] = "unknown"
        patient["delay_penalty"] = 0
        patient["postpone_penalty"] = surgeries.get("postpone_cost", 0)
        patient["release_day"] = 1
        patient["due_day"] = 7

    sched = schedule[["patient_id", "day_index"]].drop_duplicates("patient_id").copy()
    out = patient.merge(sched, on="patient_id", how="left")
    out["scheduled"] = out["day_index"].notna().astype(int)
    out["postponed"] = 1 - out["scheduled"]
    out["delay_days"] = np.where(out["scheduled"] == 1, np.maximum(0, out["day_index"].fillna(0).astype(int) - out["due_day"].fillna(7).astype(int)), 0)
    out["delay_cost"] = out["delay_days"] * out["delay_penalty"].fillna(0)
    out["postpone_cost_eval"] = out["postponed"] * out["postpone_penalty"].fillna(0)
    postponed = out[out["postponed"] == 1].copy()
    return out, postponed


# ---------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------

def evaluate_schedule(schedule: Any,
                      instance: Any,
                      transfer_priority: str = "current_first",
                      fill_preferred_surgeon: bool = True,
                      blocking_cost_per_day: float = 1000.0,
                      hard_violation_penalty: float = 100000.0) -> Dict[str, pd.DataFrame]:
    """Unified schedule evaluator.

    Parameters
    ----------
    schedule:
        DataFrame or path to schedule.csv.
    instance:
        dict or path to an instance directory.
    transfer_priority:
        Priority rule for ICU-to-ward transfers in the blocking simulator.
    fill_preferred_surgeon:
        If schedule has no surgeon_id, fill it from patient_surgeon_eligibility.
    blocking_cost_per_day:
        Cost coefficient for ICU-ready blocked patient-days.
    hard_violation_penalty:
        Penalty coefficient for execution feasibility violations.

    Returns
    -------
    dict of DataFrames.
    """
    if isinstance(instance, (str, Path)):
        inst = load_instance(instance)
    elif isinstance(instance, dict):
        inst = dict(instance)
    else:
        raise TypeError("instance must be an instance dict or directory path")

    raw_schedule = _as_dataframe(schedule, "schedule")
    sch = _ensure_schedule_fields(raw_schedule, inst, fill_preferred_surgeon=fill_preferred_surgeon)

    # Core metrics.
    shehadeh_daily = compute_shehadeh_downstream_metrics(sch, inst)
    patient_day_states, blocking_daily = simulate_blocking_states(sch, inst, transfer_priority=transfer_priority)

    # Execution metrics.
    block_metrics, block_vio = evaluate_blocks(sch, inst)
    surgeon_metrics, surgeon_vio = evaluate_surgeons(sch, inst)
    equipment_metrics, equipment_vio = evaluate_equipment(sch, inst)
    priority_metrics, postponed = evaluate_priority(sch, inst)

    violations = pd.concat([block_vio, surgeon_vio, equipment_vio], ignore_index=True)
    if violations.empty:
        violations = pd.DataFrame(columns=["type", "count", "detail"])

    daily = shehadeh_daily.merge(
        blocking_daily,
        on=["day_index", "day", "icu_capacity", "ward_capacity"],
        how="outer",
    ).fillna(0)

    costs = inst.get("costs", {})
    overtime_cost = float(costs.get("overtime_per_min", 0.0))
    idle_cost = float(costs.get("idle_per_min", 0.0))
    icu_excess_cost = float(costs.get("icu_excess_per_bed_day", 0.0))
    ward_excess_cost = float(costs.get("ward_excess_per_bed_day", 0.0))

    n_total = int(len(inst["surgeries"]))
    n_scheduled = int(sch["patient_id"].nunique())
    n_postponed = int(n_total - n_scheduled)

    high_priority = priority_metrics[priority_metrics["priority_class"].astype(str) == "high"]
    n_high = int(len(high_priority))
    n_high_scheduled = int(high_priority["scheduled"].sum()) if n_high else 0
    n_high_postponed = int(high_priority["postponed"].sum()) if n_high else 0

    or_overtime_min = float(block_metrics["or_overtime_min"].sum()) if not block_metrics.empty else 0.0
    or_idle_min = float(block_metrics["or_idle_min"].sum()) if not block_metrics.empty else 0.0

    icu_excess_sheh = float(daily["icu_excess_shehadeh"].sum())
    ward_excess_sheh = float(daily["ward_excess_shehadeh"].sum())
    icu_excess_block = float(daily["icu_excess_blocking"].sum())
    ward_excess_block = float(daily["ward_excess_blocking"].sum())
    blocked_days = float(daily["blocked_transfer_count"].sum())
    effective_loss = float(daily["effective_icu_capacity_loss"].sum())

    T = max(1, int(daily["day_index"].nunique()))
    icu_PE = float((daily["icu_excess_shehadeh"] > 0).mean())
    ward_PE = float((daily["ward_excess_shehadeh"] > 0).mean())
    icu_ME = float(daily["icu_excess_shehadeh"].mean())
    ward_ME = float(daily["ward_excess_shehadeh"].mean())

    delay_cost = float(priority_metrics["delay_cost"].sum())
    postpone_cost = float(priority_metrics["postpone_cost_eval"].sum())
    blocking_cost = blocking_cost_per_day * blocked_days
    shehadeh_downstream_cost = icu_excess_cost * icu_excess_sheh + ward_excess_cost * ward_excess_sheh
    or_cost = overtime_cost * or_overtime_min + idle_cost * or_idle_min
    violation_count = int(violations["count"].sum()) if "count" in violations.columns and len(violations) else 0
    violation_cost = hard_violation_penalty * violation_count

    total_cost_shehadeh_view = postpone_cost + delay_cost + or_cost + shehadeh_downstream_cost + violation_cost
    total_cost_blocking_view = postpone_cost + delay_cost + or_cost + blocking_cost + icu_excess_cost * icu_excess_block + ward_excess_cost * ward_excess_block + violation_cost

    summary = pd.DataFrame([{
        "n_total": n_total,
        "n_scheduled": n_scheduled,
        "n_postponed": n_postponed,
        "schedule_rate": n_scheduled / max(1, n_total),
        "n_high_priority": n_high,
        "n_high_priority_scheduled": n_high_scheduled,
        "n_high_priority_postponed": n_high_postponed,
        "high_priority_schedule_rate": n_high_scheduled / max(1, n_high),
        "or_overtime_min": round(or_overtime_min, 3),
        "or_idle_min": round(or_idle_min, 3),
        "icu_excess_bed_days_shehadeh": icu_excess_sheh,
        "ward_excess_bed_days_shehadeh": ward_excess_sheh,
        "icu_PE_shehadeh": round(icu_PE, 6),
        "ward_PE_shehadeh": round(ward_PE, 6),
        "icu_ME_shehadeh": round(icu_ME, 6),
        "ward_ME_shehadeh": round(ward_ME, 6),
        "icu_excess_bed_days_blocking": icu_excess_block,
        "ward_excess_bed_days_blocking": ward_excess_block,
        "blocked_transfer_patient_days": blocked_days,
        "effective_icu_capacity_loss": effective_loss,
        "peak_icu_ready_blocked": float(daily["icu_ready_blocked"].max()),
        "delay_cost": round(delay_cost, 3),
        "postpone_cost": round(postpone_cost, 3),
        "or_cost": round(or_cost, 3),
        "shehadeh_downstream_cost": round(shehadeh_downstream_cost, 3),
        "blocking_cost": round(blocking_cost, 3),
        "violation_count": violation_count,
        "violation_cost": round(violation_cost, 3),
        "total_cost_shehadeh_view": round(total_cost_shehadeh_view, 3),
        "total_cost_blocking_view": round(total_cost_blocking_view, 3),
    }])

    return {
        "summary": summary,
        "daily_metrics": daily,
        "patient_day_states": patient_day_states,
        "block_metrics": block_metrics,
        "surgeon_metrics": surgeon_metrics,
        "equipment_metrics": equipment_metrics,
        "priority_metrics": priority_metrics,
        "postponed_patients": postponed,
        "violations": violations,
        "normalized_schedule": sch,
    }


def save_evaluation_results(results: Dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a surgery schedule.")
    parser.add_argument("--instance-dir", required=True, help="Directory containing generated instance files.")
    parser.add_argument("--schedule", default=None, help="Path to schedule.csv. Defaults to instance-dir/schedule.csv.")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory.")
    parser.add_argument("--transfer-priority", default="current_first", choices=["current_first", "elective_first", "fifo"])
    parser.add_argument("--no-fill-preferred-surgeon", action="store_true")
    args = parser.parse_args()

    inst = load_instance(args.instance_dir)
    schedule_path = args.schedule or str(Path(args.instance_dir) / "schedule.csv")
    res = evaluate_schedule(
        schedule_path,
        inst,
        transfer_priority=args.transfer_priority,
        fill_preferred_surgeon=not args.no_fill_preferred_surgeon,
    )
    save_evaluation_results(res, args.output_dir)
    print(res["summary"].to_string(index=False))
    print(f"\nSaved evaluation results to: {args.output_dir}")


"""
Shehadeh-style benchmark generator with ICU-to-ward blocking-state extension.

This script creates semi-synthetic OR-to-downstream elective surgery instances
from the summary statistics and block schedules reported in Appendix G of
Shehadeh et al., "Operating Room-to-Downstream Elective Surgery Planning Under
Uncertainty" (EJOR, 2026).

Important:
- This does NOT reproduce confidential patient-level hospital data.
- It creates benchmark-style instances using reported means, standard deviations,
  specialty mixes, block schedules, bed-capacity settings, and cost settings.
- It additionally adds our proposed ICU-to-ward transfer-blocking states:
  ICU_TREATING, ICU_READY_BLOCKED, and WARD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import json
import numpy as np
import pandas as pd


DAYS_7 = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SURGERY_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ---------------------------------------------------------------------
# Appendix-G summary statistics
# ---------------------------------------------------------------------

CASE_STATS = {
    "CABG+AVR": {"percentage": 7.77, "mu_d": 286, "sd_d": 87, "mu_icu": 4, "sd_icu": 6.5, "mu_ward": 7, "sd_ward": 5.9},
    "AVR":      {"percentage": 11.68, "mu_d": 207, "sd_d": 125, "mu_icu": 3, "sd_icu": 3.2, "mu_ward": 6, "sd_ward": 4.5},
    "CABG":     {"percentage": 40.89, "mu_d": 247, "sd_d": 81, "mu_icu": 3, "sd_icu": 3.8, "mu_ward": 5, "sd_ward": 4.6},
    "MVR":      {"percentage": 4.53, "mu_d": 250, "sd_d": 77, "mu_icu": 3, "sd_icu": 2.9, "mu_ward": 6, "sd_ward": 5.8},
    "TAVR":     {"percentage": 7.18, "mu_d": 83, "sd_d": 36, "mu_icu": 2, "sd_icu": 2.0, "mu_ward": 4, "sd_ward": 3.8},
}

M_STATS = {
    "ENT":      {"percentage": 21.34, "mu_d": 74, "sd_d": 37, "mu_icu": 0.1, "sd_icu": 0.1, "mu_ward": 0.1, "sd_ward": 0.1},
    "OBGYN":    {"percentage": 9.26, "mu_d": 86, "sd_d": 40, "mu_icu": 2.0, "sd_icu": 2.0, "mu_ward": 2.0, "sd_ward": 2.0},
    "ORTHO":    {"percentage": 23.26, "mu_d": 107, "sd_d": 44, "mu_icu": 1.5, "sd_icu": 1.5, "mu_ward": 1.5, "sd_ward": 1.5},
    "NEURO":    {"percentage": 5.04, "mu_d": 160, "sd_d": 77, "mu_icu": 2.0, "sd_icu": 2.0, "mu_ward": 2.0, "sd_ward": 2.0},
    "GEN":      {"percentage": 22.12, "mu_d": 93, "sd_d": 49, "mu_icu": 0.05, "sd_icu": 0.05, "mu_ward": 0.05, "sd_ward": 0.05},
    "OPHTH":    {"percentage": 2.98, "mu_d": 38, "sd_d": 19, "mu_icu": 0.05, "sd_icu": 0.05, "mu_ward": 0.05, "sd_ward": 0.05},
    "VASCULAR": {"percentage": 8.20, "mu_d": 120, "sd_d": 61, "mu_icu": 3.5, "sd_icu": 3.5, "mu_ward": 3.5, "sd_ward": 3.5},
    "CARDIAC":  {"percentage": 2.44, "mu_d": 240, "sd_d": 103, "mu_icu": 2.0, "sd_icu": 2.0, "mu_ward": 3.5, "sd_ward": 3.5},
    "UROLOGY":  {"percentage": 5.36, "mu_d": 64, "sd_d": 52, "mu_icu": 0.8, "sd_icu": 0.8, "mu_ward": 0.8, "sd_ward": 0.8},
}

R_STATS = {
    "CARDIAC":  {"percentage": 3, "mu_d": 99, "sd_d": 53, "mu_icu": 2.0, "sd_icu": 2.0, "mu_ward": 2.0, "sd_ward": 2.0},
    "GASTRO":   {"percentage": 9, "mu_d": 132, "sd_d": 76, "mu_icu": 1.0, "sd_icu": 1.0, "mu_ward": 1.0, "sd_ward": 1.0},
    "GYN":      {"percentage": 9, "mu_d": 78, "sd_d": 52, "mu_icu": 2.0, "sd_icu": 2.0, "mu_ward": 2.0, "sd_ward": 2.0},
    "MED":      {"percentage": 19, "mu_d": 75, "sd_d": 72, "mu_icu": 0.05, "sd_icu": 0.05, "mu_ward": 1.0, "sd_ward": 1.0},
    "ORTHO":    {"percentage": 20, "mu_d": 142, "sd_d": 58, "mu_icu": 1.5, "sd_icu": 1.5, "mu_ward": 1.5, "sd_ward": 1.5},
    "UROLOGY":  {"percentage": 6, "mu_d": 72, "sd_d": 38, "mu_icu": 0.8, "sd_icu": 0.8, "mu_ward": 1.0, "sd_ward": 1.0},
    "ENT":      {"percentage": 18, "mu_d": 74, "sd_d": 37, "mu_icu": 0.1, "sd_icu": 0.1, "mu_ward": 0.1, "sd_ward": 0.1},
    "NEURO":    {"percentage": 5, "mu_d": 160, "sd_d": 77, "mu_icu": 2.0, "sd_icu": 2.0, "mu_ward": 2.0, "sd_ward": 2.0},
    "VASCULAR": {"percentage": 8, "mu_d": 120, "sd_d": 61, "mu_icu": 3.5, "sd_icu": 3.5, "mu_ward": 3.5, "sd_ward": 3.5},
    "OPHTH":    {"percentage": 3, "mu_d": 38, "sd_d": 19, "mu_icu": 0.05, "sd_icu": 0.05, "mu_ward": 0.05, "sd_ward": 0.05},
}

CASE_BLOCKS = (
    [(1, d, "CABG+AVR") for d in SURGERY_DAYS] +
    [(2, d, "AVR") for d in SURGERY_DAYS] +
    [(3, d, "CABG") for d in SURGERY_DAYS] +
    [(4, d, "MVR") for d in SURGERY_DAYS] +
    [(5, d, "TAVR") for d in SURGERY_DAYS]
)

M_BLOCKS = [
    (1, "Monday", "ENT"), (1, "Tuesday", "ENT"), (1, "Wednesday", "ENT"),
    (2, "Wednesday", "ENT"), (2, "Thursday", "ENT"), (2, "Friday", "ENT"),
    (3, "Monday", "OBGYN"), (3, "Wednesday", "OBGYN"), (3, "Friday", "OBGYN"),
    (4, "Monday", "ORTHO"), (4, "Tuesday", "ORTHO"), (4, "Thursday", "ORTHO"), (4, "Friday", "ORTHO"),
    (5, "Tuesday", "ORTHO"), (5, "Wednesday", "NEURO"),
    (6, "Monday", "GEN"), (6, "Tuesday", "GEN"), (6, "Wednesday", "GEN"), (6, "Thursday", "GEN"),
    (7, "Tuesday", "GEN"), (7, "Wednesday", "GEN"), (7, "Thursday", "GEN"), (7, "Friday", "GEN"),
    (8, "Monday", "OPHTH"), (8, "Tuesday", "OPHTH"), (8, "Thursday", "OPHTH"), (8, "Friday", "OPHTH"),
    (9, "Monday", "VASCULAR"), (9, "Wednesday", "CARDIAC"), (9, "Friday", "VASCULAR"),
    (10, "Monday", "UROLOGY"), (10, "Wednesday", "ORTHO"),
]

R_BLOCKS = (
    [(1, d, "ENT") for d in SURGERY_DAYS] +
    [(2, d, "GYN") for d in SURGERY_DAYS] +
    [(3, d, "ORTHO") for d in SURGERY_DAYS] +
    [(4, d, "NEURO") for d in SURGERY_DAYS] +
    [(5, d, "MED") for d in SURGERY_DAYS] +
    [(6, d, "OPHTH") for d in SURGERY_DAYS] +
    [(7, d, "VASCULAR") for d in SURGERY_DAYS] +
    [(8, d, "CARDIAC") for d in SURGERY_DAYS] +
    [(9, d, "UROLOGY") for d in SURGERY_DAYS] +
    [(10, d, "GASTRO") for d in SURGERY_DAYS]
)


def _largest_remainder_counts(total, weights):
    keys = list(weights.keys())
    w = np.array([max(0.0, float(weights[k])) for k in keys], dtype=float)
    if w.sum() <= 0:
        w[:] = 1.0
    raw = total * w / w.sum()
    counts = np.floor(raw).astype(int)
    rem = total - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:rem]:
        counts[idx] += 1
    return {k: int(v) for k, v in zip(keys, counts)}


def _equal_counts(total, keys):
    return _largest_remainder_counts(total, {k: 1.0 for k in keys})


def _lognormal_params(mean, sd):
    mean = max(float(mean), 1e-9)
    sd = max(float(sd), 1e-9)
    sigma2 = math.log(1.0 + (sd * sd) / (mean * mean))
    return math.log(mean) - 0.5 * sigma2, math.sqrt(max(sigma2, 1e-12))


def _sample_positive(rng, mean, sd, distribution="lognormal", lower=0.0, integer=False):
    if distribution == "lognormal":
        mu, sigma = _lognormal_params(mean, sd)
        val = float(rng.lognormal(mu, sigma))
    elif distribution == "truncnorm":
        val = float(rng.normal(mean, sd))
        for _ in range(100):
            if val >= lower:
                break
            val = float(rng.normal(mean, sd))
    else:
        raise ValueError("distribution must be 'lognormal' or 'truncnorm'")
    val = max(float(lower), val)
    return int(max(0, round(val))) if integer else val


def _approx_quantiles(mean, sd, distribution="lognormal", integer=False):
    rng = np.random.default_rng(1234567)
    samples = [_sample_positive(rng, mean, sd, distribution=distribution, lower=0.0, integer=False) for _ in range(15000)]
    lo, hi = np.quantile(samples, [0.20, 0.80])
    if integer:
        lo = int(max(0, math.floor(lo)))
        hi = int(max(lo, math.ceil(hi)))
    return float(lo), float(hi)


def _build_blocks(block_list, block_length_min, stats):
    rows = []
    for block_id, (or_id, day, specialty) in enumerate(block_list, start=1):
        mu = stats[specialty]["mu_d"]
        max_cases = max(1, min(12, int(math.ceil(block_length_min / max(20.0, 0.40 * mu)))))
        rows.append({
            "block_id": block_id,
            "or_id": int(or_id),
            "day": day,
            "day_index": DAYS_7.index(day) + 1,
            "specialty": specialty,
            "block_length_min": int(block_length_min),
            "max_cases": int(max_cases),
        })
    return pd.DataFrame(rows)


def _build_capacities(weekday_beds, weekend_beds, T=7):
    rows = []
    for idx, day in enumerate(DAYS_7[:T], start=1):
        weekend = day in ["Saturday", "Sunday"]
        cap = int(weekend_beds if weekend else weekday_beds)
        rows.append({"day_index": idx, "day": day, "icu_capacity": cap, "ward_capacity": cap})
    return pd.DataFrame(rows)


@dataclass
class BenchmarkConfig:
    dataset: str = "case"          # "case", "M", or "R"
    I: int = 70
    seed: int = 1
    T: int = 7
    block_length_min: int = 480
    bed_setting: Tuple[int, int] = (10, 7)
    cost_structure: int = 2        # 1 -> overtime 26; 2 -> overtime 100
    distribution: str = "lognormal"
    use_reported_percentages: bool = False
    add_blocking_extension_fields: bool = True


def generate_benchmark(config):
    rng = np.random.default_rng(config.seed)

    if config.dataset == "case":
        stats, block_list = CASE_STATS, CASE_BLOCKS
        counts = (_largest_remainder_counts(config.I, {k: v["percentage"] for k, v in stats.items()})
                  if config.use_reported_percentages else _equal_counts(config.I, list(stats.keys())))
    elif config.dataset == "M":
        stats, block_list = M_STATS, M_BLOCKS
        counts = _largest_remainder_counts(config.I, {k: v["percentage"] for k, v in stats.items()})
    elif config.dataset == "R":
        stats, block_list = R_STATS, R_BLOCKS
        counts = (_largest_remainder_counts(config.I, {k: v["percentage"] for k, v in stats.items()})
                  if config.use_reported_percentages else _equal_counts(config.I, list(stats.keys())))
    else:
        raise ValueError("dataset must be 'case', 'M', or 'R'")

    blocks = _build_blocks(block_list, config.block_length_min, stats)
    capacities = _build_capacities(config.bed_setting[0], config.bed_setting[1], config.T)

    overtime_cost = 26.0 if config.cost_structure == 1 else 100.0
    idle_cost = overtime_cost / 1.5
    waiting_cost = idle_cost
    bed_excess_cost = 1000.0
    alpha = 0.5

    rows = []
    patient_id = 1
    for specialty, n in counts.items():
        st = stats[specialty]
        d_lo, d_hi = _approx_quantiles(st["mu_d"], st["sd_d"], distribution=config.distribution, integer=False)
        icu_lo, icu_hi = _approx_quantiles(st["mu_icu"], st["sd_icu"], distribution=config.distribution, integer=True)
        ward_lo, ward_hi = _approx_quantiles(st["mu_ward"], st["sd_ward"], distribution=config.distribution, integer=True)

        block_ids = blocks.loc[blocks["specialty"] == specialty, "block_id"].astype(int).tolist()
        for j in range(int(n)):
            duration = _sample_positive(rng, st["mu_d"], st["sd_d"], distribution=config.distribution, lower=5.0, integer=False)
            icu_los = int(_sample_positive(rng, st["mu_icu"], st["sd_icu"], distribution=config.distribution, lower=0.0, integer=True))
            ward_los = int(_sample_positive(rng, st["mu_ward"], st["sd_ward"], distribution=config.distribution, lower=0.0, integer=True))

            row = {
                "patient_id": int(patient_id),
                "surgery_id": f"{specialty}_{j+1:03d}",
                "specialty": specialty,
                "duration_min": round(duration, 2),
                "duration_mean": st["mu_d"],
                "duration_sd": st["sd_d"],
                "duration_lb": round(max(1.0, d_lo), 2),
                "duration_ub": round(max(d_lo, d_hi), 2),
                "icu_los_days": int(icu_los),
                "icu_los_mean": st["mu_icu"],
                "icu_los_sd": st["sd_icu"],
                "icu_los_lb": int(icu_lo),
                "icu_los_ub": int(max(icu_lo, icu_hi)),
                "ward_los_days": int(ward_los),
                "ward_los_mean": st["mu_ward"],
                "ward_los_sd": st["sd_ward"],
                "ward_los_lb": int(ward_lo),
                "ward_los_ub": int(max(ward_lo, ward_hi)),
                "compatible_block_ids": ",".join(map(str, block_ids)),
                "assign_cost": alpha * overtime_cost,
                "postpone_cost": alpha * overtime_cost * 1.5,
            }
            if config.add_blocking_extension_fields:
                row.update({
                    "requires_icu": int(icu_los > 0),
                    "icu_treatment_days": int(icu_los),
                    "ready_for_ward_after_icu_days": int(icu_los),
                    "can_be_blocked_in_icu": int(icu_los > 0 and ward_los > 0),
                })
            rows.append(row)
            patient_id += 1

    surgeries = pd.DataFrame(rows).sort_values(["specialty", "patient_id"]).reset_index(drop=True)
    type_stats = pd.DataFrame([{"specialty": k, **v, "count": counts.get(k, 0)} for k, v in stats.items()])

    costs = {
        "overtime_per_min": overtime_cost,
        "idle_per_min": idle_cost,
        "waiting_per_min": waiting_cost,
        "icu_excess_per_bed_day": bed_excess_cost,
        "ward_excess_per_bed_day": bed_excess_cost,
        "alpha_assignment": alpha,
    }

    metadata = {
        "dataset": config.dataset,
        "I": config.I,
        "seed": config.seed,
        "T": config.T,
        "block_length_min": config.block_length_min,
        "bed_setting_weekday_weekend": list(config.bed_setting),
        "cost_structure": config.cost_structure,
        "distribution": config.distribution,
        "use_reported_percentages": config.use_reported_percentages,
        "notes": [
            "Semi-synthetic generator from Appendix-G summary statistics.",
            "Does not reproduce confidential patient-level hospital data.",
            "Case-I and R-I use equal type counts by default, matching the reported instance construction.",
        ],
    }

    return {
        "surgeries": surgeries,
        "blocks": blocks,
        "capacities": capacities,
        "costs": costs,
        "type_stats": type_stats,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------
# Blocking-state extension
# ---------------------------------------------------------------------

def generate_current_bed_state(instance, seed=None, initial_icu_occupancy_rate=0.70,
                               initial_ward_occupancy_rate=0.75, ready_fraction_among_icu=0.25):
    rng = np.random.default_rng(instance["metadata"].get("seed", 0) if seed is None else seed)
    capacities = instance["capacities"]
    cap1 = capacities.loc[capacities["day_index"] == 1].iloc[0]

    n_icu = int(round(float(cap1["icu_capacity"]) * initial_icu_occupancy_rate))
    n_ward = int(round(float(cap1["ward_capacity"]) * initial_ward_occupancy_rate))

    type_stats = instance["type_stats"]
    ward_mean = max(1.0, float(np.average(type_stats["mu_ward"], weights=np.maximum(type_stats["count"], 1))))

    icu_rows = []
    for j in range(n_icu):
        ready = rng.random() < ready_fraction_among_icu
        rem = 0 if ready else int(rng.integers(1, 4))
        ward_los = int(max(1, round(rng.lognormal(*_lognormal_params(ward_mean, max(1.0, ward_mean * 0.7))))))
        icu_rows.append({
            "patient_uid": f"CICU_{j+1:03d}",
            "source": "current_icu",
            "icu_treatment_remaining_days": rem,
            "ready_at_start": int(ready),
            "ready_for_ward_day": int(1 if ready else 1 + rem),
            "ward_los_days": int(ward_los),
        })

    ward_rows = []
    for j in range(n_ward):
        rem = int(rng.integers(1, max(2, int(round(ward_mean)) + 3)))
        ward_rows.append({
            "patient_uid": f"CWARD_{j+1:03d}",
            "source": "current_ward",
            "ward_remaining_days": int(rem),
            "planned_discharge_day": int(1 + rem),
        })

    return pd.DataFrame(icu_rows), pd.DataFrame(ward_rows)


def build_greedy_initial_schedule(instance, priority="patient_id", allow_overtime=False):
    surgeries = instance["surgeries"].copy()
    blocks = instance["blocks"].copy().sort_values(["day_index", "or_id", "block_id"])
    unscheduled = set(surgeries["patient_id"].astype(int).tolist())
    rows = []

    for _, block in blocks.iterrows():
        specialty = block["specialty"]
        block_len = float(block["block_length_min"])
        current_time = 0.0
        position = 1

        candidates = surgeries[(surgeries["patient_id"].isin(unscheduled)) & (surgeries["specialty"] == specialty)].copy()
        if priority == "short_first":
            candidates = candidates.sort_values(["duration_min", "patient_id"])
        elif priority == "long_first":
            candidates = candidates.sort_values(["duration_min", "patient_id"], ascending=[False, True])
        else:
            candidates = candidates.sort_values("patient_id")

        for _, surg in candidates.iterrows():
            dur = float(surg["duration_min"])
            if (not allow_overtime) and current_time + dur > block_len + 1e-9:
                continue

            pid = int(surg["patient_id"])
            rows.append({
                "patient_id": pid,
                "patient_uid": f"E_{pid:04d}",
                "surgery_id": surg["surgery_id"],
                "specialty": specialty,
                "block_id": int(block["block_id"]),
                "or_id": int(block["or_id"]),
                "day": block["day"],
                "day_index": int(block["day_index"]),
                "position": int(position),
                "planned_start_min": round(current_time, 2),
                "duration_min": dur,
                "planned_end_min": round(current_time + dur, 2),
                "requires_icu": int(surg.get("requires_icu", int(float(surg["icu_los_days"]) > 0))),
                "icu_treatment_days": int(surg.get("icu_treatment_days", surg["icu_los_days"])),
                "ward_los_days": int(surg["ward_los_days"]),
            })
            current_time += dur
            position += 1
            unscheduled.remove(pid)

    return pd.DataFrame(rows)


def simulate_blocking_states(instance, schedule=None, current_icu=None, current_ward=None, transfer_priority="current_first"):
    if schedule is None:
        schedule = instance.get("schedule")
    if schedule is None:
        schedule = build_greedy_initial_schedule(instance)

    if current_icu is None:
        current_icu = instance.get("current_icu")
    if current_ward is None:
        current_ward = instance.get("current_ward")
    if current_icu is None or current_ward is None:
        current_icu, current_ward = generate_current_bed_state(instance)

    capacities = instance["capacities"].copy()
    T = int(instance["metadata"].get("T", int(capacities["day_index"].max())))

    icu_active = {}
    ward_active = {}
    entry_counter = 0

    for _, row in current_icu.iterrows():
        entry_counter += 1
        icu_active[str(row["patient_uid"])] = {
            "source": "current_icu",
            "ready_day": int(row["ready_for_ward_day"]),
            "ward_los_days": int(row["ward_los_days"]),
            "entry_order": entry_counter,
            "surgery_id": None,
            "specialty": None,
        }

    for _, row in current_ward.iterrows():
        ward_active[str(row["patient_uid"])] = {
            "source": "current_ward",
            "discharge_day": int(row["planned_discharge_day"]),
            "surgery_id": None,
            "specialty": None,
        }

    schedule_by_day = {}
    if isinstance(schedule, pd.DataFrame) and len(schedule) > 0:
        for _, row in schedule.iterrows():
            schedule_by_day.setdefault(int(row["day_index"]), []).append(row.to_dict())

    patient_rows = []
    daily_rows = []

    for day in range(1, T + 1):
        cap_row = capacities.loc[capacities["day_index"] == day].iloc[0]
        icu_cap = int(cap_row["icu_capacity"])
        ward_cap = int(cap_row["ward_capacity"])

        # Ward discharge first.
        for pid in list(ward_active.keys()):
            if int(ward_active[pid]["discharge_day"]) <= day:
                del ward_active[pid]

        # New surgeries.
        direct_ward_arrivals = []
        for row in schedule_by_day.get(day, []):
            pid = str(row["patient_uid"])
            requires_icu = int(row.get("requires_icu", 0)) == 1
            icu_days = int(max(0, row.get("icu_treatment_days", 0)))
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
                }
            elif ward_days > 0:
                direct_ward_arrivals.append({
                    "patient_uid": pid,
                    "source": "elective_direct_ward",
                    "discharge_day": int(day + ward_days),
                    "surgery_id": row.get("surgery_id"),
                    "specialty": row.get("specialty"),
                })

        # ICU-ready patients attempt transfer.
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
                }
                if pid in icu_active:
                    del icu_active[pid]

        # Direct ward arrivals after ICU transfers.
        for row in direct_ward_arrivals:
            ward_active[row["patient_uid"]] = row

        # Record.
        icu_treating = 0
        icu_blocked = 0
        for pid, data in icu_active.items():
            is_ready = int(data["ready_day"]) <= day
            state = "ICU_READY_BLOCKED" if is_ready else "ICU_TREATING"
            icu_treating += int(not is_ready)
            icu_blocked += int(is_ready)
            patient_rows.append({
                "day_index": day,
                "day": cap_row["day"],
                "patient_uid": pid,
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
                "day": cap_row["day"],
                "patient_uid": pid,
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
            "day": cap_row["day"],
            "icu_capacity": icu_cap,
            "ward_capacity": ward_cap,
            "icu_treating": int(icu_treating),
            "icu_ready_blocked": int(icu_blocked),
            "icu_occupancy": int(icu_occ),
            "ward_occupancy": int(ward_occ),
            "icu_excess": int(max(0, icu_occ - icu_cap)),
            "ward_excess": int(max(0, ward_occ - ward_cap)),
            "blocked_transfer_count": int(icu_blocked),
            "effective_icu_capacity_loss": int(icu_blocked),
        })

    return pd.DataFrame(patient_rows), pd.DataFrame(daily_rows)


def extend_instance_with_blocking_states(instance, seed=None, initial_icu_occupancy_rate=0.70,
                                         initial_ward_occupancy_rate=0.75,
                                         ready_fraction_among_icu=0.25,
                                         schedule_priority="patient_id",
                                         transfer_priority="current_first"):
    extended = dict(instance)
    current_icu, current_ward = generate_current_bed_state(
        instance,
        seed=seed,
        initial_icu_occupancy_rate=initial_icu_occupancy_rate,
        initial_ward_occupancy_rate=initial_ward_occupancy_rate,
        ready_fraction_among_icu=ready_fraction_among_icu,
    )
    schedule = build_greedy_initial_schedule(instance, priority=schedule_priority)
    patient_day_states, daily_bed_states = simulate_blocking_states(
        instance,
        schedule=schedule,
        current_icu=current_icu,
        current_ward=current_ward,
        transfer_priority=transfer_priority,
    )

    extended["current_icu"] = current_icu
    extended["current_ward"] = current_ward
    extended["schedule"] = schedule
    extended["patient_day_states"] = patient_day_states
    extended["daily_bed_states"] = daily_bed_states
    extended["metadata"] = dict(instance["metadata"])
    extended["metadata"]["blocking_extension"] = {
        "initial_icu_occupancy_rate": initial_icu_occupancy_rate,
        "initial_ward_occupancy_rate": initial_ward_occupancy_rate,
        "ready_fraction_among_icu": ready_fraction_among_icu,
        "schedule_priority": schedule_priority,
        "transfer_priority": transfer_priority,
        "state_convention": "daily; ward discharges first, ICU transfers next, direct-ward surgical arrivals last",
    }
    return extended


def run_instance_sanity_checks(instance, duration_rel_tol=0.35, los_rel_tol=0.75):
    results = []

    def add(check, status, detail):
        results.append({"check": check, "status": status, "detail": detail})

    surgeries = instance.get("surgeries")
    blocks = instance.get("blocks")
    capacities = instance.get("capacities")
    type_stats = instance.get("type_stats")
    metadata = instance.get("metadata", {})

    if not isinstance(surgeries, pd.DataFrame) or surgeries.empty:
        add("surgeries table exists", "FAIL", "surgeries is missing or empty")
        return pd.DataFrame(results)

    I = int(metadata.get("I", len(surgeries)))
    add("surgery count", "PASS" if len(surgeries) == I else "FAIL", f"expected I={I}, observed {len(surgeries)}")

    if isinstance(blocks, pd.DataFrame):
        expected_blocks = {"case": 25, "M": 32, "R": 50}.get(metadata.get("dataset"))
        if expected_blocks is not None:
            add("block count", "PASS" if len(blocks) == expected_blocks else "FAIL", f"expected {expected_blocks}, observed {len(blocks)}")
        block_specialties = set(blocks["specialty"].astype(str))
        surgery_specialties = set(surgeries["specialty"].astype(str))
        missing = sorted(surgery_specialties - block_specialties)
        add("every surgery type has at least one block", "PASS" if not missing else "FAIL",
            "all surgery specialties appear in block table" if not missing else f"missing: {missing}")
        empty_compat = surgeries["compatible_block_ids"].fillna("").astype(str).str.len().eq(0).sum()
        add("compatible block ids", "PASS" if empty_compat == 0 else "FAIL", f"{empty_compat} surgeries have empty compatible_block_ids")
    else:
        add("blocks table exists", "FAIL", "blocks is missing")

    if isinstance(capacities, pd.DataFrame):
        T = int(metadata.get("T", capacities["day_index"].nunique()))
        add("capacity horizon", "PASS" if capacities["day_index"].nunique() == T else "FAIL",
            f"expected T={T}, observed {capacities['day_index'].nunique()} unique days")
        pos = ((capacities["icu_capacity"] > 0) & (capacities["ward_capacity"] > 0)).all()
        add("positive capacities", "PASS" if pos else "FAIL", "ICU and ward capacities are positive for all days")
    else:
        add("capacities table exists", "FAIL", "capacities is missing")

    if isinstance(type_stats, pd.DataFrame):
        generated_counts = surgeries.groupby("specialty").size().to_dict()
        mismatches = []
        for _, row in type_stats.iterrows():
            sp = row["specialty"]
            if int(row["count"]) != int(generated_counts.get(sp, 0)):
                mismatches.append((sp, int(row["count"]), int(generated_counts.get(sp, 0))))
        add("type counts", "PASS" if not mismatches else "FAIL",
            "all type counts match type_stats" if not mismatches else f"mismatches: {mismatches}")

        warns = []
        for sp, group in surgeries.groupby("specialty"):
            target = type_stats[type_stats["specialty"] == sp].iloc[0]
            if len(group) < 2:
                continue
            rel_d = abs(float(group["duration_min"].mean()) - float(target["mu_d"])) / max(1.0, float(target["mu_d"]))
            if rel_d > duration_rel_tol:
                warns.append(f"{sp}: duration mean rel.err={rel_d:.2f}")
            rel_i = abs(float(group["icu_los_days"].mean()) - float(target["mu_icu"])) / max(1.0, float(target["mu_icu"]))
            if rel_i > los_rel_tol:
                warns.append(f"{sp}: ICU LOS mean rel.err={rel_i:.2f}")
            rel_w = abs(float(group["ward_los_days"].mean()) - float(target["mu_ward"])) / max(1.0, float(target["mu_ward"]))
            if rel_w > los_rel_tol:
                warns.append(f"{sp}: Ward LOS mean rel.err={rel_w:.2f}")
        add("generated sample moments", "PASS" if not warns else "WARN",
            "sample means are within loose tolerance" if not warns else "; ".join(warns[:8]))
    else:
        add("type_stats table exists", "FAIL", "type_stats is missing")

    costs = instance.get("costs", {})
    expected_o = 26.0 if metadata.get("cost_structure") == 1 else 100.0
    add("cost structure", "PASS" if abs(float(costs.get("overtime_per_min", -1)) - expected_o) < 1e-9 else "FAIL",
        f"expected overtime cost {expected_o}, observed {costs.get('overtime_per_min')}")

    schedule = instance.get("schedule")
    if isinstance(schedule, pd.DataFrame) and not schedule.empty:
        dup = int(schedule["patient_id"].duplicated().sum())
        add("schedule duplicate patients", "PASS" if dup == 0 else "FAIL", f"{dup} duplicated scheduled patients")
        merged = schedule.merge(blocks[["block_id", "specialty", "block_length_min"]], on="block_id", suffixes=("", "_block"))
        bad_compat = int((merged["specialty"] != merged["specialty_block"]).sum())
        add("schedule specialty compatibility", "PASS" if bad_compat == 0 else "FAIL", f"{bad_compat} incompatible assignments")
        block_over = []
        for block_id, group in schedule.groupby("block_id"):
            end_max = float(group["planned_end_min"].max())
            block_len = float(blocks.loc[blocks["block_id"] == block_id, "block_length_min"].iloc[0])
            if end_max > block_len + 1e-6:
                block_over.append((int(block_id), round(end_max, 2), round(block_len, 2)))
        add("schedule block length", "PASS" if not block_over else "FAIL",
            "no scheduled block exceeds planned length" if not block_over else f"overloaded blocks: {block_over[:8]}")
    else:
        add("schedule table", "WARN", "no schedule table attached; run extend_instance_with_blocking_states()")

    states = instance.get("patient_day_states")
    daily = instance.get("daily_bed_states")
    if isinstance(states, pd.DataFrame) and isinstance(daily, pd.DataFrame) and not states.empty and not daily.empty:
        both = states[(states["occupies_icu"] == 1) & (states["occupies_ward"] == 1)]
        add("patient not simultaneously in ICU and ward", "PASS" if both.empty else "FAIL", f"{len(both)} patient-day rows occupy both units")
        agg = states.groupby("day_index")[["occupies_icu", "occupies_ward", "blocked_in_icu"]].sum().reset_index()
        comp = agg.merge(daily[["day_index", "icu_occupancy", "ward_occupancy", "icu_ready_blocked"]], on="day_index", how="left")
        mismatch = comp[(comp["occupies_icu"] != comp["icu_occupancy"]) |
                        (comp["occupies_ward"] != comp["ward_occupancy"]) |
                        (comp["blocked_in_icu"] != comp["icu_ready_blocked"])]
        add("daily aggregates match patient states", "PASS" if mismatch.empty else "FAIL",
            "aggregates match" if mismatch.empty else f"mismatched days: {mismatch['day_index'].tolist()}")
        add("nonnegative excess variables", "PASS" if ((daily["icu_excess"] >= 0) & (daily["ward_excess"] >= 0)).all() else "FAIL",
            "ICU/Ward excess values are nonnegative")
        add("blocking signal", "PASS", f"total blocked patient-days = {int(daily['blocked_transfer_count'].sum())}")
    else:
        add("blocking-state tables", "WARN", "no patient_day_states/daily_bed_states attached")

    return pd.DataFrame(results)


def print_sanity_report(report):
    if report.empty:
        print("No sanity-check results.")
        return
    for status in ["FAIL", "WARN", "PASS"]:
        sub = report[report["status"] == status]
        if len(sub) == 0:
            continue
        print(f"\n[{status}]")
        for _, row in sub.iterrows():
            print(f"- {row['check']}: {row['detail']}")


def save_instance(instance, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for key, obj in instance.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(out / f"{key}.csv", index=False)

    with open(out / "costs.json", "w", encoding="utf-8") as f:
        json.dump(instance.get("costs", {}), f, indent=2)
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(instance.get("metadata", {}), f, indent=2)



# ---------------------------------------------------------------------
# 6. Synthetic clinical inputs for full algorithm testing
# ---------------------------------------------------------------------

def _case_category_from_specialty(specialty, rng):
    """Map surgery type/specialty to synthetic clinical category.

    For Case-I cardiac data, this is not a true hepatobiliary label.
    It is a controllable synthetic proxy for testing priority/case-mix logic.
    """
    specialty = str(specialty)
    if specialty in ["CABG+AVR", "CABG", "MVR"]:
        # Longer, more invasive cardiac cases.
        cancer_flag = 0
        open_flag = 1
        approach = "open"
        complexity = "high" if specialty == "CABG+AVR" else "medium_high"
    elif specialty == "AVR":
        cancer_flag = 0
        open_flag = int(rng.random() < 0.65)
        approach = "open" if open_flag else "minimally_invasive"
        complexity = "medium"
    elif specialty == "TAVR":
        cancer_flag = 0
        open_flag = 0
        approach = "minimally_invasive"
        complexity = "low"
    else:
        # Generic mapping for M/R datasets.
        invasive_types = {"CARDIAC", "VASCULAR", "NEURO", "ORTHO", "GASTRO"}
        open_flag = int(specialty in invasive_types and rng.random() < 0.70)
        approach = "open" if open_flag else "minimally_invasive"
        cancer_flag = int(specialty in {"GASTRO", "GEN", "VASCULAR"} and rng.random() < 0.30)
        complexity = "medium_high" if open_flag else "low"
    return cancer_flag, open_flag, approach, complexity


def _normalize_priority_policy(policy: str) -> str:
    policy = str(policy or "legacy_acuity_linked").strip().lower().replace("-", "_")
    aliases = {
        "legacy": "legacy_acuity_linked",
        "acuity": "legacy_acuity_linked",
        "acuity_linked": "legacy_acuity_linked",
        "independent": "independent_urgency",
        "policy": "independent_urgency",
        "policy_priority": "independent_urgency",
        "policy_based": "independent_urgency",
    }
    policy = aliases.get(policy, policy)
    if policy not in {"legacy_acuity_linked", "independent_urgency"}:
        raise ValueError(
            "priority_policy must be 'legacy_acuity_linked' or 'independent_urgency'"
        )
    return policy


def _priority_count_allocation(n: int, shares: Dict[str, float]) -> Dict[str, int]:
    """Allocate exact high/medium/low counts by largest remainder."""
    classes = ["high", "medium", "low"]
    weights = {c: max(0.0, float(shares.get(c, 0.0))) for c in classes}
    if sum(weights.values()) <= 0:
        weights = {"high": 0.25, "medium": 0.45, "low": 0.30}
    return _largest_remainder_counts(int(n), weights)


def generate_patient_priority(
    instance,
    seed=None,
    hepatobiliary_proxy=True,
    priority_policy: str = "legacy_acuity_linked",
    priority_shares: Optional[Dict[str, float]] = None,
):
    """Generate synthetic priority and resource-profile attributes.

    Two policies are available.

    ``legacy_acuity_linked`` reproduces the historic experimental logic, where
    priority partially depends on cancer/open proxies and realised duration/LOS.
    It is retained only for backward compatibility and sensitivity analysis.

    ``independent_urgency`` is the publication default.  It assigns an exact
    policy urgency mix using a seeded random permutation and does *not* use
    realised duration, ICU LOS, ward LOS, or resource-profile proxies to define
    the priority class.  Release and due windows remain priority dependent,
    because this is the intended service-policy mechanism.
    """
    policy = _normalize_priority_policy(priority_policy)
    shares = dict(priority_shares or {"high": 0.25, "medium": 0.45, "low": 0.30})
    rng = np.random.default_rng(instance["metadata"].get("seed", 0) + 101 if seed is None else seed)
    surgeries = instance["surgeries"].copy().sort_values("patient_id").reset_index(drop=True)

    # Preassign policy urgency independently of all downstream-burden fields.
    assigned_class: Dict[int, str] = {}
    if policy == "independent_urgency":
        counts = _priority_count_allocation(len(surgeries), shares)
        labels = np.array(
            ["high"] * counts["high"]
            + ["medium"] * counts["medium"]
            + ["low"] * counts["low"],
            dtype=object,
        )
        rng.shuffle(labels)
        assigned_class = {
            int(pid): str(label)
            for pid, label in zip(surgeries["patient_id"].astype(int).tolist(), labels.tolist())
        }

    rows = []
    for _, row in surgeries.iterrows():
        # Resource-profile fields are retained for surgeon/equipment eligibility.
        # They are not used to define priority under independent_urgency.
        cancer_flag, open_flag, approach, complexity = _case_category_from_specialty(row["specialty"], rng)
        if policy == "legacy_acuity_linked" and hepatobiliary_proxy and row["specialty"] in ["CABG+AVR", "CABG", "MVR", "GASTRO", "GEN"]:
            cancer_flag = int(rng.random() < 0.55)

        duration = float(row["duration_min"])
        icu_days = int(row["icu_treatment_days"]) if "icu_treatment_days" in row else int(row["icu_los_days"])
        ward_days = int(row["ward_los_days"])

        if policy == "legacy_acuity_linked":
            score = 1.0
            score += 2.5 * cancer_flag
            score += 1.2 * open_flag
            score += 0.5 * int(duration >= 240)
            score += 0.7 * int(icu_days >= 3)
            score += 0.2 * int(ward_days >= 5)
            score += float(rng.normal(0, 0.15))
            score = max(0.5, score)
            if score >= 4.2:
                pclass = "high"
            elif score >= 2.5:
                pclass = "medium"
            else:
                pclass = "low"
            priority_driver = "legacy_acuity_linked"
        else:
            pclass = assigned_class[int(row["patient_id"])]
            base = {"high": 3.0, "medium": 2.0, "low": 1.0}[pclass]
            # Small tie-breaker is independent of duration/LOS and never changes class.
            score = base + float(rng.uniform(-0.04, 0.04))
            priority_driver = "policy_randomized_urgency"

        if pclass == "high":
            due_day = int(rng.integers(1, 3))
            delay_penalty, postpone_penalty = 600, 2200
        elif pclass == "medium":
            due_day = int(rng.integers(2, 5))
            delay_penalty, postpone_penalty = 300, 1200
        else:
            due_day = int(rng.integers(4, 8))
            delay_penalty, postpone_penalty = 120, 500

        release_day = int(rng.integers(1, min(due_day, 3) + 1))
        if cancer_flag and open_flag:
            case_category = "cancer_open"
        elif cancer_flag:
            case_category = "cancer_mis"
        elif open_flag:
            case_category = "benign_open"
        else:
            case_category = "benign_mis"

        rows.append({
            "patient_id": int(row["patient_id"]),
            "surgery_id": row["surgery_id"],
            "specialty": row["specialty"],
            "priority_class": pclass,
            "priority_score": round(float(score), 3),
            "priority_generation_policy": policy,
            "priority_driver": priority_driver,
            "delay_penalty": int(delay_penalty),
            "postpone_penalty": int(postpone_penalty),
            "release_day": int(release_day),
            "due_day": int(due_day),
            "cancer_flag": int(cancer_flag),
            "open_flag": int(open_flag),
            "surgery_approach": approach,
            "case_category": case_category,
            "complexity_class": complexity,
        })

    return pd.DataFrame(rows)

def generate_surgeons(instance, seed=None, n_surgeons=None):
    """Generate synthetic surgeon/team table.

    For Case-I, one or two teams are created per cardiac surgery type.
    For M/R instances, teams are created by specialty.

    Output columns:
        surgeon_id, surgeon_name, primary_specialty, skill_level,
        daily_max_minutes, can_do_open, can_do_complex, notes
    """
    rng = np.random.default_rng(instance["metadata"].get("seed", 0) + 202 if seed is None else seed)
    type_stats = instance["type_stats"].copy()
    specialties = type_stats["specialty"].tolist()

    rows = []
    sid = 1
    for sp in specialties:
        count = int(type_stats.loc[type_stats["specialty"] == sp, "count"].iloc[0])
        # More workload -> more teams.
        teams = max(1, int(round(count / 12)))
        if n_surgeons is not None:
            teams = 1
        for k in range(teams):
            skill = "senior" if (k == 0 or rng.random() < 0.35) else "standard"
            daily_max = int(360 if skill == "senior" else 300)
            rows.append({
                "surgeon_id": f"S{sid:02d}",
                "surgeon_name": f"Team_{sp}_{k+1}",
                "primary_specialty": sp,
                "skill_level": skill,
                "daily_max_minutes": daily_max,
                "can_do_open": 1 if skill == "senior" else int(rng.random() < 0.65),
                "can_do_complex": 1 if skill == "senior" else int(rng.random() < 0.30),
                "notes": "synthetic surgeon/team generated for algorithm testing",
            })
            sid += 1

    if n_surgeons is not None and n_surgeons > 0:
        rows = rows[:n_surgeons]

    return pd.DataFrame(rows)


def generate_patient_surgeon_eligibility(instance, patient_priority=None, surgeons=None, seed=None):
    """Generate patient-surgeon eligibility matrix.

    Eligibility logic:
    - primary specialty must match;
    - complex/open cases prefer senior or open-capable teams;
    - at least one eligible surgeon is guaranteed for every patient.
    """
    rng = np.random.default_rng(instance["metadata"].get("seed", 0) + 303 if seed is None else seed)
    surgeries = instance["surgeries"]
    if patient_priority is None:
        patient_priority = generate_patient_priority(instance, seed=seed)
    if surgeons is None:
        surgeons = generate_surgeons(instance, seed=seed)

    rows = []
    for _, surg in surgeries.iterrows():
        pid = int(surg["patient_id"])
        sp = surg["specialty"]
        pr = patient_priority.loc[patient_priority["patient_id"] == pid].iloc[0]
        candidates = surgeons[surgeons["primary_specialty"] == sp].copy()

        if candidates.empty:
            candidates = surgeons.copy()

        eligible_ids = []
        for _, s in candidates.iterrows():
            eligible = 1
            if int(pr["open_flag"]) == 1 and int(s["can_do_open"]) != 1:
                eligible = 0
            if str(pr["complexity_class"]) in ["high", "medium_high"] and int(s["can_do_complex"]) != 1:
                # Allow standard teams for medium_high with small probability to avoid over-restriction.
                eligible = int(str(pr["complexity_class"]) == "medium_high" and rng.random() < 0.25)
            if eligible:
                eligible_ids.append(s["surgeon_id"])
                rows.append({
                    "patient_id": pid,
                    "surgery_id": surg["surgery_id"],
                    "specialty": sp,
                    "surgeon_id": s["surgeon_id"],
                    "eligible": 1,
                    "preferred": 0,
                })

        # Guarantee at least one eligible surgeon.
        if not eligible_ids:
            fallback = candidates.sort_values(["skill_level", "surgeon_id"], ascending=[True, True]).iloc[0]
            eligible_ids = [fallback["surgeon_id"]]
            rows.append({
                "patient_id": pid,
                "surgery_id": surg["surgery_id"],
                "specialty": sp,
                "surgeon_id": fallback["surgeon_id"],
                "eligible": 1,
                "preferred": 0,
            })

        preferred = eligible_ids[0]
        for r in rows:
            if r["patient_id"] == pid and r["surgeon_id"] == preferred:
                r["preferred"] = 1
                break

    return pd.DataFrame(rows)


def generate_surgeon_calendar(instance, surgeons=None, seed=None):
    """Generate synthetic surgeon available windows after ward rounds/clinics.

    Output table contains available windows only:
        surgeon_id, day_index, day, available_start_min, available_end_min, window_label

    Convention:
    - day starts at 8:00 AM = 0 min;
    - regular OR day is 0--480 min;
    - ward round blocks early morning for most teams;
    - outpatient clinic blocks one afternoon for some teams.
    """
    rng = np.random.default_rng(instance["metadata"].get("seed", 0) + 404 if seed is None else seed)
    if surgeons is None:
        surgeons = generate_surgeons(instance, seed=seed)

    rows = []
    for _, s in surgeons.iterrows():
        clinic_day = int(rng.integers(1, 6))
        for day_index, day in enumerate(SURGERY_DAYS, start=1):
            # Default full-day availability after ward round.
            windows = [(60, 480, "after_ward_round")]

            # Senior surgeons often have outpatient clinic one afternoon.
            if day_index == clinic_day:
                windows = [(60, 240, "pre_clinic")]
            # Some days have split availability due to meetings.
            elif rng.random() < 0.15:
                windows = [(60, 210, "morning_available"), (300, 480, "afternoon_available")]

            for start, end, label in windows:
                if end > start:
                    rows.append({
                        "surgeon_id": s["surgeon_id"],
                        "day_index": day_index,
                        "day": day,
                        "available_start_min": int(start),
                        "available_end_min": int(end),
                        "window_label": label,
                    })
    return pd.DataFrame(rows)


def generate_equipment(instance, seed=None):
    """Generate synthetic equipment capacity and patient equipment requirements."""
    surgeries = instance["surgeries"]
    # Keep equipment deliberately simple for early testing.
    equipment = pd.DataFrame([
        {"equipment_type": "standard_OR_set", "quantity": 99, "notes": "non-binding default set"},
        {"equipment_type": "minimally_invasive_tower", "quantity": 2, "notes": "shared laparoscopic/robotic imaging tower"},
        {"equipment_type": "cardiac_bypass_or_complex_set", "quantity": 2, "notes": "proxy for complex high-acuity surgical equipment"},
    ])

    req_rows = []
    for _, row in surgeries.iterrows():
        req_rows.append({
            "patient_id": int(row["patient_id"]),
            "surgery_id": row["surgery_id"],
            "equipment_type": "standard_OR_set",
            "quantity_required": 1,
        })
        sp = str(row["specialty"])
        dur = float(row["duration_min"])
        if sp in ["TAVR", "AVR", "ENT", "OPHTH", "UROLOGY"] or dur < 120:
            req_rows.append({
                "patient_id": int(row["patient_id"]),
                "surgery_id": row["surgery_id"],
                "equipment_type": "minimally_invasive_tower",
                "quantity_required": 1,
            })
        if sp in ["CABG+AVR", "CABG", "MVR", "CARDIAC", "VASCULAR", "NEURO"] or dur >= 240:
            req_rows.append({
                "patient_id": int(row["patient_id"]),
                "surgery_id": row["surgery_id"],
                "equipment_type": "cardiac_bypass_or_complex_set",
                "quantity_required": 1,
            })

    return equipment, pd.DataFrame(req_rows)


def generate_uncertainty_scenarios(instance, seed=None, n_scenarios=50, distribution=None):
    """Generate scenario samples for surgery duration, ICU treatment days, and ward LOS."""
    rng = np.random.default_rng(instance["metadata"].get("seed", 0) + 505 if seed is None else seed)
    surgeries = instance["surgeries"]
    distribution = distribution or instance["metadata"].get("distribution", "lognormal")

    rows = []
    for scen in range(1, int(n_scenarios) + 1):
        for _, row in surgeries.iterrows():
            duration = _sample_positive(
                rng,
                float(row["duration_mean"]),
                float(row["duration_sd"]),
                distribution=distribution,
                lower=5.0,
                integer=False,
            )
            icu = int(_sample_positive(
                rng,
                float(row["icu_los_mean"]),
                float(row["icu_los_sd"]),
                distribution=distribution,
                lower=0.0,
                integer=True,
            ))
            ward = int(_sample_positive(
                rng,
                float(row["ward_los_mean"]),
                float(row["ward_los_sd"]),
                distribution=distribution,
                lower=0.0,
                integer=True,
            ))
            rows.append({
                "scenario_id": scen,
                "patient_id": int(row["patient_id"]),
                "surgery_id": row["surgery_id"],
                "specialty": row["specialty"],
                "duration_min": round(duration, 2),
                "icu_treatment_days": int(icu),
                "ward_los_days": int(ward),
            })
    return pd.DataFrame(rows)


def extend_instance_with_synthetic_inputs(
    instance,
    seed=None,
    n_scenarios=50,
    priority_policy: str = "legacy_acuity_linked",
    priority_shares: Optional[Dict[str, float]] = None,
    hepatobiliary_proxy: bool = True,
):
    """Add synthetic priority, surgeon calendar, equipment, and scenarios."""
    extended = dict(instance)
    patient_priority = generate_patient_priority(
        instance, seed=seed, priority_policy=priority_policy,
        priority_shares=priority_shares, hepatobiliary_proxy=hepatobiliary_proxy,
    )
    surgeons = generate_surgeons(instance, seed=seed)
    eligibility = generate_patient_surgeon_eligibility(instance, patient_priority, surgeons, seed=seed)
    surgeon_calendar = generate_surgeon_calendar(instance, surgeons, seed=seed)
    equipment, patient_equipment = generate_equipment(instance, seed=seed)
    scenarios = generate_uncertainty_scenarios(instance, seed=seed, n_scenarios=n_scenarios)

    extended["patient_priority"] = patient_priority
    extended["surgeons"] = surgeons
    extended["patient_surgeon_eligibility"] = eligibility
    extended["surgeon_calendar"] = surgeon_calendar
    extended["equipment"] = equipment
    extended["patient_equipment"] = patient_equipment
    extended["scenarios"] = scenarios

    extended["metadata"] = dict(instance.get("metadata", {}))
    extended["metadata"]["synthetic_clinical_inputs"] = {
        "seed": seed,
        "n_scenarios": int(n_scenarios),
        "priority_policy": _normalize_priority_policy(priority_policy),
        "priority_shares": dict(priority_shares or {"high": 0.25, "medium": 0.45, "low": 0.30}),
        "hepatobiliary_proxy": bool(hepatobiliary_proxy),
        "tables": [
            "patient_priority",
            "surgeons",
            "patient_surgeon_eligibility",
            "surgeon_calendar",
            "equipment",
            "patient_equipment",
            "scenarios",
        ],
        "warning": "Synthetic clinical inputs are for algorithm testing only and should not be reported as real clinical data.",
    }
    return extended


def run_extended_sanity_checks(instance):
    """Additional checks for synthetic clinical inputs."""
    base = run_instance_sanity_checks(instance)
    results = base.to_dict("records")

    def add(check, status, detail):
        results.append({"check": check, "status": status, "detail": detail})

    surgeries = instance.get("surgeries")
    pr = instance.get("patient_priority")
    surgeons = instance.get("surgeons")
    elig = instance.get("patient_surgeon_eligibility")
    cal = instance.get("surgeon_calendar")
    scenarios = instance.get("scenarios")
    equipment = instance.get("equipment")
    peq = instance.get("patient_equipment")

    I = len(surgeries) if isinstance(surgeries, pd.DataFrame) else 0

    if isinstance(pr, pd.DataFrame):
        add("patient_priority count", "PASS" if len(pr) == I else "FAIL", f"expected {I}, observed {len(pr)}")
        bad_due = pr[pr["due_day"] < pr["release_day"]]
        add("release/due consistency", "PASS" if bad_due.empty else "FAIL", f"{len(bad_due)} patients have due_day < release_day")
        # Publication priority policy must not mechanically depend on realised
        # duration or downstream LOS.  This is an audit, not a significance test.
        if "priority_generation_policy" in pr.columns:
            policy = str(pr["priority_generation_policy"].iloc[0]) if len(pr) else ""
            add("priority generation policy", "PASS" if policy else "WARN", policy or "missing policy label")
            if policy == "independent_urgency" and isinstance(surgeries, pd.DataFrame):
                q = pr[["patient_id", "priority_class"]].merge(
                    surgeries[["patient_id", "duration_min", "icu_los_days", "ward_los_days"]],
                    on="patient_id", how="left",
                )
                codes = q["priority_class"].map({"low": 1.0, "medium": 2.0, "high": 3.0})
                corrs = {}
                for c in ["duration_min", "icu_los_days", "ward_los_days"]:
                    x = pd.to_numeric(q[c], errors="coerce")
                    corrs[c] = float(pd.Series(codes).corr(x, method="spearman")) if x.notna().sum() >= 3 else 0.0
                max_abs = max(abs(v) for v in corrs.values()) if corrs else 0.0
                add("priority/downstream independence audit", "PASS" if max_abs < 0.25 else "WARN",
                    "; ".join(f"rho(priority,{k})={v:.3f}" for k, v in corrs.items()))
    else:
        add("patient_priority table", "FAIL", "missing")

    if isinstance(surgeons, pd.DataFrame) and isinstance(elig, pd.DataFrame):
        missing_patients = set(surgeries["patient_id"].astype(int)) - set(elig["patient_id"].astype(int))
        add("surgeon eligibility coverage", "PASS" if not missing_patients else "FAIL", f"patients without eligible surgeon: {sorted(list(missing_patients))[:10]}")
        unknown_surgeons = set(elig["surgeon_id"]) - set(surgeons["surgeon_id"])
        add("surgeon ids valid", "PASS" if not unknown_surgeons else "FAIL", f"unknown surgeon ids: {sorted(list(unknown_surgeons))[:10]}")
    else:
        add("surgeons/eligibility tables", "FAIL", "missing surgeons or eligibility")

    if isinstance(cal, pd.DataFrame) and isinstance(surgeons, pd.DataFrame):
        no_calendar = set(surgeons["surgeon_id"]) - set(cal["surgeon_id"])
        add("surgeon calendar coverage", "PASS" if not no_calendar else "FAIL", f"surgeons without calendar: {sorted(list(no_calendar))[:10]}")
        bad_window = cal[cal["available_end_min"] <= cal["available_start_min"]]
        add("surgeon calendar windows", "PASS" if bad_window.empty else "FAIL", f"{len(bad_window)} invalid windows")
    else:
        add("surgeon_calendar table", "FAIL", "missing")

    if isinstance(scenarios, pd.DataFrame):
        n_scen = scenarios["scenario_id"].nunique()
        expected = n_scen * I
        add("scenario matrix size", "PASS" if len(scenarios) == expected else "FAIL", f"{n_scen} scenarios × {I} patients = {expected}, observed {len(scenarios)}")
        nonneg = ((scenarios["duration_min"] > 0) & (scenarios["icu_treatment_days"] >= 0) & (scenarios["ward_los_days"] >= 0)).all()
        add("scenario nonnegative values", "PASS" if nonneg else "FAIL", "duration > 0 and LOS >= 0 for all scenarios")
    else:
        add("scenarios table", "FAIL", "missing")

    if isinstance(equipment, pd.DataFrame) and isinstance(peq, pd.DataFrame):
        missing_equipment = set(peq["equipment_type"]) - set(equipment["equipment_type"])
        add("equipment ids valid", "PASS" if not missing_equipment else "FAIL", f"unknown equipment types: {missing_equipment}")
        missing_req = set(surgeries["patient_id"].astype(int)) - set(peq["patient_id"].astype(int))
        add("patient equipment coverage", "PASS" if not missing_req else "FAIL", f"patients without equipment row: {sorted(list(missing_req))[:10]}")
    else:
        add("equipment tables", "FAIL", "missing equipment or patient_equipment")

    return pd.DataFrame(results)

if __name__ == "__main__":
    cfg = BenchmarkConfig(dataset="case", I=70, seed=7, bed_setting=(10, 7), cost_structure=2)
    inst = generate_benchmark(cfg)
    ext_inst = extend_instance_with_blocking_states(
        inst,
        seed=77,
        initial_icu_occupancy_rate=0.80,
        initial_ward_occupancy_rate=0.90,
        ready_fraction_among_icu=0.50,
        schedule_priority="patient_id",
        transfer_priority="current_first",
    )
    ext_inst = extend_instance_with_synthetic_inputs(ext_inst, seed=88, n_scenarios=50)
    save_instance(ext_inst, "case_70_full_synthetic_ext")
    report = run_extended_sanity_checks(ext_inst)
    print_sanity_report(report)
    print("\nInstance generated successfully.")
    print("Output folder: case_70_full_synthetic_ext")

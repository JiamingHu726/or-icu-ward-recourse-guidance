#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_mannino_patient_pool.py

Generate a Mannino-style candidate patient pool from the cleaned duration
statistics produced by prepare_mannino_stats.py.

This script creates an intermediate CSV. It does not replace your full
OR--ICU--ward instance generator by itself. Use the CSV to override the patient
list/duration fields in your existing benchmark generator.

Recommended use:
    python generate_mannino_patient_pool.py ^
      --stats-json mannino_prepared/mannino_duration_stats.json ^
      --case-pool mannino_prepared/mannino_case_pool.csv ^
      --n 70 ^
      --seed 7 ^
      --output mannino_prepared/mannino_pool_n70_seed7.csv ^
      --duration-mode empirical

If you do not provide --case-pool, duration-mode will fall back to normal
sampling from team-level mean/sd.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional


def allocate_counts(n: int, teams: List[Dict[str, Any]]) -> Dict[str, int]:
    raw = [(t["team"], n * float(t["percentage"]) / 100.0) for t in teams]
    base = {team: int(math.floor(x)) for team, x in raw}
    remainders = sorted([(team, x - math.floor(x)) for team, x in raw], key=lambda z: -z[1])
    missing = n - sum(base.values())
    for team, _ in remainders[:missing]:
        base[team] += 1
    return base


def read_case_pool(path: Optional[Path]) -> Dict[str, List[float]]:
    if path is None:
        return {}
    by_team: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            team = row["team"]
            dur = float(row["duration_min"])
            by_team.setdefault(team, []).append(dur)
    return by_team


def sample_nonnegative_normal(rng: random.Random, mean: float, sd: float, lower: float = 1.0) -> float:
    if sd <= 0:
        return max(lower, mean)
    for _ in range(100):
        x = rng.gauss(mean, sd)
        if x >= lower:
            return x
    return max(lower, mean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--case-pool", default=None, help="Optional mannino_case_pool.csv for empirical bootstrap durations")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-mode", choices=["empirical", "normal", "mean"], default="empirical")
    parser.add_argument("--priority-share", type=float, default=0.35, help="Fraction assigned high priority for intermediate pool")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.stats_json, encoding="utf-8") as f:
        stats = json.load(f)

    teams = []
    for team, row in stats["teams"].items():
        item = dict(row)
        item["team"] = team
        teams.append(item)

    counts = allocate_counts(args.n, teams)
    empirical = read_case_pool(Path(args.case_pool) if args.case_pool else None)

    rows = []
    pid = 1
    for team_row in teams:
        team = team_row["team"]
        for _ in range(counts[team]):
            if args.duration_mode == "mean":
                dur = float(team_row["duration_mean"])
            elif args.duration_mode == "empirical" and empirical.get(team):
                dur = rng.choice(empirical[team])
            else:
                dur = sample_nonnegative_normal(
                    rng,
                    float(team_row["duration_mean"]),
                    float(team_row["duration_sd"]),
                    lower=1.0,
                )

            priority = "high" if rng.random() < args.priority_share else "normal"

            rows.append({
                "patient_id": f"P{pid:04d}",
                "team": team,
                "specialty": team,
                "duration_min": round(dur, 2),
                "priority_class": priority,
                # These fields are intentionally left blank.
                # Mannino data do not contain ICU/ward LOS. Fill them using your
                # current synthetic downstream generator or a separate LOS policy.
                "needs_icu": "",
                "icu_los_days": "",
                "ward_los_days": "",
            })
            pid += 1

    rng.shuffle(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "team",
                "specialty",
                "duration_min",
                "priority_class",
                "needs_icu",
                "icu_los_days",
                "ward_los_days",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} patients to {out}")


if __name__ == "__main__":
    main()

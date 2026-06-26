#!/usr/bin/env python3
from __future__ import annotations

"""
batch_generate_publication_scenarios.py

Generate publication-level synthetic OR-ICU-Ward instances under nominal and
ward-transfer stress scenarios.

The stress scenarios are designed to test the value of explicit ICU-to-ward
transfer recourse. They modify ward availability after the same base instance is
generated, so differences across scenarios are caused by downstream capacity
stress rather than a different patient pool.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from publication_experiment_config import (
    PUBLICATION_SIZES,
    PUBLICATION_SEEDS,
    PUBLICATION_SCENARIOS,
    PUBLICATION_BED_SETTINGS,
    PUBLICATION_INSTANCE_ROOT,
)

from shehadeh_style_benchmark_generator_v3 import (
    BenchmarkConfig,
    generate_benchmark,
    extend_instance_with_blocking_states,
    extend_instance_with_synthetic_inputs,
    save_instance,
    run_extended_sanity_checks,
)


def _copy_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(inst)


def _scale_ward_capacity(inst: Dict[str, Any], multiplier: float, days=None) -> Dict[str, Any]:
    out = _copy_instance(inst)
    caps = out["capacities"].copy()
    mask = pd.Series(True, index=caps.index)
    if days is not None:
        days = {int(d) for d in days}
        mask = caps["day_index"].astype(int).isin(days)
    old = caps.loc[mask, "ward_capacity"].astype(float)
    caps.loc[mask, "ward_capacity"] = np.maximum(1, np.floor(old * float(multiplier))).astype(int)
    out["capacities"] = caps
    return out


def apply_scenario(base_inst: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Return a scenario-specific instance.

    nominal:
        No change.
    ward_pressure:
        Reduces ward capacity on all days by 10%.
    transfer_bottleneck:
        Reduces ward capacity on midweek days where ICU-ready transfers often
        accumulate after early-week surgeries.
    """
    if scenario == "nominal":
        inst = _copy_instance(base_inst)
        note = "No downstream stress adjustment."
    elif scenario == "ward_pressure":
        inst = _scale_ward_capacity(base_inst, multiplier=0.90, days=None)
        note = "Ward capacity reduced by 10% on all planning days."
    elif scenario == "transfer_bottleneck":
        inst = _scale_ward_capacity(base_inst, multiplier=0.75, days=[3, 4, 5])
        note = "Ward capacity reduced by 25% on days 3--5 to create a transfer bottleneck."
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    meta = dict(inst.get("metadata", {}))
    meta["publication_scenario"] = scenario
    meta["publication_scenario_note"] = note
    inst["metadata"] = meta
    return inst


def generate_base_instance(n: int, seed: int) -> Dict[str, Any]:
    bed_setting = PUBLICATION_BED_SETTINGS.get(
        int(n),
        (max(6, round(0.12 * int(n))), max(5, round(0.08 * int(n)))),
    )
    cfg = BenchmarkConfig(
        dataset="case",
        I=int(n),
        seed=int(seed),
        bed_setting=bed_setting,
        cost_structure=2,
        distribution="lognormal",
    )
    inst = generate_benchmark(cfg)
    inst = extend_instance_with_blocking_states(
        inst,
        seed=int(seed) + 70,
        initial_icu_occupancy_rate=0.80,
        initial_ward_occupancy_rate=0.90,
        ready_fraction_among_icu=0.50,
        schedule_priority="patient_id",
        transfer_priority="current_first",
    )
    inst = extend_instance_with_synthetic_inputs(inst, seed=int(seed) + 80, n_scenarios=50)
    meta = dict(inst.get("metadata", {}))
    meta.update({"n": int(n), "seed": int(seed), "bed_setting": bed_setting})
    inst["metadata"] = meta
    return inst


def save_publication_instance(inst: Dict[str, Any], output_dir: Path, scenario: str, n: int, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_instance(inst, output_dir)
    report = run_extended_sanity_checks(inst)
    (output_dir / "sanity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    caps = inst["capacities"].copy()
    meta = {
        "scenario": scenario,
        "n": int(n),
        "seed": int(seed),
        "output_dir": str(output_dir),
        "icu_capacity_by_day": dict(zip(caps["day_index"].astype(int), caps["icu_capacity"].astype(int))),
        "ward_capacity_by_day": dict(zip(caps["day_index"].astype(int), caps["ward_capacity"].astype(int))),
        "scenario_note": inst.get("metadata", {}).get("publication_scenario_note", ""),
    }
    (output_dir / "publication_instance_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate publication-level OR-ICU-Ward scenarios.")
    parser.add_argument("--sizes", nargs="+", type=int, default=PUBLICATION_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=PUBLICATION_SEEDS)
    parser.add_argument("--scenarios", nargs="+", default=PUBLICATION_SCENARIOS)
    parser.add_argument("--output-root", default=str(PUBLICATION_INSTANCE_ROOT))
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for n in args.sizes:
        for seed in args.seeds:
            print(f"\n=== Generating base instance n={n}, seed={seed} ===")
            base = generate_base_instance(n, seed)
            for scenario in args.scenarios:
                inst = apply_scenario(base, scenario)
                out = root / scenario / f"case_{n}_seed{seed}"
                print(f"  -> {scenario}: {out}")
                rows.append(save_publication_instance(inst, out, scenario, n, seed))

    index = pd.DataFrame(rows)
    index.to_csv(root / "publication_instance_index.csv", index=False)
    print(f"\nGenerated {len(rows)} scenario instances under {root}")
    print(f"Index: {root / 'publication_instance_index.csv'}")


if __name__ == "__main__":
    main()

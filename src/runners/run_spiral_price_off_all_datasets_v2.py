#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_spiral_price_off_all_datasets_v2.py

Fixed runner for guidance_mode="off" / price-off spiral LNS.

Why v2?
-------
The previous runner failed for two reasons:
1) Synthetic anchor root was wrong. Your actual synthetic result root is:
       publication_batch_results
2) GermanOR/Mannino commands reached the script but exited with return code 2,
   which is almost certainly argparse rejecting unsupported CLI options such as
       --max-or-overtime
       --include-pool
   depending on the local version of pr_glns_spiral_or_icu_ward_guidance_mode.py.

This v2 avoids CLI argument mismatch by importing
pr_glns_spiral_or_icu_ward_guidance_mode.py and calling run_spiral_pr_glns()
directly. It inspects the function signature and passes only supported keyword
arguments.

Datasets and roots used by default
----------------------------------
Synthetic:
    instances: publication_batch_instances
    anchors:   publication_batch_results

GermanOR:
    n=50/70:
        instances: german_publication_batch_instances
        anchors:   german_publication_results_full_methods
    n=100/150:
        instances: german_publication_batch_instances_100_150
        anchors:   german_publication_results_full_methods_100_150

Mannino:
    n=50/70:
        instances: mannino_publication_batch_instances
        anchors:   mannino_publication_results_full_methods
    n=100/150:
        instances: mannino_publication_batch_instances_100_150
        anchors:   mannino_publication_results_full_methods_100_150
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


DEFAULT_SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
DEFAULT_SIZES = [50, 70, 100, 150]
DEFAULT_SCENARIOS = ["nominal", "transfer_bottleneck"]


def case_name(n: int, seed: int) -> str:
    return f"case_{int(n)}_seed{int(seed)}"


def root_pair(dataset: str, n: int, args) -> Tuple[Path, Path]:
    """Return (instance_root, anchor_result_root)."""
    if dataset == "Synthetic":
        return Path(args.synthetic_instance_root), Path(args.synthetic_anchor_root)

    if dataset == "GermanOR":
        if int(n) >= 100:
            return Path(args.german_instance_root_100_150), Path(args.german_anchor_root_100_150)
        return Path(args.german_instance_root_50_70), Path(args.german_anchor_root_50_70)

    if dataset == "Mannino":
        if int(n) >= 100:
            return Path(args.mannino_instance_root_100_150), Path(args.mannino_anchor_root_100_150)
        return Path(args.mannino_instance_root_50_70), Path(args.mannino_anchor_root_50_70)

    raise ValueError(dataset)


def resolve_case_dir(root: Path, scenario: str, n: int, seed: int) -> Optional[Path]:
    cname = case_name(n, seed)

    direct = root / scenario / cname
    if direct.exists():
        return direct

    direct2 = root / cname
    if direct2.exists():
        return direct2

    if not root.exists():
        return None

    hits = list(root.rglob(cname))
    if not hits:
        return None

    scen_hits = [h for h in hits if scenario in h.parts]
    if scen_hits:
        scen_hits.sort(key=lambda p: len(str(p)))
        return scen_hits[0]

    hits.sort(key=lambda p: len(str(p)))
    return hits[0]


def read_csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return {c.strip() for c in next(csv.reader(f))}
    except Exception:
        return set()


def looks_like_schedule_csv(path: Path) -> bool:
    cols = read_csv_header(path)
    if not {"patient_id", "day_index"}.issubset(cols):
        return False
    hints = {"block_id", "or_id", "planned_start_min", "start_min", "duration_min"}
    return len(cols & hints) >= 2


def looks_like_pressure_csv(path: Path) -> bool:
    cols = read_csv_header(path)
    if "day_index" not in cols:
        return False
    if "stage3_pressure_score" in cols:
        return True
    needed = {"icu_ready_blocked_stage3", "ward_excess_stage3", "icu_excess_stage3"}
    return needed.issubset(cols)


def find_first_schedule_under(root: Path) -> Optional[Path]:
    if not root.exists():
        return None

    preferred_names = [
        "schedule.csv",
        "normalized_schedule.csv",
        "final_schedule.csv",
        "best_schedule.csv",
        "stage2_schedule.csv",
        "repaired_schedule.csv",
        "pr_glns_schedule.csv",
    ]

    # Prefer shallow known names.
    for name in preferred_names:
        for p in root.rglob(name):
            rel = str(p.relative_to(root)).lower()
            if "candidate" in rel or "trace" in rel:
                continue
            if looks_like_schedule_csv(p):
                return p

    candidates = []
    for p in root.rglob("*.csv"):
        rel = str(p.relative_to(root)).lower()
        if "candidate" in rel or "trace" in rel or "daily" in rel or "pressure" in rel:
            continue
        if looks_like_schedule_csv(p):
            candidates.append(p)
    if candidates:
        candidates.sort(key=lambda p: (len(str(p)), str(p)))
        return candidates[0]
    return None


def find_anchor_schedules(case_result_dir: Path) -> List[Tuple[str, Path]]:
    """
    Return anchors for spiral LNS.
    For fairness, avoid using previous 06/07 LNS outputs as anchors.
    """
    stage_map = [
        ("ba_hla", "01_ba_hla_v41"),
        ("execution_repair", "02_stage2_v3_volume_first"),
        ("feedback_repair", "04_hp_forced_ot_feedback"),
    ]

    anchors: List[Tuple[str, Path]] = []
    for label, folder in stage_map:
        p = find_first_schedule_under(case_result_dir / folder)
        if p is not None:
            anchors.append((label, p))

    if not anchors:
        # Last fallback: any schedule-looking file under 01/02/04 only.
        for p in sorted(case_result_dir.rglob("*.csv")):
            rel = str(p.relative_to(case_result_dir)).replace("\\", "/")
            if rel.startswith(("01_", "02_", "04_")) and looks_like_schedule_csv(p):
                anchors.append((p.parent.name, p))
                if len(anchors) >= 3:
                    break

    # Deduplicate paths and labels.
    seen_paths, used_labels, out = set(), set(), []
    for label, p in anchors:
        key = str(p.resolve()).lower()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        base = label
        k = 2
        while label in used_labels:
            label = f"{base}_{k}"
            k += 1
        used_labels.add(label)
        out.append((label, p))
    return out


def find_pressure_day_files(case_result_dir: Path) -> List[Path]:
    roots = [
        case_result_dir / "03_stage3_on_stage2_v3",
        case_result_dir / "05_stage3_on_hp_forced_ot_feedback",
        case_result_dir / "04_hp_forced_ot_feedback",
    ]
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.csv")):
            if looks_like_pressure_csv(p):
                out.append(p)

    if not out:
        for p in sorted(case_result_dir.rglob("*.csv")):
            rel = str(p.relative_to(case_result_dir)).replace("\\", "/").lower()
            if rel.startswith(("06_", "07_", "spiral_", "graph_")):
                continue
            if looks_like_pressure_csv(p):
                out.append(p)

    seen, unique = set(), []
    for p in out:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def import_spiral_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Spiral script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("spiral_guidance_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def output_dir_for(args, dataset: str, scenario: str, n: int, seed: int) -> Path:
    return Path(args.output_root) / dataset / scenario / case_name(n, seed) / "spiral_off"


def already_done(out_dir: Path) -> bool:
    candidates = [
        out_dir / "spiral_pr_glns_metadata.json",
        out_dir / "summary.csv",
        out_dir / "spiral_trace.csv",
    ]
    return any(p.exists() and p.stat().st_size > 0 for p in candidates)


def load_anchors(anchor_paths: List[Tuple[str, Path]]) -> Dict[str, pd.DataFrame]:
    return {label: pd.read_csv(path) for label, path in anchor_paths}


def call_spiral(mod, args, instance_dir: Path, output_dir: Path,
                anchors: Dict[str, pd.DataFrame], pressure_files: List[Path]) -> Dict[str, Any]:
    if not hasattr(mod, "run_spiral_pr_glns"):
        raise AttributeError("Module has no run_spiral_pr_glns()")
    if not hasattr(mod, "load_instance"):
        raise AttributeError("Module has no load_instance()")

    instance = mod.load_instance(instance_dir)
    fn = mod.run_spiral_pr_glns
    sig = inspect.signature(fn)

    kwargs: Dict[str, Any] = {
        "instance": instance,
        "anchor_schedules": anchors,
        "output_dir": output_dir,
    }

    optional = {
        "pressure_day_paths": [str(p) for p in pressure_files],
        "include_pool": args.include_pool,
        "target_volume": None,
        "high_target": None,
        "cycles": args.cycles,
        "proposals_per_cycle": args.proposals_per_cycle,
        "exact_every": args.exact_every,
        "exact_top_k": args.exact_top_k,
        "seed": args.lns_seed,
        "max_or_overtime": args.max_or_overtime,
        "time_limit_seconds": args.time_limit_seconds,
        "wallclock_limit_s": args.wallclock_limit_s if args.wallclock_limit_s > 0 else None,
        "guidance_mode": "off",
        "verbose": False,
    }

    for k, v in optional.items():
        if k in sig.parameters:
            kwargs[k] = v

    if "guidance_mode" not in sig.parameters:
        raise RuntimeError(
            "run_spiral_pr_glns() does not accept guidance_mode. "
            "You are not using the guidance_mode patched file."
        )

    return fn(**kwargs)


def write_log(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_case(mod, args, dataset: str, scenario: str, n: int, seed: int) -> Dict[str, Any]:
    instance_root, anchor_root = root_pair(dataset, n, args)
    instance_dir = resolve_case_dir(instance_root, scenario, n, seed)
    anchor_dir = resolve_case_dir(anchor_root, scenario, n, seed)
    out_dir = output_dir_for(args, dataset, scenario, n, seed)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "scenario": scenario,
        "n": int(n),
        "seed": int(seed),
        "instance_root": str(instance_root),
        "anchor_root": str(anchor_root),
        "instance_dir": str(instance_dir) if instance_dir else "",
        "anchor_result_dir": str(anchor_dir) if anchor_dir else "",
        "output_dir": str(out_dir),
        "num_anchors": 0,
        "num_pressure_files": 0,
        "status": "",
        "error": "",
    }

    if instance_dir is None:
        row["status"] = "missing_instance_dir"
        return row
    if anchor_dir is None:
        row["status"] = "missing_anchor_result_dir"
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and already_done(out_dir):
        row["status"] = "skipped_existing"
        return row

    anchor_paths = find_anchor_schedules(anchor_dir)
    pressure_files = find_pressure_day_files(anchor_dir)
    row["num_anchors"] = len(anchor_paths)
    row["num_pressure_files"] = len(pressure_files)
    row["anchor_paths"] = " | ".join(f"{label}:{path}" for label, path in anchor_paths)
    row["pressure_paths"] = " | ".join(str(p) for p in pressure_files)

    if not anchor_paths:
        row["status"] = "missing_anchor_schedules"
        return row

    print("\n" + "=" * 100)
    print(f"[{dataset}] {scenario}, n={n}, seed={seed}")
    print(f"instance: {instance_dir}")
    print(f"anchor:   {anchor_dir}")
    print(f"output:   {out_dir}")
    print(f"anchors={len(anchor_paths)}, pressure_files={len(pressure_files)}")
    for label, p in anchor_paths:
        print(f"  anchor {label}: {p}")
    for p in pressure_files:
        print(f"  pressure: {p}")

    if args.dry_run:
        row["status"] = "dry_run"
        return row

    try:
        anchors = load_anchors(anchor_paths)
        meta = call_spiral(mod, args, instance_dir, out_dir, anchors, pressure_files)
        row["status"] = "success"
        row["runtime_sec"] = meta.get("runtime_sec", "")
        row["best_source"] = meta.get("best_source", "")
        row["best_exact_score"] = meta.get("best_exact_score", "")
        row["best_fast_objective"] = meta.get("best_fast_objective", "")
    except Exception as e:
        row["status"] = "failed"
        row["error"] = repr(e)
        err_path = out_dir / "price_off_error.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAILED: {e}")
        print(f"Traceback saved to: {err_path}")

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Synthetic", "GermanOR", "Mannino"],
                    choices=["Synthetic", "GermanOR", "Mannino"])
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)

    ap.add_argument("--spiral-script", default="pr_glns_spiral_or_icu_ward_guidance_mode.py")
    ap.add_argument("--output-root", default="spiral_price_off_all_results_v2")

    ap.add_argument("--synthetic-instance-root", default="publication_batch_instances")
    ap.add_argument("--synthetic-anchor-root", default="publication_batch_results")

    ap.add_argument("--german-instance-root-50-70", default="german_publication_batch_instances")
    ap.add_argument("--german-anchor-root-50-70", default="german_publication_results_full_methods")
    ap.add_argument("--german-instance-root-100-150", default="german_publication_batch_instances_100_150")
    ap.add_argument("--german-anchor-root-100-150", default="german_publication_results_full_methods_100_150")

    ap.add_argument("--mannino-instance-root-50-70", default="mannino_publication_batch_instances")
    ap.add_argument("--mannino-anchor-root-50-70", default="mannino_publication_results_full_methods")
    ap.add_argument("--mannino-instance-root-100-150", default="mannino_publication_batch_instances_100_150")
    ap.add_argument("--mannino-anchor-root-100-150", default="mannino_publication_results_full_methods_100_150")

    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--proposals-per-cycle", type=int, default=80)
    ap.add_argument("--exact-every", type=int, default=25)
    ap.add_argument("--exact-top-k", type=int, default=3)
    ap.add_argument("--lns-seed", type=int, default=202706)
    ap.add_argument("--max-or-overtime", type=int, default=180)
    ap.add_argument("--include-pool", default="all", choices=["all", "high_only", "none"])
    ap.add_argument("--time-limit-seconds", type=float, default=None)
    ap.add_argument("--wallclock-limit-s", type=float, default=0.0)

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    # Clear contamination from previous diagnostics.
    os.environ.pop("ORSCHE_LNS_ABS_BUDGET", None)
    os.environ.pop("ORSCHE_LNS_DESTROY_POLICY", None)
    if args.wallclock_limit_s > 0:
        os.environ["ORSCHE_LNS_WALLCLOCK_LIMIT_S"] = str(args.wallclock_limit_s)
    else:
        os.environ.pop("ORSCHE_LNS_WALLCLOCK_LIMIT_S", None)

    mod = import_spiral_module(Path(args.spiral_script))

    rows: List[Dict[str, Any]] = []
    log_path = Path(args.output_root) / "price_off_run_log_v2.csv"

    total = len(args.datasets) * len(args.sizes) * len(args.scenarios) * len(args.seeds)
    idx = 0
    for dataset in args.datasets:
        for n in args.sizes:
            for scenario in args.scenarios:
                for seed in args.seeds:
                    idx += 1
                    print(f"\n### Progress {idx}/{total}")
                    row = run_case(mod, args, dataset, scenario, n, seed)
                    rows.append(row)
                    write_log(log_path, rows)

                    if row["status"] not in {"success", "dry_run", "skipped_existing"}:
                        print(f"WARNING: {row['status']} for {dataset} {scenario} n={n} seed={seed}")
                        if not args.continue_on_error:
                            raise SystemExit(1)

    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()

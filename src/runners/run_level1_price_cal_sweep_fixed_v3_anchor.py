#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_level1_price_cal_sweep_fixed_v3_anchor.py

Fixed Level-1 runner with anchor seeding.

Root cause fixed here:
    Your guided LNS batch fails with:
        FileNotFoundError: No anchor schedules found.

Why:
    You ran guided-only into a new result root. The guided batch expects anchor
    schedules from earlier methods inside the same case result folder. The new
    price_cal result root is empty, so no anchors exist.

Fix:
    Before each guided price_cal run, copy existing anchor method folders from
    the old full-method result root into the new price_cal result root, excluding
    the old guided-LNS output folder. Then run the guided arm.

Run from your repo root.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


GUIDED_EXCLUDE_TOKENS = [
    "06_downstream_aggressive_spiral_pr_glns",
    "downstream_aggressive_spiral_pr_glns",
    "spiral_pr_glns",
    "pr_glns",
    "graph_guided",
    "price_cal",
    "spiral_off",
]


def get_slope(level0_csv: Path, dataset: str, n: int, mode: str, manual: float) -> float:
    if manual >= 0:
        return float(manual)

    df = pd.read_csv(level0_csv)

    if mode == "dataset_n":
        sub = df[
            (df.get("grouping") == "dataset+n")
            & (df.get("dataset") == dataset)
            & (pd.to_numeric(df.get("n"), errors="coerce") == int(n))
        ]
        if len(sub):
            v = float(sub.iloc[0]["slope"])
            if np.isfinite(v):
                return v

    sub = df[(df.get("grouping") == "dataset") & (df.get("dataset") == dataset)]
    if len(sub):
        v = float(sub.iloc[0]["slope"])
        if np.isfinite(v):
            return v

    sub = df[df.get("grouping") == "overall"]
    if len(sub):
        v = float(sub.iloc[0]["slope"])
        if np.isfinite(v):
            return v

    raise RuntimeError(f"No usable slope for {dataset}, n={n}. Use --manual-base-weight.")


def instance_root_for(dataset: str, n: int, args) -> Path:
    if dataset == "GermanOR":
        return Path(args.german_instance_root_100_150 if int(n) >= 100 else args.german_instance_root_50_70)
    if dataset == "Mannino":
        return Path(args.mannino_instance_root_100_150 if int(n) >= 100 else args.mannino_instance_root_50_70)
    if dataset == "Synthetic":
        return Path(args.synthetic_instance_root)
    raise ValueError(dataset)


def anchor_root_for(dataset: str, n: int, args) -> Path:
    if dataset == "GermanOR":
        return Path(args.german_anchor_root_100_150 if int(n) >= 100 else args.german_anchor_root_50_70)
    if dataset == "Mannino":
        return Path(args.mannino_anchor_root_100_150 if int(n) >= 100 else args.mannino_anchor_root_50_70)
    if dataset == "Synthetic":
        return Path(args.synthetic_anchor_root)
    raise ValueError(dataset)


def runner_for(dataset: str, args) -> Path:
    if dataset == "GermanOR":
        return Path(args.german_runner)
    if dataset == "Mannino":
        return Path(args.mannino_runner)
    if dataset == "Synthetic":
        return Path(args.synthetic_runner)
    raise ValueError(dataset)


def case_name(n: int, seed: int) -> str:
    return f"case_{int(n)}_seed{int(seed)}"


def find_case_dir(root: Path, dataset: str, scenario: str, n: int, seed: int) -> Path | None:
    cname = case_name(n, seed)
    candidates = [
        root / scenario / cname,
        root / dataset / scenario / cname,
        root / f"n{n}" / scenario / cname,
        root / dataset / f"n{n}" / scenario / cname,
        root / cname,
    ]
    for p in candidates:
        if p.exists():
            return p

    if not root.exists():
        return None

    hits = list(root.rglob(cname))
    if not hits:
        return None

    scored = []
    for h in hits:
        parts = [str(x).lower() for x in h.parts]
        score = 0
        if str(scenario).lower() in parts:
            score += 5
        if str(dataset).lower() in parts:
            score += 2
        if f"n{n}".lower() in parts:
            score += 1
        scored.append((score, len(str(h)), h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def should_exclude_anchor_child(path: Path) -> bool:
    s = str(path).lower()
    return any(tok.lower() in s for tok in GUIDED_EXCLUDE_TOKENS)


def copy_anchor_case(source_case: Path, target_case: Path, clean_target: bool = False) -> dict:
    if not source_case.exists():
        raise FileNotFoundError(source_case)

    if clean_target and target_case.exists():
        shutil.rmtree(target_case)

    target_case.mkdir(parents=True, exist_ok=True)

    copied_dirs = []
    copied_files = []
    skipped = []

    # Copy top-level method folders and root files except old guided-LNS outputs.
    for child in source_case.iterdir():
        if should_exclude_anchor_child(child):
            skipped.append(str(child.name))
            continue

        dst = target_case / child.name

        if child.is_dir():
            shutil.copytree(child, dst, dirs_exist_ok=True)
            copied_dirs.append(str(child.name))
        elif child.is_file():
            # Root metadata is useful; large trace/log files are not necessary but harmless.
            if child.suffix.lower() in [".csv", ".json", ".txt"]:
                shutil.copy2(child, dst)
                copied_files.append(str(child.name))

    return {
        "source_case": str(source_case),
        "target_case": str(target_case),
        "copied_dirs": "|".join(copied_dirs),
        "copied_files": "|".join(copied_files),
        "skipped": "|".join(skipped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=r"E:\anaconda3\python.exe")
    ap.add_argument("--level0-csv", default=r"recourse_recalibration_level0\level0_pressure_regression.csv")
    ap.add_argument("--output-root", default="recourse_level1_price_cal_results_v3")
    ap.add_argument("--base-weight-mode", choices=["dataset", "dataset_n"], default="dataset")
    ap.add_argument("--manual-base-weight", type=float, default=-1.0)
    ap.add_argument("--multipliers", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2])
    ap.add_argument("--datasets", nargs="+", default=["GermanOR", "Mannino"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[50, 70, 100, 150])
    ap.add_argument("--scenarios", nargs="+", default=["nominal", "transfer_bottleneck"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7,11,19,23,29,31,37,41,43,47])

    ap.add_argument("--german-runner", default="run_german_full_methods.py")
    ap.add_argument("--mannino-runner", default="run_mannino_full_methods.py")
    ap.add_argument("--synthetic-runner", default="run_publication_batch_full_methods.py")

    ap.add_argument("--german-instance-root-50-70", default="german_publication_batch_instances")
    ap.add_argument("--german-instance-root-100-150", default="german_publication_batch_instances_100_150")
    ap.add_argument("--mannino-instance-root-50-70", default="mannino_publication_batch_instances")
    ap.add_argument("--mannino-instance-root-100-150", default="mannino_publication_batch_instances_100_150")
    ap.add_argument("--synthetic-instance-root", default="publication_batch_instances")

    ap.add_argument("--german-anchor-root-50-70", default="german_publication_results_full_methods")
    ap.add_argument("--german-anchor-root-100-150", default="german_publication_results_full_methods_100_150")
    ap.add_argument("--mannino-anchor-root-50-70", default="mannino_publication_results_full_methods")
    ap.add_argument("--mannino-anchor-root-100-150", default="mannino_publication_results_full_methods_100_150")
    ap.add_argument("--synthetic-anchor-root", default="publication_batch_results")

    ap.add_argument("--guided-cycles", type=int, default=None)
    ap.add_argument("--guided-proposals-per-cycle", type=int, default=None)
    ap.add_argument("--guided-exact-every", type=int, default=None)
    ap.add_argument("--guided-exact-top-k", type=int, default=None)

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--clean-target", action="store_true",
                    help="Delete each target case folder before copying anchors.")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_rows = []
    anchor_rows = []

    for dataset in args.datasets:
        runner = runner_for(dataset, args)
        if not runner.exists():
            msg = f"Missing runner: {runner}"
            print("ERROR:", msg)
            log_rows.append({"dataset": dataset, "status": "missing_runner", "runner": str(runner)})
            if not args.continue_on_error:
                raise FileNotFoundError(runner)
            continue

        for n in args.sizes:
            inst_root = instance_root_for(dataset, n, args)
            anch_root = anchor_root_for(dataset, n, args)

            if not inst_root.exists():
                msg = f"Missing instance root for {dataset} n={n}: {inst_root}"
                print("ERROR:", msg)
                log_rows.append({"dataset": dataset, "n": n, "status": "missing_instance_root", "instance_root": str(inst_root)})
                if not args.continue_on_error:
                    raise FileNotFoundError(inst_root)
                continue

            if not anch_root.exists():
                msg = f"Missing anchor root for {dataset} n={n}: {anch_root}"
                print("ERROR:", msg)
                log_rows.append({"dataset": dataset, "n": n, "status": "missing_anchor_root", "anchor_root": str(anch_root)})
                if not args.continue_on_error:
                    raise FileNotFoundError(anch_root)
                continue

            base_w = get_slope(Path(args.level0_csv), dataset, n, args.base_weight_mode, args.manual_base_weight)

            for mult in args.multipliers:
                weight = float(base_w) * float(mult)
                arm = f"price_cal_w{mult:g}".replace(".", "p")
                result_root = out_root / dataset / f"n{n}" / arm
                result_root.mkdir(parents=True, exist_ok=True)

                # Seed anchors into this arm's result-root.
                missing_anchors = []
                for scenario in args.scenarios:
                    for seed in args.seeds:
                        source_case = find_case_dir(anch_root, dataset, scenario, n, seed)
                        target_case = result_root / scenario / case_name(n, seed)
                        if source_case is None:
                            missing_anchors.append(f"{scenario}/{case_name(n, seed)}")
                            continue
                        info = copy_anchor_case(source_case, target_case, clean_target=args.clean_target)
                        anchor_rows.append({
                            "dataset": dataset,
                            "n": n,
                            "arm": arm,
                            "scenario": scenario,
                            "seed": seed,
                            **info,
                        })

                if missing_anchors:
                    msg = f"Missing anchors: {missing_anchors[:5]}{'...' if len(missing_anchors)>5 else ''}"
                    print("ERROR:", dataset, n, arm, msg)
                    log_rows.append({
                        "dataset": dataset, "n": n, "multiplier": mult, "pressure_weight": weight,
                        "status": "missing_anchors", "missing_anchors": "|".join(missing_anchors),
                    })
                    if not args.continue_on_error:
                        raise FileNotFoundError(msg)
                    continue

                cmd = [
                    args.python, str(runner),
                    "--instance-root", str(inst_root),
                    "--result-root", str(result_root),
                    "--sizes", str(n),
                    "--seeds", *[str(s) for s in args.seeds],
                    "--scenarios", *args.scenarios,
                    "--run-guided",
                ]

                if args.guided_cycles is not None:
                    cmd += ["--guided-cycles", str(args.guided_cycles)]
                if args.guided_proposals_per_cycle is not None:
                    cmd += ["--guided-proposals-per-cycle", str(args.guided_proposals_per_cycle)]
                if args.guided_exact_every is not None:
                    cmd += ["--guided-exact-every", str(args.guided_exact_every)]
                if args.guided_exact_top_k is not None:
                    cmd += ["--guided-exact-top-k", str(args.guided_exact_top_k)]
                if args.force:
                    cmd += ["--force"]
                if args.quiet:
                    cmd += ["--quiet"]

                env = os.environ.copy()
                env["RECOURSE_PRICING_ARM"] = "price_cal"
                env["RECOURSE_PRESSURE_WEIGHT"] = str(weight)
                env["RECOURSE_EXACT_NOPRESSURE_EVAL"] = "1"

                row = {
                    "dataset": dataset,
                    "n": n,
                    "multiplier": mult,
                    "pressure_weight": weight,
                    "instance_root": str(inst_root),
                    "anchor_root": str(anch_root),
                    "runner": str(runner),
                    "result_root": str(result_root),
                    "cmd": " ".join(cmd),
                }

                print("\n" + "=" * 90)
                print(f"{dataset} n={n} arm={arm} weight={weight}")
                print(" ".join(cmd))

                if args.dry_run:
                    row["status"] = "dry_run"
                    row["returncode"] = None
                    log_rows.append(row)
                    continue

                p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (result_root / "stdout.txt").write_text(p.stdout, encoding="utf-8", errors="ignore")
                (result_root / "stderr.txt").write_text(p.stderr, encoding="utf-8", errors="ignore")
                row["returncode"] = p.returncode
                row["status"] = "ok" if p.returncode == 0 else "failed"
                log_rows.append(row)

                if p.returncode != 0:
                    print("FAILED. Last stderr lines:")
                    print("\n".join(p.stderr.splitlines()[-20:]))
                    if not args.continue_on_error:
                        pd.DataFrame(log_rows).to_csv(out_root / "level1_run_log.csv", index=False)
                        pd.DataFrame(anchor_rows).to_csv(out_root / "anchor_seed_log.csv", index=False)
                        raise RuntimeError(f"Failed: {dataset} n={n} arm={arm}")

    pd.DataFrame(log_rows).to_csv(out_root / "level1_run_log.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(out_root / "anchor_seed_log.csv", index=False)
    print(f"\nSaved log: {out_root / 'level1_run_log.csv'}")
    print(f"Saved anchor log: {out_root / 'anchor_seed_log.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_synthetic_abclean_factorial.py

Synthetic A/B/clean factorial runner.

This runner deliberately separates:
  A = pressure-guided destroy/repair operators (operator_on/off)
  B = scalar pressure cost in the fast objective (RECOURSE_PRESSURE_WEIGHT)

Supported arms:
  off               A off, B=0. Usually not needed if Synthetic off already exists.
  price_off_clean   A off, B=0. Clean-gate arm; should match existing off.
  A_only            A on,  B=0. Alias: price_cal_w0
  B_only            A off, B=1*slope
  AB_w0p25          A on,  B=0.25*slope
  AB_w0p5           A on,  B=0.5*slope
  AB_w1             A on,  B=1*slope. Alias: price_cal_w1
  AB_w2             A on,  B=2*slope. Alias: price_cal_w2

Default roots:
  instances: publication_batch_instances
  anchors:   publication_batch_results

Important:
  Use pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py or an equivalent
  guidance module that supports guidance_mode='operator_off' and obeys
  RECOURSE_PRESSURE_WEIGHT for price/operator_off modes.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
DEFAULT_SIZES = [50, 70, 100, 150]
DEFAULT_SCENARIOS = ["nominal", "transfer_bottleneck"]
DEFAULT_ARMS = ["price_off_clean", "A_only", "B_only", "AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"]

ARM_ALIASES = {
    "clean": "price_off_clean",
    "priceoffclean": "price_off_clean",
    "price_off": "price_off_clean",
    "a": "A_only",
    "a_only": "A_only",
    "price_cal_w0": "A_only",
    "w0": "A_only",
    "b": "B_only",
    "b_only": "B_only",
    "ab": "AB_w1",
    "ab_w0.25": "AB_w0p25",
    "ab_w0p25": "AB_w0p25",
    "ab_w0.5": "AB_w0p5",
    "ab_w0p5": "AB_w0p5",
    "ab_w1": "AB_w1",
    "ab_w2": "AB_w2",
    "price_cal_w1": "AB_w1",
    "price_cal_w2": "AB_w2",
    "off": "off",
}


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    operator_on: bool
    b_multiplier: float
    guidance_mode: str
    needs_slope: bool


def canonical_arm(name: str) -> str:
    raw = str(name).strip()
    key = raw.lower().replace("-", "_")
    key = key.replace(".", "p") if key.startswith("ab_w") else key
    return ARM_ALIASES.get(key, raw)


def arm_spec(name: str, pressure_unit: Optional[float]) -> ArmSpec:
    arm = canonical_arm(name)
    if arm == "off":
        return ArmSpec(arm=arm, operator_on=False, b_multiplier=0.0, guidance_mode="off", needs_slope=False)
    if arm == "price_off_clean":
        return ArmSpec(arm=arm, operator_on=False, b_multiplier=0.0, guidance_mode="off", needs_slope=False)
    if arm == "A_only":
        return ArmSpec(arm=arm, operator_on=True, b_multiplier=0.0, guidance_mode="price", needs_slope=False)
    if arm == "B_only":
        return ArmSpec(arm=arm, operator_on=False, b_multiplier=1.0, guidance_mode="operator_off", needs_slope=True)
    m = re.fullmatch(r"AB_w([0-9]+(?:p[0-9]+)?)", arm)
    if m:
        mult = float(m.group(1).replace("p", "."))
        return ArmSpec(arm=arm, operator_on=True, b_multiplier=mult, guidance_mode="price", needs_slope=True)
    raise ValueError(f"Unknown arm {name!r}. Use one of: off, price_off_clean, A_only, B_only, AB_w0p25, AB_w0p5, AB_w1, AB_w2")


def case_name(n: int, seed: int) -> str:
    return f"case_{int(n)}_seed{int(seed)}"


def resolve_case_dir(root: Path, scenario: str, n: int, seed: int) -> Optional[Path]:
    cname = case_name(n, seed)
    candidates = [
        root / scenario / cname,
        root / "Synthetic" / scenario / cname,
        root / f"n{int(n)}" / scenario / cname,
        root / "Synthetic" / f"n{int(n)}" / scenario / cname,
        root / cname,
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    if not root.exists():
        return None
    hits = [p for p in root.rglob(cname) if p.is_dir()]
    if not hits:
        return None
    scored = []
    for h in hits:
        parts = [str(x).lower() for x in h.parts]
        score = 0
        if scenario.lower() in parts:
            score += 5
        if "synthetic" in parts:
            score += 2
        if f"n{int(n)}".lower() in parts:
            score += 1
        scored.append((score, len(str(h)), h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def read_csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return {str(c).strip() for c in next(csv.reader(f))}
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
    preferred = [
        "schedule.csv", "normalized_schedule.csv", "final_schedule.csv", "best_schedule.csv",
        "stage2_schedule.csv", "repaired_schedule.csv", "pr_glns_schedule.csv",
        "spiral_pr_glns_schedule.csv",
    ]
    for name in preferred:
        for p in root.rglob(name):
            rel = str(p.relative_to(root)).lower()
            if any(x in rel for x in ["candidate", "trace"]):
                continue
            if looks_like_schedule_csv(p):
                return p
    candidates = []
    for p in root.rglob("*.csv"):
        rel = str(p.relative_to(root)).lower()
        if any(x in rel for x in ["candidate", "trace", "daily", "pressure"]):
            continue
        if looks_like_schedule_csv(p):
            candidates.append(p)
    candidates.sort(key=lambda p: (len(str(p)), str(p)))
    return candidates[0] if candidates else None


def find_anchor_schedules(case_result_dir: Path) -> List[Tuple[str, Path]]:
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
        for p in sorted(case_result_dir.rglob("*.csv")):
            rel = str(p.relative_to(case_result_dir)).replace("\\", "/").lower()
            if rel.startswith(("01_", "02_", "04_")) and looks_like_schedule_csv(p):
                anchors.append((p.parent.name, p))
                if len(anchors) >= 3:
                    break
    seen, out = set(), []
    for label, p in anchors:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
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
        if root.exists():
            out += [p for p in sorted(root.rglob("*.csv")) if looks_like_pressure_csv(p)]
    if not out:
        for p in sorted(case_result_dir.rglob("*.csv")):
            rel = str(p.relative_to(case_result_dir)).replace("\\", "/").lower()
            if rel.startswith(("06_", "07_", "spiral_", "graph_", "price_")):
                continue
            if looks_like_pressure_csv(p):
                out.append(p)
    seen, uniq = set(), []
    for p in out:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def import_spiral_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Spiral script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("spiral_guidance_abclean_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def output_dir_for(args, arm: str, scenario: str, n: int, seed: int) -> Path:
    return Path(args.output_root) / "Synthetic" / f"n{int(n)}" / arm / scenario / case_name(n, seed) / arm


def already_done(out_dir: Path) -> bool:
    p = out_dir / "spiral_trace.csv"
    return p.exists() and p.stat().st_size > 0


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_pressure_unit_from_csv(path: Optional[Path], r2_threshold: float) -> Tuple[Optional[float], str, Optional[float]]:
    if path is None or not path.exists():
        return None, "none", None
    df = pd.read_csv(path)
    if df.empty:
        return None, f"empty:{path}", None
    # Prefer explicit Synthetic rows and explicit slope_syn.
    sub = df.copy()
    if "dataset" in sub.columns:
        syn = sub[sub["dataset"].astype(str).str.lower().eq("synthetic")]
        if len(syn):
            sub = syn
    row = sub.iloc[0]
    slope_col = None
    for c in ["slope_syn", "slope", "pressure_weight_unit", "unit_weight"]:
        if c in sub.columns:
            slope_col = c
            break
    if slope_col is None:
        return None, f"missing_slope_column:{path}", None
    slope = pd.to_numeric(pd.Series([row[slope_col]]), errors="coerce").iloc[0]
    r2 = None
    for c in ["R2_syn", "r2_syn", "R2", "r2"]:
        if c in sub.columns:
            r2 = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            break
    if not np.isfinite(slope):
        return None, f"nonfinite_slope:{path}", float(r2) if r2 is not None and np.isfinite(r2) else None
    if r2 is not None and np.isfinite(r2) and float(r2) < r2_threshold:
        return float(slope), f"low_r2:{float(r2):.6g}", float(r2)
    return float(slope), f"csv:{path}:{slope_col}", float(r2) if r2 is not None and np.isfinite(r2) else None


def resolve_pressure_unit(args, specs: List[ArmSpec]) -> Tuple[Optional[float], str, Optional[float]]:
    if not any(s.needs_slope for s in specs):
        return 0.0, "not_needed", None
    if args.pressure_weight_unit is not None:
        return float(args.pressure_weight_unit), "manual", None
    slope, source, r2 = read_pressure_unit_from_csv(Path(args.level0_synth_csv) if args.level0_synth_csv else None, args.low_r2_threshold)
    if slope is not None:
        if source.startswith("low_r2") and args.skip_b_if_low_r2:
            return 0.0, source + ":skip_b", r2
        return slope, source, r2
    if args.fallback_pressure_weight is not None:
        return float(args.fallback_pressure_weight), "fallback", None
    raise RuntimeError(
        "B/AB arms need a pressure-weight unit. Provide --pressure-weight-unit, "
        "or --level0-synth-csv with a slope_syn/slope column, or --fallback-pressure-weight."
    )


def call_spiral(mod, args, spec: ArmSpec, pressure_weight: float,
                instance_dir: Path, out_dir: Path, anchors: Dict[str, pd.DataFrame],
                pressure_files: List[Path]) -> Dict[str, Any]:
    if not hasattr(mod, "run_spiral_pr_glns") or not hasattr(mod, "load_instance"):
        raise AttributeError("Spiral module must expose load_instance() and run_spiral_pr_glns().")
    instance = mod.load_instance(instance_dir)
    fn = mod.run_spiral_pr_glns
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {"instance": instance, "anchor_schedules": anchors, "output_dir": out_dir}
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
        "wallclock_limit_s": args.wallclock_limit_s if args.wallclock_limit_s and args.wallclock_limit_s > 0 else None,
        "guidance_mode": spec.guidance_mode,
        "verbose": not args.quiet,
    }
    for k, v in optional.items():
        if k in sig.parameters:
            kwargs[k] = v
    if "guidance_mode" not in sig.parameters:
        raise RuntimeError("run_spiral_pr_glns() does not accept guidance_mode; use the abclean-patched spiral script.")

    # Isolate environment for the called heuristic.
    old_env = {k: os.environ.get(k) for k in [
        "RECOURSE_PRICING_ARM", "RECOURSE_PRESSURE_WEIGHT", "RECOURSE_EXACT_NOPRESSURE_EVAL",
        "ORSCHE_AB_OPERATOR_MODE", "ORSCHE_LNS_ABS_BUDGET", "ORSCHE_LNS_DESTROY_POLICY",
    ]}
    try:
        os.environ.pop("ORSCHE_LNS_ABS_BUDGET", None)
        os.environ.pop("ORSCHE_LNS_DESTROY_POLICY", None)
        os.environ["RECOURSE_PRICING_ARM"] = spec.arm
        os.environ["RECOURSE_PRESSURE_WEIGHT"] = repr(float(pressure_weight))
        os.environ["RECOURSE_EXACT_NOPRESSURE_EVAL"] = "1"
        os.environ["ORSCHE_AB_OPERATOR_MODE"] = "on" if spec.operator_on else "off"
        meta = fn(**kwargs) or {}
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return meta


def update_metadata(out_dir: Path, manifest: Dict[str, Any]) -> None:
    meta_path = out_dir / "spiral_pr_glns_metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta.setdefault("abclean_manifest", {}).update(manifest)
    # Also duplicate key fields at top-level for collector convenience.
    for k in [
        "dataset", "n", "scenario", "seed", "arm", "operator_on", "b_multiplier",
        "pressure_weight_unit", "actual_pressure_weight", "guidance_mode",
        "exact_nopressure_eval", "lns_seed", "cycles", "proposals_per_cycle",
        "exact_every", "exact_top_k", "max_or_overtime",
    ]:
        if k in manifest:
            meta[k] = manifest[k]
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "abclean_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def run_one(mod, args, spec: ArmSpec, pressure_unit: float, pressure_source: str, pressure_r2: Optional[float],
            scenario: str, n: int, seed: int) -> Dict[str, Any]:
    inst_root = Path(args.synthetic_instance_root)
    anch_root = Path(args.synthetic_anchor_root)
    inst_dir = resolve_case_dir(inst_root, scenario, n, seed)
    anch_dir = resolve_case_dir(anch_root, scenario, n, seed)
    pressure_weight = float(spec.b_multiplier) * float(pressure_unit or 0.0)
    out_dir = output_dir_for(args, spec.arm, scenario, n, seed)
    row: Dict[str, Any] = {
        "dataset": "Synthetic", "n": int(n), "scenario": scenario, "seed": int(seed), "arm": spec.arm,
        "operator_on": int(spec.operator_on), "b_multiplier": float(spec.b_multiplier),
        "pressure_weight_unit": float(pressure_unit or 0.0), "actual_pressure_weight": pressure_weight,
        "pressure_weight_source": pressure_source, "pressure_regression_r2": pressure_r2 if pressure_r2 is not None else "",
        "guidance_mode": spec.guidance_mode, "lns_seed": int(args.lns_seed),
        "cycles": int(args.cycles), "proposals_per_cycle": int(args.proposals_per_cycle),
        "exact_every": int(args.exact_every), "exact_top_k": int(args.exact_top_k),
        "max_or_overtime": int(args.max_or_overtime), "include_pool": args.include_pool,
        "instance_root": str(inst_root), "anchor_root": str(anch_root),
        "instance_dir": str(inst_dir or ""), "anchor_result_dir": str(anch_dir or ""), "output_dir": str(out_dir),
        "status": "", "error": "", "trace_path": str(out_dir / "spiral_trace.csv"),
    }
    if spec.needs_slope and pressure_source.endswith(":skip_b"):
        row["status"] = "skipped_low_r2"
        return row
    if inst_dir is None:
        row["status"] = "missing_instance_dir"
        return row
    if anch_dir is None:
        row["status"] = "missing_anchor_result_dir"
        return row
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and already_done(out_dir):
        row["status"] = "skipped_existing"
        return row
    anchor_paths = find_anchor_schedules(anch_dir)
    pressure_files = find_pressure_day_files(anch_dir)
    row["num_anchors"] = len(anchor_paths)
    row["num_pressure_files"] = len(pressure_files)
    row["anchor_paths"] = " | ".join(f"{lab}:{p}" for lab, p in anchor_paths)
    row["pressure_paths"] = " | ".join(map(str, pressure_files))
    if not anchor_paths:
        row["status"] = "missing_anchor_schedules"
        return row

    print("\n" + "=" * 100)
    print(f"[Synthetic] {scenario}, n={n}, seed={seed}, arm={spec.arm}, A={'on' if spec.operator_on else 'off'}, B={pressure_weight:.8g}")
    print(f"instance: {inst_dir}")
    print(f"anchor:   {anch_dir}")
    print(f"output:   {out_dir}")
    print(f"anchors={len(anchor_paths)}, pressure_files={len(pressure_files)}")

    if args.dry_run:
        row["status"] = "dry_run"
        return row
    try:
        anchors = {lab: pd.read_csv(p) for lab, p in anchor_paths}
        meta = call_spiral(mod, args, spec, pressure_weight, inst_dir, out_dir, anchors, pressure_files)
        manifest = dict(row)
        manifest.update({
            "exact_nopressure_eval": 1,
            "anchor_paths_list": [str(p) for _, p in anchor_paths],
            "pressure_paths_list": [str(p) for p in pressure_files],
        })
        update_metadata(out_dir, manifest)
        row["status"] = "success"
        for k in ["runtime_sec", "best_source", "best_exact_score", "best_fast_objective", "recourse_pressure_weight_fast"]:
            row[k] = meta.get(k, "")
    except Exception as e:
        row["status"] = "failed"
        row["error"] = repr(e)
        err_path = out_dir / "abclean_error.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"FAILED: {e}\nTraceback saved to: {err_path}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--spiral-script", default="pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py")
    ap.add_argument("--output-root", default="synthetic_abclean_factorial_results")
    ap.add_argument("--synthetic-instance-root", default="publication_batch_instances")
    ap.add_argument("--synthetic-anchor-root", default="publication_batch_results")
    ap.add_argument("--pressure-weight-unit", type=float, default=None, help="Synthetic slope unit for B. Required for B/AB unless level0/fallback is supplied.")
    ap.add_argument("--level0-synth-csv", default="level0_pressure_regression_synth.csv")
    ap.add_argument("--fallback-pressure-weight", type=float, default=None, help="Explicit fallback B unit, e.g. real-data slope. The manifest will mark it as fallback.")
    ap.add_argument("--low-r2-threshold", type=float, default=0.2)
    ap.add_argument("--skip-b-if-low-r2", action="store_true")
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
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    specs = [arm_spec(a, args.pressure_weight_unit) for a in args.arms]
    pressure_unit, pressure_source, pressure_r2 = resolve_pressure_unit(args, specs)
    mod = import_spiral_module(Path(args.spiral_script))

    # Fail early if B-only cannot be represented by the loaded module.
    if any(s.guidance_mode == "operator_off" for s in specs):
        try:
            norm = getattr(mod, "_normalize_guidance_mode")("operator_off")
            if norm != "operator_off":
                raise RuntimeError(f"operator_off normalized to {norm!r}")
        except Exception as e:
            raise RuntimeError(
                "B_only requires an abclean-patched spiral script supporting guidance_mode='operator_off'. "
                "Use pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py."
            ) from e

    rows: List[Dict[str, Any]] = []
    log_path = Path(args.output_root) / "synthetic_abclean_run_log.csv"
    total = len(args.sizes) * len(args.scenarios) * len(args.seeds) * len(specs)
    idx = 0
    for n in args.sizes:
        for scenario in args.scenarios:
            for seed in args.seeds:
                for spec in specs:
                    idx += 1
                    print(f"\n### Progress {idx}/{total}")
                    row = run_one(mod, args, spec, float(pressure_unit or 0.0), pressure_source, pressure_r2, scenario, n, seed)
                    rows.append(row)
                    write_rows(log_path, rows)
                    ok = {"success", "dry_run", "skipped_existing", "skipped_low_r2"}
                    if row["status"] not in ok:
                        print(f"WARNING: {row['status']} for Synthetic {scenario} n={n} seed={seed} arm={spec.arm}")
                        if not args.continue_on_error:
                            raise SystemExit(1)
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()

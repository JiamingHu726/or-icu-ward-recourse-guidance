#!/usr/bin/env python3
"""Audit an existing merged frozen-calibration directory.

Use this after artifacts have been copied from separately generated 50/70 and
100/150 batches into a common `data/derived/frozen_calibration` directory.
It rebuilds the authoritative manifest and audit from the files actually present,
so a prior single-directory collection run cannot undercount the final union.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SIZES = [50, 70, 100, 150]
SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected(prefix: str) -> set[str]:
    return {f"{prefix}_n{n}_seed{s}.csv" for n in SIZES for s in SEEDS}


def classify(path: Path) -> tuple[str, str]:
    name = path.name
    family = "GermanOR" if "GermanOR" in path.parts else "Mannino" if "Mannino" in path.parts else ""
    if re.fullmatch(r"(?:german|mannino)_pool_n(?:50|70|100|150)_seed\d+\.csv", name):
        return family, "seeded_or_duration_pool"
    if name == "mannino_case_pool.csv":
        return family, "empirical_duration_case_pool"
    if name in {"mannino_duration_stats.json", "mannino_duration_summary.csv"}:
        return family, "duration_calibration_summary"
    return family, "pool_generation_metadata"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_root", help="Existing data/derived/frozen_calibration directory.")
    parser.add_argument("--output-manifest", default="metadata/frozen_calibration_artifacts_manifest.csv")
    parser.add_argument("--output-audit", default="metadata/frozen_calibration_artifacts_audit.json")
    args = parser.parse_args()

    root = Path(args.frozen_root).resolve()
    if not root.exists():
        raise SystemExit(f"Frozen calibration directory not found: {root}")

    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        family, role = classify(path)
        rows.append({
            "dataset_family": family,
            "artifact_role": role,
            "source_file_name": path.name,
            "release_relative_path": f"data/derived/frozen_calibration/{path.relative_to(root).as_posix()}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset_family", "artifact_role", "source_file_name",
                        "release_relative_path", "bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    german_dir = root / "GermanOR"
    mannino_dir = root / "Mannino"
    german_names = {p.name for p in german_dir.glob("*.csv")} if german_dir.exists() else set()
    mannino_names = {p.name for p in mannino_dir.glob("*.csv")} if mannino_dir.exists() else set()
    missing_g = sorted(expected("german_pool") - german_names)
    missing_m = sorted(expected("mannino_pool") - mannino_names)
    raw = [
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xls", ".xlsx", ".zip", ".rar"}
    ]
    audit = {
        "expected_sizes": SIZES,
        "expected_seeds": SEEDS,
        "expected_pools_per_family": len(SIZES) * len(SEEDS),
        "german_pools_found": len(expected("german_pool") & german_names),
        "mannino_pools_found": len(expected("mannino_pool") & mannino_names),
        "missing_german_pools": missing_g,
        "missing_mannino_pools": missing_m,
        "artifact_count": len(rows),
        "complete": not missing_g and not missing_m,
        "raw_workbook_or_archive_files_detected": raw,
        "notes": [
            "This audit scans the final merged output folder.",
            "A complete result requires 40 seeded pools per family.",
            "Raw Mannino workbook files should not appear here unless reuse terms are verified.",
        ],
    }
    out_audit = Path(args.output_audit)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} artifacts to {out_manifest}")
    print(f"Wrote {out_audit}")
    print(
        "Coverage: "
        f"GermanOR {audit['german_pools_found']}/40; "
        f"Mannino {audit['mannino_pools_found']}/40; "
        f"complete={audit['complete']}"
    )
    if not audit["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

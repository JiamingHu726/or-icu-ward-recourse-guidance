#!/usr/bin/env python3
"""Collect frozen calibration artifacts from one or more prepared directories per family.

Use plural options because the publication batches may be split, for example:
- 50/70 pools in one directory;
- 100/150 pools in a second directory.

The script merges all seed-specific pools. Shared files are retained under
`source_snapshots/<source-label>/` to avoid silently overwriting batch-specific
indexes, metadata, or empirical duration pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
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


def safe_label(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name or "prepared")


def copy_one(source: Path, target: Path, family: str, role: str, label: str, rows: list[dict]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    rows.append({
        "dataset_family": family,
        "artifact_role": role,
        "source_label": label,
        "source_file_name": source.name,
        "release_relative_path": target.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256(source),
    })


def collect_family(
    family: str,
    roots: list[Path],
    pool_prefix: str,
    shared_names: list[str],
    dest: Path,
    rows: list[dict],
) -> tuple[list[str], list[str]]:
    expected_names = expected(pool_prefix)
    found: dict[str, Path] = {}
    duplicates: list[str] = []

    for root in roots:
        if not root.exists():
            print(f"WARNING: missing {family} prepared directory: {root}")
            continue
        label = safe_label(root)
        for file in sorted(root.glob(f"{pool_prefix}_n*_seed*.csv")):
            if file.name not in expected_names:
                continue
            if file.name in found:
                # Preserve provenance by noting duplicate candidate, but only one
                # seed-specific pool may represent a fixed historical cell.
                duplicates.append(f"{file.name}: {found[file.name]} | {file}")
                continue
            found[file.name] = file
            copy_one(file, dest / family / file.name, family, "seeded_or_duration_pool", label, rows)

        for name in shared_names:
            file = root / name
            if file.exists() and file.is_file():
                role = (
                    "empirical_duration_case_pool" if name == "mannino_case_pool.csv"
                    else "duration_calibration_summary" if name.startswith("mannino_duration")
                    else "pool_generation_metadata"
                )
                copy_one(
                    file,
                    dest / family / "source_snapshots" / label / name,
                    family,
                    role,
                    label,
                    rows,
                )
    missing = sorted(expected_names - set(found))
    return missing, duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--german-prepared-dirs", nargs="+", required=True)
    parser.add_argument("--mannino-prepared-dirs", nargs="+", required=True)
    parser.add_argument("--output-data-dir", default="data/derived/frozen_calibration")
    parser.add_argument("--output-manifest", default="metadata/frozen_calibration_artifacts_manifest.csv")
    parser.add_argument("--output-audit", default="metadata/frozen_calibration_artifacts_audit.json")
    args = parser.parse_args()

    rows: list[dict] = []
    dest = Path(args.output_data_dir)
    missing_g, dup_g = collect_family(
        "GermanOR",
        [Path(x) for x in args.german_prepared_dirs],
        "german_pool",
        ["german_pool_index.csv", "german_pool_generation_metadata.json", "german_pool_fallback_log.csv"],
        dest,
        rows,
    )
    missing_m, dup_m = collect_family(
        "Mannino",
        [Path(x) for x in args.mannino_prepared_dirs],
        "mannino_pool",
        ["mannino_duration_stats.json", "mannino_duration_summary.csv", "mannino_case_pool.csv"],
        dest,
        rows,
    )

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset_family", "artifact_role", "source_label", "source_file_name",
              "release_relative_path", "bytes", "sha256"]
    with out_manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "expected_sizes": SIZES,
        "expected_seeds": SEEDS,
        "expected_pools_per_family": len(SIZES) * len(SEEDS),
        "german_pools_found": len(expected("german_pool")) - len(missing_g),
        "mannino_pools_found": len(expected("mannino_pool")) - len(missing_m),
        "missing_german_pools": missing_g,
        "missing_mannino_pools": missing_m,
        "duplicate_german_pool_candidates": dup_g,
        "duplicate_mannino_pool_candidates": dup_m,
        "artifact_count": len(rows),
        "complete": not missing_g and not missing_m,
    }
    out_audit = Path(args.output_audit)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print(f"Copied {len(rows)} frozen calibration artifacts.")
    print(f"Coverage: GermanOR {audit['german_pools_found']}/40; Mannino {audit['mannino_pools_found']}/40")
    if not audit["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

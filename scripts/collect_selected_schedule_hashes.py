#!/usr/bin/env python3
"""Create an auditable selected-schedule hash manifest.

This script records both:
1. raw SHA-256 hashes of schedule CSV files; and
2. canonical SHA-256 hashes after normalizing CSV row order and column order.

The canonical hash is the relevant equality check when two schedule files differ
only in row order, whitespace, or CSV serialization. The raw hash remains useful
as a byte-level provenance record.

The script never fabricates missing schedules. It reports incomplete arm groups
explicitly in the generated summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FILENAME = "spiral_pr_glns_schedule.csv"
REAL_ARMS = ["price_cal_w0", "price_cal_w0p25", "price_cal_w0p5", "price_cal_w1", "price_cal_w2"]
SYNTHETIC_ARMS = ["price_off_clean", "A_only", "B_only", "AB_w0p25", "AB_w0p5", "AB_w1", "AB_w2"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_value(value: Any) -> str:
    """Normalize only serialization artefacts; do not coerce scientific values."""
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical_csv_sha256(path: Path) -> tuple[str, int, int]:
    """Hash a CSV after deterministic header and row normalization."""
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = sorted(clean_value(x) for x in (reader.fieldnames or []))
        rows = []
        for record in reader:
            normalized = tuple(clean_value(record.get(field, "")) for field in fields)
            rows.append(normalized)

    digest.update(("\x1f".join(fields) + "\n").encode("utf-8"))
    for row in sorted(rows):
        digest.update(("\x1f".join(row) + "\n").encode("utf-8"))
    return digest.hexdigest(), len(rows), len(fields)


def infer_dataset(text: str) -> str:
    text = text.replace("\\", "/")
    tests = [
        ("GermanOR", r"(?:^|/)GermanOR(?:/|$)|german[_-]"),
        ("Mannino", r"(?:^|/)Mannino(?:/|$)|mannino"),
        ("Synthetic", r"(?:^|/)Synthetic(?:/|$)|synthetic|publication_batch_instances"),
    ]
    for label, pattern in tests:
        if re.search(pattern, text, flags=re.I):
            return label
    return ""


def capture(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.I)
    return match.group(1) if match else ""


def infer_arm(text: str) -> str:
    text = text.replace("\\", "/")
    arms = REAL_ARMS + SYNTHETIC_ARMS
    # Longest first protects w0p25 from matching w0.
    for arm in sorted(arms, key=len, reverse=True):
        if re.search(rf"(?:^|/){re.escape(arm)}(?:/|$)", text, flags=re.I):
            return arm
    return ""


def infer_fields(path: Path) -> dict[str, str]:
    text = str(path).replace("\\", "/")
    return {
        "dataset": infer_dataset(text),
        "n": capture(text, r"/n(50|70|100|150)(?:/|$)") or capture(text, r"/case_(50|70|100|150)_seed"),
        "scenario": capture(text, r"/(nominal|ward_pressure|transfer_bottleneck)(?:/|$)"),
        "seed": capture(text, r"case_\d+_seed(\d+)"),
        "arm": infer_arm(text),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def nested_get(data: Any, candidate_keys: list[str]) -> str:
    """Find the first matching scalar key in a nested manifest."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key in candidate_keys and not isinstance(value, (dict, list)):
                return clean_value(value)
        for value in data.values():
            found = nested_get(value, candidate_keys)
            if found != "":
                return found
    elif isinstance(data, list):
        for value in data:
            found = nested_get(value, candidate_keys)
            if found != "":
                return found
    return ""


def nearest_metadata(schedule: Path) -> dict[str, str]:
    """Collect selected non-sensitive metadata from the nearest run directory."""
    keys = {
        "selected_exact_score": ["selected_exact_score", "final_exact_score", "exact_nopressure_score", "exact_score"],
        "selected_iteration": ["selected_iteration", "selection_iteration", "final_iteration"],
        "guidance_mode": ["guidance_mode"],
        "actual_pressure_weight": ["actual_pressure_weight", "pressure_weight"],
        "lns_seed": ["lns_seed"],
    }
    out = {name: "" for name in keys}
    filenames = [
        "spiral_pr_glns_metadata.json",
        "abclean_run_manifest.json",
        "run_metadata.json",
    ]
    for parent in [schedule.parent, *schedule.parents]:
        for filename in filenames:
            candidate = parent / filename
            if candidate.exists():
                data = read_json(candidate)
                for out_key, candidates in keys.items():
                    if not out[out_key]:
                        out[out_key] = nested_get(data, candidates)
        if any(out.values()):
            break
    return out


def root_relative(path: Path, roots: list[Path]) -> tuple[str, str]:
    """Return a public-safe path relative to the most specific supplied root."""
    candidates = []
    for root in roots:
        try:
            rel = path.resolve().relative_to(root.resolve())
            candidates.append((len(str(root.resolve())), root.name or "root", rel.as_posix()))
        except ValueError:
            continue
    if candidates:
        _, label, rel = max(candidates, key=lambda item: item[0])
        return label, rel
    return "", path.name


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="One or more result-root directories to scan.")
    parser.add_argument("--output-dir", default="metadata", help="Directory for detailed and summary CSV files.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Schedule filename to hash.")
    parser.add_argument("--fail-on-incomplete-real-groups", action="store_true")
    args = parser.parse_args()

    roots = [Path(item) for item in args.roots]
    valid_roots = []
    for root in roots:
        if root.exists():
            valid_roots.append(root)
        else:
            print(f"WARNING: root not found; skipped: {root}")

    discovered: set[Path] = set()
    for root in valid_roots:
        discovered.update(path.resolve() for path in root.rglob(args.filename))

    rows: list[dict[str, Any]] = []
    for path in sorted(discovered):
        metadata = nearest_metadata(path)
        root_label, public_path = root_relative(path, valid_roots)
        row = infer_fields(path)
        canonical_hash, row_count, column_count = canonical_csv_sha256(path)
        row.update({
            "root_label": root_label,
            "schedule_file_relative": public_path,
            "raw_sha256": sha256_file(path),
            "canonical_sha256": canonical_hash,
            "bytes": path.stat().st_size,
            "row_count": row_count,
            "column_count": column_count,
        })
        row.update(metadata)
        rows.append(row)

    rows.sort(key=lambda r: (
        r["dataset"], int(r["n"] or 0), r["scenario"], int(r["seed"] or 0), r["arm"], r["schedule_file_relative"]
    ))

    out_dir = Path(args.output_dir)
    detailed_fields = [
        "dataset", "n", "scenario", "seed", "arm",
        "root_label", "schedule_file_relative",
        "raw_sha256", "canonical_sha256", "bytes", "row_count", "column_count",
        "selected_exact_score", "selected_iteration",
        "guidance_mode", "actual_pressure_weight", "lns_seed",
    ]
    write_csv(out_dir / "selected_schedule_hashes.csv", rows, detailed_fields)

    # Summarize all arm groups.
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["n"], row["scenario"], row["seed"])].append(row)

    group_rows = []
    real_pairwise_rows = []
    synthetic_pairwise_rows = []
    incomplete_real_groups = 0

    for key, group in sorted(groups.items()):
        dataset, n, scenario, seed = key
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_arm[row["arm"]].append(row)

        present_arms = sorted(arm for arm in by_arm if arm)
        unknown_arm_files = sum(len(by_arm[arm]) for arm in by_arm if not arm)
        canonical_hashes = {row["canonical_sha256"] for row in group}
        group_rows.append({
            "dataset": dataset,
            "n": n,
            "scenario": scenario,
            "seed": seed,
            "schedule_files": len(group),
            "arms_found": "|".join(present_arms),
            "unknown_arm_files": unknown_arm_files,
            "unique_canonical_hashes": len(canonical_hashes),
        })

        present_real = [arm for arm in REAL_ARMS if arm in by_arm]
        if present_real:
            missing = [arm for arm in REAL_ARMS if arm not in by_arm]
            if missing:
                incomplete_real_groups += 1
            reference = by_arm.get("price_cal_w0", [])
            for arm in REAL_ARMS:
                current = by_arm.get(arm, [])
                status = "ok"
                if not reference:
                    status = "missing_reference_w0"
                elif not current:
                    status = "missing_arm"
                elif len(reference) != 1 or len(current) != 1:
                    status = "duplicate_schedule_files"
                real_pairwise_rows.append({
                    "dataset": dataset,
                    "n": n,
                    "scenario": scenario,
                    "seed": seed,
                    "reference_arm": "price_cal_w0",
                    "comparison_arm": arm,
                    "reference_present": len(reference),
                    "comparison_present": len(current),
                    "same_raw_sha256": (
                        reference[0]["raw_sha256"] == current[0]["raw_sha256"]
                        if len(reference) == len(current) == 1 else ""
                    ),
                    "same_canonical_sha256": (
                        reference[0]["canonical_sha256"] == current[0]["canonical_sha256"]
                        if len(reference) == len(current) == 1 else ""
                    ),
                    "status": status,
                })

        # Synthetic control check: only record comparable control arms when they coexist.
        control = by_arm.get("price_off_clean", [])
        for arm in SYNTHETIC_ARMS:
            current = by_arm.get(arm, [])
            if control or current:
                status = "ok"
                if not control:
                    status = "missing_control_price_off_clean"
                elif not current:
                    status = "missing_arm"
                elif len(control) != 1 or len(current) != 1:
                    status = "duplicate_schedule_files"
                synthetic_pairwise_rows.append({
                    "dataset": dataset,
                    "n": n,
                    "scenario": scenario,
                    "seed": seed,
                    "reference_arm": "price_off_clean",
                    "comparison_arm": arm,
                    "reference_present": len(control),
                    "comparison_present": len(current),
                    "same_raw_sha256": (
                        control[0]["raw_sha256"] == current[0]["raw_sha256"]
                        if len(control) == len(current) == 1 else ""
                    ),
                    "same_canonical_sha256": (
                        control[0]["canonical_sha256"] == current[0]["canonical_sha256"]
                        if len(control) == len(current) == 1 else ""
                    ),
                    "status": status,
                })

    write_csv(
        out_dir / "selected_schedule_hash_group_summary.csv",
        group_rows,
        ["dataset", "n", "scenario", "seed", "schedule_files", "arms_found", "unknown_arm_files", "unique_canonical_hashes"],
    )
    write_csv(
        out_dir / "price_arm_schedule_equality.csv",
        real_pairwise_rows,
        ["dataset", "n", "scenario", "seed", "reference_arm", "comparison_arm",
         "reference_present", "comparison_present", "same_raw_sha256", "same_canonical_sha256", "status"],
    )
    write_csv(
        out_dir / "synthetic_arm_schedule_equality.csv",
        synthetic_pairwise_rows,
        ["dataset", "n", "scenario", "seed", "reference_arm", "comparison_arm",
         "reference_present", "comparison_present", "same_raw_sha256", "same_canonical_sha256", "status"],
    )

    summary = [
        "# Selected-schedule hash audit",
        "",
        f"- Supplied result roots found: {len(valid_roots)}",
        f"- Schedule files discovered: {len(rows)}",
        f"- Dataset/size/scenario/seed groups discovered: {len(groups)}",
        f"- Real price-calibration groups with missing expected arms: {incomplete_real_groups}",
        "",
        "## Interpretation",
        "",
        "- `raw_sha256` tests byte-for-byte file identity.",
        "- `canonical_sha256` tests schedule equality after CSV row order and column order are normalized.",
        "- A missing arm, duplicate schedule file, or unknown arm is reported explicitly; it is never counted as equality.",
        "- The public manifest stores paths relative to the supplied result roots and does not expose private absolute paths.",
        "",
        "## Required review before release",
        "",
        "1. Inspect `price_arm_schedule_equality.csv` for all real price-calibration groups.",
        "2. Inspect `synthetic_arm_schedule_equality.csv` for Synthetic factorial groups.",
        "3. Confirm that all expected arms are present for the manuscript comparisons.",
        "4. Retain the detailed hash CSV and both equality tables in `metadata/`.",
    ]
    (out_dir / "selected_schedule_hash_audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} schedule hashes to {out_dir / 'selected_schedule_hashes.csv'}")
    print(f"Wrote {len(real_pairwise_rows)} real price-arm comparisons to {out_dir / 'price_arm_schedule_equality.csv'}")
    print(f"Wrote {len(synthetic_pairwise_rows)} Synthetic arm comparisons to {out_dir / 'synthetic_arm_schedule_equality.csv'}")
    if incomplete_real_groups:
        print(f"WARNING: {incomplete_real_groups} real price-calibration groups are missing one or more expected arms.")

    if args.fail_on_incomplete_real_groups and incomplete_real_groups:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

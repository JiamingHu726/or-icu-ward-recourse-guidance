#!/usr/bin/env python3
"""Audit the actual historical source-injection settings in generated case folders."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_nested(data: dict, key: str, default=""):
    if key in data:
        return data.get(key, default)
    for value in data.values():
        if isinstance(value, dict):
            result = get_nested(value, key, None)
            if result is not None:
                return result
    return default


def infer_family(path: Path) -> str:
    text = str(path).lower()
    if "german" in text:
        return "GermanOR"
    if "mannino" in text:
        return "Mannino"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="Generated-instance roots to scan.")
    parser.add_argument("--output", default="metadata/benchmark_instance_provenance.csv")
    args = parser.parse_args()

    rows = []
    for root_s in args.roots:
        root = Path(root_s)
        if not root.exists():
            print(f"WARNING: root not found: {root}")
            continue
        for path in sorted(root.rglob("metadata.json")):
            meta = read_json(path)
            family = meta.get("dataset_family") or infer_family(path)
            if family not in {"GermanOR", "Mannino"}:
                continue
            rows.append({
                "dataset_family": family,
                "case_metadata_relative_path": path.relative_to(root).as_posix(),
                "base_instance_id": get_nested(meta, "base_instance_id"),
                "scenario": get_nested(meta, "publication_scenario", get_nested(meta, "scenario")),
                "source_injection_mode": (
                    get_nested(meta, "german_injection_mode") if family == "GermanOR"
                    else get_nested(meta, "mannino_injection_mode")
                ),
                "source_duration_mode": (
                    get_nested(meta, "german_duration_mode") if family == "GermanOR" else ""
                ),
                "source_duration_origin": (
                    get_nested(meta, "german_or_duration_source") if family == "GermanOR"
                    else get_nested(meta, "mannino_duration_source")
                ),
                "priority_policy": get_nested(meta, "priority_policy"),
                "publication_experiment_version": get_nested(meta, "publication_experiment_version"),
            })

    rows.sort(key=lambda x: (x["dataset_family"], x["case_metadata_relative_path"]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset_family", "case_metadata_relative_path", "base_instance_id", "scenario",
        "source_injection_mode", "source_duration_mode", "source_duration_origin",
        "priority_policy", "publication_experiment_version",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} instance provenance records to {out}")
    if rows:
        modes = sorted({(r["dataset_family"], r["source_injection_mode"], r["source_duration_mode"]) for r in rows})
        print("Observed source-injection settings:")
        for entry in modes:
            print("  ", entry)


if __name__ == "__main__":
    main()

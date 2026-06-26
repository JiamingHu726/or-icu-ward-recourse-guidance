#!/usr/bin/env python3
"""Locate candidate source-calibration/preprocessing scripts for release review."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

KEYWORDS = [
    "GermanOR", "Mannino", "MSS-Adjusts", "SINTEF", "Korzhenevich",
    "Zander", "zenodo.7147921", "case mix", "case_mix",
    "duration distribution", "duration_distribution",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="Root of the final research source tree.")
    parser.add_argument("--output", default="metadata/source_preprocessing_candidates.csv")
    args = parser.parse_args()

    root = Path(args.source_root)
    if not root.exists():
        raise SystemExit(f"Source root not found: {root}")

    rows = []
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        matched = [term for term in KEYWORDS if re.search(re.escape(term), text, flags=re.I)]
        if matched:
            rows.append({
                "relative_script_path": path.relative_to(root).as_posix(),
                "matched_terms": "|".join(matched),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_script_path", "matched_terms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidate preprocessing scripts to {out}")
    if not rows:
        print("WARNING: no source-specific calibration/preprocessing candidate was found.")


if __name__ == "__main__":
    main()

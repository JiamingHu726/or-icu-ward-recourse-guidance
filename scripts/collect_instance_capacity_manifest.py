#!/usr/bin/env python3
"""Build an auditable instance/capacity manifest from generated case directories.

The script records:
- source-family inference, n, seed, and scenario;
- distribution and nested generation metadata;
- daily ICU and ward capacity summaries;
- initial ICU/ward patient counts and transfer-ready counts;
- scenario capacity ratios relative to the matched nominal case.

The capacity ratios are empirical summaries derived from the released daily capacity
profiles. They are not treated as a separately declared generator parameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required; install the repository environment first.") from exc


N_VALUES = {50, 70, 100, 150}
SCENARIOS = {"nominal", "ward_pressure", "transfer_bottleneck"}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def nested_value(mapping: dict, key: str, default=""):
    """Read a field from top level, then common nested generator blocks."""
    if key in mapping:
        return mapping.get(key, default)
    for container in ("blocking_extension", "initial_state", "generation", "generator"):
        nested = mapping.get(container, {})
        if isinstance(nested, dict) and key in nested:
            return nested.get(key, default)
    return default


def numeric_summary(df: pd.DataFrame, candidates: list[str]) -> tuple[float | None, float | None, float | None]:
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.min()), float(values.max()), float(values.mean())
    return None, None, None


def infer_dataset(text: str) -> str:
    """Infer family from a directory segment or conventional source-root name."""
    normalized = text.replace("\\", "/")
    tests = [
        ("GermanOR", r"(?:^|/)GermanOR(?:/|$)|german[_-]"),
        ("Mannino", r"(?:^|/)Mannino(?:/|$)|mannino"),
        ("Synthetic", r"(?:^|/)Synthetic(?:/|$)|synthetic|publication_batch_instances"),
    ]
    for label, pattern in tests:
        if re.search(pattern, normalized, flags=re.I):
            return label
    return ""


def parse_path(case: Path) -> dict:
    text = str(case).replace("\\", "/")

    def capture(pattern: str) -> str:
        match = re.search(pattern, text, flags=re.I)
        return match.group(1) if match else ""

    return {
        "case_dir": str(case),
        "dataset": infer_dataset(text),
        "n": capture(r"case_(50|70|100|150)_seed"),
        "seed": capture(r"case_\d+_seed(\d+)"),
        "scenario": capture(r"/(nominal|ward_pressure|transfer_bottleneck)/"),
    }


def count_initial_state(case: Path) -> dict:
    out = {
        "initial_icu_patient_count": "",
        "initial_ward_patient_count": "",
        "initial_icu_ready_patient_count": "",
        "observed_ready_fraction_among_initial_icu": "",
    }
    icu = case / "current_icu.csv"
    ward = case / "current_ward.csv"

    if icu.exists():
        try:
            df = pd.read_csv(icu)
            out["initial_icu_patient_count"] = int(len(df))
            ready_col = next((c for c in ["ready_at_start", "ready_for_transfer", "icu_ready"] if c in df.columns), None)
            if ready_col is not None:
                ready = pd.to_numeric(df[ready_col], errors="coerce").fillna(0).astype(float)
                ready_count = int((ready > 0).sum())
                out["initial_icu_ready_patient_count"] = ready_count
                if len(df) > 0:
                    out["observed_ready_fraction_among_initial_icu"] = ready_count / len(df)
        except Exception:
            pass

    if ward.exists():
        try:
            out["initial_ward_patient_count"] = int(len(pd.read_csv(ward)))
        except Exception:
            pass
    return out


def capacity_metadata(case: Path) -> dict:
    """Read scenario note and daily capacities from publication metadata if present."""
    pmeta = read_json(case / "publication_instance_metadata.json")
    return {
        "publication_scenario_note": pmeta.get("scenario_note", ""),
        "icu_capacity_profile_json": json.dumps(pmeta.get("icu_capacity_by_day", {}), sort_keys=True),
        "ward_capacity_profile_json": json.dumps(pmeta.get("ward_capacity_by_day", {}), sort_keys=True),
    }


def enrich_relative_capacity(rows: list[dict]) -> None:
    """Add capacity multipliers relative to the matched nominal instance."""
    index: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if str(row.get("scenario", "")) == "nominal":
            index[(str(row.get("dataset", "")), str(row.get("n", "")), str(row.get("seed", "")))] = row

    measures = [
        "icu_capacity_min", "icu_capacity_mean", "icu_capacity_max",
        "ward_capacity_min", "ward_capacity_mean", "ward_capacity_max",
    ]
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("n", "")), str(row.get("seed", "")))
        nominal = index.get(key)
        for measure in measures:
            target_key = f"{measure}_multiplier_vs_nominal"
            if nominal is None:
                row[target_key] = ""
                continue
            try:
                baseline = float(nominal.get(measure, ""))
                current = float(row.get(measure, ""))
                row[target_key] = current / baseline if baseline != 0 else ""
            except (TypeError, ValueError):
                row[target_key] = ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="Case-root directories to scan.")
    parser.add_argument("--output", default="metadata/instance_capacity_manifest.csv")
    args = parser.parse_args()

    metadata_paths: set[Path] = set()
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"WARNING: root not found; skipped: {root_path}")
            continue
        metadata_paths.update(root_path.rglob("metadata.json"))

    rows: list[dict] = []
    for meta_path in sorted(metadata_paths):
        case = meta_path.parent
        meta = read_json(meta_path)
        row = parse_path(case)

        # Prefer values embedded in metadata; use parsed values as a fallback.
        row["n"] = meta.get("n", meta.get("I", row["n"]))
        row["seed"] = meta.get("seed", row["seed"])
        row["scenario"] = meta.get("publication_scenario", row["scenario"])
        row.update({
            "distribution": meta.get("distribution", ""),
            "generator_dataset_label": meta.get("dataset", ""),
            "initial_icu_occupancy_rate": nested_value(meta, "initial_icu_occupancy_rate"),
            "initial_ward_occupancy_rate": nested_value(meta, "initial_ward_occupancy_rate"),
            "ready_fraction_among_icu": nested_value(meta, "ready_fraction_among_icu"),
            "bed_setting_json": json.dumps(meta.get("bed_setting", meta.get("bed_setting_weekday_weekend", ""))),
        })
        row.update(count_initial_state(case))
        row.update(capacity_metadata(case))

        cap = case / "capacities.csv"
        if cap.exists():
            try:
                capacities = pd.read_csv(cap)
                for prefix, candidates in {
                    "icu_capacity": ["icu_capacity", "ICU_capacity", "icu_beds"],
                    "ward_capacity": ["ward_capacity", "Ward_capacity", "ward_beds"],
                }.items():
                    low, high, mean = numeric_summary(capacities, candidates)
                    row[f"{prefix}_min"] = low
                    row[f"{prefix}_max"] = high
                    row[f"{prefix}_mean"] = mean
            except Exception as exc:
                row["capacity_read_error"] = repr(exc)

        rows.append(row)

    enrich_relative_capacity(rows)
    rows.sort(key=lambda r: (
        str(r.get("dataset", "")), int(r.get("n", 0) or 0),
        str(r.get("scenario", "")), int(r.get("seed", 0) or 0),
    ))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["case_dir"]
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    if not frame.empty:
        print("\\nRows by dataset/scenario/n:")
        print(frame.groupby(["dataset", "scenario", "n"], dropna=False).size().to_string())
        missing = [
            c for c in [
                "initial_icu_occupancy_rate", "initial_ward_occupancy_rate",
                "ready_fraction_among_icu", "icu_capacity_mean", "ward_capacity_mean",
            ] if c in frame.columns and frame[c].replace("", pd.NA).isna().any()
        ]
        if missing:
            print(f"WARNING: missing values remain in: {', '.join(missing)}")
        else:
            print("Core generator, state, and capacity fields are complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect benchmark-calibration scripts and local dependencies for public release.

This utility targets the source-specific pipeline identified by the repository scan:
- build_calibration_source_table.py
- prepare_german_or_patient_pool.py
- batch_generate_german_publication_instances.py
- prepare_mannino_stats.py
- generate_mannino_patient_pool.py
- batch_generate_mannino_publication_instances.py

It copies only project source code and recursively discovered local Python modules.
It does not copy raw third-party data. It also emits a list of referenced data/config
files so the release author can distinguish shareable frozen calibration tables from
restricted upstream source records.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import re
import shutil
from collections import deque
from pathlib import Path


TARGETS = [
    "build_calibration_source_table.py",
    "prepare_german_or_patient_pool.py",
    "batch_generate_german_publication_instances.py",
    "prepare_mannino_stats.py",
    "generate_mannino_patient_pool.py",
    "batch_generate_mannino_publication_instances.py",
]
DATA_REF = re.compile(r"""['"]([^'"]+\.(?:csv|tsv|xlsx|xls|json|ya?ml|txt))['"]""", re.I)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module_candidates(module: str, current: Path, root: Path) -> list[Path]:
    if not module:
        return []
    rel = Path(*module.split("."))
    options = [
        current.parent / f"{rel}.py",
        current.parent / rel / "__init__.py",
        root / f"{rel}.py",
        root / rel / "__init__.py",
    ]
    # Deduplicate while preserving meaningful resolution order.
    seen = set()
    unique = []
    for path in options:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def local_imports(path: Path, root: Path) -> set[Path]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    found = set()
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                for candidate in module_candidates(module, path, root):
                    if candidate.exists():
                        found.add(candidate.resolve())
                        break
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # Resolve relative import module against the importing file.
                base = path.parent
                for _ in range(max(node.level - 1, 0)):
                    base = base.parent
                if module:
                    candidates = [base / f"{Path(*module.split('.'))}.py",
                                  base / Path(*module.split('.')) / "__init__.py"]
                else:
                    candidates = []
                for candidate in candidates:
                    if candidate.exists():
                        found.add(candidate.resolve())
                        break
            else:
                for candidate in module_candidates(module, path, root):
                    if candidate.exists():
                        found.add(candidate.resolve())
                        break
    return found


def role_for(path: Path) -> str:
    return "source_preprocessing" if path.name in TARGETS else "local_dependency"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="Root of the original project source tree.")
    parser.add_argument("--output-src", default="src/benchmark_calibration")
    parser.add_argument("--output-metadata", default="metadata")
    parser.add_argument("--no-dependencies", action="store_true",
                        help="Copy only the six targeted scripts; do not follow local imports.")
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    if not root.exists():
        raise SystemExit(f"Source root not found: {root}")

    output_src = Path(args.output_src)
    output_meta = Path(args.output_metadata)
    output_src.mkdir(parents=True, exist_ok=True)
    output_meta.mkdir(parents=True, exist_ok=True)

    targets = {}
    for name in TARGETS:
        matches = sorted(root.rglob(name))
        # Avoid recursively selecting an existing release copy.
        matches = [m for m in matches if output_src.resolve() not in m.resolve().parents]
        if not matches:
            print(f"WARNING: target not found: {name}")
        elif len(matches) > 1:
            print(f"WARNING: multiple matches for {name}; using {matches[0].relative_to(root)}")
            targets[name] = matches[0].resolve()
        else:
            targets[name] = matches[0].resolve()

    queue = deque(targets.values())
    selected = set(targets.values())
    if not args.no_dependencies:
        while queue:
            current = queue.popleft()
            for dep in local_imports(current, root):
                # Only include project-local files under the specified source root.
                try:
                    dep.relative_to(root)
                except ValueError:
                    continue
                if dep not in selected:
                    selected.add(dep)
                    queue.append(dep)

    manifest_rows = []
    reference_rows = []
    for source in sorted(selected):
        rel = source.relative_to(root)
        destination = output_src / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest_rows.append({
            "role": role_for(source),
            "source_relative_path": rel.as_posix(),
            "release_relative_path": destination.as_posix(),
            "sha256": digest(source),
            "bytes": source.stat().st_size,
        })

        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for item in sorted(set(DATA_REF.findall(text))):
            reference_rows.append({
                "referencing_script": rel.as_posix(),
                "referenced_file_literal": item,
                "release_action": (
                    "Review manually: deposit only frozen calibration/config artifacts that "
                    "may be shared; do not copy restricted raw third-party data."
                ),
            })

    with (output_meta / "benchmark_calibration_release_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["role", "source_relative_path", "release_relative_path", "sha256", "bytes"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (output_meta / "benchmark_calibration_file_references.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["referencing_script", "referenced_file_literal", "release_action"],
        )
        writer.writeheader()
        writer.writerows(reference_rows)

    readme = """# Benchmark-calibration source bundle

This directory was generated by `scripts/collect_benchmark_calibration_release_files.py`.

It contains the source-specific Mannino and GermanOR calibration / patient-pool /
instance-generation scripts, together with recursively detected project-local Python
dependencies. It does not include raw third-party source records.

Review `metadata/benchmark_calibration_file_references.csv` before public release.
For each referenced table or config, decide whether it is:
1. a shareable frozen calibration artifact that must be committed; or
2. restricted upstream data that must be retrieved from its original provider.
"""
    (output_src / "README.md").write_text(readme, encoding="utf-8")

    print(f"Copied {len(manifest_rows)} Python files to {output_src}")
    print(f"Wrote {output_meta / 'benchmark_calibration_release_manifest.csv'}")
    print(f"Wrote {output_meta / 'benchmark_calibration_file_references.csv'}")


if __name__ == "__main__":
    main()

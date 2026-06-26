#!/usr/bin/env python3
"""Collect normalized Gurobi version/thread evidence from archived stdout logs."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

VERSION = re.compile(r"Gurobi Optimizer version (?P<version>.+)")
THREADS = re.compile(
    r"Thread count:\s+(?P<physical>\d+)\s+physical cores,\s+"
    r"(?P<logical>\d+)\s+logical processors,\s+using up to\s+(?P<used>\d+)\s+threads"
)
EXPLICIT = re.compile(r"Set parameter Threads to value (?P<threads>\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory containing archived stdout/log files")
    parser.add_argument("--output", default="metadata/solver_log_evidence.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".log"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        version = None
        physical = logical = used = explicit = None
        for line in lines:
            mv = VERSION.search(line)
            mt = THREADS.search(line)
            me = EXPLICIT.search(line)
            if mv:
                version = mv.group("version")
            if mt:
                physical = mt.group("physical")
                logical = mt.group("logical")
                used = mt.group("used")
            if me:
                explicit = me.group("threads")
        if version or used or explicit:
            rows.append({
                "relative_log_path": path.relative_to(root).as_posix(),
                "gurobi_version_line": version or "",
                "physical_cores": physical or "",
                "logical_processors": logical or "",
                "observed_thread_limit": used or "",
                "explicit_threads_parameter": explicit or "",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "relative_log_path", "gurobi_version_line", "physical_cores",
            "logical_processors", "observed_thread_limit", "explicit_threads_parameter",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} with {len(rows)} solver-log records.")


if __name__ == "__main__":
    main()

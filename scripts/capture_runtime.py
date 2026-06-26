#!/usr/bin/env python3
"""Capture runtime, package, hardware, and documented solver-thread policy."""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def powershell_json(command: str):
    if platform.system() != "Windows" or shutil.which("powershell") is None:
        return None
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return json.loads(raw) if raw else None
    except Exception:
        return None


def hardware() -> dict:
    record = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        record["physical_cpu_count"] = psutil.cpu_count(logical=False)
        record["memory_bytes"] = psutil.virtual_memory().total
    except Exception:
        record["physical_cpu_count"] = None
        record["memory_bytes"] = None

    record["windows_processor"] = powershell_json(
        "Get-CimInstance Win32_Processor | "
        "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
    )
    record["windows_memory"] = powershell_json(
        "Get-CimInstance Win32_ComputerSystem | "
        "Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress"
    )
    return record


def gurobi_info() -> dict:
    info = {"gurobipy_version": package_version("gurobipy"), "gurobi_runtime_version": None}
    try:
        import gurobipy as gp
        info["gurobi_runtime_version"] = ".".join(str(x) for x in gp.gurobi.version())
    except Exception as exc:
        info["gurobi_import_error"] = repr(exc)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="metadata")
    parser.add_argument(
        "--historical-thread-parameter",
        default="NOT_EXPLICITLY_SET",
        help="Use NOT_EXPLICITLY_SET unless the archived code explicitly set model.Params.Threads.",
    )
    parser.add_argument(
        "--observed-auto-thread-limit",
        type=int,
        default=16,
        help="Observed upper limit from archived Gurobi logs; use 16 for this release.",
    )
    args = parser.parse_args()

    thread_vars = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "packages": {
            name: package_version(name)
            for name in ["numpy", "pandas", "matplotlib", "gurobipy", "PyYAML", "psutil"]
        },
        "solver": gurobi_info(),
        "thread_environment": {name: os.getenv(name, "NOT_SET") for name in thread_vars},
        "historical_solver_execution": {
            "threads_parameter": args.historical_thread_parameter,
            "selection_mode": "Gurobi automatic thread selection",
            "observed_effective_thread_limit": args.observed_auto_thread_limit,
        },
        "hardware": hardware(),
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'runtime_manifest.json'}")


if __name__ == "__main__":
    main()

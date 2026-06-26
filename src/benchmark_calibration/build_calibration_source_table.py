#!/usr/bin/env python3
from __future__ import annotations
"""Build a traceable calibration/source table for the manuscript appendix."""
import argparse
from pathlib import Path
import pandas as pd

ROWS = [
    ("OR duration and case-mix", "Direct / derived", "German OR Benchmarking process-time distributions or Mannino/SINTEF duration pool", "Empirical or distributional OR-duration injection; source mode recorded per instance", "OR-side source condition"),
    ("Base specialty mix, block schedule, ICU/ward LOS moments", "Derived", "Appendix-G benchmark statistics and block templates", "Common semi-synthetic benchmark generator; exact source stats retained in type_stats.csv", "Common downstream baseline"),
    ("Priority class and due-date windows", "Calibrated", "Policy-based synthetic urgency generator", "Exact high/medium/low shares assigned independently of realised duration, ICU LOS, and ward LOS; due windows depend on policy tier", "Primary access policy"),
    ("Surgeon eligibility and availability", "Calibrated", "Synthetic execution-layer generator", "Specialty-compatible teams, seeded calendars, and workload limits", "Execution feasibility"),
    ("Equipment requirements and capacities", "Calibrated", "Synthetic execution-layer generator", "Procedure-resource mapping and seeded capacity table", "Execution feasibility"),
    ("Initial ICU/ward occupancy", "Stress / calibrated", "Blocking-state extension", "Initial ICU occupancy 0.80, ward occupancy 0.90, and ICU-ready fraction 0.50 in publication generator", "Initial downstream load"),
    ("Nominal scenario", "Scenario", "Publication scenario definition", "No capacity reduction", "Primary"),
    ("Transfer bottleneck", "Scenario", "Publication scenario definition", "Ward capacity multiplied by 0.75 on planning days 3--5", "Primary paired stress test"),
    ("Ward pressure", "Scenario", "Publication scenario definition", "Ward capacity multiplied by 0.90 across the horizon", "Sensitivity only"),
]

def latex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="calibration_source_table")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS, columns=["Parameter category", "Source type", "Source / calibration basis", "Generation rule", "Analysis role"])
    df.to_csv(out / "calibration_source_traceability.csv", index=False)
    lb = "\\\\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Calibration and source traceability for the semi-synthetic OR--ICU--ward instance generator.}",
        r"\label{tab:calibration-source}",
        r"\scriptsize",
        r"\begin{tabular}{p{0.18\textwidth}p{0.11\textwidth}p{0.24\textwidth}p{0.31\textwidth}p{0.11\textwidth}}",
        r"\toprule",
        "Parameter category & Source type & Source / calibration basis & Generation rule & Analysis role " + lb,
        r"\midrule",
    ]
    for row in ROWS:
        lines.append(" & ".join(latex_escape(x) for x in row) + " " + lb)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "calibration_source_traceability.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'calibration_source_traceability.tex'}")

if __name__ == '__main__':
    main()

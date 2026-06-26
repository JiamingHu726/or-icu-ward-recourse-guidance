#!/usr/bin/env python3
"""Check core source modules required by the final reproducibility scope."""
from pathlib import Path

required = [
    'src/core/surgery_schedule_evaluator.py',
    'src/core/stage2_priority_soft_gurobi_repair_v3_fixed.py',
    'src/core/shehadeh_style_integrated_mip_baseline_v2_fixed.py',
    'src/core/stage3_icu_ward_blocking_flow_mip_fixed.py',
    'src/core/pr_glns_spiral_or_icu_ward_guidance_mode.py',
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    print('CORE CODE INCOMPLETE:')
    print('\n'.join(f' - {p}' for p in missing))
    raise SystemExit(2)
print('Core source dependencies present.')

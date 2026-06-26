#!/usr/bin/env python3
from __future__ import annotations

"""
publication_experiment_config.py

Publication-level configuration for the OR-ICU-Ward computational study.
This file deliberately separates public experiment names from internal code
folder names. Internal folder names can remain unchanged; tables and figures
should use PUBLIC_METHOD_LABELS.
"""

from pathlib import Path

# Candidate-pool sizes. These are not performed weekly surgeries; they are the
# elective patient pools considered by the weekly planning model.
PUBLICATION_SIZES = [50, 70, 100, 150]
PUBLICATION_SEEDS = [7, 11, 19, 23, 29, 31, 37, 41, 43, 47]
PUBLICATION_SCENARIOS = ["nominal", "ward_pressure", "transfer_bottleneck"]

# Analysis roles are frozen before the final batch is run.  The primary analysis
# uses matched nominal and transfer-bottleneck scenarios.  Ward pressure is a
# prespecified sensitivity scenario and must not be pooled into primary tests.
PRIMARY_SCENARIOS = ["nominal", "transfer_bottleneck"]
SENSITIVITY_SCENARIOS = ["ward_pressure"]
SCENARIO_ANALYSIS_ROLE = {
    "nominal": "primary",
    "transfer_bottleneck": "primary",
    "ward_pressure": "sensitivity",
}

# The primary publication setting uses policy-based synthetic urgency that is
# deliberately independent of realised duration, ICU LOS, and ward LOS.  This
# avoids building the downstream trade-off into the priority label itself.
PUBLICATION_PRIORITY_POLICY = "independent_urgency"
PUBLICATION_PRIORITY_SHARES = {"high": 0.25, "medium": 0.45, "low": 0.30}
PUBLICATION_HEPATOBILIARY_PROXY = False
PUBLICATION_EXPERIMENT_VERSION = "priority-independent-temporal-guidance-v1"

# Bed settings are scaled from the tuned case_70 setting ICU=10, Ward=7.
# The n=150 setting should be interpreted as a larger candidate pool rather
# than a claim that all 150 patients are scheduled.
PUBLICATION_BED_SETTINGS = {
    50: (8, 6),
    70: (10, 7),
    100: (13, 9),
    150: (17, 12),
}

PUBLICATION_INSTANCE_ROOT = Path("publication_batch_instances")
PUBLICATION_RESULT_ROOT = Path("publication_batch_results")
PUBLICATION_TABLE_ROOT = Path("publication_tables")
PUBLICATION_FIGURE_ROOT = Path("publication_figures")

# Parameters reused from the current implementation.
BA_HLA_PARAMS = {
    "iterations": 18,
    "trials_per_iteration": 10,
    "refill_rounds": 8,
    "eps_block": 8.0,
    "eps_bed": 15.0,
    "eps_violation": 8.0,
}

EXECUTION_REPAIR_PARAMS = {
    "slot_minutes": 30,
    "max_or_overtime": 60,
    "time_limit": 300,
    "mip_gap": 0.03,
    "high_drop_limit": 0,
    "max_total_drop": 2,
    "objective_mode": "volume_first",
}

TRANSFER_RECOURSE_PARAMS = {
    "time_limit": 300,
    "mip_gap": 0.01,
    "objective_mode": "capacity_first",
    "allow_ward_excess": True,
}

LNS_PARAMS = {
    "cycles": 6,
    "proposals_per_cycle": 80,
    "exact_every": 20,
    "exact_top_k": 5,
    "max_or_overtime": 210,
}

UNGUIDED_LNS_PARAMS = {
    "iterations": 200,
}

# Internal folder names retained for compatibility with the existing code.
METHOD_SPECS = [
    {
        "internal": "01_ba_hla_v41",
        "public": "Initial access heuristic",
        "short": "Initial",
        "category": "warm_start",
    },
    {
        "internal": "02_stage2_v3_volume_first",
        "public": "Execution-repaired baseline",
        "short": "Execution repair",
        "category": "baseline",
    },
    {
        "internal": "04_hp_forced_ot_feedback",
        "public": "Recourse-feedback repair",
        "short": "Feedback repair",
        "category": "proposed",
    },
    {
        "internal": "07_v41_weak_pr_glns",
        "public": "Unguided LNS control",
        "short": "Unguided LNS",
        "category": "ablation",
    },
    {
        "internal": "06_downstream_aggressive_spiral_pr_glns",
        "public": "Recourse-guided LNS",
        "short": "Guided LNS",
        "category": "intensification",
    },
    {
        "internal": "10_shehadeh_adaptive_access",
        "public": "Adaptive integrated MIP",
        "short": "Adaptive MIP",
        "category": "benchmark",
    },
    {
        "internal": "10_shehadeh_adaptive_free",
        "public": "Free-admission integrated MIP",
        "short": "Free MIP",
        "category": "sensitivity",
    },
]

PUBLIC_METHOD_LABELS = {m["internal"]: m["public"] for m in METHOD_SPECS}
PUBLIC_METHOD_SHORT_LABELS = {m["internal"]: m["short"] for m in METHOD_SPECS}
METHOD_CATEGORIES = {m["internal"]: m["category"] for m in METHOD_SPECS}

# Ordering for tables and figures.
PUBLIC_METHOD_ORDER = [
    "Execution-repaired baseline",
    "Recourse-feedback repair",
    "Unguided LNS control",
    "Recourse-guided LNS",
    "Adaptive integrated MIP",
    "Free-admission integrated MIP",
]

SCENARIO_LABELS = {
    "nominal": "Nominal",
    "ward_pressure": "Ward pressure",
    "transfer_bottleneck": "Transfer bottleneck",
}

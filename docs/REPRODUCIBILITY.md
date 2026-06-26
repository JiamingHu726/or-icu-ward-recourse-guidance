# Reproducibility protocol

## What the final release reproduces

1. Seed-clustered M1 figure and its exact/Monte-Carlo Page trend tests.
2. Seed-clustered M2 heatmaps and the n=150 blocked-transfer / high-priority-access scatter.
3. M3 scalar-pricing schedule-identity panel from SHA-256 audit records.
4. Baseline-load validation and fixed-schedule weight-sensitivity tables.

## Statistical unit

The primary unit is the **base-seed cluster**. For a family, candidate-pool size, and base seed, paired differences are averaged across the matched `nominal` and `transfer_bottleneck` scenarios before cross-seed summaries are computed. The `ward_pressure` scenario is a prespecified sensitivity scenario and is not pooled into the primary analyses.

## What is deliberately not promised

The release does not offer a one-command rerun of every historical raw pipeline. The historical monolithic batch runner depended on local modules not required for the final evidence, and it was removed to avoid a misleading reproducibility claim. The release instead supports reproducible regeneration of the final figures/tables from deposited derived data, configurations, and selected-schedule audits.

## Commands

See the commands in the root `README.md`. Run:

```bash
python scripts/check_core_source_dependencies.py
python scripts/check_release.py
```

Use `python scripts/check_release.py --public` only after replacing metadata placeholders.

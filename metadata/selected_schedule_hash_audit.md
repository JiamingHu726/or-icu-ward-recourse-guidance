# Selected-schedule hash audit

## Scope

This audit has two scopes:

1. **Primary M3 scalar-price audit (complete):** GermanOR and Mannino, with the five scalar-price arms
   `price_cal_w0`, `price_cal_w0p25`, `price_cal_w0p5`, `price_cal_w1`, and `price_cal_w2`.
2. **Auxiliary Synthetic A-effect audit (partial):** `price_off_clean` versus `A_only` only. It does not
   contain the Synthetic `B_only` or `AB_*` schedules and must not be interpreted as a full Synthetic
   price-factorial schedule audit.

## Primary M3 result: complete schedule-level equality evidence

- Real price-calibration cells: **160**
  - 2 benchmark-calibrated families: GermanOR and Mannino;
  - 4 candidate-pool sizes: 50, 70, 100, and 150;
  - 2 scenarios: nominal and transfer bottleneck;
  - 10 seeds per dataset-size-scenario cell.
- Expected scalar-price arms per cell: **5**
- Real selected schedule files hashed: **800**
- Missing real price arms: **0**
- Cells with exactly five real price arms: **160 / 160**
- Cells with one unique canonical schedule hash across all five price arms: **160 / 160**
- Non-reference comparisons to `price_cal_w0`: **640**
- Raw byte-level ties: **640 / 640**
- Canonical schedule ties: **640 / 640**
- Non-OK comparison records: **0**

Therefore, within every audited real calibration cell, the final selected schedule is identical across
the zero-price and all four positive scalar-price multipliers. This is the schedule-level evidence
underlying the M3 statement that scalar pressure pricing did not change the selected schedules in
the audited real price-calibration sweep.

## Auxiliary Synthetic A-effect audit: intentionally partial

The scan also found **40** Synthetic schedule files in **20** cells. Each such cell contains only
`price_off_clean` and `A_only`, yielding two distinct canonical schedules per cell. The original
v7 script reported absent `B_only` and `AB_*` paths as `missing_arm` because those result roots
were not supplied to that scan. Those rows are not evidence of failed experiments or missing
manuscript data; they only show that the scan was scoped to the A-only output root.

The partial Synthetic comparison is retained under:

```text
metadata/auxiliary/synthetic_Aonly_vs_control_schedule_equality_partial.csv
```

It is not a primary release requirement and should not be used to claim complete schedule-level
coverage of all Synthetic factorial arms.

## Files retained

```text
metadata/selected_schedule_hashes.csv
metadata/selected_schedule_hash_group_summary.csv
metadata/price_arm_schedule_equality.csv
metadata/selected_schedule_hash_audit.md
metadata/auxiliary/synthetic_Aonly_vs_control_schedule_equality_partial.csv
```

## Hash interpretation

- `raw_sha256` checks byte-for-byte identity.
- `canonical_sha256` checks equality after CSV row-order, column-order, and incidental whitespace
  normalization.
- The primary M3 conclusion holds under both definitions.

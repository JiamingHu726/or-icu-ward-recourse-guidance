# Synthetic A/B/clean factorial scripts

Files:

- `pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py`
  - patched spiral module;
  - adds `guidance_mode='operator_off'` for B-only;
  - keeps `spiral_pressure_cost` alive for B-only;
  - obeys `RECOURSE_PRESSURE_WEIGHT` in price/operator-off modes;
  - logs `best_stage3_ward_excess`, `best_pressure_cost`, `target_volume`, and `high_target` into `spiral_trace.csv`.

- `run_synthetic_abclean_factorial.py`
  - runner for `price_off_clean`, `A_only`, `B_only`, `AB_w0p25`, `AB_w0p5`, `AB_w1`, `AB_w2`;
  - writes per-run `abclean_run_manifest.json` and global `synthetic_abclean_run_log.csv`.

- `analyze_synthetic_abclean_factorial.py`
  - strict iso-60 analyzer;
  - compares treatment arms against the existing Synthetic off baseline;
  - outputs clean gate, A effect, B-only effect, B dose effect, A-only-vs-off check, and Synthetic Level-0 slope.

## Recommended run order

### 1) Clean gate + A-only smoke

```powershell
python run_synthetic_abclean_factorial.py `
  --sizes 100 150 `
  --seeds 7 `
  --scenarios nominal `
  --arms price_off_clean A_only `
  --spiral-script pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py `
  --output-root synthetic_abclean_smoke `
  --synthetic-instance-root publication_batch_instances `
  --synthetic-anchor-root publication_batch_results `
  --continue-on-error
```

Analyze:

```powershell
python analyze_synthetic_abclean_factorial.py `
  --ab-root synthetic_abclean_smoke `
  --off-root spiral_price_off_all_results_v2 `
  --output-dir synthetic_abclean_smoke_analysis `
  --sizes 100 150 `
  --seeds 7 `
  --scenarios nominal
```

If the existing Synthetic off baseline is not under `spiral_price_off_all_results_v2`, rerun the analyzer with the correct `--off-root`.

### 2) Estimate Synthetic pressure slope

Check:

```text
synthetic_abclean_smoke_analysis/level0_pressure_regression_synth.csv
```

If `R2_syn < 0.2`, do not prioritize B; run only `price_off_clean` and `A_only` full, or run B arms only as a labeled negative check.

### 3) B/AB smoke with Synthetic slope

Use the slope from the previous output:

```powershell
python run_synthetic_abclean_factorial.py `
  --sizes 100 150 `
  --seeds 7 `
  --scenarios nominal `
  --arms B_only AB_w0p25 AB_w0p5 AB_w1 AB_w2 `
  --pressure-weight-unit <slope_syn> `
  --spiral-script pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py `
  --output-root synthetic_abclean_smoke `
  --synthetic-instance-root publication_batch_instances `
  --synthetic-anchor-root publication_batch_results `
  --continue-on-error
```

Analyze again with the same analyzer command.

### 4) Full run

Only after the smoke and clean gate pass:

```powershell
python run_synthetic_abclean_factorial.py `
  --sizes 50 70 100 150 `
  --seeds 7 11 19 23 29 31 37 41 43 47 `
  --scenarios nominal transfer_bottleneck `
  --arms price_off_clean A_only B_only AB_w0p25 AB_w0p5 AB_w1 AB_w2 `
  --pressure-weight-unit <slope_syn> `
  --spiral-script pr_glns_spiral_or_icu_ward_guidance_mode_abclean.py `
  --output-root synthetic_abclean_factorial_results `
  --synthetic-instance-root publication_batch_instances `
  --synthetic-anchor-root publication_batch_results `
  --continue-on-error
```

Then:

```powershell
python analyze_synthetic_abclean_factorial.py `
  --ab-root synthetic_abclean_factorial_results `
  --off-root spiral_price_off_all_results_v2 `
  --output-dir synthetic_abclean_factorial_analysis
```

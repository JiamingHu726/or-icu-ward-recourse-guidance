# A-effect inferential statistics package

This package implements Claude's inferential-statistics specification for the A-effect.

## Main script

```text
analyze_A_effect_inferential_stats.py
```

## Recommended command

```powershell
$PY = "E:\anaconda3\python.exe"

& $PY analyze_A_effect_inferential_stats.py `
  --input after_clean_gate_attribution\A_effect_w0_vs_clean_detailed.csv `
  --input synthetic_abclean_Aonly_full_analysis_v2\A_effect_w0_vs_clean_detailed_synth.csv `
  --output-dir A_effect_inferential_stats `
  --bootstrap 10000 `
  --jt-permutations 20000 `
  --seed 202706 `
  --p-adjust holm
```

## Outputs

```text
A_effect_input_standardized.csv
A_effect_inferential_stats.csv
A_effect_trend_by_dataset.csv
A_effect_median_gap_bootstrap_CI.png
A_effect_inferential_stats_report.md
```

## Notes

- `gap > 0`: A is better than clean.
- GermanOR/Mannino use one-sided Wilcoxon signed-rank tests.
- Synthetic uses one-sided sign tests because heavy-tailed/bimodal effects violate the Wilcoxon symmetry assumption.
- Synthetic can be statistically positive but still labelled fragile when the distribution is bimodal/heavy-tailed.
- No Cohen's d is reported.

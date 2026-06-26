# Synthetic tail and mechanism diagnostics

This package implements `synthetic_tail_and_mechanism_diagnostics.md`.

## Script

```text
synthetic_tail_and_mechanism_diagnostics.py
```

## Required Synthetic input

Use the detailed A-only-vs-clean file from the v2 analyzer, preferably:

```text
synthetic_abclean_Aonly_full_analysis_v2\A_effect_w0_vs_clean_detailed_synth.csv
```

or a same-schema file such as:

```text
final_pairwise_iso60_detailed_synth.csv
```

The script infers the gap column, component columns, base/clean columns, and treatment/A columns when possible.

## Optional real-data input for Part 2

To run the full cross-dataset mechanism reconciliation, pass the GermanOR/Mannino detailed file:

```text
final_pairwise_iso60_detailed.csv
```

If GermanOR and Mannino are in separate files, pass `--real-detailed` twice.

Without real data, the script still produces Synthetic Part 1 and Synthetic-only Part 2 outputs, but C1/unified-framework decisions are marked incomplete.

## PowerShell command

```powershell
$PY = "E:\anaconda3\python.exe"

& $PY synthetic_tail_and_mechanism_diagnostics.py `
  --synthetic-detailed synthetic_abclean_Aonly_full_analysis_v2\A_effect_w0_vs_clean_detailed_synth.csv `
  --real-detailed real_abclean_analysis\final_pairwise_iso60_detailed.csv `
  --output-dir synthetic_tail_mechanism_diagnostics
```

Synthetic-only version:

```powershell
$PY = "E:\anaconda3\python.exe"

& $PY synthetic_tail_and_mechanism_diagnostics.py `
  --synthetic-detailed synthetic_abclean_Aonly_full_analysis_v2\A_effect_w0_vs_clean_detailed_synth.csv `
  --output-dir synthetic_tail_mechanism_diagnostics_synth_only
```

## Main outputs

Part 1:

```text
synth_tail_per_instance.csv
synth_tail_driver_summary.csv
synth_gap_distribution_stats.csv
synth_tail_predictor_rules.csv
synth_tail_decision_tree.csv
synth_selection_artifact_check.csv
synth_gap_distribution_n150.png
```

Part 2:

```text
mechanism_C1_downstream_signs.csv
mechanism_C2_violation_signs.csv
mechanism_coupling_scatter_data.csv
mechanism_coupling_correlations.csv
mechanism_binding_profile.csv
mechanism_tail_link_summary.csv
mechanism_delta_blocked_delta_viol_scatter.png
diagnostic_decision_report.md
```

## Interpretation

- `gap > 0`: A_only is better than clean.
- `gap_component__exact_X > 0`: A improves component X.
- `delta_blocked = base_exact_blocked - treatment_exact_blocked`; positive means A reduces blocking.
- `delta_viol = treatment_exact_violation - base_exact_violation`; positive means A increases violations.
- `extra_high_missed_by_A = treatment_exact_high_deficit - base_exact_high_deficit`; positive means A misses more high-priority patients.

## Caveat

`mechanism_binding_profile.csv` reports raw baseline component medians. It does not invent capacity-normalized quantities unless the detailed files already contain enough information. The normalization status is therefore marked `raw_only_no_capacity_metadata`.

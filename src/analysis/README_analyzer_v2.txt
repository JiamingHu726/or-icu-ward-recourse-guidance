Synthetic A/B/clean analyzer v2
===============================

Use this file instead of the previous analyze_synthetic_abclean_factorial.py.

Key changes:
1. Clean gate official check no longer uses iso-60.
   - clean_gate_final_summary_synth.csv: official clean gate
   - clean_gate_common_iteration_summary_synth.csv: robustness check
   - clean_gate_iso60_diagnostic_summary_synth.csv: diagnostic only

2. Treatment effects are compared against matched new price_off_clean at iso-60.
   - A_effect_w0_vs_clean_summary_synth.csv
   - B_only_effect_vs_clean_summary_synth.csv
   - B_dose_effect_vs_w0_summary_synth.csv

3. Level-0 Synthetic pressure regression uses treatment - price_off_clean, not treatment - old off.
   - level0_pressure_regression_synth.csv

4. final_pairwise_iso60_detailed_synth.csv is kept as an alias, but now means vs-clean official pairwise.
   For clarity, prefer final_pairwise_iso60_vs_clean_detailed_synth.csv.

Example:
python analyze_synthetic_abclean_factorial_v2.py ^
  --ab-root synthetic_abclean_Aonly_check ^
  --off-root spiral_price_off_all_results_v2 ^
  --output-dir synthetic_abclean_Aonly_check_analysis_v2 ^
  --sizes 100 150 ^
  --seeds 7 11 19 23 29 ^
  --scenarios nominal transfer_bottleneck

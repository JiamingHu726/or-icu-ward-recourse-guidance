# OR--ICU--Ward Recourse Guidance: Reproducibility Package

This repository supports the manuscript:

> **When Does Downstream Recourse Guidance Improve Surgical Planning? Evidence Across Candidate-Pool Load Regimes**

It contains the final figure-generation code, derived result tables, source-provenance records, configuration files, selected-schedule hash audits, and benchmark-calibrated generated artifacts needed to reproduce the paper's reported computational evidence.

## Release status

This repository is prepared for public archival release `v1.0.0`.

* Repository: https://github.com/JiamingHu726/or-icu-ward-recourse-guidance
* Intended archival platform: Zenodo
* License: BSD 3-Clause License
* Public release status: pending final metadata review and Zenodo archival DOI

The repository will be made public immediately before the archival release and formal manuscript submission. The Zenodo version DOI will be added after the `v1.0.0` archive is created.


## Scope of reproducibility

The release supports reproduction of the final M1--M3 figures, baseline-load validation table, fixed-schedule weight-sensitivity table, and scalar-pricing schedule-identity audit. It does **not** claim to re-run every historical experiment from raw benchmark downloads in a single command. The original historical batch runner was excluded because it depends on frozen private/local modules that are not part of the final manuscript evidence chain.

## Main evidence files

| Evidence | Location |
|---|---|
| M1 seed-clustered load response | `figures/M1_seed_clustered_load_response.pdf`; `results/derived_tables/M1_seed_clustered_*.csv` |
| M2 mechanism and access trade-off | `figures/M2_mechanism_decomposition_and_access_tradeoff.pdf`; `results/derived_tables/M2_seed_clustered_*.csv` |
| M3 attribution validity | `figures/M3_attribution_validity.pdf`; `metadata/price_arm_schedule_equality.csv` |
| Baseline-load validation | `results/derived_tables/baseline_load_table_values.csv` |
| Fixed-schedule weight sensitivity | `results/derived_tables/weight_sensitivity_table_values.csv` |
| Exact objective weights | `config/objective_weights.json` |
| Algorithm settings | `config/algorithm_parameters.yml` |
| Source-data / calibration provenance | `metadata/source_data_manifest.csv`; `docs/DATA_PROVENANCE.md` |

## Quick start

Create the documented environment, then reproduce the final figures:

```bash
python src/figures/make_M1_seed_clustered.py \
  --input results/derived_tables/A_effect_input_standardized.csv \
  --output-dir reproduced_figures/M1

python src/figures/make_M2_seed_clustered_mechanism.py \
  --effects results/derived_tables/all_standardized_effect_rows.csv \
  --output-dir reproduced_figures/M2

python src/figures/make_M3_attribution_validity.py \
  --alignment results/derived_tables/M3_zero_intervention_alignment.csv \
  --hash-audit metadata/price_arm_schedule_equality.csv \
  --output-dir reproduced_figures/M3
```

The M1 and M2 scripts require `numpy`, `pandas`, `matplotlib`, and `scipy`. M3 requires `pandas` and `matplotlib`.

## Public data policy

- **GermanOR:** frozen duration pools are included with CC-BY-4.0 attribution; see `data/derived/frozen_calibration/GermanOR/ATTRIBUTION.md`.
- **Mannino:** raw workbook and derived pools are intentionally not redistributed because no explicit redistribution licence was confirmed. This repository provides retrieval instructions, deterministic preprocessing/generation code, and SHA-256 verification manifests.
- **Synthetic:** generator configuration and derived study artifacts are public in this repository.

See `docs/PUBLIC_DATA_ACCESS_POLICY.md` and `DATA_CODE_AVAILABILITY.md`.

## License

Project code is distributed under the BSD 3-Clause License. Third-party benchmark data retain their original terms and are not relicensed by this repository.

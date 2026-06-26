# Source-code layout

- `core/`: scheduling evaluator, repair, transfer-recourse, and PR-GLNS source modules used by the final implementation.
- `benchmark_calibration/`: source-specific GermanOR/Mannino duration calibration and semi-synthetic instance generation utilities.
- `figures/`: final M1--M3 figure-generation scripts.
- `analysis/`: supporting analysis utilities.
- `runners/`: archived run wrappers and result collectors.
- `generation/`: scenario-generation utilities. The historical monolithic batch pipeline is intentionally excluded; see `docs/REPRODUCIBILITY.md`.

The release is organized to reproduce the final paper evidence rather than to re-execute every historical local experiment batch.

# Data provenance and calibration boundary

The study instances are **semi-synthetic OR--ICU--ward planning instances**. GermanOR and Mannino contribute empirical operating-room duration calibration only. Their source records do not directly supply the ICU/ward pathways, capacities, initial occupancy, transfer bottlenecks, urgency labels, surgeon calendars, equipment requirements, or other execution-layer attributes used in this study. Those elements are generated using documented rules and scenario parameters.

The final manuscript therefore uses the terms **source-data-derived benchmark families** and **duration-calibrated semi-synthetic families**. It does not describe GermanOR or Mannino as fully observed end-to-end hospital pathways.

`metadata/source_data_manifest.csv` is the machine-readable provenance record. All benchmark-derived cases in the frozen study use `duration_only` injection.

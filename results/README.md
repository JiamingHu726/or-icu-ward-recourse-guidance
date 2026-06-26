# Derived result tables

The tables in `derived_tables/` are the final analysis inputs and outputs used in the load-regime manuscript. They contain derived, non-raw outputs and do not include private benchmark workbooks.

`all_standardized_effect_rows_v21.csv` (and its generic alias `all_standardized_effect_rows.csv`) is sanitized to remove local filesystem paths. It retains the dataset, seed, scenario, exact-score gap, and component contributions required to regenerate M1 and M2.

# Source and derived data policy

## External sources

| Family | Source | Repository policy |
|---|---|---|
| Mannino | MSS-Adjusts Surgery Data, SINTEF ICT | Do not redistribute the raw source data unless its terms explicitly permit redistribution. Provide retrieval instructions and code that transforms the source into the study format. |
| GermanOR | Korzhenevich and Zander (2024), ready-to-use surgical process data set | Do not redistribute raw source data unless permitted. Store only derived study manifests and scripts that recreate study-ready instances. |
| Synthetic | Study generator | Archive generator source, frozen random seeds, group-level parameter tables, generated instance/capacity manifests, and result summaries. |

## What must be deposited

1. `metadata/source_data_manifest.csv` with the source, access date, version, citation key, and transformation script for every benchmark family.
2. `metadata/instance_capacity_manifest.csv`, generated from every released case directory.
3. The synthetic generation parameter table, including procedure-class weights, duration and length-of-stay statistics, ICU-use probabilities, capacity profiles, and stress multipliers.
4. Derived outputs required for the paper tables and figures. Avoid uploading raw third-party records if redistribution is not permitted.


## Terminology for the manuscript and repository

GermanOR and Mannino should be described as **benchmark-calibrated instance families derived from public surgical process data**, not as raw patient-level scheduling records. The released planning cases are generated instances. Their source-derived characteristics include procedure mix and process-duration information; the case generator then applies the common planning and downstream execution framework. Synthetic is a deliberately stress-focused generated family.

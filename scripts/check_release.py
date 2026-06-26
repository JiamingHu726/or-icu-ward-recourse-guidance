#!/usr/bin/env python3
"""Validate the final load-regime reproducibility staging release.

Default mode checks a private staging release. Use --public to also require
final author/URL/DOI metadata before a public GitHub/Zenodo release.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--public', action='store_true')
args = ap.parse_args()

required = [
    'README.md', 'LICENSE', 'DATA_CODE_AVAILABILITY.md', '.zenodo.json', 'CITATION.cff',
    'environment.project.yml', 'config/objective_weights.json', 'config/algorithm_parameters.yml',
    'config/solver_configuration.yml', 'metadata/source_data_manifest.csv',
    'metadata/selected_schedule_hashes.csv', 'metadata/price_arm_schedule_equality.csv',
    'metadata/instance_capacity_manifest.csv', 'metadata/benchmark_instance_provenance.csv',
    'data/derived/frozen_calibration/GermanOR/german_pool_n50_seed7.csv',
    'data/derived/frozen_calibration/GermanOR/german_pool_n150_seed47.csv',
    'data/restricted/Mannino/README.md',
    'src/core/pr_glns_spiral_or_icu_ward_guidance_mode.py',
    'src/figures/make_M1_seed_clustered.py',
    'src/figures/make_M2_seed_clustered_mechanism.py',
    'src/figures/make_M3_attribution_validity.py',
    'figures/M1_seed_clustered_load_response.pdf',
    'figures/M2_mechanism_decomposition_and_access_tradeoff.pdf',
    'figures/M3_attribution_validity.pdf',
    'results/derived_tables/M1_seed_clustered_summary.csv',
    'results/derived_tables/M2_seed_clustered_component_summary.csv',
    'results/derived_tables/baseline_load_table_values.csv',
    'results/derived_tables/weight_sensitivity_table_values.csv',
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    print('RELEASE BLOCKED. Missing required artifacts:')
    print('\n'.join(f' - {p}' for p in missing))
    raise SystemExit(1)

for forbidden in [
    'figures/M1_regime_threshold.pdf',
    'figures/M2_mechanism_components_and_coupling.pdf',
    'src/figures/make_mechanism_figures.py',
    'src/generation/batch_run_publication_pipeline.py',
]:
    if Path(forbidden).exists():
        print(f'RELEASE BLOCKED. Superseded artifact remains: {forbidden}')
        raise SystemExit(2)

cache = list(Path('.').rglob('__pycache__'))
if cache:
    print('WARNING. __pycache__ directories are ignored by Git; remove them before packaging a ZIP release.')

if args.public:
    citation = Path('CITATION.cff').read_text(encoding='utf-8')
    zenodo = Path('.zenodo.json').read_text(encoding='utf-8')
    placeholders = ['REPLACE_WITH_OWNER', '0.9.0-staging', '"creators": []']
    found = [x for x in placeholders if x in citation or x in zenodo]
    if found:
        print('PUBLIC RELEASE BLOCKED. Replace public metadata placeholders:')
        print('\n'.join(f' - {x}' for x in found))
        raise SystemExit(4)
    try:
        metadata = json.loads(zenodo)
        if not metadata.get('creators'):
            print('PUBLIC RELEASE BLOCKED. .zenodo.json requires final creators.')
            raise SystemExit(5)
    except json.JSONDecodeError:
        print('PUBLIC RELEASE BLOCKED. Invalid .zenodo.json.')
        raise SystemExit(6)
    print('Public release check passed.')
else:
    print('Staging release check passed. Public metadata placeholders remain intentionally.')

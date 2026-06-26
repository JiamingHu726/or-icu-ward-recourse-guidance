# Release status

## Completed in v0.9.0 staging candidate

- BSD-3-Clause project-code licence added.
- Current load-regime title and terminology applied to repository documentation.
- Superseded threshold figures, stacked-median M2, and malformed M3 figure removed.
- Final M1, M2, and M3 figures, result tables, and figure scripts added.
- Baseline-load and weight-sensitivity result tables added.
- Source provenance updated to duration-only benchmark injection.
- Historical monolithic pipeline runner removed from the public release scope.
- Public-data policy distinguishes GermanOR redistribution from Mannino non-redistribution.

## Required before public v1.0.0 / Zenodo

1. Confirm legal ownership and acceptance of the BSD-3-Clause licence.
2. Add final release authors, GitHub owner, public repository URL, and release date.
3. Create the public GitHub repository with the chosen name: `or-icu-ward-recourse-guidance`.
4. Create the Zenodo archive and insert the version/concept DOI.
5. Run `python scripts/check_release.py --public`.
6. Replace the manuscript data/code-availability placeholder with the version DOI.

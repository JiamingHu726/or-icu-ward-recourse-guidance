# Environment documentation

## Two different artifacts are intentionally deposited

### 1. Archived workstation snapshot

The original experiments were run from a broad Conda `base` environment on the Windows workstation. The following files document that workstation state:

- `environment.win64.no_builds.yml`
- `environment.win64.explicit.txt`
- `metadata/archival_conda_pip_section_do_not_install.txt`
- `metadata/archival_pip_overlay_do_not_install.txt`

They are **provenance records**, not the recommended installation route. The base environment contained development, notebook, browser, machine-learning, and CUDA packages unrelated to this manuscript. In particular, the archival pip section includes a CUDA-specific Torch requirement (`torch==2.11.0+cu126`) that is not available from the default PyPI index and is not required by this project.

### 2. Curated executable project environment

Use `environment.project.yml` to reproduce the released code. It contains the pinned packages required by the archived project runners:

- Python 3.12.7
- NumPy 1.26.4
- pandas 2.2.2
- matplotlib 3.9.2
- SciPy 1.13.1
- PyYAML 6.0.1
- openpyxl 3.1.5
- psutil 7.0.0
- gurobipy 13.0.1

The project runners inspected for this release import NumPy, pandas, SciPy, and Gurobi-related code; they do **not** import Torch, torchvision, or torchaudio.

## Installation

```powershell
$CONDA = "E:\anaconda3\Scripts\conda.exe"

& $CONDA env create `
  -f .\environment.project.yml `
  -n or-icu-ward-recourse
```

Gurobi requires a valid separately installed license. Never deposit a `gurobi.lic` file, license token, or private license-server address in the repository.

## Why the earlier pip-overlay installation failed

The original base environment's pip overlay included `torch==2.11.0+cu126`. The `+cu126` build is a CUDA-specific wheel and is not found on the default package index used by a plain `pip install -r ...` command. That package is unrelated to the OR--ICU--ward research code and is deliberately excluded from the curated environment.

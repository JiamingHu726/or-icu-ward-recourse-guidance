# Reproduction commands

Run these commands from the repository root.

## 0. Capture and freeze the original Python environment

The captured Python executable is `E:\anaconda3\python.exe`, which normally corresponds to the Conda `base` environment. Confirm with `conda env list` if necessary.

```powershell
$PY = "E:\anaconda3\python.exe"
$CONDA = "E:\anaconda3\Scripts\conda.exe"

& .\scripts\export_environment.ps1 `
  -CondaExe $CONDA `
  -EnvironmentName base `
  -PythonExe $PY
```

This writes `environment.yml`, `environment.lock.yml`, `environment.explicit.txt`, `metadata\pip_freeze.txt`, and an updated runtime manifest.

## 1. Capture complete Gurobi log evidence

```powershell
& $PY .\scripts\collect_solver_log_evidence.py `
  .\recourse_level1_price_cal_results_v3 `
  --output .\metadata\solver_log_evidence.csv
```

The historical evidence should show Gurobi 13.0.1 and automatic selection `using up to 16 threads`.

## 2. Build derived audit manifests

```powershell
& $PY .\scripts\collect_instance_capacity_manifest.py `
  .\german_publication_batch_instances `
  .\german_publication_batch_instances_100_150 `
  .\mannino_publication_batch_instances `
  .\mannino_publication_batch_instances_100_150 `
  .\publication_batch_instances `
  --output .\metadata\instance_capacity_manifest.csv

& $PY .\scripts\collect_selected_schedule_hashes.py `
  .\recourse_level1_price_cal_results_v3 `
  .\synthetic_abclean_factorial_results `
  --output-dir .\metadata
```

## 3. Pin threads for future release reruns

`GUROBI_THREADS` is not a native Gurobi parameter. For a future reproduction run, ensure that every Gurobi model creation calls:

```python
model.Params.Threads = 16
```

The release helper is `src\solver_runtime_configuration.py`. Verify this in the solver log before reporting new wall-clock comparisons.

## 4. Run the release check

```powershell
& $PY .\scripts\check_release.py
```

## 5. Create the curated project environment

Do **not** install the archived base-environment pip overlay. It includes unrelated
CUDA-specific Torch packages and is retained only as provenance.

```powershell
$CONDA = "E:\anaconda3\Scripts\conda.exe"

& $CONDA env create `
  -f .\environment.project.yml `
  -n or-icu-ward-recourse
```

The archived workstation snapshots remain available under
`environment.win64.no_builds.yml`, `environment.win64.explicit.txt`, and
`metadata/`, but they are not the recommended executable environment.

## 6. Run the schedule-hash audit on the original machine

For the current audit task, use the original interpreter that produced the
results; no new environment is required:

```powershell
$PY = "E:\anaconda3\python.exe"

& $PY .\scripts\collect_selected_schedule_hashes.py `
  .\recourse_level1_price_cal_results_v3 `
  .\synthetic_abclean_Aonly_check `
  --output-dir .\metadata
```

Replace the second result root with the actual Synthetic output directory found
by the schedule-file search.

# Hardware and software execution environment

## Original experiment machine

| Item | Recorded value |
|---|---|
| Operating system | Windows 11, version 10.0.22631 |
| Processor | 12th Gen Intel(R) Core(TM) i5-12500H |
| Physical CPU cores | 12 |
| Logical processors | 16 |
| Installed memory | 16,908,140,544 bytes (15.75 GiB) |
| Python | 3.12.7, packaged by Anaconda, Inc. |
| Gurobi Optimizer runtime | 13.0.1 |
| `gurobipy` | 13.0.1 |
| Gurobi build observed in solver logs | v13.0.1rc0, win64, Windows 11.0 (22631.2) |

## Historical thread policy

The archived code did **not** explicitly set the Gurobi `Threads` parameter. The archived production solver logs report:

```text
Thread count: 12 physical cores, 16 logical processors, using up to 16 threads
```

Thus, the historical experiments used Gurobi automatic thread selection with an observed upper limit of 16 threads on the original workstation. This should not be described as an explicitly fixed `Threads = 16` historical setting.

## Fixed policy for future reproduction runs

For deterministic documentation of future reruns, the release code should explicitly apply:

```python
model.Params.Threads = 16
```

immediately after every `gurobipy.Model(...)` construction. The corresponding solver log must contain a thread-setting record or the standard thread-count line. The historical and future-reproduction settings are intentionally distinguished in `config/solver_configuration.yml`.

## Thread-related environment variables captured on the original machine

- `OMP_NUM_THREADS`: not set
- `MKL_NUM_THREADS`: not set
- `OPENBLAS_NUM_THREADS`: not set
- `NUMEXPR_NUM_THREADS`: not set
- `GUROBI_THREADS`: not set

`GUROBI_THREADS` is a repository metadata variable only. Gurobi does not automatically use this variable as its `Threads` parameter; explicit model configuration is required for future reruns.

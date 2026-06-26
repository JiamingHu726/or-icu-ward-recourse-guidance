# Archived solver-log evidence

This file records the solver information extracted from archived `stdout.txt` files. Private local drive prefixes are intentionally removed.

## Production calibration output

Normalized relative log path:

```text
results/real_price_calibration/Mannino/n70/price_cal_w2/stdout.txt
```

Observed repeated lines:

```text
Gurobi Optimizer version 13.0.1 build v13.0.1rc0 (win64 - Windows 11.0 (22631.2))
Thread count: 12 physical cores, 16 logical processors, using up to 16 threads
```

The same version and thread-count message also appeared in archived smoke-run logs for GermanOR and Mannino.

## Interpretation

- No archived code hit explicitly set `model.Params.Threads` or `model.setParam("Threads", ...)`.
- The historical code therefore used Gurobi automatic thread selection.
- On the original Windows workstation, the observed automatic upper limit was 16 threads.
- The public release distinguishes this historical automatic setting from the recommended future fixed setting, `Threads = 16`.

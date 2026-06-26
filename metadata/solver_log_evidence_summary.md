# Solver-log evidence summary

The repository contains `40` archived solver-log records.

## Consistency checks

- Recorded Gurobi version/build entries: 1
- Observed thread-limit values: 16 
- Explicit `Threads` parameter entries: 0

All scanned records report the same Gurobi runtime/build and the same automatic upper limit of 16 threads. No log records an explicit `Set parameter Threads ...` command. Accordingly, the historical execution policy is documented as Gurobi automatic thread selection with an observed limit of 16 threads, rather than a historically fixed `Threads = 16` setting.

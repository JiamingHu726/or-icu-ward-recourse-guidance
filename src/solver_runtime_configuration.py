"""Use this helper in every release-code path that constructs a Gurobi model."""
from __future__ import annotations

import gurobipy as gp

REPRODUCTION_THREADS = 16


def configure_gurobi_model(model: gp.Model, *, threads: int = REPRODUCTION_THREADS) -> gp.Model:
    """Apply the fixed reproduction policy before optimization."""
    model.Params.Threads = threads
    return model

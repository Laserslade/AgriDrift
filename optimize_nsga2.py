"""
NSGA-II optimization of spray settings for a fixed weather scenario.
See README.md for the objective and decision variable definitions.
"""

import numpy as np
import pandas as pd

from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Choice
from pymoo.core.mixed import (
    MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination,
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from surrogate import SpraySurrogate, NOZZLE_TYPES

# Default scenario: wind_dir_deg=90 is full crosswind, the worst case for
# effective crosswind speed. Edit freely or pass a scenario dict into run().
SCENARIO = dict(
    wind_speed=4.0,             # m/s
    wind_dir_deg=90.0,          # deg, 90 = full crosswind (worst case)
    atmospheric_stability="neutral",
    temperature_c=24.0,         # deg C
    relative_humidity=55.0,     # %
)

PRESSURE_BOUNDS = (1.0, 6.0)
BOOM_HEIGHT_BOUNDS = (0.3, 1.3)


class SpraySettingsProblem(Problem):
    """Mixed-variable (1 categorical + 2 continuous), 4-objective problem."""

    def __init__(self, surrogate: SpraySurrogate, scenario: dict,
                 boom_height_bounds=BOOM_HEIGHT_BOUNDS):
        self.surrogate = surrogate
        self.scenario = scenario
        variables = {
            "nozzle_type": Choice(options=NOZZLE_TYPES),
            "pressure_bar": Real(bounds=PRESSURE_BOUNDS),
            "boom_height_m": Real(bounds=boom_height_bounds),
        }
        super().__init__(vars=variables, n_obj=4, n_ieq_constr=0)

    def _evaluate(self, X, out, *args, **kwargs):
        # X is a list of dicts, one per individual, in mixed-variable mode
        nozzle = np.array([x["nozzle_type"] for x in X])
        pressure = np.array([x["pressure_bar"] for x in X])
        boom = np.array([x["boom_height_m"] for x in X])

        pred = self.surrogate.predict(self.scenario, nozzle, pressure, boom)

        f1 = pred["drift_fraction"]
        f2 = 1.0 - pred["coverage_fraction"]
        f3 = pred["evaporated_fraction"]
        f4 = pressure / PRESSURE_BOUNDS[1]  # normalized to [0,1] like the others
        out["F"] = np.column_stack([f1, f2, f3, f4])


def run(scenario: dict = None, pop_size: int = 200, n_gen: int = 150, seed: int = 1,
        boom_height_bounds=BOOM_HEIGHT_BOUNDS):
    scenario = scenario or SCENARIO
    surrogate = SpraySurrogate(model_prefix="models/xgb_model")
    problem = SpraySettingsProblem(surrogate, scenario, boom_height_bounds=boom_height_bounds)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)

    rows = []
    for x, f in zip(res.X, res.F):
        diag = surrogate.predict(scenario, np.array([x["nozzle_type"]]),
                                  np.array([x["pressure_bar"]]),
                                  np.array([x["boom_height_m"]]))
        # Fractions are bounded in [0, 1]; clip small out-of-range surrogate noise.
        drift = float(np.clip(f[0], 0.0, 1.0))
        coverage = float(np.clip(1.0 - f[1], 0.0, 1.0))
        evap = float(np.clip(f[2], 0.0, 1.0))
        pressure_norm = float(f[3])  # already in [0,1] by construction
        escaped = float(np.clip(diag["escaped_fraction"][0], 0.0, 1.0))
        rows.append({
            "nozzle_type": x["nozzle_type"],
            "pressure_bar": round(float(x["pressure_bar"]), 2),
            "boom_height_m": round(float(x["boom_height_m"]), 2),
            "drift_fraction": round(drift, 5),
            "coverage_fraction": round(coverage, 5),
            "evaporated_fraction": round(evap, 6),
            "pressure_norm": round(pressure_norm, 5),
            "escaped_fraction_diag": round(escaped, 6),
        })

    pareto_df = pd.DataFrame(rows)
    # Collapse near-duplicate solutions before presenting.
    pareto_df = pareto_df.drop_duplicates(
        subset=["nozzle_type", "pressure_bar", "boom_height_m",
                "drift_fraction", "coverage_fraction", "evaporated_fraction"]
    ).sort_values("drift_fraction").reset_index(drop=True)

    # Clipping can turn illusory non-dominated points into genuinely
    # dominated ones. Re-check dominance on the clipped values.
    pareto_df = _drop_dominated(pareto_df)

    return pareto_df, res, scenario


def _drop_dominated(pareto_df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = ["drift_fraction", "coverage_fraction", "evaporated_fraction", "pressure_norm"]
    vals = pareto_df[obj_cols].to_numpy()
    minimize_dir = np.array([1.0, -1.0, 1.0, 1.0])  # coverage maximized, flip sign
    signed = vals * minimize_dir
    keep = np.ones(len(pareto_df), dtype=bool)
    for i in range(len(pareto_df)):
        for j in range(len(pareto_df)):
            if i == j:
                continue
            dominates_i = np.all(signed[j] <= signed[i]) and np.any(signed[j] < signed[i])
            if dominates_i:
                keep[i] = False
                break
    return pareto_df[keep].sort_values("drift_fraction").reset_index(drop=True)


def thin_for_display(pareto_df: pd.DataFrame, max_rows: int = 25) -> pd.DataFrame:
    """Pick an evenly spread subset across the drift range for a readable table."""
    if len(pareto_df) <= max_rows:
        return pareto_df
    idx = np.linspace(0, len(pareto_df) - 1, max_rows).round().astype(int)
    idx = sorted(set(idx))
    return pareto_df.iloc[idx].reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 70)
    print("RUN 1: unconstrained boom height (0.3-1.3 m, full simulator range)")
    print("=" * 70)
    pareto_df, res, scenario = run()
    print("Scenario:", scenario)
    print(f"Non-dominated set: {len(pareto_df)} solutions")
    print(thin_for_display(pareto_df).to_string(index=False))
    pareto_df.to_csv("results/pareto_front_unconstrained.csv", index=False)

    print()
    print("=" * 70)
    print("RUN 2: operational boom-height floor (0.5-1.3 m)")
    print("=" * 70)
    pareto_df2, res2, scenario2 = run(boom_height_bounds=(0.5, 1.3))
    print("Scenario:", scenario2)
    print(f"Non-dominated set: {len(pareto_df2)} solutions")
    print(thin_for_display(pareto_df2).to_string(index=False))
    pareto_df2.to_csv("results/pareto_front_operational.csv", index=False)

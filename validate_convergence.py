"""
Convergence validation for the NSGA-II spray-settings optimizer.
See README.md for the grid cross-check and multi-seed methodology.
"""

import numpy as np
import pandas as pd

from surrogate import SpraySurrogate, NOZZLE_TYPES
from optimize_nsga2 import run as run_nsga2, PRESSURE_BOUNDS, BOOM_HEIGHT_BOUNDS


def grid_ground_truth(scenario: dict, surrogate: SpraySurrogate,
                       n_pressure: int = 80, n_boom: int = 80,
                       boom_height_bounds=BOOM_HEIGHT_BOUNDS) -> pd.DataFrame:
    """Brute-force grid over the decision space, used as ground truth."""
    pressures = np.linspace(*PRESSURE_BOUNDS, n_pressure)
    booms = np.linspace(*boom_height_bounds, n_boom)
    pp, bb = np.meshgrid(pressures, booms)
    pp = pp.ravel()
    bb = bb.ravel()

    rows = []
    for nt in NOZZLE_TYPES:
        nozzle_arr = np.full(pp.shape, nt)
        pred = surrogate.predict(scenario, nozzle_arr, pp, bb)
        drift = np.clip(pred["drift_fraction"], 0.0, 1.0)
        coverage = np.clip(pred["coverage_fraction"], 0.0, 1.0)
        evap = np.clip(pred["evaporated_fraction"], 0.0, 1.0)
        best_idx = np.argmin(drift)  # global best-drift point for this nozzle
        rows.append({
            "nozzle_type": nt,
            "pressure_bar": round(float(pp[best_idx]), 3),
            "boom_height_m": round(float(bb[best_idx]), 3),
            "drift_fraction": round(float(drift[best_idx]), 6),
            "coverage_fraction": round(float(coverage[best_idx]), 6),
            "evaporated_fraction": round(float(evap[best_idx]), 7),
        })
    grid_df = pd.DataFrame(rows).sort_values("drift_fraction").reset_index(drop=True)
    return grid_df


def multi_seed_stability(scenario: dict, seeds=(1, 2, 3, 4, 5),
                          pop_size: int = 100, n_gen: int = 100,
                          boom_height_bounds=BOOM_HEIGHT_BOUNDS) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        pareto_df, res, _ = run_nsga2(scenario, pop_size=pop_size, n_gen=n_gen, seed=seed,
                                       boom_height_bounds=boom_height_bounds)
        best = pareto_df.sort_values("drift_fraction").iloc[0]
        rows.append({
            "seed": seed,
            "nozzle_type": best.nozzle_type,
            "pressure_bar": best.pressure_bar,
            "boom_height_m": best.boom_height_m,
            "drift_fraction": best.drift_fraction,
            "coverage_fraction": best.coverage_fraction,
        })
    return pd.DataFrame(rows)


def validate_scenario(name: str, scenario: dict, surrogate: SpraySurrogate,
                       boom_height_bounds=BOOM_HEIGHT_BOUNDS) -> dict:
    grid_df = grid_ground_truth(scenario, surrogate, boom_height_bounds=boom_height_bounds)
    grid_best = grid_df.iloc[0]

    seeds_df = multi_seed_stability(scenario, boom_height_bounds=boom_height_bounds)

    nsga_drift = seeds_df.drift_fraction.astype(float)
    nozzle_agreement = (seeds_df.nozzle_type == seeds_df.nozzle_type.iloc[0]).all()
    drift_spread = float(nsga_drift.max() - nsga_drift.min())
    gap_vs_grid = float(nsga_drift.min() - grid_best.drift_fraction)

    verdict = "CONVERGED"
    if not nozzle_agreement or drift_spread > 0.01:
        verdict = "UNSTABLE (seeds disagree)"
    elif gap_vs_grid > 0.01:
        verdict = "SUSPECT (NSGA-II worse than grid ground truth)"

    return {
        "scenario": name,
        "grid_best_nozzle": grid_best.nozzle_type,
        "grid_best_drift": grid_best.drift_fraction,
        "nsga_best_nozzle_mode": seeds_df.nozzle_type.mode().iloc[0],
        "nsga_best_drift_min": float(nsga_drift.min()),
        "nsga_drift_spread_across_seeds": drift_spread,
        "gap_nsga_minus_grid": gap_vs_grid,
        "all_seeds_agree_on_nozzle": bool(nozzle_agreement),
        "verdict": verdict,
        "grid_table": grid_df,
        "seeds_table": seeds_df,
    }


if __name__ == "__main__":
    from scenarios import SCENARIOS
    OPERATIONAL_BOOM_BOUNDS = (0.5, 1.3)
    surrogate = SpraySurrogate()
    summary_rows = []
    for name, scenario in SCENARIOS.items():
        print(f"\n=== {name} ===")
        result = validate_scenario(name, scenario, surrogate, boom_height_bounds=OPERATIONAL_BOOM_BOUNDS)
        print(f"Grid ground truth best: {result['grid_best_nozzle']} "
              f"drift={result['grid_best_drift']:.5f}")
        print(f"NSGA-II across 5 seeds: mode={result['nsga_best_nozzle_mode']} "
              f"best_drift={result['nsga_best_drift_min']:.5f} "
              f"spread={result['nsga_drift_spread_across_seeds']:.6f} "
              f"gap_vs_grid={result['gap_nsga_minus_grid']:.6f}")
        print(f"Verdict: {result['verdict']}")
        summary_rows.append({k: v for k, v in result.items()
                              if k not in ("grid_table", "seeds_table")})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/convergence_validation_summary_operational.csv", index=False)
    print("\n\n=== SUMMARY (operational boom floor 0.5-1.3m) ===")
    print(summary_df.to_string(index=False))

"""
Run the full NSGA-II Pareto front for every scenario in scenarios.py,
using the operational boom-height floor (0.5-1.3 m).
"""

import pandas as pd
from optimize_nsga2 import run as run_nsga2, thin_for_display
from scenarios import SCENARIOS

OPERATIONAL_BOOM_BOUNDS = (0.5, 1.3)


def main():
    all_rows = []
    best_rows = []
    for name, scenario in SCENARIOS.items():
        pareto_df, res, _ = run_nsga2(scenario, pop_size=200, n_gen=150, seed=1,
                                       boom_height_bounds=OPERATIONAL_BOOM_BOUNDS)
        pareto_df = pareto_df.copy()
        pareto_df.insert(0, "scenario", name)
        all_rows.append(pareto_df)

        best = pareto_df.sort_values("drift_fraction").iloc[0]
        best_rows.append({
            "scenario": name,
            "wind_speed": scenario["wind_speed"],
            "atmospheric_stability": scenario["atmospheric_stability"],
            "temperature_c": scenario["temperature_c"],
            "relative_humidity": scenario["relative_humidity"],
            "n_pareto_solutions": len(pareto_df),
            "recommended_nozzle": best.nozzle_type,
            "recommended_pressure_bar": best.pressure_bar,
            "recommended_boom_height_m": best.boom_height_m,
            "drift_fraction": best.drift_fraction,
            "coverage_fraction": best.coverage_fraction,
            "evaporated_fraction": best.evaporated_fraction,
        })
        print(f"{name:22s} n_pareto={len(pareto_df):3d}  "
              f"best -> {best.nozzle_type:20s} P={best.pressure_bar} bar "
              f"H={best.boom_height_m} m  drift={best.drift_fraction:.5f} "
              f"cov={best.coverage_fraction:.4f}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv("results/pareto_all_scenarios.csv", index=False)

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv("results/recommended_settings_by_scenario.csv", index=False)
    print("\nSaved results/pareto_all_scenarios.csv and results/recommended_settings_by_scenario.csv")
    return combined, best_df


if __name__ == "__main__":
    main()

"""
Generate a training dataset for an XGBoost surrogate of SpraySimulator.
See README.md for the sampling design and target list.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale

from spray_sim.state import SystemState, Environment, Settings
from spray_sim.simulator import SpraySimulator, NOZZLE_PARAMS, STABILITY_TURB_MULT

# input parameter ranges (physically plausible for boom spraying)
CONTINUOUS_BOUNDS = {
    "wind_speed":        (0.3, 9.0),     # m/s
    "wind_dir_deg":       (0.0, 180.0),  # deg
    "temperature_c":     (5.0, 38.0),    # deg C
    "relative_humidity": (15.0, 95.0),   # %
    "pressure_bar":      (1.0, 6.0),     # bar
    "boom_height_m":     (0.3, 1.3),     # m
}
NOZZLE_TYPES = list(NOZZLE_PARAMS.keys())
STABILITY_CLASSES = list(STABILITY_TURB_MULT.keys())

CONT_COLS = list(CONTINUOUS_BOUNDS.keys())


def sample_inputs(n_samples: int, seed: int = 0) -> pd.DataFrame:
    """Latin hypercube over continuous vars, uniform random over categoricals."""
    rng = np.random.default_rng(seed)

    sampler = LatinHypercube(d=len(CONT_COLS), seed=seed)
    unit = sampler.random(n=n_samples)
    lo = np.array([CONTINUOUS_BOUNDS[c][0] for c in CONT_COLS])
    hi = np.array([CONTINUOUS_BOUNDS[c][1] for c in CONT_COLS])
    cont = scale(unit, lo, hi)

    df = pd.DataFrame(cont, columns=CONT_COLS)
    df["nozzle_type"] = rng.choice(NOZZLE_TYPES, size=n_samples)
    df["atmospheric_stability"] = rng.choice(STABILITY_CLASSES, size=n_samples)
    return df


def run_simulator_for_row(sim: SpraySimulator, row, n_particles: int) -> dict:
    env = Environment(
        wind_speed=row["wind_speed"],
        wind_dir_deg=row["wind_dir_deg"],
        atmospheric_stability=row["atmospheric_stability"],
        temperature_c=row["temperature_c"],
        relative_humidity=row["relative_humidity"],
    )
    settings = Settings(
        nozzle_type=row["nozzle_type"],
        pressure_bar=row["pressure_bar"],
        boom_height_m=row["boom_height_m"],
    )
    state = SystemState(environment=env, settings=settings)
    res = sim.run(state, n_particles=n_particles)
    return {
        "coverage_fraction": res.coverage_fraction,
        "drift_fraction": res.drift_fraction,
        "evaporated_fraction": res.evaporated_fraction,
        "escaped_fraction": res.escaped_fraction,
        "mean_droplet_diameter_um": res.mean_droplet_diameter_um,
    }


def generate(n_samples: int = 4000, n_particles: int = 1500, base_seed: int = 123,
             out_path: str = "data/spray_dataset.csv") -> pd.DataFrame:
    inputs_df = sample_inputs(n_samples, seed=base_seed)

    records = []
    t0 = time.time()
    for i, row in inputs_df.iterrows():
        # vary the simulator seed per sample so turbulence isn't identical every row
        sim = SpraySimulator(seed=base_seed * 1000 + i)
        out = run_simulator_for_row(sim, row, n_particles)
        records.append(out)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples} sims done ({elapsed:.1f}s elapsed, "
                  f"{elapsed/(i+1)*1000:.2f}s/1000 sims)")

    outputs_df = pd.DataFrame(records)
    full_df = pd.concat([inputs_df.reset_index(drop=True), outputs_df], axis=1)
    full_df.to_csv(out_path, index=False)
    print(f"Saved {len(full_df)} rows to {out_path} in {time.time()-t0:.1f}s total")
    return full_df


if __name__ == "__main__":
    generate()

"""
Wrapper around the trained XGBoost surrogate models.
Lets callers query simulator-equivalent outputs without running SpraySimulator.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

NOZZLE_TYPES = [
    "air_induction_cvi",
    "air_induction_id",
    "air_induction_tti",
    "flat_fan_classic",
]
STABILITY_CLASSES = ["neutral", "stable", "unstable"]

# Exact column order the models were trained on. Order matters because the
# sklearn XGBRegressor wrapper validates feature names against the booster.
FEATURE_COLUMNS = [
    "wind_speed", "wind_dir_deg", "temperature_c", "relative_humidity",
    "pressure_bar", "boom_height_m",
    "nozzle_type_air_induction_cvi", "nozzle_type_air_induction_id",
    "nozzle_type_air_induction_tti", "nozzle_type_flat_fan_classic",
    "atmospheric_stability_neutral", "atmospheric_stability_stable",
    "atmospheric_stability_unstable",
]

TARGETS = [
    "coverage_fraction", "drift_fraction", "evaporated_fraction",
    "escaped_fraction", "mean_droplet_diameter_um",
]


class SpraySurrogate:
    def __init__(self, model_prefix: str = "models/xgb_model"):
        self.models = {}
        for target in TARGETS:
            m = xgb.XGBRegressor()
            m.load_model(f"{model_prefix}_{target}.json")
            self.models[target] = m

    def _feature_frame(self, scenario: dict, nozzle_type, pressure_bar, boom_height_m) -> pd.DataFrame:
        """Build a feature frame for one scenario and a batch of decision variables.
        nozzle_type/pressure_bar/boom_height_m may be arrays or scalars."""
        nozzle_type = np.atleast_1d(nozzle_type)
        pressure_bar = np.atleast_1d(pressure_bar).astype(float)
        boom_height_m = np.atleast_1d(boom_height_m).astype(float)
        n = max(len(nozzle_type), len(pressure_bar), len(boom_height_m))
        nozzle_type = np.broadcast_to(nozzle_type, (n,))
        pressure_bar = np.broadcast_to(pressure_bar, (n,))
        boom_height_m = np.broadcast_to(boom_height_m, (n,))

        data = {
            "wind_speed": np.full(n, scenario["wind_speed"], dtype=float),
            "wind_dir_deg": np.full(n, scenario["wind_dir_deg"], dtype=float),
            "temperature_c": np.full(n, scenario["temperature_c"], dtype=float),
            "relative_humidity": np.full(n, scenario["relative_humidity"], dtype=float),
            "pressure_bar": pressure_bar,
            "boom_height_m": boom_height_m,
        }
        for nt in NOZZLE_TYPES:
            data[f"nozzle_type_{nt}"] = (nozzle_type == nt).astype(float)
        for sc in STABILITY_CLASSES:
            val = float(scenario["atmospheric_stability"] == sc)
            data[f"atmospheric_stability_{sc}"] = np.full(n, val, dtype=float)

        return pd.DataFrame(data, columns=FEATURE_COLUMNS)

    def predict(self, scenario: dict, nozzle_type, pressure_bar, boom_height_m) -> dict:
        X = self._feature_frame(scenario, nozzle_type, pressure_bar, boom_height_m)
        out = {}
        for target, model in self.models.items():
            out[target] = model.predict(X)
        return out

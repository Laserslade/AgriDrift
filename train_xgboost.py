"""
Train XGBoost surrogate models for SpraySimulator.
See README.md for target list and reported metrics.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGETS = [
    "coverage_fraction",
    "drift_fraction",
    "evaporated_fraction",
    "escaped_fraction",
    "mean_droplet_diameter_um",
]
CATEGORICAL_COLS = ["nozzle_type", "atmospheric_stability"]
NUMERIC_COLS = ["wind_speed", "wind_dir_deg", "temperature_c",
                 "relative_humidity", "pressure_bar", "boom_height_m"]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X_num = df[NUMERIC_COLS].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
    return pd.concat([X_num, X_cat], axis=1)


def train_and_eval(csv_path: str = "data/spray_dataset.csv", test_size: float = 0.2,
                    random_state: int = 42, model_out_prefix: str = "models/xgb_model"):
    df = pd.read_csv(csv_path)
    X = build_feature_frame(df)
    feature_names = list(X.columns)

    results = {}
    models = {}

    for target in TARGETS:
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # escaped_fraction and evaporated_fraction are near-zero for most of
        # the domain, so give them more depth/rounds than the other targets.
        rare = target in ("escaped_fraction", "evaporated_fraction")
        model = xgb.XGBRegressor(
            n_estimators=400 if rare else 300,
            max_depth=6 if rare else 5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        # second random split, to sanity check R^2 isn't a fluke of one split
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=test_size, random_state=random_state + 1
        )
        model2 = xgb.XGBRegressor(**model.get_params())
        model2.fit(X_train2, y_train2, verbose=False)
        r2_check = r2_score(y_test2, model2.predict(X_test2))

        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)

        results[target] = {
            "r2": r2,
            "r2_alt_split": r2_check,
            "mae": mae,
            "rmse": rmse,
            "y_mean": float(np.mean(y)),
            "y_std": float(np.std(y)),
            "top_features": importances.head(6).to_dict(),
        }
        models[target] = model
        model.save_model(f"{model_out_prefix}_{target}.json")

    with open("models/xgb_training_report.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, models, feature_names


def print_report(results: dict):
    print(f"{'target':<28} {'R2':>8} {'R2(alt split)':>14} {'MAE':>10} {'RMSE':>10}")
    print("-" * 74)
    for target, r in results.items():
        print(f"{target:<28} {r['r2']:>8.4f} {r['r2_alt_split']:>14.4f} "
              f"{r['mae']:>10.5f} {r['rmse']:>10.5f}")
    print()
    for target, r in results.items():
        print(f"[{target}]  (mean={r['y_mean']:.4f}, std={r['y_std']:.4f})")
        for feat, imp in r["top_features"].items():
            print(f"    {feat:<28} {imp:.4f}")
        print()


if __name__ == "__main__":
    results, models, feature_names = train_and_eval()
    print_report(results)

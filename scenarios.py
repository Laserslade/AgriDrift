"""
Weather scenario battery for testing whether optimal spray settings vary by condition.
wind_dir_deg is fixed at 90 (worst-case crosswind) across all scenarios.
"""

SCENARIOS = {
    "calm_wind": dict(
        wind_speed=1.0, wind_dir_deg=90.0, atmospheric_stability="stable",
        temperature_c=18.0, relative_humidity=65.0,
    ),
    "moderate_wind": dict(
        wind_speed=3.0, wind_dir_deg=90.0, atmospheric_stability="neutral",
        temperature_c=24.0, relative_humidity=55.0,
    ),
    "strong_wind": dict(
        wind_speed=7.0, wind_dir_deg=90.0, atmospheric_stability="neutral",
        temperature_c=24.0, relative_humidity=55.0,
    ),
    "stable_atmosphere": dict(
        wind_speed=3.0, wind_dir_deg=90.0, atmospheric_stability="stable",
        temperature_c=15.0, relative_humidity=70.0,
    ),
    "unstable_atmosphere": dict(
        wind_speed=3.0, wind_dir_deg=90.0, atmospheric_stability="unstable",
        temperature_c=32.0, relative_humidity=35.0,
    ),
    "hot_dry": dict(
        wind_speed=3.0, wind_dir_deg=90.0, atmospheric_stability="neutral",
        temperature_c=36.0, relative_humidity=18.0,
    ),
    "cool_humid": dict(
        wind_speed=3.0, wind_dir_deg=90.0, atmospheric_stability="neutral",
        temperature_c=10.0, relative_humidity=90.0,
    ),
    "worst_case_combo": dict(
        wind_speed=8.0, wind_dir_deg=90.0, atmospheric_stability="unstable",
        temperature_c=34.0, relative_humidity=22.0,
    ),
}

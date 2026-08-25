"""
State container for SpraySimulator.
See README.md for the full field reference.
"""

from dataclasses import dataclass


@dataclass
class Environment:
    wind_speed: float             # m/s
    wind_dir_deg: float           # degrees, angle of wind relative to boom line
    atmospheric_stability: str    # "stable" | "neutral" | "unstable"
    temperature_c: float          # deg C
    relative_humidity: float      # percent, 0-100


@dataclass
class Settings:
    nozzle_type: str              # one of NOZZLE_PARAMS keys in simulator.py
    pressure_bar: float           # bar
    boom_height_m: float          # m above ground


@dataclass
class SystemState:
    environment: Environment
    settings: Settings

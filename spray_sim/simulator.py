"""
Simplified 2D Lagrangian particle simulator for boom-sprayer drift.
See README.md for the physics model and coordinate system details.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from .state import SystemState

# --- physical constants ---
G = 9.81                # m/s^2
RHO_WATER = 1000.0      # kg/m^3
RHO_AIR = 1.225         # kg/m^3
MU_AIR = 1.81e-5        # Pa.s

# Nozzle baseline droplet size (VMD), pressure sensitivity, and fan geometry.
# discharge_coeff is an approximate placeholder pending calibration.
NOZZLE_PARAMS = {
    "flat_fan_classic":  {"vmd_ref_um": 250.0, "pressure_exp": -0.35, "ref_pressure_bar": 2.0, "fan_angle_deg": 110.0, "discharge_coeff": 0.4},
    "air_induction_cvi": {"vmd_ref_um": 450.0, "pressure_exp": -0.15, "ref_pressure_bar": 2.0, "fan_angle_deg": 110.0, "discharge_coeff": 0.4},
    "air_induction_id":  {"vmd_ref_um": 500.0, "pressure_exp": -0.15, "ref_pressure_bar": 2.0, "fan_angle_deg": 120.0, "discharge_coeff": 0.4},
    "air_induction_tti": {"vmd_ref_um": 600.0, "pressure_exp": -0.12, "ref_pressure_bar": 2.0, "fan_angle_deg": 110.0, "discharge_coeff": 0.4},
}
LOGNORMAL_SIGMA = 0.35   # spread of droplet size distribution (relative)

# turbulence intensity multiplier by stability class
STABILITY_TURB_MULT = {"stable": 0.5, "neutral": 1.0, "unstable": 1.7}

MIN_DIAMETER_UM = 20.0   # below this, treat droplet as fully evaporated / lost


@dataclass
class SimulationResult:
    coverage_fraction: float          # fraction of emitted volume deposited on-target (0 <= x < swath_width)
    drift_fraction: float             # fraction deposited off-target within domain (x >= swath_width)
    evaporated_fraction: float        # fraction lost to evaporation before landing
    escaped_fraction: float           # fraction that exited the domain without landing or evaporating
    downwind_profile: Dict[float, float] = field(default_factory=dict)  # distance bin (m) -> fraction deposited there
    mean_droplet_diameter_um: float = 0.0
    n_particles: int = 0
    diagnostics: list = field(default_factory=list)  # per-sample-particle trace (see run(diagnostic_n=...))


class SpraySimulator:
    def __init__(self, swath_width_m: float = 2.0, domain_x_max_m: float = 50.0,
                 dt: float = 0.02, max_time_s: float = 60.0, seed: int = None,
                 profile_bins=None, nozzle_params: dict = None):
        self.swath_width_m = swath_width_m
        self.domain_x_max_m = domain_x_max_m
        self.dt = dt
        self.max_time_s = max_time_s
        self.rng = np.random.default_rng(seed)

        # distance bins for the downwind deposition profile
        self._profile_bins = profile_bins if profile_bins is not None else [1, 3, 5, 10, 25, 50]

        # nozzle physical params can be overridden for calibration
        self.nozzle_params = nozzle_params if nozzle_params is not None else NOZZLE_PARAMS
        # turbulence intensity scaling factor, multiplies wind_effective to get turbulence std
        self.turb_coefficient = 0.15

        # Lagrangian integral time scale (s) for the Ornstein-Uhlenbeck turbulence model.
        # Literature range for near-surface agricultural spraying is roughly 1-5s.
        self.lagrangian_time_scale = 2.0

    def _droplet_diameters(self, nozzle_type: str, pressure_bar: float, n: int) -> np.ndarray:
        p = self.nozzle_params[nozzle_type]
        vmd = p["vmd_ref_um"] * (pressure_bar / p["ref_pressure_bar"]) ** p["pressure_exp"]
        mu_log = np.log(vmd)
        d_um = self.rng.lognormal(mean=mu_log, sigma=LOGNORMAL_SIGMA, size=n)
        return np.clip(d_um, 5.0, 2000.0)  # microns

    def run(self, state: SystemState, n_particles: int = 4000, diagnostic_n: int = 0) -> SimulationResult:
        env = state.environment
        settings = state.settings
        nozzle_p = self.nozzle_params[settings.nozzle_type]

        d_um = self._droplet_diameters(settings.nozzle_type, settings.pressure_bar, n_particles)
        d_m = d_um * 1e-6

        # effective crosswind component of wind speed, drives lateral/downwind drift
        wind_effective = env.wind_speed * abs(np.sin(np.radians(env.wind_dir_deg)))
        wind_effective = max(wind_effective, 0.05)  # avoid degenerate zero-wind case

        turb_mult = STABILITY_TURB_MULT.get(env.atmospheric_stability, 1.0)
        # turbulence std scales with wind speed and height above ground (simplified)
        turb_sigma = self.turb_coefficient * wind_effective * turb_mult

        n = n_particles

        # Nozzle release physics: droplets leave with a pressure-dependent exit
        # velocity, direction sampled from a Gaussian cone around the nozzle axis.
        pressure_pa = settings.pressure_bar * 1e5
        exit_velocity = nozzle_p["discharge_coeff"] * np.sqrt(2 * pressure_pa / RHO_WATER)  # m/s

        fan_angle_deg = nozzle_p["fan_angle_deg"]
        # Gaussian cone std chosen so ~95% of the fan angle is covered within +-2*std
        angle_std_deg = fan_angle_deg / 4.0
        launch_angle_deg = self.rng.normal(0.0, angle_std_deg, size=n)
        launch_angle_rad = np.radians(launch_angle_deg)

        # release position: near-point source at the boom (x=0); spatial spread
        # across the fan comes from the launch angle, not a position offset
        x = np.zeros(n)
        z = np.full(n, settings.boom_height_m)

        vx = exit_velocity * np.sin(launch_angle_rad)
        vz_arr = -exit_velocity * np.cos(launch_angle_rad)  # negative = downward, nozzle axis points down

        active = np.ones(n, dtype=bool)          # still airborne
        evaporated = np.zeros(n, dtype=bool)
        escaped = np.zeros(n, dtype=bool)
        landed_x = np.full(n, np.nan)
        landed_t = np.full(n, np.nan)

        # instrumentation: record initial state for a small sample of particles
        diag_n = min(diagnostic_n, n)
        diagnostics = []
        if diag_n > 0:
            for i in range(diag_n):
                diagnostics.append({
                    "particle_id": i,
                    "initial_diameter_um": float(d_um[i]),
                    "exit_velocity_ms": float(exit_velocity),
                    "launch_angle_deg": float(launch_angle_deg[i]),
                    "initial_vx": float(vx[i]),
                    "initial_vz": float(vz_arr[i]),
                    "landing_x": None,
                    "flight_time_s": None,
                    "outcome": None,
                })

        evap_k = self._evaporation_rate(env.temperature_c, env.relative_humidity)  # m^2/s, per-particle constant

        # Ornstein-Uhlenbeck correlated turbulence state, per particle.
        # Produces heavier dispersion tails than independent per-step noise.
        T_L = max(self.lagrangian_time_scale, self.dt)  # guard against T_L < dt
        ou_decay    = np.exp(-self.dt / T_L)
        ou_noise_std = np.sqrt(1.0 - ou_decay ** 2)     # preserves steady-state variance
        u_turb = np.zeros(n)   # per-particle horizontal turbulent velocity, m/s
        w_turb = np.zeros(n)   # per-particle vertical   turbulent velocity, m/s

        n_steps = int(self.max_time_s / self.dt)
        t = 0.0
        for _ in range(n_steps):
            t += self.dt
            if not active.any():
                break
            idx = active

            # relative velocity to wind (wind only horizontal)
            rvx = vx[idx] - wind_effective
            rvz = vz_arr[idx]
            rel_speed = np.sqrt(rvx ** 2 + rvz ** 2) + 1e-12

            re = RHO_AIR * rel_speed * d_m[idx] / MU_AIR
            cd = np.where(
                re < 1000,
                (24.0 / np.maximum(re, 1e-6)) * (1 + 0.15 * re ** 0.687),
                0.44,
            )
            area = np.pi * (d_m[idx] ** 2) / 4.0
            mass = RHO_WATER * (np.pi / 6.0) * d_m[idx] ** 3

            # Linearized drag coefficient k, treated as constant over this step.
            # Solved exactly (exponential update) since explicit Euler is unstable here.
            k = 0.5 * RHO_AIR * cd * area * rel_speed  # N per (m/s) of relative velocity
            tau = mass / np.maximum(k, 1e-30)          # relaxation time, s
            decay = np.exp(-self.dt / tau)

            # x: no external force besides drag, relative velocity decays to 0
            rvx_new = rvx * decay
            # z: gravity + drag, exact solution of d(rvz)/dt = -G - rvz/tau
            rvz_new = (rvz + G * tau) * decay - G * tau

            vx[idx] = wind_effective + rvx_new
            vz_arr[idx] = rvz_new

            # turbulence: Ornstein-Uhlenbeck (Langevin) correlated update,
            # decays toward zero over T_L seconds while injecting new noise.
            n_active = idx.sum()
            u_turb[idx] = (u_turb[idx] * ou_decay
                           + turb_sigma * ou_noise_std
                           * self.rng.standard_normal(n_active))
            w_turb[idx] = (w_turb[idx] * ou_decay
                           + turb_sigma * 0.3 * ou_noise_std
                           * self.rng.standard_normal(n_active))

            # u_turb / w_turb are turbulent velocities (m/s) that advect the
            # particle alongside the drag-computed inertial velocity.
            x[idx] = x[idx] + (vx[idx] + u_turb[idx]) * self.dt
            z[idx] = z[idx] + (vz_arr[idx] + w_turb[idx]) * self.dt

            # evaporation: d(d^2)/dt = -evap_k, update diameter
            d2 = d_m[idx] ** 2 - evap_k * self.dt
            d_m[idx] = np.sqrt(np.clip(d2, 0.0, None))

            newly_evaporated = idx.copy()
            newly_evaporated[idx] = d_m[idx] < (MIN_DIAMETER_UM * 1e-6)
            evaporated |= newly_evaporated
            active &= ~newly_evaporated

            newly_landed = idx.copy()
            newly_landed[idx] = z[idx] <= 0.0
            newly_landed &= active
            if newly_landed.any():
                landed_x[newly_landed] = x[newly_landed]
                landed_t[newly_landed] = t
            active &= ~newly_landed

            newly_escaped = idx.copy()
            newly_escaped[idx] = x[idx] > self.domain_x_max_m
            newly_escaped &= active
            escaped |= newly_escaped
            active &= ~newly_escaped

        # anything still active when time runs out counts as escaped
        escaped |= active

        landed_mask = ~np.isnan(landed_x)
        # on-target includes x<0, since the fan-angle release can launch
        # droplets slightly upwind and they still land on the target field
        on_target = landed_mask & (landed_x < self.swath_width_m)
        off_target = landed_mask & (landed_x >= self.swath_width_m)

        # Volume weighting: deposition is measured by tracer volume in field
        # trials, so weight each particle by d^3 (proportional to volume).
        vol_weights = d_um ** 3                        # proportional to droplet volume
        total_weight = vol_weights.sum()

        coverage_fraction = vol_weights[on_target].sum() / total_weight
        drift_fraction = vol_weights[off_target].sum() / total_weight
        evaporated_fraction = vol_weights[evaporated].sum() / total_weight
        escaped_fraction = vol_weights[escaped].sum() / total_weight

        profile = {}
        edges = [0] + self._profile_bins
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = landed_mask & (landed_x >= lo) & (landed_x < hi)
            width = hi - lo
            profile[hi] = (vol_weights[in_bin].sum() / total_weight) / width  # volume-fraction per metre

        if diag_n > 0:
            for i in range(diag_n):
                if landed_mask[i]:
                    diagnostics[i]["outcome"] = "landed"
                    diagnostics[i]["landing_x"] = float(landed_x[i])
                    diagnostics[i]["flight_time_s"] = float(landed_t[i])
                elif evaporated[i]:
                    diagnostics[i]["outcome"] = "evaporated"
                elif escaped[i]:
                    diagnostics[i]["outcome"] = "escaped"

        return SimulationResult(
            coverage_fraction=coverage_fraction,
            drift_fraction=drift_fraction,
            evaporated_fraction=evaporated_fraction,
            escaped_fraction=escaped_fraction,
            downwind_profile=profile,
            mean_droplet_diameter_um=float(np.mean(d_um)),
            n_particles=n,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _evaporation_rate(temperature_c: float, relative_humidity: float) -> float:
        """d^2-law evaporation constant, m^2/s.
        Increases with temperature, decreases with humidity."""
        base = 2.5e-9
        temp_factor = np.exp((temperature_c - 20.0) / 20.0)
        humidity_factor = max(0.05, 1.0 - relative_humidity / 100.0)
        return base * temp_factor * humidity_factor

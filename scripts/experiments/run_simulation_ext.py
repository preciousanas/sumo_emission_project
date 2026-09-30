#!/usr/bin/env python3
"""
run_simulation_ext.py — Extended simulation wrapper for penetration rate experiments.

Wraps run_simulation.py to add a --penetration-rate argument.
The original run_simulation.py is NEVER modified.

When penetration_rate < 1.0, only a random sample of the vehicles on each
approach edge is used to compute the emission-based switching decision.
This models partial sensor coverage (e.g. 40% of vehicles broadcast V2X).

All other logic (baseline control, data recording, TraCI loop) is unchanged.

Called by run_penetration_experiment.py.
Not intended for direct user invocation.
"""

import sys
import os
import optparse
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate run_simulation in the parent scripts/ directory
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Parse arguments — superset of run_simulation's options
# ---------------------------------------------------------------------------
opt_parser = optparse.OptionParser()
opt_parser.add_option('--seed',              type='int',    default=42)
opt_parser.add_option('--method',            type='string', default='emission_based')
opt_parser.add_option('--nogui',             action='store_true', default=False)
opt_parser.add_option('--output',            type='string', default='simulation_data.csv')
opt_parser.add_option('--route',             type='string', default=None)
opt_parser.add_option('--penetration-rate',  type='float',  default=1.0,
                      dest='penetration_rate',
                      help='Fraction of vehicles visible to emission sensor (0-1, default 1.0).')
opt_parser.add_option('--emission-threshold', type='float', default=50.0,
                      dest='emission_threshold',
                      help='Emission threshold in mg/s (default 50.0).')
options, _ = opt_parser.parse_args()

penetration_rate = max(0.0, min(1.0, options.penetration_rate))

# ---------------------------------------------------------------------------
# Import run_simulation and patch globals
# ---------------------------------------------------------------------------
import run_simulation as rs

rs.EMISSION_THRESHOLD = options.emission_threshold

# ---------------------------------------------------------------------------
# Patch emission_based_control_logic to apply the penetration mask
# ---------------------------------------------------------------------------
# We replace the function at module level so all existing call sites in
# run_simulation.py's main loop pick up the patched version.

import traci
import traci.exceptions

_orig_emission_logic = rs.emission_based_control_logic
_seed_rng            = random.Random(options.seed)   # seeded for reproducibility


def _sampled_vehicle_ids(edge: str) -> list:
    """Return a randomly sampled subset of vehicles on `edge`."""
    try:
        all_vehs = list(traci.edge.getLastStepVehicleIDs(edge))
    except traci.exceptions.TraCIException:
        return []
    if penetration_rate >= 1.0:
        return all_vehs
    k = max(0, round(len(all_vehs) * penetration_rate))
    return _seed_rng.sample(all_vehs, min(k, len(all_vehs)))


def _patched_emission_logic(current_green_phase, phase_duration):
    """Penetration-rate-aware emission controller."""
    if phase_duration < rs.MIN_GREEN:
        return False
    if phase_duration >= rs.MAX_GREEN:
        return True

    opposing_phase = 2 if current_green_phase == 0 else 0

    def _safe_co2(veh_id):
        try:
            return traci.vehicle.getCO2Emission(veh_id)
        except traci.exceptions.TraCIException:
            return None

    co2_green = sum(
        traci.vehicle.getCO2Emission(v)
        for edge in rs.GREEN_PHASES[current_green_phase]
        for v in _sampled_vehicle_ids(edge)
        if _safe_co2(v) is not None
    )
    co2_red = sum(
        traci.vehicle.getCO2Emission(v)
        for edge in rs.GREEN_PHASES[opposing_phase]
        for v in _sampled_vehicle_ids(edge)
        if _safe_co2(v) is not None
    )

    if co2_red > (co2_green + rs.EMISSION_THRESHOLD):
        return True
    return False


# Only patch when actually running emission_based; baseline is unaffected
if options.method == 'emission_based' and penetration_rate < 1.0:
    rs.emission_based_control_logic = _patched_emission_logic

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
rs.run_simulation(
    seed        = options.seed,
    method      = options.method,
    use_gui     = not options.nogui,
    output_file = options.output,
    route_file  = options.route,
)

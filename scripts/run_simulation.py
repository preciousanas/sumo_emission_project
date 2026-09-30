#!/usr/bin/env python3
"""
SUMO Simulation Runner for Traffic Signal Control Comparison.

Runs a single simulation instance for either:
  - The baseline actuated control (gap-based, vehicle presence)
  - The proposed emission-based actuated control

Outputs three files per run (all in the same directory as --output):
  result_seed_<N>.csv      — per-timestep per-vehicle data (TraCI)
  tripinfo_seed_<N>.xml    — full-trip summary per vehicle (SUMO native)
  statistics_seed_<N>.xml  — aggregate simulation statistics (SUMO native)

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
No functional changes in this file relative to the version used in the
previous re-run.  It is included here as the definitive reference copy
that matches run_experiment.py and analyze_results.py.

The --route option (added in an earlier revision) is the key integration
point: run_experiment.py passes a volume-specific route file path here,
which causes a temporary sumocfg to be built pointing at that file instead
of the default config/traffic.sumocfg.  This is what enables multi-volume
sweeps without any manual editing of config files.

The Unicode arrow characters in the two print() calls after traci.close()
have been replaced with ASCII -> to avoid UnicodeEncodeError on Windows
CP1252 terminals (the crash described in the analysis report).
"""

import os
import sys
import csv
import optparse
import random
from collections import defaultdict
import traci
import traci.exceptions

# ---------------------------------------------------------------------------
# Configuration and default parameters
# ---------------------------------------------------------------------------

RANDOM_SEED        = 42
SIMULATION_METHOD  = 'baseline'   # 'baseline' or 'emission_based'
MIN_GREEN          = 15           # seconds minimum green before switching allowed
MAX_GREEN          = 60           # seconds maximum green before forced switch
EMISSION_THRESHOLD = 50.0         # mg/s CO2 differential to trigger emission switch (reduced from 100 to 50 on 2026-04-23)
MAX_GAP            = 3.0          # seconds gap threshold for baseline control

# 300m tracking box centred on the intersection
BOX_SIZE             = 300
INTERSECTION_CENTER  = (500, 500)
BOX_BOUNDS = (
    INTERSECTION_CENTER[0] - BOX_SIZE / 2,   # min_x
    INTERSECTION_CENTER[0] + BOX_SIZE / 2,   # max_x
    INTERSECTION_CENTER[1] - BOX_SIZE / 2,   # min_y
    INTERSECTION_CENTER[1] + BOX_SIZE / 2,   # max_y
)

# Traffic light and edge configuration
TL_ID         = "center"
GREEN_PHASES  = {0: ['north_in', 'south_in'],   # NS green
                 2: ['east_in',  'west_in']}     # EW green
YELLOW_PHASES = {1, 3}


# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option(
        '--seed', type='int', default=RANDOM_SEED,
        help='Random seed for reproducible vehicle generation.')
    opt_parser.add_option(
        '--method', type='string', default=SIMULATION_METHOD,
        help="Control method: 'baseline' or 'emission_based'.")
    opt_parser.add_option(
        '--nogui', action='store_true', default=False,
        help='Run headless sumo (faster, no visualisation).')
    opt_parser.add_option(
        '--output', type='string', default='simulation_data.csv',
        help='Path to the per-timestep CSV output file.')
    opt_parser.add_option(
        '--route', type='string', default=None,
        help=(
            'Path to a specific .rou.xml file. '
            'When supplied, a temporary sumocfg is built pointing at this '
            'route file instead of the default config/traffic.sumocfg. '
            'Used by run_experiment.py for multi-volume sweeps.'
        ))
    options, _ = opt_parser.parse_args()
    return options


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

def run_simulation(seed, method, use_gui, output_file, route_file=None):
    """
    Run one SUMO simulation and write all outputs.

    Parameters
    ----------
    seed : int
    method : str   'baseline' or 'emission_based'
    use_gui : bool
    output_file : str   path to result_seed_N.csv
    route_file : str or None
        When provided, a temporary sumocfg is created pointing at this
        route file.  When None, config/traffic.sumocfg is used as-is.
    """

    # 1. SUMO_HOME
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. Derive sibling output paths from the CSV path
    #    CSV    : data/<vol>/<method>/result_seed_<N>.csv
    #    tripinfo: data/<vol>/<method>/tripinfo_seed_<N>.xml
    #    stats  : data/<vol>/<method>/statistics_seed_<N>.xml
    stem            = os.path.basename(os.path.splitext(output_file)[0])
    out_dir         = os.path.dirname(output_file)
    tripinfo_file   = os.path.join(out_dir, stem.replace('result_', 'tripinfo_')   + '.xml')
    statistics_file = os.path.join(out_dir, stem.replace('result_', 'statistics_') + '.xml')

    # 3. Build SUMO config path
    if route_file is not None:
        from capacity.sumo_config_builder import build_sumocfg_for_run
        from pathlib import Path
        import tempfile

        net_file = Path(PROJECT_ROOT) / 'config' / 'traffic.net.xml'
        tmp_cfg  = Path(tempfile.mktemp(suffix='.sumocfg'))
        build_sumocfg_for_run(
            base_sumocfg_path   = Path(PROJECT_ROOT) / 'config' / 'traffic.sumocfg',
            output_sumocfg_path = tmp_cfg,
            net_file_path       = net_file,
            route_file_path     = Path(route_file),
            begin               = 0,
            end                 = 3600,
            step_length         = 1.0,
        )
        config_file = str(tmp_cfg)
    else:
        config_file = os.path.join(PROJECT_ROOT, 'config', 'traffic.sumocfg')

    sumo_binary = 'sumo-gui' if use_gui else 'sumo'

    # 4. SUMO command
    sumo_cmd = [
        sumo_binary,
        '-c',             config_file,
        '--step-length',  '1',
        '--random',
        '--seed',         str(seed),
        '--tripinfo-output',  tripinfo_file,
        '--statistic-output', statistics_file,
    ]

    # 5. Start TraCI
    try:
        traci.start(sumo_cmd)
    except traci.exceptions.FatalTraCIError as e:
        sys.exit(f"Failed to start SUMO: {e}")

    # 6. State tracking
    last_phase_change  = 0.0
    current_phase      = -1
    vehicle_data       = {}
    emissions_data     = defaultdict(lambda: {'co2': 0.0, 'time': 0.0})
    departed_vehicles  = set()
    arrived_vehicles   = set()
    phase_switch_count = 0

    # 7. Simulation loop
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'sim_time', 'veh_id', 'veh_type', 'speed', 'wait_time',
            'co2_emission', 'edge_id', 'in_box', 'phase', 'phase_duration'
        ])
        writer.writerow([f'# Seed: {seed}, Method: {method}'])

        try:
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                current_time = traci.simulation.getTime()

                # Traffic light state
                try:
                    new_phase = traci.trafficlight.getPhase(TL_ID)
                    if new_phase != current_phase:
                        last_phase_change = current_time
                        current_phase     = new_phase
                    phase_duration = current_time - last_phase_change
                except traci.exceptions.TraCIException:
                    phase_duration = 0
                    new_phase      = -1

                # Control logic
                if current_phase not in YELLOW_PHASES and current_phase != -1:
                    if method == 'baseline':
                        should_switch = baseline_control_logic(current_phase, phase_duration)
                    else:
                        should_switch = emission_based_control_logic(current_phase, phase_duration)

                    if should_switch:
                        yellow_phase = 1 if current_phase == 0 else 3
                        traci.trafficlight.setPhase(TL_ID, yellow_phase)
                        phase_switch_count += 1
                        print(f"Step {step}: Switching from phase {current_phase} "
                              f"to {yellow_phase}")

                # Per-vehicle data collection
                for veh_id in traci.vehicle.getIDList():
                    try:
                        pos      = traci.vehicle.getPosition(veh_id)
                        speed    = traci.vehicle.getSpeed(veh_id)
                        veh_type = traci.vehicle.getTypeID(veh_id)
                        edge     = traci.vehicle.getRoadID(veh_id)
                        co2      = traci.vehicle.getCO2Emission(veh_id)
                        in_box   = is_in_box(pos)

                        if veh_id not in vehicle_data:
                            vehicle_data[veh_id] = {
                                'wait_start': None, 'total_wait': 0.0,
                                'type': veh_type
                            }
                            departed_vehicles.add(veh_id)

                        # Stop-line wait (approach edges only)
                        if (speed < 0.1 and
                                edge in ['north_in', 'south_in', 'east_in', 'west_in']):
                            if vehicle_data[veh_id]['wait_start'] is None:
                                vehicle_data[veh_id]['wait_start'] = current_time
                        else:
                            if vehicle_data[veh_id]['wait_start'] is not None:
                                waited = current_time - vehicle_data[veh_id]['wait_start']
                                vehicle_data[veh_id]['total_wait'] += waited
                                vehicle_data[veh_id]['wait_start']  = None

                        if in_box:
                            emissions_data[veh_type]['co2']  += co2
                            emissions_data[veh_type]['time'] += 1

                        wait_start = vehicle_data[veh_id]['wait_start']
                        wait_val   = (current_time - wait_start) if wait_start else 0.0
                        writer.writerow([
                            current_time, veh_id, veh_type, speed, wait_val,
                            co2, edge, in_box, current_phase, phase_duration
                        ])

                    except traci.exceptions.TraCIException:
                        continue

                arrived_vehicles.update(traci.simulation.getArrivedIDList())
                step += 1

        except Exception as e:
            print(f"Simulation error at step {step}: {e}")
        finally:
            traci.close()
            print(f"\nSimulation completed.  Seed: {seed}, Method: {method}")
            print(f"Phase switches: {phase_switch_count}")
            print(f"Tripinfo written  -> {tripinfo_file}")
            print(f"Statistics written-> {statistics_file}")
            print_summary_statistics(
                vehicle_data, departed_vehicles, arrived_vehicles, emissions_data
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_in_box(position):
    x, y = position
    return (BOX_BOUNDS[0] <= x <= BOX_BOUNDS[1] and
            BOX_BOUNDS[2] <= y <= BOX_BOUNDS[3])


def baseline_control_logic(current_green_phase, phase_duration):
    """Gap-based actuated control. Returns True to switch."""
    if phase_duration < MIN_GREEN:
        return False
    if phase_duration >= MAX_GREEN:
        print(f"Baseline: max green ({MAX_GREEN}s) reached. Switching.")
        return True

    for edge in GREEN_PHASES[current_green_phase]:
        try:
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            if not vehicles:
                print(f"Baseline: no vehicles on {edge}. Switching.")
                return True
            last_veh = vehicles[-1]
            tls_list = traci.vehicle.getNextTLS(last_veh)
            if not tls_list:
                print(f"Baseline: vehicle {last_veh} has no upcoming TLS. Switching.")
                return True
            gap = tls_list[0][2]
            if gap > MAX_GAP:
                print(f"Baseline: gap {gap:.2f}m > {MAX_GAP}m on {edge}. Switching.")
                return True
        except (traci.exceptions.TraCIException, IndexError):
            return True

    return False


def emission_based_control_logic(current_green_phase, phase_duration):
    """Emission-aware actuated control. Returns True to switch."""
    if phase_duration < MIN_GREEN:
        return False
    if phase_duration >= MAX_GREEN:
        return True

    opposing_phase = 2 if current_green_phase == 0 else 0

    co2_green = sum(
        traci.vehicle.getCO2Emission(v)
        for edge in GREEN_PHASES[current_green_phase]
        for v in traci.edge.getLastStepVehicleIDs(edge)
        if _safe_co2(v) is not None
    )
    co2_red = sum(
        traci.vehicle.getCO2Emission(v)
        for edge in GREEN_PHASES[opposing_phase]
        for v in traci.edge.getLastStepVehicleIDs(edge)
        if _safe_co2(v) is not None
    )

    if co2_red > (co2_green + EMISSION_THRESHOLD):
        print(f"Emission switch: red={co2_red:.2f}, green={co2_green:.2f}")
        return True
    return False


def _safe_co2(veh_id):
    try:
        return traci.vehicle.getCO2Emission(veh_id)
    except traci.exceptions.TraCIException:
        return None


def print_summary_statistics(vehicle_data, departed, arrived, emissions_data):
    total_dep    = len(departed)
    total_arr    = len(arrived)
    avg_wait     = (
        sum(v['total_wait'] for v in vehicle_data.values()) / total_arr
        if total_arr > 0 else 0.0
    )
    print("\n--- Simulation Summary ---")
    print(f"Departed Vehicles:    {total_dep}")
    print(f"Arrived Vehicles:     {total_arr}")
    print(f"Average Waiting Time: {avg_wait:.2f}s  (full-route stop-line wait)")
    print("\n--- Avg CO2 Emission Rates (mg/s) inside intersection box ---")
    for veh_type, data in sorted(emissions_data.items()):
        if data['time'] > 0:
            print(f"  {veh_type}: {data['co2'] / data['time']:.2f} mg/s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    options = get_options()
    run_simulation(
        seed        = options.seed,
        method      = options.method,
        use_gui     = not options.nogui,
        output_file = options.output,
        route_file  = options.route,
    )

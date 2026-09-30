#!/usr/bin/env python3
"""
run_composition_experiment.py — Task 3: Vehicle Composition Sensitivity.

PURPOSE
-------
Run both baseline and emission_based simulations for each vehicle fleet
composition, at the 4 representative volumes.  The composition changes
the mix of vehicle types (and therefore the per-vehicle CO2 output),
but the total demand (total veh/hr) is kept constant at the target volume.

COMPOSITIONS
  comp_baseline — 40% car, 15% bus, 15% truck, 10% van, 10% gen, 10% coach
  comp_heavy    — 20% car, 20% bus, 25% truck, 15% van,  5% gen, 15% coach
  comp_light    — 80% car,  2% bus,  2% truck,  5% van, 10% gen,  1% coach
  comp_mixed    — 55% car, 12% bus, 18% truck,  7% van,  5% gen,  3% coach

HOW SCALING WORKS
  Each composition route file defines flows at 568 veh/hr (4 approaches ×
  142 veh/hr).  This script scales all vehsPerHour values by
  (target_volume / 568) and writes a temp .rou.xml before passing to SUMO.
  Original composition files are never modified.

USAGE
-----
  # Run all compositions at all 4 representative volumes:
  python scripts/experiments/run_composition_experiment.py

  # Run specific compositions:
  python scripts/experiments/run_composition_experiment.py --compositions heavy light

  # Run specific volumes:
  python scripts/experiments/run_composition_experiment.py --volumes 1500 5000

  # comp_baseline results can be compared to existing data/ as validation:
  python scripts/experiments/run_composition_experiment.py --compositions baseline

OUTPUT STRUCTURE
    experiments/composition/data/comp_XXX/V####/{baseline,emission_based}/
        result_seed_N.csv   tripinfo_seed_N.xml   statistics_seed_N.xml
        log_seed_N.txt

NOTE: Original files in data/, scripts/, config/ are never modified.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / 'scripts'
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'composition'
COMP_CFG_DIR = EXP_ROOT / 'config'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_VOLUME    = 568.0    # veh/hr — the reference total in composition files
REP_VOLUMES    = [1500, 2900, 3300, 5000]
SEEDS          = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]
METHODS        = ['baseline', 'emission_based']
DEFAULT_THRESHOLD = 50.0

COMPOSITIONS = ['baseline', 'heavy', 'light', 'mixed']

COMP_DESCRIPTIONS = {
    'baseline': '40% car, 15% bus, 15% truck, 10% van, 10% gen, 10% coach',
    'heavy':    '20% car, 20% bus, 25% truck, 15% van,  5% gen, 15% coach',
    'light':    '80% car,  2% bus,  2% truck,  5% van, 10% gen,  1% coach',
    'mixed':    '55% car, 12% bus, 18% truck,  7% van,  5% gen,  3% coach',
}

REGIME_LABELS = {
    1500: 'Free-flow',
    2900: 'Near-saturation',
    3300: 'Breakdown onset',
    5000: 'Gridlock',
}

# ---------------------------------------------------------------------------
# Route file scaling
# ---------------------------------------------------------------------------

def scale_route_file(template_path: Path, target_volume: int,
                     out_path: Path) -> None:
    """
    Parse template_path, scale all vehsPerHour values by (target_volume/BASE_VOLUME),
    and write the scaled XML to out_path.
    """
    scale = target_volume / BASE_VOLUME
    tree  = ET.parse(template_path)
    root  = tree.getroot()

    for flow_elem in root.iter('flow'):
        if 'vehsPerHour' in flow_elem.attrib:
            orig = float(flow_elem.attrib['vehsPerHour'])
            flow_elem.set('vehsPerHour', f'{orig * scale:.4f}')

    os.makedirs(out_path.parent, exist_ok=True)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def build_temp_sumocfg(net_file: Path, route_file: Path, out_path: Path) -> None:
    """Build a minimal .sumocfg pointing at the given net and route files."""
    cfg = ET.Element('configuration')

    inp = ET.SubElement(cfg, 'input')
    ET.SubElement(inp, 'net-file',    value=str(net_file))
    ET.SubElement(inp, 'route-files', value=str(route_file))

    time_elem = ET.SubElement(cfg, 'time')
    ET.SubElement(time_elem, 'begin',  value='0')
    ET.SubElement(time_elem, 'end',    value='3600')

    proc = ET.SubElement(cfg, 'processing')
    ET.SubElement(proc, 'step-length', value='1.0')

    os.makedirs(out_path.parent, exist_ok=True)
    ET.ElementTree(cfg).write(out_path, encoding='utf-8', xml_declaration=True)


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def run_one(seed: int, volume: int, composition: str, method: str,
            threshold: float, use_gui: bool,
            grand_idx: int, grand_total: int) -> bool:
    vol_label  = f'V{volume:04d}'
    comp_label = f'comp_{composition}'

    out_dir = EXP_ROOT / 'data' / comp_label / vol_label / method
    os.makedirs(out_dir, exist_ok=True)

    output_csv = out_dir / f'result_seed_{seed}.csv'
    log_file   = out_dir / f'log_seed_{seed}.txt'

    if output_csv.exists() and output_csv.stat().st_size > 0:
        print(f"  [{grand_idx}/{grand_total}] SKIP (exists): "
              f"{comp_label}/{vol_label}/{method}/seed={seed}")
        return True

    print(f"  [{grand_idx}/{grand_total}] "
          f"comp={composition:<8}  vol={vol_label}  method={method:<13}  seed={seed}",
          flush=True)

    # Build scaled route file in a temp dir
    template     = COMP_CFG_DIR / f'comp_{composition}.rou.xml'
    net_file     = PROJECT_ROOT / 'config' / 'traffic.net.xml'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_route = Path(tmpdir) / f'traffic_{comp_label}_{vol_label}.rou.xml'
        tmp_cfg   = Path(tmpdir) / f'run_{comp_label}_{vol_label}_{seed}.sumocfg'

        scale_route_file(template, volume, tmp_route)
        build_temp_sumocfg(net_file, tmp_route, tmp_cfg)

        # Choose wrapper: threshold wrapper for emission_based
        if method == 'emission_based':
            sim_script = str(SCRIPT_DIR / 'run_simulation_threshold.py')
            cmd = [
                sys.executable, sim_script,
                '--seed',               str(seed),
                '--method',             method,
                '--output',             str(output_csv),
                '--route',              str(tmp_route),
                '--emission-threshold', str(threshold),
            ]
        else:
            sim_script = str(SCRIPTS_DIR / 'run_simulation.py')
            cmd = [
                sys.executable, sim_script,
                '--seed',   str(seed),
                '--method', method,
                '--output', str(output_csv),
                '--route',  str(tmp_route),
            ]

        if not use_gui:
            cmd.append('--nogui')

        try:
            with open(log_file, 'w') as log:
                subprocess.run(
                    cmd,
                    cwd=str(SCRIPTS_DIR),
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            return True
        except subprocess.CalledProcessError:
            print(f"    ERROR — see {log_file.relative_to(PROJECT_ROOT)}")
            return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Vehicle composition sensitivity experiment.'
    )
    ap.add_argument(
        '--compositions', nargs='+', default=COMPOSITIONS,
        choices=COMPOSITIONS, metavar='C',
        help=f'Compositions to run (default: all). Choices: {COMPOSITIONS}.'
    )
    ap.add_argument(
        '--volumes', nargs='+', type=int, default=REP_VOLUMES, metavar='V',
        help=f'Demand volumes (default: {REP_VOLUMES}).'
    )
    ap.add_argument(
        '--seeds', nargs='+', type=int, default=SEEDS, metavar='S',
        help='Random seeds (default: all 10).'
    )
    ap.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help=f'Emission threshold in mg/s (default: {DEFAULT_THRESHOLD}).'
    )
    ap.add_argument(
        '--methods', nargs='+', default=METHODS,
        choices=METHODS,
        help='Control methods to run (default: both baseline and emission_based).'
    )
    ap.add_argument(
        '--gui', action='store_true', default=False,
        help='Run SUMO with GUI (debugging only).'
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    compositions = args.compositions
    volumes      = sorted(args.volumes)
    seeds        = args.seeds
    methods      = args.methods

    print("\n" + "=" * 65)
    print("  Vehicle Composition Experiment")
    print("=" * 65)
    for c in compositions:
        print(f"  {f'comp_{c}':<16}: {COMP_DESCRIPTIONS.get(c, '')}")
    print(f"  Volumes     : {[f'V{v:04d}' for v in volumes]}")
    print(f"  Methods     : {methods}")
    print(f"  Seeds       : {seeds}")
    print(f"  Threshold   : {args.threshold} mg/s")
    print(f"  Output      : experiments/composition/data/")

    grand_total = len(compositions) * len(volumes) * len(seeds) * len(methods)
    grand_idx   = 0
    grand_start = time.time()
    errors      = []

    print(f"\n  Total simulations: {grand_total}\n")

    for comp in compositions:
        comp_label = f'comp_{comp}'
        print(f"\n{'─'*65}")
        print(f"  Composition: {comp_label}  ({COMP_DESCRIPTIONS.get(comp, '')})")
        print(f"{'─'*65}")
        for volume in volumes:
            regime = REGIME_LABELS.get(volume, '?')
            print(f"\n  Volume V{volume:04d} ({regime})")
            for method in methods:
                for seed in seeds:
                    grand_idx += 1
                    ok = run_one(seed, volume, comp, method, args.threshold,
                                 args.gui, grand_idx, grand_total)
                    if not ok:
                        errors.append((comp, volume, method, seed))

    elapsed = time.time() - grand_start
    print(f"\n{'='*65}")
    print(f"  Completed in {elapsed:.1f}s")
    if errors:
        print(f"  ERRORS ({len(errors)}): {errors}")
    else:
        print("  All simulations succeeded.")
    print(f"{'='*65}")
    print("\nNext step — analyse results:")
    print("  python scripts/experiments/analyze_composition.py")


if __name__ == '__main__':
    main()

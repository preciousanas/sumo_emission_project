#!/usr/bin/env python3
"""
run_penetration_experiment.py — Task 2: Penetration Rate Experiments.

PURPOSE
-------
Test how the emission-based controller performs when only a fraction of
vehicles are visible to the sensor (partial V2X / probe-vehicle coverage).

Penetration rate = fraction of vehicles whose instantaneous CO2 is used
in the switching decision.  The fleet itself is unchanged; only the
information available to the controller changes.

pen=1.0 (100%) results are read from the existing data/ folder — no re-run.
pen=0.2, 0.4, 0.6, 0.8 are run fresh using run_simulation_ext.py.

USAGE
-----
  # Run all penetration rates at all 4 representative volumes:
  python scripts/experiments/run_penetration_experiment.py

  # Run specific penetration rates:
  python scripts/experiments/run_penetration_experiment.py --penetration-rates 0.2 0.4

  # Run specific volumes:
  python scripts/experiments/run_penetration_experiment.py --volumes 1500 5000

OUTPUT STRUCTURE
    experiments/penetration_rate/data/pen_XXX/V####/emission_based/
        result_seed_N.csv   tripinfo_seed_N.xml   statistics_seed_N.xml
        log_seed_N.txt
    (pen_100 reads directly from data/V####/emission_based/ — no copy made)

NOTE: Original files in data/, scripts/, config/ are never modified.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / 'scripts'
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'penetration_rate'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PENETRATION_RATES = [0.2, 0.4, 0.6, 0.8]   # 1.0 reused from data/
REFERENCE_PENETRATION     = 1.0
REP_VOLUMES               = [1500, 2900, 3300, 5000]
SEEDS                     = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]
DEFAULT_THRESHOLD         = 50.0

REGIME_LABELS = {
    1500: 'Free-flow',
    2900: 'Near-saturation',
    3300: 'Breakdown onset',
    5000: 'Gridlock',
}


def route_file_for_volume(volume: int) -> Path:
    vol_label  = f'V{volume:04d}'
    route_file = (PROJECT_ROOT / 'data' / 'capacity_analysis' / 'volumes'
                  / vol_label / f'traffic_{vol_label}.rou.xml')
    if not route_file.exists():
        raise FileNotFoundError(
            f"Route file not found for V{volume:04d}:\n  {route_file}"
        )
    return route_file


def pen_label(rate: float) -> str:
    return f'pen_{int(round(rate * 100)):03d}'


def run_one(seed: int, volume: int, penetration: float, threshold: float,
            use_gui: bool, grand_idx: int, grand_total: int) -> bool:
    vol_label  = f'V{volume:04d}'
    plabel     = pen_label(penetration)
    route_file = route_file_for_volume(volume)

    out_dir = EXP_ROOT / 'data' / plabel / vol_label / 'emission_based'
    os.makedirs(out_dir, exist_ok=True)

    output_csv = out_dir / f'result_seed_{seed}.csv'
    log_file   = out_dir / f'log_seed_{seed}.txt'

    if output_csv.exists() and output_csv.stat().st_size > 0:
        print(f"  [{grand_idx}/{grand_total}] SKIP (exists): "
              f"{plabel}/{vol_label}/seed={seed}")
        return True

    print(f"  [{grand_idx}/{grand_total}] "
          f"pen={penetration:.0%}  vol={vol_label}  seed={seed}", flush=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / 'run_simulation_ext.py'),
        '--seed',             str(seed),
        '--method',           'emission_based',
        '--output',           str(output_csv),
        '--route',            str(route_file),
        '--penetration-rate', str(penetration),
        '--emission-threshold', str(threshold),
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


def parse_args():
    ap = argparse.ArgumentParser(
        description='Run penetration rate experiments for emission-based controller.'
    )
    ap.add_argument(
        '--penetration-rates', nargs='+', type=float,
        default=DEFAULT_PENETRATION_RATES, metavar='P',
        help='Penetration rates to test, e.g. 0.2 0.4 0.6 0.8 (default: all four).'
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
        '--gui', action='store_true', default=False,
        help='Run SUMO with GUI (for debugging).'
    )
    return ap.parse_args()


def main():
    args = parse_args()

    penetration_rates = sorted(set(args.penetration_rates))
    volumes           = sorted(args.volumes)
    seeds             = args.seeds

    print("\n" + "=" * 65)
    print("  Penetration Rate Experiment")
    print("=" * 65)
    print(f"  Penetration rates : {[f'{p:.0%}' for p in penetration_rates]}")
    print(f"  (pen=100% reused from existing data/ — not re-run)")
    print(f"  Volumes           : {[f'V{v:04d}' for v in volumes]}")
    print(f"  Seeds             : {seeds}")
    print(f"  Threshold         : {args.threshold} mg/s")
    print(f"  Output            : experiments/penetration_rate/data/")

    grand_total = len(penetration_rates) * len(volumes) * len(seeds)
    grand_idx   = 0
    grand_start = time.time()
    errors      = []

    print(f"\n  Total new simulations: {grand_total}\n")

    for pen in penetration_rates:
        plabel = pen_label(pen)
        print(f"\n{'─'*65}")
        print(f"  Penetration = {pen:.0%}  [{plabel}]")
        print(f"{'─'*65}")
        for volume in volumes:
            regime = REGIME_LABELS.get(volume, '?')
            print(f"\n  Volume V{volume:04d} ({regime})")
            for seed in seeds:
                grand_idx += 1
                ok = run_one(seed, volume, pen, args.threshold,
                             args.gui, grand_idx, grand_total)
                if not ok:
                    errors.append((pen, volume, seed))

    elapsed = time.time() - grand_start
    print(f"\n{'='*65}")
    print(f"  Completed in {elapsed:.1f}s")
    if errors:
        print(f"  ERRORS ({len(errors)}): {errors}")
    else:
        print("  All simulations succeeded.")
    print(f"{'='*65}")
    print("\nNext step — analyse results:")
    print("  python scripts/experiments/analyze_penetration.py")


if __name__ == '__main__':
    main()

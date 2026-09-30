#!/usr/bin/env python3
"""
run_threshold_search.py — Task 1: Coarse + Fine Grid Search for Optimal Emission Threshold.

PURPOSE
-------
Run Pipeline-B emission_based simulations across a range of EMISSION_THRESHOLD
values to find the value that maximises CO2 savings while keeping throughput
degradation above -1% and wait time from increasing.

Baseline results (method='baseline') are threshold-independent — they are
re-used from the existing data/ folder and are NEVER re-run here.

COARSE GRID (run first)
    python scripts/experiments/run_threshold_search.py --phase coarse

FINE GRID (run after reviewing analyze_threshold_search.py output)
    python scripts/experiments/run_threshold_search.py --phase fine --fine-thresholds 30 40 45 55 60 65

CUSTOM VOLUMES / SEEDS
    python scripts/experiments/run_threshold_search.py --volumes 1500 2900 --seeds 42 10

OUTPUT STRUCTURE
    experiments/threshold_search/data/thr_XXX/V####/emission_based/
        result_seed_N.csv
        tripinfo_seed_N.xml
        statistics_seed_N.xml
        log_seed_N.txt

NOTE: Original files in data/, scripts/, config/ are never modified.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / 'scripts'
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'threshold_search'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
COARSE_THRESHOLDS   = [10, 25, 75, 100, 150, 200, 300]   # skip 50 — already in data/
REFERENCE_THRESHOLD = 50                                   # existing baseline run
REP_VOLUMES         = [1500, 2900, 3300, 5000]            # one per traffic regime
SEEDS               = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]

REGIME_LABELS = {
    1500: 'Free-flow',
    2900: 'Near-saturation',
    3300: 'Breakdown onset',
    5000: 'Gridlock',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def route_file_for_volume(volume: int) -> Path:
    """Return the scaled route file already produced by Pipeline A."""
    vol_label  = f'V{volume:04d}'
    route_file = (PROJECT_ROOT / 'data' / 'capacity_analysis' / 'volumes'
                  / vol_label / f'traffic_{vol_label}.rou.xml')
    if not route_file.exists():
        raise FileNotFoundError(
            f"Route file not found for V{volume:04d}:\n  {route_file}\n"
            "Run Pipeline A first (run_capacity_experiment.py)."
        )
    return route_file


def run_one(seed: int, volume: int, threshold: float, use_gui: bool,
            grand_idx: int, grand_total: int) -> bool:
    """
    Run a single emission_based simulation with the given threshold.
    Returns True on success.
    """
    vol_label  = f'V{volume:04d}'
    thr_label  = f'thr_{int(threshold):03d}'
    route_file = route_file_for_volume(volume)

    out_dir = EXP_ROOT / 'data' / thr_label / vol_label / 'emission_based'
    os.makedirs(out_dir, exist_ok=True)

    output_csv = out_dir / f'result_seed_{seed}.csv'
    log_file   = out_dir / f'log_seed_{seed}.txt'

    # Skip if already done
    if output_csv.exists() and output_csv.stat().st_size > 0:
        print(f"  [{grand_idx}/{grand_total}] SKIP (exists): "
              f"{thr_label}/{vol_label}/seed={seed}")
        return True

    print(f"  [{grand_idx}/{grand_total}] "
          f"thr={threshold:>5.1f}  vol={vol_label}  seed={seed}", flush=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / 'run_simulation_threshold.py'),
        '--seed',               str(seed),
        '--method',             'emission_based',
        '--output',             str(output_csv),
        '--route',              str(route_file),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Threshold grid search for optimal emission controller threshold.'
    )
    ap.add_argument(
        '--phase', choices=['coarse', 'fine'], default='coarse',
        help='Which grid phase to run (default: coarse).'
    )
    ap.add_argument(
        '--fine-thresholds', nargs='+', type=float, default=None,
        metavar='T',
        help='Threshold values for the fine grid phase (e.g. 30 40 45 55 60 65).'
    )
    ap.add_argument(
        '--volumes', nargs='+', type=int, default=REP_VOLUMES,
        metavar='V',
        help=f'Demand volumes to test (default: {REP_VOLUMES}).'
    )
    ap.add_argument(
        '--seeds', nargs='+', type=int, default=SEEDS,
        metavar='S',
        help='Random seeds to use (default: all 10).'
    )
    ap.add_argument(
        '--gui', action='store_true', default=False,
        help='Run SUMO with GUI (slow; for debugging only).'
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.phase == 'coarse':
        thresholds = COARSE_THRESHOLDS
        print("\n" + "=" * 65)
        print("  Threshold Grid Search — COARSE PHASE")
        print(f"  Thresholds : {thresholds}  (thr=50 reused from data/)")
        print("=" * 65)
    else:
        if not args.fine_thresholds:
            print("[ERROR] --fine-thresholds required for fine phase.")
            print("  Review analyze_threshold_search.py output first,")
            print("  then re-run with e.g.:  --fine-thresholds 30 40 45 55 60")
            sys.exit(1)
        thresholds = sorted(args.fine_thresholds)
        print("\n" + "=" * 65)
        print("  Threshold Grid Search — FINE PHASE")
        print(f"  Thresholds : {thresholds}")
        print("=" * 65)

    volumes = sorted(args.volumes)
    seeds   = args.seeds

    print(f"  Volumes    : {[f'V{v:04d}' for v in volumes]}")
    print(f"  Seeds      : {seeds}")
    print(f"  Output     : experiments/threshold_search/data/")

    grand_total = len(thresholds) * len(volumes) * len(seeds)
    grand_idx   = 0
    grand_start = time.time()
    errors      = []

    print(f"\n  Total simulations: {grand_total}\n")

    for thr in thresholds:
        thr_label = f'thr_{int(thr):03d}'
        print(f"\n{'─'*65}")
        print(f"  Threshold = {thr} mg/s  [{thr_label}]")
        print(f"{'─'*65}")
        for volume in volumes:
            regime = REGIME_LABELS.get(volume, '?')
            print(f"\n  Volume V{volume:04d} ({regime})")
            for seed in seeds:
                grand_idx += 1
                ok = run_one(seed, volume, thr, args.gui, grand_idx, grand_total)
                if not ok:
                    errors.append((thr, volume, seed))

    elapsed = time.time() - grand_start
    print(f"\n{'='*65}")
    print(f"  Completed in {elapsed:.1f}s")
    if errors:
        print(f"  ERRORS ({len(errors)}): {errors}")
    else:
        print("  All simulations succeeded.")
    print(f"{'='*65}")
    print("\nNext step — analyse results:")
    print("  python scripts/experiments/analyze_threshold_search.py")


if __name__ == '__main__':
    main()

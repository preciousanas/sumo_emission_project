#!/usr/bin/env python3
"""
run_experiment.py — Pipeline B batch runner with multi-volume sweep support.

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
1. Added argparse (replaces no argument parsing at all).
   New optional argument: --volumes
   If omitted, the script auto-discovers volumes from the capacity analysis
   folder (data/capacity_analysis/volumes/V####/) — exactly mirroring how
   Pipeline A organises its data.  If provided, the listed integers are used
   directly instead.

2. Auto-discovery of volumes.
   discover_volumes() scans data/capacity_analysis/volumes/ for sub-folders
   whose names match V#### and whose traffic_V####.rou.xml file exists.
   This means you never have to type volume numbers manually — just run:
       python scripts/run_experiment.py
   and all volumes that Pipeline A produced will automatically be swept.

3. Volume-organised output folder structure.
   Outputs now live under data/<vol_label>/<method>/ instead of data/<method>/.
   This mirrors Pipeline A's structure under data/capacity_analysis/volumes/.
   Example for seed 42, baseline, V0600:
       data/V0600/baseline/result_seed_42.csv
       data/V0600/baseline/log_seed_42.txt
       data/V0600/baseline/tripinfo_seed_42.xml
       data/V0600/baseline/statistics_seed_42.xml

4. Scaled route files are read directly from the capacity analysis folder.
   Instead of calling route_scaler.py to re-generate the files, the script
   re-uses the already-scaled traffic_V####.rou.xml files that Pipeline A
   produced.  This avoids duplication and guarantees that Pipeline B uses
   identical route files to the ones whose capacity was measured.
   The route file path passed to run_simulation.py is therefore:
       data/capacity_analysis/volumes/V####/traffic_V####.rou.xml

5. --route argument passed to run_simulation.py.
   run_simulation.py already accepts --route; this script now always passes
   it so that each volume sweep uses the correct scaled route file.

6. Progress reporting now includes the volume label so the terminal output
   is easy to follow across a long multi-volume run.

USAGE
-----
# Sweep all volumes auto-discovered from Pipeline A output (recommended):
    python scripts/run_experiment.py

# Sweep specific volumes only:
    python scripts/run_experiment.py --volumes 600 1200 2000

# Run with GUI (for debugging a single seed):
    python scripts/run_experiment.py --volumes 600 --gui

FOLDER STRUCTURE PRODUCED
--------------------------
data/
  V0200/
    baseline/       result_seed_*.csv, log_seed_*.txt,
                    tripinfo_seed_*.xml, statistics_seed_*.xml
    emission_based/ (same files)
  V0400/
    baseline/
    emission_based/
  ... (one folder per discovered volume)
"""

import os
import subprocess
import time
import argparse
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS        = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]
METHODS      = ['baseline', 'emission_based']
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / 'data'

# Location where Pipeline A stores its per-volume route files.
# Each sub-folder is named V#### and contains traffic_V####.rou.xml.
CAPACITY_VOLUMES_DIR = DATA_DIR / 'capacity_analysis' / 'volumes'


# ---------------------------------------------------------------------------
# Volume discovery
# ---------------------------------------------------------------------------

def discover_volumes() -> list:
    """
    Scan CAPACITY_VOLUMES_DIR for sub-folders whose route file exists.

    Returns a sorted list of integer volume values, e.g. [200, 400, 600, ...].
    Raises SystemExit with a helpful message if the folder does not exist or
    contains no valid volume sub-folders.
    """
    if not CAPACITY_VOLUMES_DIR.exists():
        raise SystemExit(
            f"\n[ERROR] Capacity volumes folder not found:\n"
            f"  {CAPACITY_VOLUMES_DIR}\n\n"
            f"Pipeline A must be run first so its route files exist.\n"
            f"Run:  python scripts/run_capacity_experiment.py\n"
            f"Or pass volumes manually:  python scripts/run_experiment.py "
            f"--volumes 600 1200 2000"
        )

    volumes = []
    for entry in sorted(CAPACITY_VOLUMES_DIR.iterdir()):
        # Folder name must match V#### pattern
        if not (entry.is_dir() and entry.name.startswith('V') and
                entry.name[1:].isdigit()):
            continue
        vol_int   = int(entry.name[1:])
        route_file = entry / f'traffic_{entry.name}.rou.xml' # file name change
        if route_file.exists():
            volumes.append(vol_int)
        else:
            print(f"  [discover] Skipping {entry.name}: "
                  f"route file not found ({route_file.name})")

    if not volumes:
        raise SystemExit(
            f"\n[ERROR] No valid volume folders found in:\n"
            f"  {CAPACITY_VOLUMES_DIR}\n\n"
            f"Each volume folder must contain traffic_V####.rou.xml.\n"  # file name change
            f"Re-run Pipeline A or pass volumes manually with --volumes."
        )

    return volumes


def route_file_for_volume(volume: int) -> Path:
    """
    Return the absolute path to the scaled route file for a given volume.

    Uses the file already produced by Pipeline A so both pipelines share
    identical route files.
    """
    vol_label  = f'V{volume:04d}'
    route_file = CAPACITY_VOLUMES_DIR / vol_label / f'traffic_{vol_label}.rou.xml'  # file name change
    if not route_file.exists():
        raise FileNotFoundError(
            f"Route file not found for volume {volume}:\n"
            f"  {route_file}\n"
            f"Re-run Pipeline A to regenerate it."
        )
    return route_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pipeline B batch runner. Sweeps baseline and emission-based "
            "control across all seeds for one or more demand volumes. "
            "When --volumes is omitted, volumes are auto-discovered from "
            "the Pipeline A output folder."
        )
    )
    ap.add_argument(
        '--volumes', nargs='+', type=int, default=None,
        metavar='N',
        help=(
            "Space-separated list of demand volumes to sweep "
            "(e.g. --volumes 600 1200 2000). "
            "Omit to auto-discover all volumes from "
            "data/capacity_analysis/volumes/."
        )
    )
    ap.add_argument(
        '--gui', action='store_true', default=False,
        help="Run SUMO with the GUI (default: headless / --nogui)."
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Resolve volumes to sweep
    # ------------------------------------------------------------------
    if args.volumes is not None:
        volumes = sorted(set(args.volumes))
        print(f"\nVolumes specified on command line: {volumes}")
    else:
        volumes = discover_volumes()
        print(f"\nAuto-discovered {len(volumes)} volumes from Pipeline A: {volumes}")

    # ------------------------------------------------------------------
    # Run the sweep
    # ------------------------------------------------------------------
    grand_total = len(volumes) * len(SEEDS) * len(METHODS)
    grand_count = 0
    grand_start = time.time()

    print(f"Total simulations to run: {grand_total} "
          f"({len(volumes)} volumes x {len(SEEDS)} seeds x {len(METHODS)} methods)\n")

    for volume in volumes:
        vol_label  = f'V{volume:04d}'
        route_file = route_file_for_volume(volume)

        vol_total = len(SEEDS) * len(METHODS)
        vol_count = 0
        vol_start = time.time()

        print(f"\n{'='*65}")
        print(f"Volume: {vol_label}  ({volume} veh/hr)  |  "
              f"{vol_total} simulations  |  route: {route_file.name}")
        print(f"{'='*65}")

        for seed in SEEDS:
            for method in METHODS:
                grand_count += 1
                vol_count   += 1

                # Output paths: data/<vol_label>/<method>/
                out_dir = DATA_DIR / vol_label / method
                out_dir.mkdir(parents=True, exist_ok=True)

                output_file = out_dir / f'result_seed_{seed}.csv'
                log_file    = out_dir / f'log_seed_{seed}.txt'

                print(f"\n  [{grand_count:3d}/{grand_total}] "
                      f"vol={vol_label}  seed={seed}  method={method}")

                cmd = [
                    'python', 'run_simulation.py',
                    '--seed',   str(seed),
                    '--method', method,
                    '--output', str(output_file),
                    '--route',  str(route_file),
                ]
                if not args.gui:
                    cmd.append('--nogui')

                try:
                    with open(log_file, 'w') as log:
                        subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT / 'scripts'),
                            check=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    print(f"    OK  -> {output_file.relative_to(PROJECT_ROOT)}")
                except subprocess.CalledProcessError:
                    print(f"    ERROR — see {log_file.relative_to(PROJECT_ROOT)}")

        vol_elapsed = time.time() - vol_start
        print(f"\n  Volume {vol_label} completed in {vol_elapsed:.1f}s")

    grand_elapsed = time.time() - grand_start
    print(f"\n{'='*65}")
    print(f"All {len(volumes)} volumes completed in {grand_elapsed:.1f}s")
    print(f"{'='*65}")
    print("\nNext step — analyze each volume:")
    for volume in volumes:
        vol_label = f'V{volume:04d}'
        print(f"  python scripts/analyze_results.py "
              f"--data-dir data/{vol_label} "
              f"--out-dir visuals/{vol_label}")
    print("\nOr analyze all volumes in one shell loop:")
    labels = ' '.join(f'V{v:04d}' for v in volumes)
    print(f"  for vol in {labels}; do")
    print(f"    python scripts/analyze_results.py "
          f"--data-dir data/$vol --out-dir visuals/$vol")
    print(f"  done")


if __name__ == '__main__':
    main()

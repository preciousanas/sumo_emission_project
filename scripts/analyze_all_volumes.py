#!/usr/bin/env python3
"""
analyze_all_volumes.py

Run this ONE script to analyze every volume folder produced by
run_experiment.py in a single command:

    python scripts/analyze_all_volumes.py

It automatically:
  1. Scans data/ for every V#### sub-folder that has both baseline/ and
     emission_based/ results inside.
  2. Runs the full analyze_results.py pipeline for each volume, saving
     plots to visuals/V####/.
  3. Produces a cross-volume comparison plot in visuals/cross_volume/.

Optional arguments
------------------
--data-dir   Parent data folder to scan.   (default: data)
--out-dir    Parent visuals folder.        (default: visuals)
--volumes    Only process these volumes.  e.g. --volumes 600 1200 2000
             Omit to process all discovered volumes.

Examples
--------
# Analyse everything auto-discovered:
    python scripts/analyze_all_volumes.py

# Analyse only three specific volumes:
    python scripts/analyze_all_volumes.py --volumes 600 1200 2000

# Custom data and output folders:
    python scripts/analyze_all_volumes.py --data-dir data --out-dir visuals
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the project root and scripts folder regardless of where this is run
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add scripts/ to path so analyze_results can be imported directly
sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Analyse all Pipeline B volume folders in one command.'
    )
    ap.add_argument(
        '--data-dir', type=str, default='data',
        help='Parent data folder containing V#### sub-folders. (default: data)'
    )
    ap.add_argument(
        '--out-dir', type=str, default='visuals',
        help='Parent visuals folder for all plots. (default: visuals)'
    )
    ap.add_argument(
        '--volumes', nargs='+', type=int, default=None,
        metavar='N',
        help=(
            'Only process these specific volumes '
            '(e.g. --volumes 600 1200 2000). '
            'Omit to auto-discover all V#### folders.'
        )
    )
    ap.add_argument(
        '--skip-cross-volume', action='store_true', default=False,
        help='Skip the cross-volume comparison plot at the end.'
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Volume discovery
# ---------------------------------------------------------------------------

def discover_volume_folders(data_dir: str) -> list:
    """
    Scan data_dir for V#### sub-folders that contain BOTH baseline/ and
    emission_based/ sub-folders with at least one result CSV file each.

    Returns a sorted list of (volume_int, vol_label, vol_path) tuples.
    """
    found = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path.resolve()}")
        return found

    for entry in sorted(data_path.iterdir()):
        if not (entry.is_dir() and re.match(r'^V\d{4}$', entry.name)):
            continue

        vol_int   = int(entry.name[1:])
        baseline_csvs  = list((entry / 'baseline').glob('result_seed_*.csv'))
        emission_csvs  = list((entry / 'emission_based').glob('result_seed_*.csv'))

        if not baseline_csvs:
            print(f"  [discover] Skipping {entry.name}: "
                  f"no baseline CSVs found in {entry / 'baseline'}")
            continue
        if not emission_csvs:
            print(f"  [discover] Skipping {entry.name}: "
                  f"no emission_based CSVs found in {entry / 'emission_based'}")
            continue

        found.append((vol_int, entry.name, str(entry)))

    return found


# ---------------------------------------------------------------------------
# Per-volume analysis
# ---------------------------------------------------------------------------

def analyse_one_volume(vol_label: str, data_dir: str, out_dir: str) -> bool:
    """
    Run the full analyze_results.py pipeline for a single volume folder.

    Imports analyze_results as a module and calls main() after patching
    the module-level globals so it points at the correct folders.
    This avoids subprocess overhead and gives cleaner error messages.

    Returns True on success, False on failure.
    """
    import importlib

    # Force a fresh import each time so module-level globals reset cleanly
    if 'analyze_results' in sys.modules:
        del sys.modules['analyze_results']

    try:
        import analyze_results as ar
    except ImportError as e:
        print(f"  [ERROR] Could not import analyze_results: {e}")
        print(f"  Make sure analyze_results.py is in {SCRIPT_DIR}")
        return False

    # Override the parsed CLI args that analyze_results read at import time
    ar._ARGS.data_dir     = data_dir
    ar._ARGS.out_dir      = out_dir
    ar._ARGS.vol_label    = vol_label
    ar._ARGS.cross_volume = False

    # Update the module globals that main() re-reads
    ar.DATA_DIR         = data_dir
    ar.OUTPUT_PLOTS_DIR = out_dir
    ar.VOL_LABEL        = vol_label

    # Reset phase switches so they are reloaded from this volume's logs
    ar.PHASE_SWITCHES.clear()

    try:
        ar.main()
        return True
    except Exception as e:
        print(f"\n  [ERROR] Analysis failed for {vol_label}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Cross-volume comparison
# ---------------------------------------------------------------------------

def run_cross_volume(data_dir: str, out_dir: str) -> None:
    """Run the cross-volume comparison using analyze_results.run_cross_volume_comparison()."""
    if 'analyze_results' in sys.modules:
        del sys.modules['analyze_results']

    try:
        import analyze_results as ar
        ar._ARGS.data_dir     = data_dir
        ar._ARGS.out_dir      = out_dir
        ar._ARGS.cross_volume = True
        ar.DATA_DIR         = data_dir
        ar.OUTPUT_PLOTS_DIR = out_dir
        ar.VOL_LABEL        = ''
        ar.run_cross_volume_comparison(data_dir, out_dir)
    except Exception as e:
        print(f"\n  [ERROR] Cross-volume comparison failed: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve paths relative to project root so the script works from any CWD
    data_dir = str(PROJECT_ROOT / args.data_dir)
    out_dir  = str(PROJECT_ROOT / args.out_dir)

    print("\n" + "="*65)
    print("  Pipeline B — Batch Volume Analysis")
    print("="*65)
    print(f"  Data folder : {data_dir}")
    print(f"  Output folder: {out_dir}")

    # ------------------------------------------------------------------
    # Discover or filter volumes
    # ------------------------------------------------------------------
    all_found = discover_volume_folders(data_dir)

    if not all_found:
        print(f"\n[ERROR] No valid V#### folders found in {data_dir}")
        print("  Run 'python scripts/run_experiment.py' first to generate results.")
        sys.exit(1)

    if args.volumes is not None:
        requested = set(args.volumes)
        volumes   = [(vi, vl, vp) for vi, vl, vp in all_found
                     if vi in requested]
        missing   = requested - {vi for vi, _, _ in volumes}
        if missing:
            print(f"\n  WARNING: These requested volumes were not found "
                  f"(no results yet): "
                  f"{sorted(missing)}")
    else:
        volumes = all_found

    if not volumes:
        print("\n[ERROR] No volumes to process after filtering.")
        sys.exit(1)

    vol_labels = [vl for _, vl, _ in volumes]
    print(f"\n  Volumes to analyse ({len(volumes)}): {vol_labels}\n")

    # ------------------------------------------------------------------
    # Analyse each volume
    # ------------------------------------------------------------------
    grand_start = time.time()
    results     = {}

    for vol_int, vol_label, vol_data_path in volumes:
        vol_out_path = str(Path(out_dir) / vol_label)
        os.makedirs(vol_out_path, exist_ok=True)

        print("\n" + "-"*65)
        print(f"  Analysing {vol_label}  "
              f"({vol_int} veh/hr)")
        print(f"  data -> {vol_data_path}")
        print(f"  plots -> {vol_out_path}")
        print("-"*65)

        t0      = time.time()
        success = analyse_one_volume(vol_label, vol_data_path, vol_out_path)
        elapsed = time.time() - t0

        results[vol_label] = success
        status = "OK" if success else "FAILED"
        print(f"\n  [{status}] {vol_label} completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Cross-volume comparison
    # ------------------------------------------------------------------
    if not args.skip_cross_volume:
        print("\n" + "="*65)
        print("  Cross-Volume Comparison")
        print("="*65)
        run_cross_volume(data_dir, out_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    grand_elapsed = time.time() - grand_start
    print("\n" + "="*65)
    print(f"  Batch complete in {grand_elapsed:.1f}s")
    print("="*65)
    successes = [vl for vl, ok in results.items() if ok]
    failures  = [vl for vl, ok in results.items() if not ok]
    print(f"  Succeeded: {successes}")
    if failures:
        print(f"  FAILED:    {failures}")
    print()
    print("  Output structure:")
    for vol_int, vol_label, _ in volumes:
        status = "OK" if results.get(vol_label) else "FAILED"
        print(f"    visuals/{vol_label}/   [{status}]")
    if not args.skip_cross_volume:
        print(f"    visuals/cross_volume/cross_volume_comparison.png")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
run_threshold_x_scenarios.py — Task 4: Threshold Grid × Penetration & Composition.

PURPOSE
-------
Determine whether the optimal emission threshold changes when:
  (A) Penetration rate varies (partial sensor coverage)
  (B) Vehicle composition varies (fleet mix)

Runs the COARSE threshold grid [10,25,33,44,75,100,114,129,150,200,300] mg/s
for each combination of:
  - 5 penetration rates  x 11 new thresholds (thr=50 reused from Task 2)
  - 4 compositions       x 11 new thresholds (thr=50 reused from Task 3)

thr=50 results are always reused from the respective Task 2 or Task 3 folders
(or from the original data/ folder for pen=100 / comp=baseline).

OUTPUT STRUCTURE
    experiments/threshold_x_scenarios/data/
      pen_XXX_thr_YYY/V####/emission_based/
      comp_XXX_thr_YYY/V####/{baseline,emission_based}/

USAGE
-----
  # Run everything:
  python scripts/experiments/run_threshold_x_scenarios.py

  # Run only the penetration × threshold cross:
  python scripts/experiments/run_threshold_x_scenarios.py --mode penetration

  # Run only the composition × threshold cross:
  python scripts/experiments/run_threshold_x_scenarios.py --mode composition

  # Custom thresholds:
  python scripts/experiments/run_threshold_x_scenarios.py --thresholds 10 25 75 100

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
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'threshold_x_scenarios'
COMP_CFG_DIR = PROJECT_ROOT / 'experiments' / 'composition' / 'config'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
COARSE_THRESHOLDS   = [10, 25, 33, 44, 75, 100, 114, 129, 150, 200, 300]  # 44=V1500/V2900 opt, 114=V5000 opt, 129=V3300 opt; skip 50 — reused
REFERENCE_THRESHOLD = 50
REP_VOLUMES         = [1500, 2900, 3300, 5000]
SEEDS               = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]

PENETRATION_RATES = [0.2, 0.4, 0.6, 0.8, 1.0]
COMPOSITIONS      = ['baseline', 'light', 'mixed', 'heavy']

BASE_VOLUME = 568.0   # reference veh/hr in composition templates

REGIME_LABELS = {
    1500: 'Free-flow',
    2900: 'Near-saturation',
    3300: 'Breakdown onset',
    5000: 'Gridlock',
}


# ---------------------------------------------------------------------------
# Route helpers (composition scaling)
# ---------------------------------------------------------------------------

def scale_route_file(template_path: Path, target_volume: int, out_path: Path):
    scale = target_volume / BASE_VOLUME
    tree  = ET.parse(template_path)
    root  = tree.getroot()
    for flow_elem in root.iter('flow'):
        if 'vehsPerHour' in flow_elem.attrib:
            orig = float(flow_elem.attrib['vehsPerHour'])
            flow_elem.set('vehsPerHour', f'{orig * scale:.4f}')
    os.makedirs(out_path.parent, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)


def build_temp_sumocfg(net_file: Path, route_file: Path, out_path: Path):
    cfg  = ET.Element('configuration')
    inp  = ET.SubElement(cfg, 'input')
    ET.SubElement(inp, 'net-file',    value=str(net_file))
    ET.SubElement(inp, 'route-files', value=str(route_file))
    te   = ET.SubElement(cfg, 'time')
    ET.SubElement(te, 'begin',  value='0')
    ET.SubElement(te, 'end',    value='3600')
    proc = ET.SubElement(cfg, 'processing')
    ET.SubElement(proc, 'step-length', value='1.0')
    os.makedirs(out_path.parent, exist_ok=True)
    ET.ElementTree(cfg).write(out_path, encoding='utf-8', xml_declaration=True)


def route_file_for_volume(volume: int) -> Path:
    """Return the Pipeline-A scaled route file for a given volume."""
    vol_label  = f'V{volume:04d}'
    route_file = (PROJECT_ROOT / 'data' / 'capacity_analysis' / 'volumes'
                  / vol_label / f'traffic_{vol_label}.rou.xml')
    if not route_file.exists():
        raise FileNotFoundError(
            f"Route file not found for V{volume:04d}:\n  {route_file}"
        )
    return route_file


# ---------------------------------------------------------------------------
# Penetration × threshold runs
# ---------------------------------------------------------------------------

def run_pen_thr(seed: int, volume: int, penetration: float, threshold: float,
                use_gui: bool, idx: int, total: int, dry_run: bool = False) -> bool:
    pen_int   = int(round(penetration * 100))
    thr_int   = int(threshold)
    pen_lbl   = f'pen_{pen_int:03d}'
    thr_lbl   = f'thr_{thr_int:03d}'
    vol_label = f'V{volume:04d}'
    folder    = f'{pen_lbl}_{thr_lbl}'

    out_dir    = EXP_ROOT / 'data' / folder / vol_label / 'emission_based'
    output_csv = out_dir / f'result_seed_{seed}.csv'
    log_file   = out_dir / f'log_seed_{seed}.txt'

    if output_csv.exists() and output_csv.stat().st_size > 0:
        if not dry_run:
            print(f"  [{idx}/{total}] SKIP: {folder}/{vol_label}/seed={seed}")
        return True

    if dry_run:
        print(f"  [{idx}/{total}] WOULD RUN: pen={penetration:.0%}  thr={threshold:>5.1f}  "
              f"vol={vol_label}  seed={seed}")
        return True

    os.makedirs(out_dir, exist_ok=True)
    print(f"  [{idx}/{total}] pen={penetration:.0%}  thr={threshold:>5.1f}  "
          f"vol={vol_label}  seed={seed}", flush=True)

    route_file = route_file_for_volume(volume)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / 'run_simulation_ext.py'),
        '--seed',               str(seed),
        '--method',             'emission_based',
        '--output',             str(output_csv),
        '--route',              str(route_file),
        '--penetration-rate',   str(penetration),
        '--emission-threshold', str(threshold),
    ]
    if not use_gui:
        cmd.append('--nogui')

    try:
        with open(log_file, 'w') as log:
            subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=True,
                           stdout=log, stderr=subprocess.STDOUT, text=True)
        return True
    except subprocess.CalledProcessError:
        print(f"    ERROR — {log_file.relative_to(PROJECT_ROOT)}")
        return False


# ---------------------------------------------------------------------------
# Composition × threshold runs
# ---------------------------------------------------------------------------

def run_comp_thr(seed: int, volume: int, composition: str, threshold: float,
                 method: str, use_gui: bool, idx: int, total: int,
                 dry_run: bool = False) -> bool:
    thr_int   = int(threshold)
    thr_lbl   = f'thr_{thr_int:03d}'
    comp_lbl  = f'comp_{composition}'
    folder    = f'{comp_lbl}_{thr_lbl}'
    vol_label = f'V{volume:04d}'

    # For baseline runs: if Task 3 data already exists, skip new run entirely —
    # the analyzer will use the Task 3 folder as a fallback automatically.
    if method == 'baseline' and composition != 'baseline':
        task3_bl = (PROJECT_ROOT / 'experiments' / 'composition' / 'data'
                    / comp_lbl / vol_label / 'baseline'
                    / f'result_seed_{seed}.csv')
        if task3_bl.exists() and task3_bl.stat().st_size > 0:
            if not dry_run:
                print(f"  [{idx}/{total}] SKIP (Task3 exists): "
                      f"{comp_lbl}/{vol_label}/baseline/seed={seed}")
            return True

    out_dir    = EXP_ROOT / 'data' / folder / vol_label / method
    output_csv = out_dir / f'result_seed_{seed}.csv'
    log_file   = out_dir / f'log_seed_{seed}.txt'

    if output_csv.exists() and output_csv.stat().st_size > 0:
        if not dry_run:
            print(f"  [{idx}/{total}] SKIP: {folder}/{vol_label}/{method}/seed={seed}")
        return True

    if dry_run:
        print(f"  [{idx}/{total}] WOULD RUN: comp={composition:<8}  thr={threshold:>5.1f}  "
              f"vol={vol_label}  {method}  seed={seed}")
        return True

    os.makedirs(out_dir, exist_ok=True)
    print(f"  [{idx}/{total}] comp={composition:<8}  thr={threshold:>5.1f}  "
          f"vol={vol_label}  {method}  seed={seed}", flush=True)

    template = COMP_CFG_DIR / f'comp_{composition}.rou.xml'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_route = Path(tmpdir) / f'{comp_lbl}_{vol_label}.rou.xml'
        scale_route_file(template, volume, tmp_route)

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
                subprocess.run(cmd, cwd=str(SCRIPTS_DIR), check=True,
                               stdout=log, stderr=subprocess.STDOUT, text=True)
            return True
        except subprocess.CalledProcessError:
            print(f"    ERROR — {log_file.relative_to(PROJECT_ROOT)}")
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description='Task 4: Threshold grid × penetration rates and compositions.'
    )
    ap.add_argument(
        '--mode', choices=['penetration', 'composition', 'both'],
        default='both',
        help='Which cross to run (default: both).'
    )
    ap.add_argument(
        '--thresholds', nargs='+', type=float, default=COARSE_THRESHOLDS,
        metavar='T',
        help=f'Threshold values (default: {COARSE_THRESHOLDS}; thr=50 reused).'
    )
    ap.add_argument(
        '--penetration-rates', nargs='+', type=float, default=PENETRATION_RATES,
        metavar='P',
        help=f'Penetration rates (default: {PENETRATION_RATES}).'
    )
    ap.add_argument(
        '--compositions', nargs='+', default=COMPOSITIONS,
        metavar='C',
        help=f'Compositions (default: {COMPOSITIONS}).'
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
        '--gui', action='store_true', default=False,
        help='Run SUMO with GUI (debugging only).'
    )
    ap.add_argument(
        '--dry-run', action='store_true', default=False, dest='dry_run',
        help='Print what would run without executing any simulations.'
    )
    return ap.parse_args()


def main():
    args = parse_args()

    thresholds   = sorted(args.thresholds)
    penetrations = sorted(args.penetration_rates)
    compositions = args.compositions
    volumes      = sorted(args.volumes)
    seeds        = args.seeds
    dry_run      = args.dry_run

    # --- Simulation count estimates ---
    n_pen_new  = len(thresholds) * len(penetrations) * len(volumes) * len(seeds)
    n_comp_em  = len(thresholds) * len(compositions) * len(volumes) * len(seeds)
    n_comp_bl  = len([c for c in compositions if c != 'baseline']) * len(volumes) * len(seeds)
    n_total    = ((n_pen_new  if args.mode in ('penetration', 'both') else 0) +
                  (n_comp_em + n_comp_bl if args.mode in ('composition', 'both') else 0))

    print("\n" + "=" * 65)
    print("  Task 4: Threshold × Scenario Cross-Experiment")
    print("=" * 65)
    print(f"  Mode           : {args.mode}")
    if dry_run:
        print(f"  *** DRY RUN — no simulations will actually execute ***")
    print(f"  Thresholds     : {thresholds}  (+thr=50 reused from earlier runs)")
    print(f"  Penetrations   : {[f'{p:.0%}' for p in penetrations]}")
    print(f"  Compositions   : {compositions}")
    print(f"  Volumes        : {[f'V{v:04d}' for v in volumes]}")
    print(f"  Seeds          : {seeds}")
    print(f"  Max new sims   : {n_total}  (already-done files are skipped)")
    print(f"  Est. wall-time : ~{n_total * 3 // 60}–{n_total * 6 // 60} min "
          f"(3–6 s/sim, no GUI)")

    grand_start = time.time()
    errors      = []

    # ------------------------------------------------------------------
    # A: Penetration × Threshold
    # ------------------------------------------------------------------
    if args.mode in ('penetration', 'both'):
        pen_thrs  = [(p, t) for p in penetrations for t in thresholds]
        pen_total = len(pen_thrs) * len(volumes) * len(seeds)
        pen_idx   = 0

        print(f"\n{'='*65}")
        print(f"  A: Penetration × Threshold  ({pen_total} max simulations)")
        print(f"{'='*65}")

        for pen, thr in pen_thrs:
            pen_int = int(round(pen * 100))
            thr_int = int(thr)
            if not dry_run:
                print(f"\n  pen={pen:.0%}  thr={thr:.0f} mg/s  "
                      f"[pen_{pen_int:03d}_thr_{thr_int:03d}]")
            for vol in volumes:
                for seed in seeds:
                    pen_idx += 1
                    ok = run_pen_thr(seed, vol, pen, thr, args.gui,
                                     pen_idx, pen_total, dry_run=dry_run)
                    if not ok:
                        errors.append(('pen', pen, thr, vol, seed))

    # ------------------------------------------------------------------
    # B: Composition x Threshold
    # ------------------------------------------------------------------
    if args.mode in ('composition', 'both'):
        comp_thrs  = [(c, t) for c in compositions for t in thresholds]
        comp_total = len(comp_thrs) * len(volumes) * len(seeds)
        bl_total   = len([c for c in compositions if c != 'baseline']) * len(volumes) * len(seeds)
        comp_idx   = 0

        print(f"\n{'='*65}")
        print(f"  B: Composition x Threshold")
        print(f"     {comp_total} emission_based + up to {bl_total} baseline (Task3 data reused)")
        print(f"{'='*65}")

        # Baseline per non-baseline composition (threshold-independent).
        # Smart skip: run_comp_thr checks Task3 folder before running.
        bl_idx = 0
        for comp in compositions:
            if comp == 'baseline':
                continue   # baseline composition uses data/ directly
            if not dry_run:
                print(f"\n  Baseline for comp_{comp}")
            for vol in volumes:
                for seed in seeds:
                    bl_idx += 1
                    ok = run_comp_thr(seed, vol, comp, REFERENCE_THRESHOLD,
                                      'baseline', args.gui,
                                      bl_idx, bl_total, dry_run=dry_run)
                    if not ok:
                        errors.append(('comp_baseline', comp, vol, seed))

        for comp, thr in comp_thrs:
            thr_int = int(thr)
            if not dry_run:
                print(f"\n  comp={comp}  thr={thr:.0f} mg/s")
            for vol in volumes:
                for seed in seeds:
                    comp_idx += 1
                    ok = run_comp_thr(seed, vol, comp, thr, 'emission_based',
                                      args.gui, comp_idx, comp_total,
                                      dry_run=dry_run)
                    if not ok:
                        errors.append(('comp_em', comp, thr, vol, seed))

    elapsed = time.time() - grand_start
    print(f"\n{'='*65}")
    if dry_run:
        print(f"  Dry run complete in {elapsed:.1f}s -- no simulations were run.")
    else:
        print(f"  Completed in {elapsed:.1f}s")
        if errors:
            print(f"  ERRORS ({len(errors)}): {errors}")
        else:
            print("  All simulations succeeded.")
    print(f"{'='*65}")
    if not dry_run:
        print("\nNext step: analyse cross-scenario results:")
        print("  python scripts/experiments/analyze_threshold_x_scenarios.py")


if __name__ == '__main__':
    main()

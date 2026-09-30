#!/usr/bin/env python3
"""
analyze_composition.py — Task 3 Analysis: Vehicle Composition Sensitivity.

Reads results from:
  experiments/composition/data/comp_XXX/V####/{baseline,emission_based}/
  data/V####/{baseline,emission_based}/  [comp_baseline reference, read directly]

Computes per-composition, per-volume:
  - CO2 savings %         (emission_based vs baseline, same composition)
  - Throughput change %   (emission_based vs baseline, same composition)
  - Wait time change %    (emission_based vs baseline, same composition)
  - Absolute CO2 per veh  (to show fleet-level absolute differences)

Produces:
  experiments/composition/results/composition_summary.csv
  experiments/composition/visuals/co2_savings_by_composition.png
  experiments/composition/visuals/absolute_co2_by_composition.png
  experiments/composition/visuals/all_metrics_by_composition.png

USAGE
-----
  python scripts/experiments/analyze_composition.py
"""

import csv
import os
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'composition'
DATA_ROOT    = PROJECT_ROOT / 'data'

VISUALS_DIR = EXP_ROOT / 'visuals'
RESULTS_DIR = EXP_ROOT / 'results'
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

REP_VOLUMES = [1500, 2900, 3300, 5000]
REGIME_LABELS = {
    1500: 'Free-flow',
    2900: 'Near-saturation',
    3300: 'Breakdown onset',
    5000: 'Gridlock',
}
REGIME_COLORS = {
    'Free-flow':        '#2ca02c',
    'Near-saturation':  '#1f77b4',
    'Breakdown onset':  '#ff7f0e',
    'Gridlock':         '#d62728',
}

COMPOSITIONS    = ['baseline', 'light', 'mixed', 'heavy']  # ordered light->heavy
COMP_LABELS     = {
    'baseline': 'Baseline\n(40% car)',
    'heavy':    'Heavy\n(25% truck)',
    'light':    'Light\n(80% car)',
    'mixed':    'Mixed\n(55% car)',
}
COMP_COLORS     = {
    'baseline': '#7f7f7f',
    'heavy':    '#d62728',
    'light':    '#2ca02c',
    'mixed':    '#ff7f0e',
}
THROUGHPUT_CONSTRAINT = -1.0


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_result_csv(csv_path: Path) -> dict:
    rows = []
    try:
        with open(csv_path, newline='', errors='replace') as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                if header is None:
                    header = [c.strip() for c in row]
                    continue
                if len(row) < len(header):
                    continue
                rows.append(dict(zip(header, row)))
    except Exception:
        return None
    if not rows:
        return None

    vehicle_ids = set()
    co2_sum, wait_sum, wait_count = 0.0, 0.0, 0
    for r in rows:
        try:
            veh_id = r.get('veh_id', '')
            if not veh_id or veh_id.startswith('#'):
                continue
            vehicle_ids.add(veh_id)                       # ALL vehicles → throughput
            if r.get('in_box', '').strip().lower() == 'true':
                co2_sum   += float(r.get('co2_emission', 0))
                wait_sum  += float(r.get('wait_time', 0))
                wait_count += 1
        except (ValueError, KeyError):
            continue
    n_veh = len(vehicle_ids)
    if n_veh == 0:
        return None
    return {
        'throughput':  n_veh,
        'avg_wait_s':  wait_sum / wait_count if wait_count > 0 else 0.0,
        'co2_per_veh': co2_sum / n_veh,
    }


def load_folder(folder: Path) -> dict:
    results = {}
    if not folder.exists():
        return results
    for csv_file in sorted(folder.glob('result_seed_*.csv')):
        m = re.search(r'result_seed_(\d+)\.csv', csv_file.name)
        if not m:
            continue
        seed    = int(m.group(1))
        metrics = _parse_result_csv(csv_file)
        if metrics is not None:
            results[seed] = metrics
    return results


def folder_mean(seed_results: dict) -> dict:
    if not seed_results:
        return None
    means = {}
    for k in ['throughput', 'avg_wait_s', 'co2_per_veh']:
        vals = [v[k] for v in seed_results.values() if k in v]
        means[k] = np.mean(vals) if vals else 0.0
    means['n_seeds'] = len(seed_results)
    return means


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(volumes: list, compositions: list) -> dict:
    """
    Returns nested dict: results[volume][composition] = {
        'baseline': {metrics}, 'emission_based': {metrics}
    }
    """
    results = {}
    for vol in volumes:
        vol_label = f'V{vol:04d}'
        results[vol] = {}
        for comp in compositions:
            comp_label = f'comp_{comp}'
            if comp == 'baseline':
                # Use existing data/ folder as reference
                bl_folder = DATA_ROOT / vol_label / 'baseline'
                em_folder = DATA_ROOT / vol_label / 'emission_based'
            else:
                bl_folder = EXP_ROOT / 'data' / comp_label / vol_label / 'baseline'
                em_folder = EXP_ROOT / 'data' / comp_label / vol_label / 'emission_based'

            bl_mean = folder_mean(load_folder(bl_folder))
            em_mean = folder_mean(load_folder(em_folder))

            results[vol][comp] = {
                'baseline':       bl_mean,
                'emission_based': em_mean,
            }
    return results


def compute_deltas(results: dict) -> list:
    rows = []
    for vol, comp_dict in sorted(results.items()):
        regime = REGIME_LABELS.get(vol, '?')
        for comp, data in comp_dict.items():
            bl = data['baseline']
            em = data['emission_based']
            if bl is None or em is None:
                continue

            def pct(new, old):
                return (new - old) / old * 100.0 if old else 0.0

            rows.append({
                'volume':                vol,
                'regime':                regime,
                'composition':           comp,
                'bl_co2_per_veh':        round(bl['co2_per_veh'], 1),
                'em_co2_per_veh':        round(em['co2_per_veh'], 1),
                'co2_savings_pct':       round(pct(bl['co2_per_veh'], em['co2_per_veh']), 3),
                'throughput_change_pct': round(pct(em['throughput'],  bl['throughput']),  3),
                'wait_change_pct':       round(pct(em['avg_wait_s'],  bl['avg_wait_s']),  3),
                'n_seeds_em':            em.get('n_seeds', 0),
                'n_seeds_bl':            bl.get('n_seeds', 0),
            })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_co2_savings(delta_rows: list):
    """4-panel grouped bar: CO2 savings per composition per regime."""
    by_vol = defaultdict(lambda: defaultdict(dict))
    for row in delta_rows:
        by_vol[row['volume']][row['composition']] = row

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(
        'CO₂ Savings by Fleet Composition (emission-based vs gap-based baseline)',
        fontsize=13, fontweight='bold'
    )

    x     = np.arange(len(COMPOSITIONS))
    width = 0.55

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes_flat[ax_idx]
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS.get(regime, 'black')

        co2_vals  = []
        wait_vals = []
        tp_vals   = []
        colors    = []

        for comp in COMPOSITIONS:
            row = by_vol[vol].get(comp)
            if row:
                co2_vals.append(row['co2_savings_pct'])
                wait_vals.append(row['wait_change_pct'])
                tp_vals.append(row['throughput_change_pct'])
            else:
                co2_vals.append(0)
                wait_vals.append(0)
                tp_vals.append(0)
            colors.append(COMP_COLORS.get(comp, 'grey'))

        if not any(v != 0 for v in co2_vals):
            ax.text(0.5, 0.5, 'No data available.\nRun run_composition_experiment.py first.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            ax.set_title(f'V{vol:04d} — {regime}')
            continue

        bars = ax.bar(x, co2_vals, width, color=colors, alpha=0.82,
                      edgecolor='white', linewidth=1.2, label='CO₂ savings %')

        # Overlay throughput only (wait time removed for publication clarity)
        ax2 = ax.twinx()
        ax2.plot(x, tp_vals, '^:', color='navy', alpha=0.55,
                 linewidth=1.2, markersize=6, label='Throughput Δ %')
        ax2.axhline(THROUGHPUT_CONSTRAINT, color='red', linestyle='--',
                    linewidth=1.0, alpha=0.7)
        ax2.set_ylabel('Throughput Δ (%)', fontsize=7.5, color='navy')
        ax2.tick_params(axis='y', labelcolor='navy', labelsize=7)

        # --- smart y-axis limits (give headroom for annotations) ---
        ymin = min(co2_vals) if co2_vals else 0
        ymax = max(co2_vals) if co2_vals else 0
        y_range = max(abs(ymax - ymin), 0.5)          # avoid division by zero
        ax.set_ylim(ymin - y_range * 0.35, ymax + y_range * 0.45)

        # --- Annotate bars with collision-aware placement ---
        used_y = []   # track occupied y positions to avoid text overlap
        for bar, val in zip(bars, co2_vals):
            if val == 0:
                continue
            bx    = bar.get_x() + bar.get_width() / 2
            # Base position above/below bar with a small clearance
            if val >= 0:
                base_y = bar.get_height() + y_range * 0.04
                va     = 'bottom'
            else:
                base_y = bar.get_height() - y_range * 0.04
                va     = 'top'
            # Push up if too close to zero (avoids x-axis label overlap)
            if abs(base_y) < y_range * 0.08:
                base_y = y_range * 0.08 * (1 if val >= 0 else -1)
            # Collision detection: nudge if another label is within 12 % of y_range
            for prev_y in used_y:
                if abs(base_y - prev_y) < y_range * 0.12:
                    base_y += y_range * 0.12 * (1 if val >= 0 else -1)
            used_y.append(base_y)
            ax.text(bx, base_y, f'{val:+.2f}%',
                    ha='center', va=va, fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.5, alpha=0.4)
        ax.set_title(f'V{vol:04d} — {regime}', fontsize=11, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([COMP_LABELS.get(c, c) for c in COMPOSITIONS], fontsize=8)
        ax.set_ylabel('CO₂ Savings (%)', fontsize=9)
        ax.set_xlabel('Fleet Composition', fontsize=9)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        # Place legend in lower-left to avoid overlap with throughput markers
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='lower left',
                  framealpha=0.85, borderpad=0.5)

    out = VISUALS_DIR / 'co2_savings_by_composition.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_absolute_co2(delta_rows: list):
    """Grouped bar: absolute CO2 per vehicle (mg) for baseline and emission_based per composition."""
    by_vol = defaultdict(lambda: defaultdict(dict))
    for row in delta_rows:
        by_vol[row['volume']][row['composition']] = row

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(
        'Absolute CO₂ per Vehicle by Fleet Composition\n'
        '(lower = better; shows fleet-level CO₂ even when savings are similar)',
        fontsize=12, fontweight='bold'
    )

    x      = np.arange(len(COMPOSITIONS))
    width  = 0.3

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes_flat[ax_idx]
        regime = REGIME_LABELS.get(vol, '?')

        bl_vals = []
        em_vals = []
        colors  = []

        for comp in COMPOSITIONS:
            row = by_vol[vol].get(comp)
            if row:
                bl_vals.append(row['bl_co2_per_veh'] / 1e6)   # convert mg to kg
                em_vals.append(row['em_co2_per_veh'] / 1e6)
            else:
                bl_vals.append(0)
                em_vals.append(0)
            colors.append(COMP_COLORS.get(comp, 'grey'))

        if not any(v != 0 for v in bl_vals):
            ax.text(0.5, 0.5, 'No data available.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            ax.set_title(f'V{vol:04d} — {regime}')
            continue

        ax.bar(x - width/2, bl_vals, width, color=colors, alpha=0.6,
               edgecolor='black', linewidth=0.8, label='Baseline')
        ax.bar(x + width/2, em_vals, width, color=colors, alpha=0.95,
               edgecolor='black', linewidth=0.8, label='Emission-based', hatch='///')

        ax.set_title(f'V{vol:04d} — {regime}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([COMP_LABELS.get(c, c) for c in COMPOSITIONS], fontsize=8)
        ax.set_ylabel('CO₂ per Vehicle (kg)', fontsize=9)
        ax.set_xlabel('Fleet Composition', fontsize=9)
        ax.legend(fontsize=8)

    out = VISUALS_DIR / 'absolute_co2_by_composition.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(delta_rows: list):
    out = RESULTS_DIR / 'composition_summary.csv'
    fields = ['volume', 'regime', 'composition',
              'bl_co2_per_veh', 'em_co2_per_veh',
              'co2_savings_pct', 'throughput_change_pct', 'wait_change_pct',
              'n_seeds_em', 'n_seeds_bl']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(delta_rows)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  Vehicle Composition Analysis")
    print("=" * 65)

    results    = load_all(REP_VOLUMES, COMPOSITIONS)
    delta_rows = compute_deltas(results)

    if not delta_rows:
        print("\n  [WARNING] No data found.")
        print("  Run run_composition_experiment.py first.")
        sys.exit(0)

    print(f"\n  Loaded {len(delta_rows)} data points.")
    print("\n  Generating plots ...")
    plot_co2_savings(delta_rows)
    plot_absolute_co2(delta_rows)

    print("\n  Saving CSVs ...")
    save_csv(delta_rows)

    print("\n" + "=" * 65)
    print("  Results summary (CO₂ savings %):")
    print(f"  {'Regime':<20} {'Vol':>5}", end='')
    for c in COMPOSITIONS:
        print(f"  {c:>10}", end='')
    print()
    print("  " + "-" * 70)

    by_vol = defaultdict(lambda: defaultdict(dict))
    for row in delta_rows:
        by_vol[row['volume']][row['composition']] = row

    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        print(f"  {regime:<20} {vol:>5}", end='')
        for c in COMPOSITIONS:
            row = by_vol[vol].get(c)
            val = f"{row['co2_savings_pct']:+.2f}%" if row else '  N/A '
            print(f"  {val:>10}", end='')
        print()
    print("=" * 65)


if __name__ == '__main__':
    main()

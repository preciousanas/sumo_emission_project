#!/usr/bin/env python3
"""
analyze_penetration.py — Task 2 Analysis: Penetration Rate Experiments.

Reads results from:
  experiments/penetration_rate/data/pen_XXX/V####/emission_based/
  data/V####/emission_based/  [pen_100 — read directly, no copy]
  data/V####/baseline/        [reference baseline for all penetration levels]

Computes per-penetration-rate, per-volume:
  - CO2 savings %         (emission_based vs baseline)
  - Throughput change %   (emission_based vs baseline)
  - Wait time change %    (emission_based vs baseline)

Produces:
  experiments/penetration_rate/results/penetration_summary.csv
  experiments/penetration_rate/visuals/co2_savings_by_penetration.png
  experiments/penetration_rate/visuals/all_metrics_by_penetration.png

USAGE
-----
  python scripts/experiments/analyze_penetration.py
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
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'penetration_rate'
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
ALL_PENETRATIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
THROUGHPUT_CONSTRAINT = -1.0


# ---------------------------------------------------------------------------
# CSV parsing (shared with analyze_threshold_search)
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

def discover_penetrations() -> list:
    found = [1.0]
    data_dir = EXP_ROOT / 'data'
    if data_dir.exists():
        for entry in data_dir.iterdir():
            m = re.match(r'^pen_(\d+)$', entry.name)
            if m and entry.is_dir():
                p = int(m.group(1)) / 100.0
                if p not in found:
                    found.append(p)
    return sorted(found)


def load_all(volumes: list, penetrations: list) -> dict:
    results = {}
    for vol in volumes:
        vol_label = f'V{vol:04d}'
        results[vol] = {}

        bl_folder = DATA_ROOT / vol_label / 'baseline'
        bl_seeds  = load_folder(bl_folder)
        bl_mean   = folder_mean(bl_seeds)

        for pen in penetrations:
            pen_int   = int(round(pen * 100))
            pen_lbl   = f'pen_{pen_int:03d}'

            if pen >= 1.0:
                em_folder = DATA_ROOT / vol_label / 'emission_based'
            else:
                em_folder = EXP_ROOT / 'data' / pen_lbl / vol_label / 'emission_based'

            em_seeds = load_folder(em_folder)
            em_mean  = folder_mean(em_seeds)

            results[vol][pen] = {
                'baseline':       bl_mean,
                'emission_based': em_mean,
            }
    return results


def compute_deltas(results: dict) -> list:
    rows = []
    for vol, pen_dict in sorted(results.items()):
        regime = REGIME_LABELS.get(vol, '?')
        for pen, data in sorted(pen_dict.items()):
            bl = data['baseline']
            em = data['emission_based']
            if bl is None or em is None:
                continue

            def pct(new, old):
                return (new - old) / old * 100.0 if old else 0.0

            rows.append({
                'volume':                vol,
                'regime':                regime,
                'penetration_pct':       int(round(pen * 100)),
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
    """4-panel line chart: CO2 savings % vs penetration rate per regime.
    V5000 (Gridlock) negative savings are explicitly annotated so they are
    not misread as missing bars.
    """
    by_vol = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(
        'CO2 Savings vs Sensor Penetration Rate\n'
        '(emission-based vs gap-based baseline; thr=50 mg/s)',
        fontsize=13, fontweight='bold'
    )

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes_flat[ax_idx]
        rows   = sorted(by_vol[vol], key=lambda r: r['penetration_pct'])
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS.get(regime, 'black')

        if not rows:
            ax.text(0.5, 0.5, 'No data available.\nRun run_penetration_experiment.py first.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            ax.set_title(f'V{vol:04d} -- {regime}')
            continue

        pen_pcts    = [r['penetration_pct']       for r in rows]
        co2_savings = [r['co2_savings_pct']        for r in rows]
        wait_chg    = [r['wait_change_pct']        for r in rows]
        tp_chg      = [r['throughput_change_pct']  for r in rows]

        ax2 = ax.twinx()

        l1, = ax.plot(pen_pcts, co2_savings, 'o-', color=color,
                      linewidth=2, markersize=7, label='CO2 savings (%)')
        l3, = ax2.plot(pen_pcts, tp_chg, '^:', color='#555555', alpha=0.7,
                       linewidth=1.5, markersize=5, label='Throughput (%) change')
        ax2.axhline(THROUGHPUT_CONSTRAINT, color='red', linestyle='--',
                    linewidth=1.3, alpha=0.85, label='-1% limit')

        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # --- Explicitly annotate ALL data points with their values ---
        # This ensures negative values (e.g. V5000 Gridlock) are never misread as absent.
        co2_arr = np.array(co2_savings)
        yrange  = max(abs(co2_arr.max() - co2_arr.min()), 0.5)
        for px, pv in zip(pen_pcts, co2_savings):
            va  = 'bottom' if pv >= 0 else 'top'
            off = yrange * 0.06 if pv >= 0 else -yrange * 0.06
            ax.text(px, pv + off, f'{pv:+.2f}%',
                    ha='center', va=va, fontsize=7.5, fontweight='bold',
                    color='darkred' if pv < 0 else 'darkgreen')

        # Ensure y-axis has enough room to show all annotations clearly
        y_pad = yrange * 0.45
        ax.set_ylim(co2_arr.min() - y_pad, co2_arr.max() + y_pad)

        # For gridlock (all/mostly negative), add a clear note
        n_negative = sum(1 for v in co2_savings if v < 0)
        if n_negative == len(co2_savings):
            ax.text(0.5, 0.96,
                    'All values negative: controller hurts CO2 in this regime',
                    transform=ax.transAxes, fontsize=8, color='darkred',
                    ha='center', va='top', style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0',
                              edgecolor='#cc4444', alpha=0.85))
        elif n_negative > 0:
            ax.text(0.5, 0.96,
                    f'{n_negative} of {len(co2_savings)} penetration levels show negative savings',
                    transform=ax.transAxes, fontsize=8, color='darkred',
                    ha='center', va='top', style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0',
                              edgecolor='#cc4444', alpha=0.85))

        # Mark crossover: lowest penetration where CO2 savings > 0
        positive = [r['penetration_pct'] for r in rows if r['co2_savings_pct'] > 0]
        if positive and n_negative < len(co2_savings):
            ax.axvline(min(positive), color='green', linestyle=':', linewidth=1.2, alpha=0.7)

        ax.set_xlabel('Penetration Rate (%)', fontsize=9)
        ax.set_ylabel('CO2 Savings vs Baseline (%)', fontsize=9)
        ax2.set_ylabel('Throughput Change (%)', fontsize=8, color='#555555')
        ax2.tick_params(axis='y', labelcolor='#555555', labelsize=7)
        ax.set_title(f'V{vol:04d} -- {regime}', fontsize=11, fontweight='bold', color=color)
        ax.set_xticks(pen_pcts)
        ax.tick_params(axis='both', labelsize=8)

        lines  = [l1, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7.5, loc='lower left')
        ax2.legend(fontsize=7.5, loc='lower right')

        if rows:
            ax.text(0.01, 0.01, f'n_seeds={rows[0]["n_seeds_em"]}',
                    transform=ax.transAxes, fontsize=7, color='grey', va='bottom')

    out = VISUALS_DIR / 'co2_savings_by_penetration.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")
def plot_summary_bars(delta_rows: list):
    """Grouped bar chart: CO2 savings at each penetration rate, grouped by regime.
    Negative bars (Gridlock) are visually distinct and explicitly labelled.
    """
    pens   = sorted(set(r['penetration_pct'] for r in delta_rows))
    vols   = REP_VOLUMES
    n_vols = len(vols)
    n_pens = len(pens)

    by_vol_pen = defaultdict(dict)
    for row in delta_rows:
        by_vol_pen[row['volume']][row['penetration_pct']] = row['co2_savings_pct']

    x     = np.arange(n_pens)
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 7))

    all_vals = [v for pen_d in by_vol_pen.values() for v in pen_d.values()]
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_pad = max(abs(y_max - y_min), 0.5) * 0.45

    for i, vol in enumerate(vols):
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS.get(regime, 'grey')
        vals   = [by_vol_pen[vol].get(p, 0) for p in pens]
        xpos   = x + i * width - (n_vols - 1) * width / 2

        # Use hatching for gridlock to distinguish negative bars visually
        hatch = '//' if vol == 5000 else None
        edgec = '#880000' if vol == 5000 else 'white'
        bars  = ax.bar(xpos, vals, width,
                       label=f'V{vol:04d} ({regime})',
                       color=color, alpha=0.82,
                       edgecolor=edgec, linewidth=1.2,
                       hatch=hatch)

        # Value labels on every bar
        for bx, bv in zip(xpos, vals):
            if bv == 0:
                continue
            va     = 'bottom' if bv >= 0 else 'top'
            y_off  = max(abs(y_max - y_min), 0.5) * 0.03
            y_pos  = bv + y_off if bv >= 0 else bv - y_off
            fc     = 'darkred' if bv < -0.1 else ('darkgreen' if bv > 0.1 else '#555555')
            ax.text(bx, y_pos, f'{bv:+.2f}%',
                    ha='center', va=va, fontsize=6.5, color=fc, fontweight='bold')

    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xlabel('Penetration Rate (%)', fontsize=10)
    ax.set_ylabel('CO2 Savings vs Baseline (%)', fontsize=10)
    ax.set_title('CO2 Savings by Penetration Rate and Traffic Regime\n'
                 '(emission-based vs gap-based; thr=50 mg/s)\n'
                 'Note: negative values = emission-based controller increases CO2 vs baseline',
                 fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}%' for p in pens])
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.legend(fontsize=8, loc='upper right')

    # Annotation box explaining negative V5000 values
    if any(by_vol_pen[5000].get(p, 0) < 0 for p in pens):
        ax.text(0.01, 0.03,
                'V5000 (Gridlock) bars are negative:\n'
                'emission-based control does not improve CO2 in gridlock.',
                transform=ax.transAxes, fontsize=8, color='darkred',
                va='bottom', ha='left', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
                          edgecolor='#cc4444', alpha=0.85))

    out = VISUALS_DIR / 'co2_savings_bars_by_penetration.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(delta_rows: list):
    out = RESULTS_DIR / 'penetration_summary.csv'
    fields = ['volume', 'regime', 'penetration_pct',
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
    print("  Penetration Rate Analysis")
    print("=" * 65)

    penetrations = discover_penetrations()
    print(f"  Penetration rates found: {[f'{p:.0%}' for p in penetrations]}")

    results    = load_all(REP_VOLUMES, penetrations)
    delta_rows = compute_deltas(results)

    if not delta_rows:
        print("\n  [WARNING] No data found.")
        print("  Run run_penetration_experiment.py first.")
        sys.exit(0)

    print(f"\n  Loaded {len(delta_rows)} data points.")
    print("\n  Generating plots ...")
    plot_co2_savings(delta_rows)
    plot_summary_bars(delta_rows)

    print("\n  Saving CSVs ...")
    save_csv(delta_rows)

    print("\n" + "=" * 65)
    print("  Results summary:")
    print(f"  {'Regime':<20} {'Vol':>5} {'Pen%':>6} {'CO2Sav%':>9} {'Tput%':>7} {'Wait%':>7}")
    print("  " + "-" * 60)
    for row in delta_rows:
        print(f"  {row['regime']:<20} {row['volume']:>5} "
              f"{row['penetration_pct']:>5}% "
              f"{row['co2_savings_pct']:>+9.2f}% "
              f"{row['throughput_change_pct']:>+7.2f}% "
              f"{row['wait_change_pct']:>+7.2f}%")
    print("=" * 65)


if __name__ == '__main__':
    main()

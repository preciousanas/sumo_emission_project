#!/usr/bin/env python3
"""
analyze_threshold_x_scenarios.py — Task 4 Analysis.

Reads results from:
  experiments/threshold_x_scenarios/data/pen_XXX_thr_YYY/V####/emission_based/
  experiments/threshold_x_scenarios/data/comp_XXX_thr_YYY/V####/{baseline,emission_based}/
  data/V####/baseline/    (reference baseline for penetration experiments)
  data/V####/emission_based/  (reference thr=50, pen=100 or comp=baseline)

Produces:
  Heatmap A: penetration rate (y) × threshold (x) — CO2 savings, one subplot per regime
  Heatmap B: composition (y)     × threshold (x) — CO2 savings, one subplot per regime
  Summary table CSV: optimal threshold per (penetration, composition)

  experiments/threshold_x_scenarios/results/pen_x_threshold_summary.csv
  experiments/threshold_x_scenarios/results/comp_x_threshold_summary.csv
  experiments/threshold_x_scenarios/results/optimal_threshold_matrix.csv
  experiments/threshold_x_scenarios/visuals/heatmap_pen_x_threshold.png
  experiments/threshold_x_scenarios/visuals/heatmap_comp_x_threshold.png

USAGE
-----
  python scripts/experiments/analyze_threshold_x_scenarios.py
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
import matplotlib.colors as mcolors
import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'threshold_x_scenarios'
DATA_ROOT    = PROJECT_ROOT / 'data'
PEN_EXP_ROOT = PROJECT_ROOT / 'experiments' / 'penetration_rate'
COMP_EXP_ROOT = PROJECT_ROOT / 'experiments' / 'composition'

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

ALL_THRESHOLDS   = [10, 25, 33, 44, 50, 75, 100, 114, 129, 150, 200, 300]  # 44=V1500/V2900 opt, 114=V5000 opt, 129=V3300 opt
ALL_PENETRATIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
ALL_COMPOSITIONS = ['light', 'baseline', 'mixed', 'heavy']   # light -> heavy

THROUGHPUT_CONSTRAINT = -1.0


# ---------------------------------------------------------------------------
# CSV parsing helpers (shared pattern)
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


def pct(new, old):
    return (new - old) / old * 100.0 if old else 0.0


# ---------------------------------------------------------------------------
# Penetration × Threshold data loading
# ---------------------------------------------------------------------------

def load_pen_x_thr(volumes, penetrations, thresholds) -> list:
    rows = []
    for vol in volumes:
        vol_label = f'V{vol:04d}'
        regime    = REGIME_LABELS.get(vol, '?')

        # Baseline is always from data/ (thr-independent)
        bl_mean = folder_mean(load_folder(DATA_ROOT / vol_label / 'baseline'))

        for pen in penetrations:
            pen_int = int(round(pen * 100))
            pen_lbl = f'pen_{pen_int:03d}'

            for thr in thresholds:
                thr_int = int(thr)
                thr_lbl = f'thr_{thr_int:03d}'
                folder  = f'{pen_lbl}_{thr_lbl}'

                # thr=50 at pen=100 comes from data/; at pen<100 comes from pen experiment
                if thr == 50 and pen >= 1.0:
                    em_folder = DATA_ROOT / vol_label / 'emission_based'
                elif thr == 50:
                    em_folder = PEN_EXP_ROOT / 'data' / pen_lbl / vol_label / 'emission_based'
                else:
                    em_folder = EXP_ROOT / 'data' / folder / vol_label / 'emission_based'

                em_mean = folder_mean(load_folder(em_folder))

                if bl_mean is None or em_mean is None:
                    continue

                rows.append({
                    'volume':                vol,
                    'regime':                regime,
                    'penetration_pct':       pen_int,
                    'threshold':             thr,
                    'co2_savings_pct':       round(pct(bl_mean['co2_per_veh'], em_mean['co2_per_veh']), 3),
                    'throughput_change_pct': round(pct(em_mean['throughput'],  bl_mean['throughput']),  3),
                    'wait_change_pct':       round(pct(em_mean['avg_wait_s'],  bl_mean['avg_wait_s']),  3),
                })
    return rows


# ---------------------------------------------------------------------------
# Composition × Threshold data loading
# ---------------------------------------------------------------------------

def load_comp_x_thr(volumes, compositions, thresholds) -> list:
    rows = []
    for vol in volumes:
        vol_label = f'V{vol:04d}'
        regime    = REGIME_LABELS.get(vol, '?')

        for comp in compositions:
            comp_lbl = f'comp_{comp}'

            # Baseline per composition (thr-independent)
            if comp == 'baseline':
                bl_folder = DATA_ROOT / vol_label / 'baseline'
            else:
                bl_folder = EXP_ROOT / 'data' / f'{comp_lbl}_thr_050' / vol_label / 'baseline'
                if not bl_folder.exists():
                    bl_folder = COMP_EXP_ROOT / 'data' / comp_lbl / vol_label / 'baseline'

            bl_mean = folder_mean(load_folder(bl_folder))

            for thr in thresholds:
                thr_int = int(thr)
                thr_lbl = f'thr_{thr_int:03d}'
                folder  = f'{comp_lbl}_{thr_lbl}'

                # thr=50 comes from composition experiment or existing data/
                if thr == 50 and comp == 'baseline':
                    em_folder = DATA_ROOT / vol_label / 'emission_based'
                elif thr == 50:
                    em_folder = COMP_EXP_ROOT / 'data' / comp_lbl / vol_label / 'emission_based'
                else:
                    em_folder = EXP_ROOT / 'data' / folder / vol_label / 'emission_based'

                em_mean = folder_mean(load_folder(em_folder))

                if bl_mean is None or em_mean is None:
                    continue

                rows.append({
                    'volume':                vol,
                    'regime':                regime,
                    'composition':           comp,
                    'threshold':             thr,
                    'co2_savings_pct':       round(pct(bl_mean['co2_per_veh'], em_mean['co2_per_veh']), 3),
                    'throughput_change_pct': round(pct(em_mean['throughput'],  bl_mean['throughput']),  3),
                    'wait_change_pct':       round(pct(em_mean['avg_wait_s'],  bl_mean['avg_wait_s']),  3),
                })
    return rows


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def _build_heatmap_matrix(rows: list, row_key: str, row_vals: list,
                           col_key: str, col_vals: list,
                           metric: str = 'co2_savings_pct') -> np.ndarray:
    matrix = np.full((len(row_vals), len(col_vals)), np.nan)
    lookup = {(r[row_key], r[col_key]): r[metric] for r in rows}
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            val = lookup.get((rv, cv), np.nan)
            matrix[i, j] = val
    return matrix


def plot_heatmap_pen(pen_rows: list):
    """Heatmap: penetration (y) × threshold (x), per regime, CO2 savings."""
    thresholds   = sorted(set(r['threshold']       for r in pen_rows))
    penetrations = sorted(set(r['penetration_pct'] for r in pen_rows))
    pen_labels   = [f'{p}%' for p in penetrations]
    thr_labels   = [str(t) for t in thresholds]

    by_vol = defaultdict(list)
    for row in pen_rows:
        by_vol[row['volume']].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    fig.suptitle(
        'CO₂ Savings (%) — Penetration Rate × Emission Threshold\n'
        '(emission-based vs gap-based baseline; red border = violates −1% throughput)',
        fontsize=12, fontweight='bold'
    )

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes.flatten()[ax_idx]
        regime = REGIME_LABELS.get(vol, '?')
        rows   = [r for r in by_vol[vol]]

        matrix_co2 = _build_heatmap_matrix(
            rows, 'penetration_pct', penetrations, 'threshold', thresholds, 'co2_savings_pct')
        matrix_tp  = _build_heatmap_matrix(
            rows, 'penetration_pct', penetrations, 'threshold', thresholds, 'throughput_change_pct')

        if np.all(np.isnan(matrix_co2)):
            ax.text(0.5, 0.5, 'No data.\nRun run_threshold_x_scenarios.py first.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            ax.set_title(f'V{vol:04d} — {regime}')
            continue

        vmax = max(np.nanmax(np.abs(matrix_co2)), 0.01)
        im   = ax.imshow(matrix_co2, aspect='auto',
                         cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='CO₂ savings (%)')

        # Red border on cells that violate throughput constraint
        for i in range(len(penetrations)):
            for j in range(len(thresholds)):
                tp = matrix_tp[i, j]
                v  = matrix_co2[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if abs(v) > vmax * 0.5 else 'black')
                if not np.isnan(tp) and tp < THROUGHPUT_CONSTRAINT:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

        ax.set_xticks(range(len(thresholds)))
        ax.set_yticks(range(len(penetrations)))
        ax.set_xticklabels(thr_labels, fontsize=8)
        ax.set_yticklabels(pen_labels, fontsize=8)
        ax.set_xlabel('Emission Threshold (mg/s)', fontsize=9)
        ax.set_ylabel('Penetration Rate', fontsize=9)
        ax.set_title(f'V{vol:04d} — {regime}', fontsize=11, fontweight='bold')

    out = VISUALS_DIR / 'heatmap_pen_x_threshold.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_heatmap_comp(comp_rows: list):
    """Heatmap: composition (y) × threshold (x), per regime, CO2 savings."""
    thresholds   = sorted(set(r['threshold']    for r in comp_rows))
    compositions = [c for c in ALL_COMPOSITIONS if c in set(r['composition'] for r in comp_rows)]
    comp_labels  = compositions
    thr_labels   = [str(t) for t in thresholds]

    by_vol = defaultdict(list)
    for row in comp_rows:
        by_vol[row['volume']].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    fig.suptitle(
        'CO₂ Savings (%) — Fleet Composition × Emission Threshold\n'
        '(emission-based vs gap-based baseline; red border = violates −1% throughput)',
        fontsize=12, fontweight='bold'
    )

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes.flatten()[ax_idx]
        regime = REGIME_LABELS.get(vol, '?')
        rows   = [r for r in by_vol[vol]]

        matrix_co2 = _build_heatmap_matrix(
            rows, 'composition', compositions, 'threshold', thresholds, 'co2_savings_pct')
        matrix_tp  = _build_heatmap_matrix(
            rows, 'composition', compositions, 'threshold', thresholds, 'throughput_change_pct')

        if np.all(np.isnan(matrix_co2)):
            ax.text(0.5, 0.5, 'No data.\nRun run_threshold_x_scenarios.py first.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            ax.set_title(f'V{vol:04d} — {regime}')
            continue

        vmax = max(np.nanmax(np.abs(matrix_co2)), 0.01)
        im   = ax.imshow(matrix_co2, aspect='auto',
                         cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='CO₂ savings (%)')

        for i in range(len(compositions)):
            for j in range(len(thresholds)):
                tp = matrix_tp[i, j]
                v  = matrix_co2[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if abs(v) > vmax * 0.5 else 'black')
                if not np.isnan(tp) and tp < THROUGHPUT_CONSTRAINT:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

        ax.set_xticks(range(len(thresholds)))
        ax.set_yticks(range(len(compositions)))
        ax.set_xticklabels(thr_labels, fontsize=8)
        ax.set_yticklabels(comp_labels, fontsize=8)
        ax.set_xlabel('Emission Threshold (mg/s)', fontsize=9)
        ax.set_ylabel('Fleet Composition', fontsize=9)
        ax.set_title(f'V{vol:04d} — {regime}', fontsize=11, fontweight='bold')

    out = VISUALS_DIR / 'heatmap_comp_x_threshold.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Line plots (threshold on x, CO2 savings on y, lines per scenario)
# ---------------------------------------------------------------------------

PEN_COLORS = {
    20:  '#d62728',
    40:  '#ff7f0e',
    60:  '#bcbd22',
    80:  '#1f77b4',
    100: '#2ca02c',
}
COMP_LINE_COLORS = {
    'baseline': '#7f7f7f',
    'light':    '#2ca02c',
    'mixed':    '#ff7f0e',
    'heavy':    '#d62728',
}

REGIME_COLORS_LINE = {
    'Free-flow':        '#2ca02c',
    'Near-saturation':  '#1f77b4',
    'Breakdown onset':  '#ff7f0e',
    'Gridlock':         '#d62728',
}


def plot_linechart_pen(pen_rows: list):
    """Line chart: threshold (x) × CO2 savings (y), one line per penetration rate, per regime."""
    by_vol = defaultdict(list)
    for row in pen_rows:
        by_vol[row['volume']].append(row)

    thresholds_sorted = sorted(set(r['threshold'] for r in pen_rows))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        'CO₂ Savings vs Emission Threshold — by Penetration Rate\n'
        '(solid = within throughput constraint; dashed red = violates −1% throughput)',
        fontsize=12, fontweight='bold'
    )

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes.flatten()[ax_idx]
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS_LINE.get(regime, 'black')
        rows   = by_vol[vol]

        penetrations_present = sorted(set(r['penetration_pct'] for r in rows))

        for pen in penetrations_present:
            pen_rows_v = sorted(
                [r for r in rows if r['penetration_pct'] == pen],
                key=lambda x: x['threshold']
            )
            xs = [r['threshold']       for r in pen_rows_v]
            ys = [r['co2_savings_pct'] for r in pen_rows_v]
            tp = [r['throughput_change_pct'] for r in pen_rows_v]
            lc = PEN_COLORS.get(pen, 'grey')

            # Plot segments: solid where throughput OK, dashed red where violated
            for i in range(len(xs) - 1):
                violated = tp[i] < THROUGHPUT_CONSTRAINT or tp[i+1] < THROUGHPUT_CONSTRAINT
                ax.plot(xs[i:i+2], ys[i:i+2],
                        color='red' if violated else lc,
                        linestyle='--' if violated else '-',
                        linewidth=1.6, alpha=0.85)

            ax.plot(xs, ys, 'o', color=lc, markersize=4, alpha=0.9,
                    label=f'{pen}%')
            # Mark throughput-violating points
            for x, y, t in zip(xs, ys, tp):
                if t < THROUGHPUT_CONSTRAINT:
                    ax.plot(x, y, 'x', color='red', markersize=7, markeredgewidth=1.5)

        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.set_title(f'V{vol:04d} — {regime}', fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Emission Threshold (mg/s)', fontsize=9)
        ax.set_ylabel('CO₂ Savings (%)', fontsize=9)
        ax.set_xticks(thresholds_sorted)
        ax.set_xticklabels([str(t) for t in thresholds_sorted], fontsize=8)
        ax.legend(title='Penetration', fontsize=7, title_fontsize=8,
                  loc='best', framealpha=0.85)
        ax.grid(axis='both', alpha=0.2)

    out = VISUALS_DIR / 'linechart_pen_x_threshold.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_linechart_comp(comp_rows: list):
    """Line chart: threshold (x) × CO2 savings (y), one line per composition, per regime."""
    by_vol = defaultdict(list)
    for row in comp_rows:
        by_vol[row['volume']].append(row)

    thresholds_sorted = sorted(set(r['threshold'] for r in comp_rows))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        'CO₂ Savings vs Emission Threshold — by Fleet Composition\n'
        '(solid = within throughput constraint; dashed red = violates −1% throughput)',
        fontsize=12, fontweight='bold'
    )

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes.flatten()[ax_idx]
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS_LINE.get(regime, 'black')
        rows   = by_vol[vol]

        comps_present = [c for c in ALL_COMPOSITIONS if c in set(r['composition'] for r in rows)]

        for comp in comps_present:
            comp_rows_v = sorted(
                [r for r in rows if r['composition'] == comp],
                key=lambda r: r['threshold']
            )
            xs = [r['threshold']             for r in comp_rows_v]
            ys = [r['co2_savings_pct']       for r in comp_rows_v]
            tp = [r['throughput_change_pct'] for r in comp_rows_v]
            lc = COMP_LINE_COLORS.get(comp, 'grey')

            for i in range(len(xs) - 1):
                violated = tp[i] < THROUGHPUT_CONSTRAINT or tp[i+1] < THROUGHPUT_CONSTRAINT
                ax.plot(xs[i:i+2], ys[i:i+2],
                        color='red' if violated else lc,
                        linestyle='--' if violated else '-',
                        linewidth=1.6, alpha=0.85)

            ax.plot(xs, ys, 'o', color=lc, markersize=4, alpha=0.9, label=comp)
            for xv, yv, tv in zip(xs, ys, tp):
                if tv < THROUGHPUT_CONSTRAINT:
                    ax.plot(xv, yv, 'x', color='red', markersize=7, markeredgewidth=1.5)

        ax.axhline(0, color='black', linewidth=0.7, alpha=0.4)
        ax.set_title(f'V{vol:04d} -- {regime}', fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Emission Threshold (mg/s)', fontsize=9)
        ax.set_ylabel('CO2 Savings (%)', fontsize=9)
        ax.set_xticks(thresholds_sorted)
        ax.set_xticklabels([str(t) for t in thresholds_sorted], fontsize=8)
        ax.legend(title='Composition', fontsize=7, title_fontsize=8,
                  loc='best', framealpha=0.85)
        ax.grid(axis='both', alpha=0.2)

    out = VISUALS_DIR / 'linechart_comp_x_threshold.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_optimal_shift(pen_rows: list, comp_rows: list):
    """
    Summary plot: how much does the optimal threshold shift as scenario changes?
    One grouped bar per regime. Only generated when data spans multiple thresholds.
    """
    def _opt(rows_subset):
        valid = [(r['co2_savings_pct'], r['threshold'])
                 for r in rows_subset if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if not valid:
            return None
        return max(valid, key=lambda x: x[0])[1]

    pen_opts  = {}
    comp_opts = {}

    by_vol_pen = defaultdict(lambda: defaultdict(list))
    for r in pen_rows:
        by_vol_pen[r['volume']][r['penetration_pct']].append(r)
    for vol in REP_VOLUMES:
        for pen in sorted(by_vol_pen[vol]):
            pen_opts[(vol, pen)] = _opt(by_vol_pen[vol][pen])

    by_vol_comp = defaultdict(lambda: defaultdict(list))
    for r in comp_rows:
        by_vol_comp[r['volume']][r['composition']].append(r)
    for vol in REP_VOLUMES:
        for comp in ALL_COMPOSITIONS:
            comp_opts[(vol, comp)] = _opt(by_vol_comp[vol].get(comp, []))

    all_opts = [v for v in list(pen_opts.values()) + list(comp_opts.values())
                if v is not None]
    if len(set(all_opts)) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(
        'Optimal Emission Threshold Shift by Scenario\n'
        '(threshold maximising CO2 savings subject to throughput >= -1%)',
        fontsize=12, fontweight='bold'
    )

    penetrations_present = sorted(set(k[1] for k in pen_opts))
    comps_present = [c for c in ALL_COMPOSITIONS
                     if any(comp_opts.get((vol, c)) is not None for vol in REP_VOLUMES)]
    x = np.arange(len(REP_VOLUMES))

    ax = axes[0]
    width = 0.8 / max(len(penetrations_present), 1)
    for i, pen in enumerate(penetrations_present):
        vals = [pen_opts.get((vol, pen), np.nan) for vol in REP_VOLUMES]
        offset = (i - len(penetrations_present) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=PEN_COLORS.get(pen, 'grey'), alpha=0.82,
                      label=f'{pen}%', edgecolor='white')
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        str(int(val)), ha='center', va='bottom', fontsize=7)
    ax.set_title('Penetration Rate', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS.get(v, str(v)) for v in REP_VOLUMES],
                       fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Optimal Threshold (mg/s)', fontsize=9)
    ax.legend(title='Penetration', fontsize=8, title_fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1]
    width = 0.8 / max(len(comps_present), 1)
    for i, comp in enumerate(comps_present):
        vals = [comp_opts.get((vol, comp), np.nan) for vol in REP_VOLUMES]
        offset = (i - len(comps_present) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=COMP_LINE_COLORS.get(comp, 'grey'), alpha=0.82,
                      label=comp, edgecolor='white')
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        str(int(val)), ha='center', va='bottom', fontsize=7)
    ax.set_title('Fleet Composition', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS.get(v, str(v)) for v in REP_VOLUMES],
                       fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Optimal Threshold (mg/s)', fontsize=9)
    ax.legend(title='Composition', fontsize=8, title_fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    out = VISUALS_DIR / 'optimal_threshold_shift.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CSV export + optimal threshold matrix
# ---------------------------------------------------------------------------

def save_pen_csv(rows: list):
    out    = RESULTS_DIR / 'pen_x_threshold_summary.csv'
    fields = ['volume', 'regime', 'penetration_pct', 'threshold',
              'co2_savings_pct', 'throughput_change_pct', 'wait_change_pct']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def save_comp_csv(rows: list):
    out    = RESULTS_DIR / 'comp_x_threshold_summary.csv'
    fields = ['volume', 'regime', 'composition', 'threshold',
              'co2_savings_pct', 'throughput_change_pct', 'wait_change_pct']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


def save_optimal_matrix(pen_rows: list, comp_rows: list):
    """One CSV showing optimal threshold per (vol, penetration or composition)."""
    out_rows = []

    by_vol_pen = defaultdict(lambda: defaultdict(list))
    for r in pen_rows:
        by_vol_pen[r['volume']][r['penetration_pct']].append(r)
    for vol in REP_VOLUMES:
        for pen in sorted(by_vol_pen[vol]):
            rows = by_vol_pen[vol][pen]
            valid = [(r['co2_savings_pct'], r['threshold'])
                     for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
            opt = max(valid, key=lambda x: x[0])[1] if valid else None
            out_rows.append({'scenario_type':    'penetration',
                             'scenario_value':   f'{pen}%',
                             'volume':           vol,
                             'regime':           REGIME_LABELS.get(vol, '?'),
                             'optimal_threshold': opt})

    by_vol_comp = defaultdict(lambda: defaultdict(list))
    for r in comp_rows:
        by_vol_comp[r['volume']][r['composition']].append(r)
    for vol in REP_VOLUMES:
        for comp in ALL_COMPOSITIONS:
            rows = by_vol_comp[vol].get(comp, [])
            valid = [(r['co2_savings_pct'], r['threshold'])
                     for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
            opt = max(valid, key=lambda x: x[0])[1] if valid else None
            out_rows.append({'scenario_type':    'composition',
                             'scenario_value':   comp,
                             'volume':           vol,
                             'regime':           REGIME_LABELS.get(vol, '?'),
                             'optimal_threshold': opt})

    if not out_rows:
        return
    out = RESULTS_DIR / 'optimal_threshold_matrix.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Saved: {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  Task 4: Threshold x Scenarios Analysis")
    print("=" * 65)

    thresholds   = ALL_THRESHOLDS
    penetrations = [int(round(p * 100)) for p in ALL_PENETRATIONS]
    compositions = ALL_COMPOSITIONS

    print("  Loading penetration x threshold data ...")
    pen_rows  = load_pen_x_thr(REP_VOLUMES, ALL_PENETRATIONS, thresholds)
    print("  Loading composition x threshold data ...")
    comp_rows = load_comp_x_thr(REP_VOLUMES, compositions, thresholds)

    print(f"  pen_rows : {len(pen_rows)}")
    print(f"  comp_rows: {len(comp_rows)}")

    if not pen_rows and not comp_rows:
        print("\n  [WARNING] No data found.")
        print("  Run run_threshold_x_scenarios.py first.")
        sys.exit(0)

    print("\n  Generating plots ...")
    if pen_rows:
        plot_heatmap_pen(pen_rows)
        plot_linechart_pen(pen_rows)
    if comp_rows:
        plot_heatmap_comp(comp_rows)
        plot_linechart_comp(comp_rows)
    if pen_rows or comp_rows:
        plot_optimal_shift(pen_rows or [], comp_rows or [])

    print("\n  Saving CSVs ...")
    if pen_rows:
        save_pen_csv(pen_rows)
    if comp_rows:
        save_comp_csv(comp_rows)
    if pen_rows or comp_rows:
        save_optimal_matrix(pen_rows or [], comp_rows or [])

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("  Visuals : experiments/threshold_x_scenarios/visuals/")
    print("    heatmap_pen_x_threshold.png    -- pen x thr heatmap")
    print("    heatmap_comp_x_threshold.png   -- comp x thr heatmap")
    print("    linechart_pen_x_threshold.png  -- pen x thr line plots")
    print("    linechart_comp_x_threshold.png -- comp x thr line plots")
    print("    optimal_threshold_shift.png    -- how optimum shifts by scenario")
    print("  Results : experiments/threshold_x_scenarios/results/")
    print("=" * 65)


if __name__ == '__main__':
    main()

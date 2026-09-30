#!/usr/bin/env python3
"""
analyze_threshold_search.py — Task 1 Analysis.

Reads results from:
  experiments/threshold_search/data/thr_XXX/V####/emission_based/
  data/V####/baseline/                                (reference baseline)
  data/V####/emission_based/  [thr_050]               (reference thr=50)

Computes per-threshold, per-volume:
  - CO2 savings %          (emission_based vs baseline)
  - Throughput change %    (emission_based vs baseline)
  - Wait time change %     (emission_based vs baseline)

Produces:
  experiments/threshold_search/results/threshold_summary.csv
  experiments/threshold_search/visuals/co2_vs_threshold_by_regime.png
  experiments/threshold_search/visuals/all_metrics_vs_threshold.png

USAGE
-----
  python scripts/experiments/analyze_threshold_search.py
  python scripts/experiments/analyze_threshold_search.py --phase fine
"""

import argparse
import csv
import datetime
import os
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'threshold_search'
DATA_ROOT    = PROJECT_ROOT / 'data'

VISUALS_DIR = EXP_ROOT / 'visuals'
RESULTS_DIR = EXP_ROOT / 'results'
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Representative volumes and their regimes
# ---------------------------------------------------------------------------
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

REFERENCE_THRESHOLD = 50   # results already in data/ — always included

# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _parse_result_csv(csv_path: Path) -> dict:
    """
    Parse one result_seed_N.csv file.
    Returns dict with keys: throughput, avg_wait_s, co2_per_veh
    Returns None if file is unreadable or empty.
    """
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

    # Unique vehicles
    vehicle_ids = set()
    co2_sum     = 0.0
    wait_sum    = 0.0
    wait_count  = 0

    for r in rows:
        try:
            veh_id = r.get('veh_id', '')
            if not veh_id or veh_id.startswith('#'):
                continue
            vehicle_ids.add(veh_id)                       # ALL vehicles → throughput
            if r.get('in_box', '').strip().lower() == 'true':
                co2_sum  += float(r.get('co2_emission', 0))
                wait_val  = float(r.get('wait_time', 0))
                wait_sum += wait_val
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
    """
    Load all result_seed_*.csv in a folder.
    Returns dict seed -> metrics dict.
    """
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
    """Average metrics across all seeds in a folder."""
    if not seed_results:
        return None
    keys = ['throughput', 'avg_wait_s', 'co2_per_veh']
    means = {}
    for k in keys:
        vals = [v[k] for v in seed_results.values() if k in v]
        means[k] = np.mean(vals) if vals else 0.0
    means['n_seeds'] = len(seed_results)
    return means


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_thresholds(phase: str) -> list:
    """Return sorted list of threshold values found in experiments/threshold_search/data/."""
    found = [REFERENCE_THRESHOLD]   # always include thr=50 from data/
    data_dir = EXP_ROOT / 'data'
    if data_dir.exists():
        for entry in data_dir.iterdir():
            m = re.match(r'^thr_(\d+)$', entry.name)
            if m and entry.is_dir():
                t = int(m.group(1))
                if t not in found:
                    found.append(t)
    return sorted(found)


def load_all(volumes: list, thresholds: list) -> dict:
    """
    Returns nested dict: results[volume][threshold] = {
        'baseline': {metrics},
        'emission_based': {metrics},
    }
    """
    results = {}

    for vol in volumes:
        vol_label = f'V{vol:04d}'
        results[vol] = {}

        # Load baseline (threshold-independent — always from data/)
        bl_folder   = DATA_ROOT / vol_label / 'baseline'
        bl_seeds    = load_folder(bl_folder)
        bl_mean     = folder_mean(bl_seeds)

        for thr in thresholds:
            thr_label = f'thr_{thr:03d}'

            # emission_based: thr=50 lives in data/, others in experiments/
            if thr == REFERENCE_THRESHOLD:
                em_folder = DATA_ROOT / vol_label / 'emission_based'
            else:
                em_folder = EXP_ROOT / 'data' / thr_label / vol_label / 'emission_based'

            em_seeds = load_folder(em_folder)
            em_mean  = folder_mean(em_seeds)

            results[vol][thr] = {
                'baseline':      bl_mean,
                'emission_based': em_mean,
            }

    return results


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(results: dict) -> list:
    """
    Returns list of dicts with columns:
      volume, regime, threshold, co2_savings_pct, throughput_change_pct,
      wait_change_pct, n_seeds_em, n_seeds_bl
    """
    rows = []
    for vol, thr_dict in sorted(results.items()):
        regime = REGIME_LABELS.get(vol, '?')
        for thr, data in sorted(thr_dict.items()):
            bl = data['baseline']
            em = data['emission_based']
            if bl is None or em is None:
                continue

            def pct_change(new, old):
                return (new - old) / old * 100.0 if old else 0.0

            # CO2 savings: positive = emission_based is BETTER (lower CO2)
            co2_savings     = pct_change(bl['co2_per_veh'], em['co2_per_veh'])
            throughput_chg  = pct_change(em['throughput'],  bl['throughput'])
            wait_chg        = pct_change(em['avg_wait_s'],  bl['avg_wait_s'])

            rows.append({
                'volume':              vol,
                'regime':              regime,
                'threshold':           thr,
                'co2_savings_pct':     round(co2_savings,    3),
                'throughput_change_pct': round(throughput_chg, 3),
                'wait_change_pct':     round(wait_chg,       3),
                'n_seeds_em':          em.get('n_seeds', 0),
                'n_seeds_bl':          bl.get('n_seeds', 0),
            })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

THROUGHPUT_CONSTRAINT = -1.0   # % — must stay above this


COARSE_THRESHOLDS = [10, 25, 50, 75, 100, 150, 200, 300]   # used for x-axis labelling


def plot_main(delta_rows: list):
    """
    4-panel figure: one panel per regime volume.
    Each panel shows CO2 savings %, approach-wait change %, throughput change %
    vs threshold value.  A horizontal dashed red line marks -1% throughput.
    A green shaded band marks the 'sweet spot'.

    X-axis: all data points plotted; tick labels shown only at coarse grid
    values + the optimal, to avoid label clutter after fine-grid runs.
    """
    volumes = REP_VOLUMES
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(
        'Emission Controller: CO₂ Savings & Throughput vs Emission Threshold\n'
        '(emission-based vs gap-based baseline; mean across 10 seeds)',
        fontsize=13, fontweight='bold'
    )

    # Group by volume
    by_vol = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    for ax_idx, vol in enumerate(volumes):
        ax   = axes_flat[ax_idx]
        rows = sorted(by_vol[vol], key=lambda r: r['threshold'])
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS.get(regime, 'black')

        if not rows:
            ax.set_title(f'V{vol:04d} — {regime}\n(no data)')
            ax.text(0.5, 0.5, 'No simulation data available yet.\nRun run_threshold_search.py first.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='grey')
            continue

        thresholds    = [r['threshold']           for r in rows]
        co2_savings   = [r['co2_savings_pct']     for r in rows]
        throughput_chg = [r['throughput_change_pct'] for r in rows]
        wait_chg      = [r['wait_change_pct']     for r in rows]

        ax2 = ax.twinx()

        # ---- CO2 savings (left y-axis, solid) ----
        l1, = ax.plot(thresholds, co2_savings, 'o-', color=color,
                      linewidth=2, markersize=6, label='CO₂ savings (%)', zorder=4)

        # ---- Throughput change (right y-axis, grey dotted) ----
        l3, = ax2.plot(thresholds, throughput_chg, '^:', color='#555555', alpha=0.75,
                       linewidth=1.5, markersize=5, label='Throughput Δ (%)', zorder=3)

        # ---- Throughput constraint line at -1% ----
        ax2.axhline(THROUGHPUT_CONSTRAINT, color='red', linestyle='--',
                    linewidth=1.3, alpha=0.85, label='−1% throughput limit', zorder=5)

        # ---- Sweet-spot shading: CO2 savings > 0 AND throughput change > -1% ----
        thr_arr   = np.array(thresholds)
        co2_arr   = np.array(co2_savings)
        tp_arr    = np.array(throughput_chg)

        # Shade between adjacent points where both conditions hold
        for i in range(len(thr_arr) - 1):
            both_ok = (co2_arr[i] > 0 and tp_arr[i] > THROUGHPUT_CONSTRAINT and
                       co2_arr[i+1] > 0 and tp_arr[i+1] > THROUGHPUT_CONSTRAINT)
            if both_ok:
                ax.axvspan(thr_arr[i], thr_arr[i+1], alpha=0.08,
                           color='green', zorder=1)

        # ---- Mark optimal threshold (max CO2 saving with throughput > -1%) ----
        valid = [(r['co2_savings_pct'], r['threshold'])
                 for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if valid:
            best_co2, best_thr = max(valid, key=lambda x: x[0])
            ax.axvline(best_thr, color='green', linestyle=':', linewidth=1.5,
                       alpha=0.9, zorder=6)
            ax.annotate(f'Opt={best_thr}',
                        xy=(best_thr, best_co2),
                        xytext=(6, 4), textcoords='offset points',
                        fontsize=8, color='green', fontweight='bold')

        # ---- Zero reference ----
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        # ---- Axes labels ----
        ax.set_xlabel('Emission Threshold (mg/s)', fontsize=9)
        ax.set_ylabel('Change vs Baseline (%)', fontsize=9)
        ax2.set_ylabel('Throughput Change (%)', fontsize=8, color='#555555')
        ax2.tick_params(axis='y', labelcolor='#555555', labelsize=7)
        ax.set_title(f'V{vol:04d} — {regime}', fontsize=11, fontweight='bold',
                     color=color)
        ax.tick_params(axis='both', labelsize=8)

        # Show tick labels only at coarse grid values + the optimal threshold
        # so the x-axis stays readable after fine-grid runs add many points.
        thr_set    = set(thresholds)
        label_thrs = sorted(thr_set & set(COARSE_THRESHOLDS))
        if valid:
            label_thrs = sorted(set(label_thrs) | {best_thr})
        ax.set_xticks(thresholds)               # minor positions (all points)
        ax.set_xticklabels(
            [str(t) if t in label_thrs else '' for t in thresholds]
        )
        ax.tick_params(axis='x', rotation=60, labelsize=8)

        # ---- Legend ----
        lines  = [l1, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='upper right')
        ax2.legend(fontsize=8, loc='lower right')

        # ---- Annotate n_seeds ----
        if rows:
            ax.text(0.01, 0.01, f'n_seeds={rows[0]["n_seeds_em"]}',
                    transform=ax.transAxes, fontsize=7, color='grey', va='bottom')

    out_path = VISUALS_DIR / 'co2_vs_threshold_by_regime.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_optimal_summary(delta_rows: list):
    """
    Bar chart: optimal threshold per regime, coloured by regime.
    """
    by_vol = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    regimes   = []
    opt_thrs  = []
    colors    = []
    co2_vals  = []

    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        rows   = by_vol[vol]
        valid  = [(r['co2_savings_pct'], r['threshold'])
                  for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if valid:
            best_co2, best_thr = max(valid, key=lambda x: x[0])
            regimes.append(regime)
            opt_thrs.append(best_thr)
            co2_vals.append(best_co2)
            colors.append(REGIME_COLORS.get(regime, 'grey'))

    if not regimes:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(regimes, opt_thrs, color=colors, alpha=0.82, edgecolor='white', linewidth=1.2)
    for bar, co2 in zip(bars, co2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f'{bar.get_height():.0f} mg/s\n(CO₂ Δ{co2:+.2f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Traffic Regime', fontsize=10)
    ax.set_ylabel('Optimal Threshold (mg/s)', fontsize=10)
    ax.set_title('Optimal Emission Threshold per Traffic Regime\n'
                 '(maximises CO₂ savings, throughput change > −1%)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(opt_thrs) * 1.3 if opt_thrs else 350)

    out_path = VISUALS_DIR / 'optimal_threshold_by_regime.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_stable_region(delta_rows: list):
    """
    4-panel figure: rolling-average CO2 savings with +/-0.3% stability band.

    Helps justify threshold selection by showing:
      - The raw per-threshold CO2 savings scatter (all data points)
      - A 5-point centred rolling average to smooth stochastic noise
      - A +/-0.3% band around the rolling average
      - A green shaded plateau where the rolling average exceeds (peak - 0.3%)
      - The constrained-optimal threshold marked with a vertical line

    This addresses the professor's request to justify threshold selection via
    a stable-region / rolling-average analysis rather than a single peak.
    """
    from collections import defaultdict as _dd
    by_vol = _dd(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(20, 13), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(
        'CO2 Savings Stable-Region Analysis\n'
        '(rolling average +/- 0.3% band; emission-based vs baseline)',
        fontsize=13, fontweight='bold'
    )

    for ax_idx, vol in enumerate(REP_VOLUMES):
        ax     = axes_flat[ax_idx]
        rows   = sorted(by_vol[vol], key=lambda r: r['threshold'])
        regime = REGIME_LABELS.get(vol, '?')
        color  = REGIME_COLORS.get(regime, 'black')

        if not rows:
            ax.set_title(f'V{vol:04d} -- {regime} (no data)')
            continue

        thresholds  = np.array([r['threshold']       for r in rows], dtype=float)
        co2_savings = np.array([r['co2_savings_pct'] for r in rows], dtype=float)
        tp_chg      = np.array([r['throughput_change_pct'] for r in rows], dtype=float)

        # -- rolling mean (centred, window=5, padded at edges) --
        win = 5
        padded = np.pad(co2_savings, (win // 2, win // 2), mode='edge')
        rolling = np.array([np.mean(padded[i:i + win]) for i in range(len(co2_savings))])

        # -- stable plateau: rolling avg >= (peak_rolling - 0.3%) AND throughput OK --
        valid_mask  = tp_chg > THROUGHPUT_CONSTRAINT
        valid_roll  = rolling.copy()
        valid_roll[~valid_mask] = np.nan
        peak_roll   = np.nanmax(valid_roll) if np.any(valid_mask) else np.nan
        stable_mask = valid_mask & (rolling >= peak_roll - 0.3) if not np.isnan(peak_roll) else np.zeros(len(rows), bool)

        # -- plot raw scatter --
        ax.scatter(thresholds, co2_savings, color=color, alpha=0.55,
                   zorder=3, s=30, label='Per-threshold CO2 savings')

        # -- rolling average line --
        ax.plot(thresholds, rolling, color=color, linewidth=2.2,
                zorder=4, label='Rolling avg (window=5)')

        # -- +/-0.3% band --
        ax.fill_between(thresholds, rolling - 0.3, rolling + 0.3,
                        alpha=0.15, color=color, label='+/-0.3% band')

        # -- stable plateau shading --
        for i in range(len(thresholds) - 1):
            if stable_mask[i] and stable_mask[i + 1]:
                ax.axvspan(thresholds[i], thresholds[i + 1],
                           alpha=0.18, color='green', zorder=1)

        # -- optimal threshold marker --
        valid_pts = [(r['co2_savings_pct'], r['threshold'])
                     for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if valid_pts:
            best_co2, best_thr = max(valid_pts, key=lambda x: x[0])
            ax.axvline(best_thr, color='darkgreen', linestyle='--', linewidth=1.8,
                       alpha=0.9, zorder=6)
            ax.annotate(f'Opt={best_thr} mg/s\n({best_co2:+.2f}%)',
                        xy=(best_thr, best_co2),
                        xytext=(8, 6), textcoords='offset points',
                        fontsize=8, color='darkgreen', fontweight='bold')

        # -- stable count annotation --
        n_stable = int(np.sum(stable_mask))
        n_valid  = int(np.sum(valid_mask))
        ax.text(0.98, 0.97, f'Stable: {n_stable}/{n_valid} thresholds',
                transform=ax.transAxes, fontsize=8, color='darkgreen',
                ha='right', va='top', fontweight='bold')

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Emission Threshold (mg/s)', fontsize=9)
        ax.set_ylabel('CO2 Savings vs Baseline (%)', fontsize=9)
        ax.set_title(f'V{vol:04d} -- {regime}', fontsize=11, fontweight='bold', color=color)
        ax.tick_params(axis='both', labelsize=8)

        # x-ticks: only at coarse values to keep readable
        coarse = [t for t in thresholds if t in set(COARSE_THRESHOLDS)]
        if valid_pts:
            coarse = sorted(set(coarse) | {best_thr})
        ax.set_xticks(thresholds)
        ax.set_xticklabels([str(int(t)) if t in coarse else '' for t in thresholds])
        ax.tick_params(axis='x', rotation=60, labelsize=8)

        ax.legend(fontsize=7.5, loc='upper right')

        if rows:
            ax.text(0.01, 0.01, f'n_seeds={rows[0]["n_seeds_em"]}',
                    transform=ax.transAxes, fontsize=7, color='grey', va='bottom')

    # Add green patch to legend for stable plateau
    stable_patch = mpatches.Patch(color='green', alpha=0.35, label='Stable plateau (within 0.3% of peak)')
    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles + [stable_patch], labels + [stable_patch.get_label()],
                      fontsize=7, loc='upper right')
            break

    out_path = VISUALS_DIR / 'stable_region_analysis.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_summary_csv(delta_rows: list):
    out_path = RESULTS_DIR / 'threshold_summary.csv'
    fieldnames = [
        'volume', 'regime', 'threshold',
        'co2_savings_pct', 'throughput_change_pct', 'wait_change_pct',
        'n_seeds_em', 'n_seeds_bl',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(delta_rows)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def save_optimal_csv(delta_rows: list):
    """One row per regime: the threshold value that is optimal."""
    by_vol = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    rows_out = []
    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        rows   = by_vol[vol]
        valid  = [(r['co2_savings_pct'], r['threshold'], r)
                  for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if not valid:
            rows_out.append({
                'volume': vol, 'regime': regime,
                'optimal_threshold': None, 'co2_savings_pct': None,
                'throughput_change_pct': None, 'wait_change_pct': None,
            })
            continue
        _, best_thr, best_row = max(valid, key=lambda x: x[0])
        rows_out.append({
            'volume':                vol,
            'regime':                regime,
            'optimal_threshold':     best_thr,
            'co2_savings_pct':       best_row['co2_savings_pct'],
            'throughput_change_pct': best_row['throughput_change_pct'],
            'wait_change_pct':       best_row['wait_change_pct'],
        })

    out_path = RESULTS_DIR / 'optimal_threshold_per_regime.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Fine-grid recommendation helper
# ---------------------------------------------------------------------------

def _compute_fine_recommendation(delta_rows: list, thresholds: list) -> dict:
    """
    For each regime, identify the coarse bracket [lower, upper] around the
    optimal threshold and suggest 6 evenly-spaced fill-in values.

    Returns:
      per_regime : {regime_name: {'optimal': T, 'lower': L, 'upper': U,
                                   'suggested': [...]}} or None if no valid data
      combined   : sorted list of all unique suggested values (across regimes)
      command    : ready-to-run --fine-thresholds command string
    """
    sorted_thrs = sorted(thresholds)
    by_vol: dict = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    per_regime: dict = {}
    all_suggested: set = set()

    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        rows   = by_vol[vol]
        valid  = [(r['co2_savings_pct'], r['threshold'], r)
                  for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if not valid:
            per_regime[regime] = None
            continue

        _, best_thr, _ = max(valid, key=lambda x: x[0])

        if best_thr in sorted_thrs:
            idx = sorted_thrs.index(best_thr)
        else:
            idx = -1

        if idx == -1:
            per_regime[regime] = {'optimal': best_thr,
                                   'lower': None, 'upper': None, 'suggested': []}
            continue

        lower = sorted_thrs[idx - 1] if idx > 0 else sorted_thrs[0]
        upper = sorted_thrs[idx + 1] if idx < len(sorted_thrs) - 1 else sorted_thrs[-1]

        if lower == upper or lower == best_thr == upper:
            suggested: list = []
        else:
            n    = 6
            step = (upper - lower) / (n + 1)
            raw  = [lower + step * (i + 1) for i in range(n)]
            # Round to nearest integer, drop values already in the coarse grid
            suggested = sorted({int(round(v)) for v in raw} - set(sorted_thrs))

        per_regime[regime] = {
            'optimal':   best_thr,
            'lower':     lower,
            'upper':     upper,
            'suggested': suggested,
        }
        all_suggested.update(suggested)

    combined = sorted(all_suggested)
    if combined:
        thr_str = ' '.join(str(int(t)) for t in combined)
        command = (
            'python scripts/experiments/run_threshold_search.py \\\n'
            f'      --phase fine --fine-thresholds {thr_str}'
        )
    else:
        command = (
            '(All optima are already at the grid boundary — '
            'consider expanding the coarse grid before running a fine search.)'
        )

    return {'per_regime': per_regime, 'combined': combined, 'command': command}


# ---------------------------------------------------------------------------
# Analysis report (human-readable text file)
# ---------------------------------------------------------------------------

def save_analysis_report(delta_rows, thresholds, phase):
    """Write a human-readable report to results/coarse_analysis_report.txt
    (or fine_analysis_report.txt).  Called automatically at end of main().
    Contains: run parameters, full scan table, optimal-threshold table,
    and the ready-to-paste Step 3 --fine-thresholds command.
    """
    fname    = 'fine_analysis_report.txt' if phase == 'fine' else 'coarse_analysis_report.txt'
    out_path = RESULTS_DIR / fname

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    by_vol = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    lines = []

    # ---- Header ----
    lines += [
        '=' * 67,
        '  Threshold Grid Search -- Analysis Report',
        '  Phase    : ' + phase,
        '  Generated: ' + ts,
        '=' * 67,
        '',
        '  Thresholds analysed : ' + str(sorted(thresholds)),
        '  Volumes             : ' + str(['V{:04d}'.format(v) for v in REP_VOLUMES]),
        '  Data points loaded  : ' + str(len(delta_rows)),
        '',
    ]

    # ---- Optimal threshold table ----
    # ---- Methodology framing ----
    lines += [
        '  METHODOLOGY',
        '  ' + '-' * 67,
        '  Approach : Constrained Exhaustive Grid Search',
        '  The emission threshold parameter was searched exhaustively over a',
        '  ' + str(len(thresholds)) + '-point coarse grid (10-300 mg/s) followed by a fine-grid',
        '  refinement around the coarse optimum. All combinations of threshold',
        '  x volume (4 regimes) x seed (10) were simulated.',
        '  Constraint: candidate thresholds are rejected if throughput change',
        '  falls below -1% vs baseline (ensures traffic performance is preserved).',
        '  Optimum: the threshold maximising mean in-box CO2 savings among all',
        '  throughput-valid thresholds. In-box CO2 = CO2 emitted within the',
        '  300 m intersection detection zone only (rows where in_box==True).',
        '',
    ]

    lines += [
        '  OPTIMAL THRESHOLD PER REGIME  (throughput constraint > -1%)',
        '  {:<20} {:>6} {:>8} {:>10} {:>8} {:>8}'.format(
            'Regime', 'Volume', 'Opt Thr', 'CO2 Sav%', 'Tput%', 'Wait%'),
        '  ' + '-' * 67,
    ]
    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        rows   = by_vol[vol]
        valid  = [(r['co2_savings_pct'], r['threshold'], r)
                  for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if valid:
            _, best_thr, best_row = max(valid, key=lambda x: x[0])
            lines.append('  {:<20} {:>6} {:>8} {:>+10.2f} {:>+8.2f} {:>+8.2f}'.format(
                regime, vol, best_thr,
                best_row['co2_savings_pct'],
                best_row['throughput_change_pct'],
                best_row['wait_change_pct']))
        else:
            lines.append(
                '  {:<20} {:>6}     N/A  '
                '(all thresholds violate throughput constraint)'.format(regime, vol))

    lines += ['', '=' * 67, '']

    # ---- Full scan table ----
    lines += [
        '  FULL THRESHOLD SCAN  (all data points)',
        '  {:<20} {:>5} {:>5} {:>9} {:>8} {:>8}  {:>4}'.format(
            'Regime', 'Vol', 'Thr', 'CO2%', 'Tput%', 'Wait%', 'OK?'),
        '  ' + '-' * 67,
    ]
    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        rows   = sorted(by_vol[vol], key=lambda r: r['threshold'])
        for r in rows:
            ok = 'YES' if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT else 'NO '
            lines.append(
                '  {:<20} {:>5} {:>5} {:>+9.2f} {:>+8.2f} {:>+8.2f}  {}'.format(
                    regime, vol, r['threshold'],
                    r['co2_savings_pct'],
                    r['throughput_change_pct'],
                    r['wait_change_pct'],
                    ok))

    lines += ['', '=' * 67, '']

    if phase == 'fine':
        # ---- Fine phase: analysis is complete, direct user to Step 5 ----
        lines += [
            '  FINE-GRID ANALYSIS COMPLETE',
            '  ' + '-' * 67,
            '  The optimal thresholds above are your final confirmed values.',
            '  No further threshold searching is required.',
            '',
            '  NEXT STEP -- proceed to Step 5 (Task 2: Penetration Rate):',
            '  ' + '-' * 62,
            '  python scripts/experiments/run_penetration_experiment.py',
            '  ' + '-' * 62,
            '',
            '  These optimal thresholds will be used automatically by Tasks 2, 3,',
            '  and 4 via the default emission threshold setting (thr=50). The fine',
            '  search results are available in threshold_summary.csv for reference.',
            '',
            '=' * 67,
        ]
    else:
        # ---- Coarse phase: recommend fine-grid values for Step 3 ----
        rec = _compute_fine_recommendation(delta_rows, thresholds)

        lines += [
            '  FINE-GRID RECOMMENDATION',
            "  (bracket = [lower_coarse, upper_coarse] around each regime's optimum;",
            '   fine values are 6 evenly-spaced integers inside that bracket)',
            '',
        ]
        for vol in REP_VOLUMES:
            regime = REGIME_LABELS.get(vol, '?')
            info   = rec['per_regime'].get(regime)
            if info is None:
                lines.append('  {:<22}: no valid threshold found'.format(regime))
            elif info.get('lower') is None:
                lines.append(
                    '  {:<22}: optimal={} mg/s  (cannot determine bracket)'.format(
                        regime, info['optimal']))
            else:
                bracket = '[{}, {}]'.format(info['lower'], info['upper'])
                sug_str = (', '.join(str(int(v)) for v in info['suggested'])
                           if info['suggested']
                           else '(none -- optimum already at grid boundary)')
                lines.append(
                    '  {:<22}: optimal={:>5} mg/s  bracket={:<12}  fine={}'.format(
                        regime, info['optimal'], bracket, sug_str))

        lines += [
            '',
            '  Combined fine-grid values: ' + str(rec['combined']),
            '',
            '  STEP 3 -- copy-paste this command to run the fine grid:',
            '  ' + '-' * 62,
            '  ' + rec['command'],
            '  ' + '-' * 62,
            '',
            '  After Step 3, re-analyse with:',
            '    python scripts/experiments/analyze_threshold_search.py --phase fine',
            '',
            '=' * 67,
        ]

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    print('\n  Analysis report saved --> ' + str(out_path.relative_to(PROJECT_ROOT)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Analyse threshold grid search results.')
    ap.add_argument('--phase', choices=['coarse', 'fine', 'all'], default='all',
                    help='Which phase results to include (default: all).')
    return ap.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 65)
    print("  Threshold Grid Search -- Analysis")
    print("=" * 65)

    thresholds = discover_thresholds(args.phase)
    print("  Thresholds found : " + str(thresholds))
    print("  Volumes          : " + str(['V{:04d}'.format(v) for v in REP_VOLUMES]))

    results    = load_all(REP_VOLUMES, thresholds)
    delta_rows = compute_deltas(results)

    if not delta_rows:
        print("\n  [WARNING] No data found. Run run_threshold_search.py first.")
        print("  (Results for thr=50 at the 4 volumes are expected in data/.)")
        sys.exit(0)

    print("\n  Loaded {} data points across {} thresholds and {} volumes.".format(
        len(delta_rows), len(thresholds), len(REP_VOLUMES)))

    print("\n  Generating plots ...")
    plot_main(delta_rows)
    plot_optimal_summary(delta_rows)
    plot_stable_region(delta_rows)

    print("\n  Saving CSVs ...")
    save_summary_csv(delta_rows)
    save_optimal_csv(delta_rows)

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("  Visuals : experiments/threshold_search/visuals/")
    print("  Results : experiments/threshold_search/results/")
    print("=" * 65)

    # Print optimal table to terminal
    by_vol = defaultdict(list)
    for row in delta_rows:
        by_vol[row['volume']].append(row)

    print("\n  Optimal threshold per regime (throughput constraint > -1%):")
    print("  {:<20} {:>6} {:>8} {:>10} {:>8} {:>8}".format(
        'Regime', 'Volume', 'Opt Thr', 'CO2 Sav%', 'Tput%', 'Wait%'))
    print("  " + "-" * 65)
    for vol in REP_VOLUMES:
        regime = REGIME_LABELS.get(vol, '?')
        rows   = by_vol[vol]
        valid  = [(r['co2_savings_pct'], r['threshold'], r)
                  for r in rows if r['throughput_change_pct'] > THROUGHPUT_CONSTRAINT]
        if valid:
            _, best_thr, best_row = max(valid, key=lambda x: x[0])
            print("  {:<20} {:>6} {:>8} {:>+10.2f} {:>+8.2f} {:>+8.2f}".format(
                regime, vol, best_thr,
                best_row['co2_savings_pct'],
                best_row['throughput_change_pct'],
                best_row['wait_change_pct']))
        else:
            print("  {:<20} {:>6}    N/A (all runs exceed throughput constraint)".format(
                regime, vol))

    print("\n  Next step -- if coarse analysis shows a clear optimum region,")
    print("  run the fine grid:")
    print("    python scripts/experiments/run_threshold_search.py \\")
    print("      --phase fine --fine-thresholds <T1 T2 T3 ...>")
    print("  (The exact command with pre-filled thresholds is in the report below.)")

    # Save the full analysis report (table + fine-grid command) to file
    save_analysis_report(delta_rows, thresholds, args.phase)


if __name__ == '__main__':
    main()

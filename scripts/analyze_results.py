"""analyze_results.py

SUMO Project (Paper 1) — Enhanced Multi-Seed & Multi-Volume Analysis

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
1. Command-line arguments added (argparse).
   New arguments:
     --data-dir   Root folder containing baseline/ and emission_based/
                  sub-folders.  Default: 'data'.
                  Pass 'data/V0600' to analyse one specific volume.
     --out-dir    Output folder for plots.  Default: 'visuals'.
                  Pass 'visuals/V0600' to keep each volume's plots in
                  its own folder.
     --vol-label  Human-readable label added to plot titles
                  (e.g. 'V0600 — 600 veh/hr').  Auto-derived from
                  --data-dir when omitted.
     --cross-volume
                  When present, scan --data-dir for V#### sub-folders
                  and produce a cross-volume summary instead of
                  per-volume plots.

   DATA_DIR and OUTPUT_PLOTS_DIR are now set from CLI args at start-up
   so all downstream functions automatically use the correct paths.

2. Cross-volume comparison plot  [NEW].
   run_cross_volume_comparison() scans a parent data directory for
   V#### sub-folders, loads the results from each, computes mean metrics,
   and produces a 6-panel figure showing how each metric evolves across
   demand levels for both methods.  Also saves a CSV summary.
   Output: <out-dir>/cross_volume/cross_volume_comparison.png

3. _title() helper.
   All plot title strings now call _title(base) which prepends the
   volume label so figures from different volumes are distinguishable
   when placed side by side in the paper.

4. _save() helper.
   Consolidates the repeated savefig/close/print pattern into one call.

5. load_phase_switches_from_logs() and load_capacity_from_json() are
   unchanged in behaviour; they use DATA_DIR which is now CLI-controlled.
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats


# =============================================================================
# CLI argument parsing
# =============================================================================

def _parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Analyse Pipeline B results for one volume level (default) "
            "or produce a cross-volume comparison (--cross-volume)."
        )
    )
    ap.add_argument(
        '--data-dir', type=str, default='data',
        help=(
            "Root data directory containing baseline/ and emission_based/ "
            "sub-folders.  Pass 'data/V0600' to analyse one volume. "
            "Pass 'data' with --cross-volume to scan all V#### sub-folders. "
            "(default: 'data')"
        )
    )
    ap.add_argument(
        '--out-dir', type=str, default='visuals',
        help=(
            "Output directory for plots.  Pass 'visuals/V0600' to keep "
            "each volume's plots in its own folder. (default: 'visuals')"
        )
    )
    ap.add_argument(
        '--vol-label', type=str, default=None,
        help=(
            "Human-readable volume label added to plot titles "
            "(e.g. 'V0600 - 600 veh/hr').  Auto-derived from --data-dir "
            "when omitted."
        )
    )
    ap.add_argument(
        '--cross-volume', action='store_true', default=False,
        help=(
            "Scan --data-dir for V#### sub-folders and produce a "
            "cross-volume summary plot instead of per-volume plots."
        )
    )
    args, _ = ap.parse_known_args()
    return args


_ARGS = _parse_args()

DATA_DIR         = _ARGS.data_dir
OUTPUT_PLOTS_DIR = _ARGS.out_dir


def _derive_vol_label(data_dir: str, explicit: str) -> str:
    if explicit:
        return explicit
    base = os.path.basename(os.path.normpath(data_dir))
    if re.match(r'^V\d{4}$', base):
        return base
    return ''


VOL_LABEL = _derive_vol_label(DATA_DIR, _ARGS.vol_label)


# =============================================================================
# Configuration
# =============================================================================

METHODS          = ['baseline', 'emission_based']
SEEDS            = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]
VEHICLES_PER_DAY = 14_400
DAYS_PER_YEAR    = 365
CO2_PER_KM_CAR_G = 120

PHASE_SWITCHES: Dict[str, Dict[int, int]] = {}

plt.style.use('default')
sns.set_palette('colorblind')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size']      = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# =============================================================================
# Helpers
# =============================================================================

def _title(base: str) -> str:
    """Prepend volume label to a plot title when one is set."""
    return f"[{VOL_LABEL}]  {base}" if VOL_LABEL else base


def _save(fig, filename: str) -> None:
    """Save and close a figure into OUTPUT_PLOTS_DIR."""
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_PLOTS_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out}")


# =============================================================================
# Data loading
# =============================================================================

def load_and_process_data() -> Tuple[Dict[str, Dict[int, pd.DataFrame]],
                                      Dict[str, pd.DataFrame]]:
    all_results: Dict[str, Dict[int, pd.DataFrame]] = {}
    raw_data:    Dict[str, pd.DataFrame]             = {}

    for method in METHODS:
        method_data: Dict[int, pd.DataFrame] = {}
        frames: List[pd.DataFrame] = []
        for fpath in sorted(glob.glob(
                os.path.join(DATA_DIR, method, 'result_seed_*.csv'))):
            try:
                seed = int(os.path.splitext(fpath)[0].split('_')[-1])
            except Exception:
                print(f"Skipping unexpected filename: {fpath}")
                continue
            try:
                df = pd.read_csv(fpath, comment='#', low_memory=False)
                df = clean_data(df)
                method_data[seed] = df
                frames.append(df)
                print(f"Loaded {len(df):,} rows from {fpath}")
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
        all_results[method] = method_data
        raw_data[method]    = (pd.concat(frames, ignore_index=True)
                               if frames else pd.DataFrame())
    return all_results, raw_data


def load_tripinfo_data() -> Dict[str, Dict[int, pd.DataFrame]]:
    tripinfo: Dict[str, Dict[int, pd.DataFrame]] = {}
    for method in METHODS:
        method_trips: Dict[int, pd.DataFrame] = {}
        for fpath in sorted(glob.glob(
                os.path.join(DATA_DIR, method, 'tripinfo_seed_*.xml'))):
            try:
                seed = int(os.path.splitext(fpath)[0].split('_')[-1])
            except Exception:
                print(f"Skipping unexpected filename: {fpath}")
                continue
            try:
                tree = ET.parse(fpath)
                rows = [
                    {'id':          t.attrib.get('id', ''),
                     'depart':      float(t.attrib.get('depart',      0)),
                     'arrival':     float(t.attrib.get('arrival',     0)),
                     'duration':    float(t.attrib.get('duration',    0)),
                     'waitingTime': float(t.attrib.get('waitingTime', 0)),
                     'timeLoss':    float(t.attrib.get('timeLoss',    0)),
                     'routeLength': float(t.attrib.get('routeLength', 0))}
                    for t in tree.findall('.//tripinfo')
                ]
                if rows:
                    method_trips[seed] = pd.DataFrame(rows)
                    print(f"Loaded tripinfo ({len(rows)} trips) from {fpath}")
                else:
                    print(f"Warning: no <tripinfo> entries in {fpath}")
            except Exception as e:
                print(f"Error parsing {fpath}: {e}")
        tripinfo[method] = method_trips
    return tripinfo


def load_capacity_from_json(capacity_json_path: str = None) -> int:
    if capacity_json_path is None:
        capacity_json_path = os.path.join(
            'data', 'capacity_analysis', 'summary', 'capacity_summary.json'
        )
    if os.path.exists(capacity_json_path):
        with open(capacity_json_path) as f:
            cap = int(json.load(f)['capacity_veh_per_cycle'])
        print(f"  [capacity] Loaded {cap} veh/cycle from {capacity_json_path}")
        return cap
    fallback = 79
    print(f"  [capacity] WARNING: {capacity_json_path} not found. "
          f"Using fallback {fallback}.")
    return fallback


def load_phase_switches_from_logs(data_dir: str = None) -> Dict[str, Dict[int, int]]:
    if data_dir is None:
        data_dir = DATA_DIR
    result: Dict[str, Dict[int, int]] = {m: {} for m in METHODS}
    for method in METHODS:
        log_files = sorted(glob.glob(
            os.path.join(data_dir, method, 'log_seed_*.txt')
        ))
        if not log_files:
            print(f"  [phase_switches] WARNING: no logs for '{method}' in "
                  f"{os.path.join(data_dir, method)}/")
            continue
        for fpath in log_files:
            seed = switches = log_method = None
            try:
                with open(fpath, encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        m = re.search(r'Seed:\s*(\d+),\s*Method:\s*(\w+)', line)
                        if m:
                            seed, log_method = int(m.group(1)), m.group(2)
                        m2 = re.search(r'Phase switches:\s*(\d+)', line)
                        if m2:
                            switches = int(m2.group(1))
            except OSError as e:
                print(f"  [phase_switches] Could not read {fpath}: {e}")
                continue
            if seed is None or switches is None:
                print(f"  [phase_switches] WARNING: incomplete data in {fpath}")
                continue
            if log_method and log_method != method:
                print(f"  [phase_switches] NOTE: {fpath} method mismatch — "
                      f"using folder name '{method}'.")
            result[method][seed] = switches
            print(f"  [phase_switches] {method}/seed {seed}: {switches}")
    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['sim_time', 'speed', 'wait_time', 'co2_emission', 'phase_duration']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'in_box' in df.columns:
        df['in_box'] = (
            df['in_box'].astype(str).str.lower()
            .map({'true': True, 'false': False, '1': True, '0': False,
                  'yes': True, 'no': False})
            .fillna(False)
        )
    essential = [c for c in ['veh_id', 'sim_time'] if c in df.columns]
    if essential:
        df = df.dropna(subset=essential)
    if 'sim_time' in df.columns:
        df = df[np.isfinite(df['sim_time'])]
    return df


# =============================================================================
# Metric calculators
# =============================================================================

def calculate_throughput(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    n_veh = df['veh_id'].nunique()
    sim_t = pd.to_numeric(df['sim_time'], errors='coerce').dropna()
    dur_h = (sim_t.max() - sim_t.min()) / 3600.0
    return (n_veh / dur_h) if dur_h > 0 else 0.0


def calculate_emission_metrics(df: pd.DataFrame) -> Dict[str, float]:
    in_box = df[df['in_box'] == True]
    if in_box.empty:
        return {'total_co2': 0.0, 'avg_co2_per_vehicle': 0.0,
                'avg_co2_per_second': 0.0}
    total_co2 = float(in_box['co2_emission'].sum())
    nveh      = int(in_box['veh_id'].nunique())
    sim_t     = pd.to_numeric(in_box['sim_time'], errors='coerce').dropna()
    dur_s     = float(sim_t.max() - sim_t.min()) if not sim_t.empty else 0.0
    return {'total_co2':           total_co2,
            'avg_co2_per_vehicle': total_co2 / nveh if nveh > 0 else 0.0,
            'avg_co2_per_second':  total_co2 / dur_s if dur_s > 0 else 0.0}


def calculate_wait_time_metrics(df: pd.DataFrame) -> Dict[str, float]:
    in_box = df[df['in_box'] == True]
    if in_box.empty:
        return {'total_wait_time': 0.0, 'avg_wait_time_per_vehicle': 0.0}
    veh_wait = in_box.groupby('veh_id')['wait_time'].max()
    return {'total_wait_time':           float(veh_wait.sum()),
            'avg_wait_time_per_vehicle': float(veh_wait.mean())}


def calculate_travel_time_metrics(df: pd.DataFrame) -> Dict[str, float]:
    times = [
        float(vdf[vdf['in_box']==True]['sim_time'].max()) -
        float(vdf[vdf['in_box']==True]['sim_time'].min())
        for _, vdf in df.groupby('veh_id')
        if not vdf[vdf['in_box']==True].empty
    ]
    if not times:
        return {'avg_travel_time': 0.0, 'median_travel_time': 0.0,
                'min_travel_time': 0.0, 'max_travel_time': 0.0}
    arr = np.array(times, dtype=float)
    return {'avg_travel_time':    float(arr.mean()),
            'median_travel_time': float(np.median(arr)),
            'min_travel_time':    float(arr.min()),
            'max_travel_time':    float(arr.max())}


def calculate_speed_metrics(df: pd.DataFrame) -> Dict[str, float]:
    in_box = df[df['in_box'] == True]
    if in_box.empty:
        return {'avg_speed': 0.0, 'median_speed': 0.0}
    return {'avg_speed':    float(in_box['speed'].mean()),
            'median_speed': float(in_box['speed'].median())}


# =============================================================================
# Per-seed rollup
# =============================================================================

@dataclass(frozen=True)
class MetricSpec:
    key:              str
    title:            str
    unit:             str
    higher_is_better: bool


METRICS: List[MetricSpec] = [
    MetricSpec('throughput',          'Throughput',                 'veh/h', True),
    MetricSpec('avg_wait_time',       'Avg Wait Time (in-box)',     's',     False),
    MetricSpec('avg_fulltrip_wait',   'Avg Wait Time (full-trip)',  's',     False),
    MetricSpec('avg_travel_time',     'Average Travel Time',        's',     False),
    MetricSpec('avg_speed',           'Average Speed',              'm/s',   True),
    MetricSpec('avg_co2_per_vehicle', 'CO2 Emissions per Vehicle',  'mg',    False),
]

SEED_COLORS = {42: '#1f77b4', 10: '#ff7f0e', 25: '#2ca02c',
               50: '#d62728', 75: '#9467bd',
               13: '#8c564b', 17: '#e377c2', 37: '#7f7f7f',
               5:  '#bcbd22', 77: '#17becf'}

METRIC_DECIMALS = {
    'throughput': 1, 'avg_wait_time': 1, 'avg_fulltrip_wait': 1,
    'avg_travel_time': 1, 'avg_speed': 2, 'avg_co2_per_vehicle': 0,
}


def _seed_color(seed: int) -> str:
    return SEED_COLORS.get(seed, 'black')


def _fixed_x_offsets(n: int, span: float = 0.14) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-span / 2.0, span / 2.0, n)


def compute_seed_rollups(
    all_results: Dict[str, Dict[int, pd.DataFrame]],
    tripinfo:    Dict[str, Dict[int, pd.DataFrame]],
) -> Dict[str, Dict[int, Dict[str, float]]]:

    rollups: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in METHODS}
    for method in METHODS:
        for seed, df in all_results.get(method, {}).items():
            if df.empty:
                continue
            sim_t = pd.to_numeric(df['sim_time'], errors='coerce').dropna()
            co2   = calculate_emission_metrics(df)
            wait  = calculate_wait_time_metrics(df)
            trav  = calculate_travel_time_metrics(df)
            spd   = calculate_speed_metrics(df)
            ft_wait = 0.0
            td = tripinfo.get(method, {}).get(seed)
            if td is not None and not td.empty:
                ft_wait = float(td['waitingTime'].mean())
            rollups[method][seed] = {
                'total_vehicles':        float(df['veh_id'].nunique()),
                'total_simulation_time': float(sim_t.max()-sim_t.min())
                                         if not sim_t.empty else 0.0,
                'throughput':            float(calculate_throughput(df)),
                'total_co2':             float(co2['total_co2']),
                'avg_co2_per_vehicle':   float(co2['avg_co2_per_vehicle']),
                'avg_co2_per_second':    float(co2['avg_co2_per_second']),
                'total_wait_time':       float(wait['total_wait_time']),
                'avg_wait_time':         float(wait['avg_wait_time_per_vehicle']),
                'avg_fulltrip_wait':     ft_wait,
                'avg_travel_time':       float(trav['avg_travel_time']),
                'avg_speed':             float(spd['avg_speed']),
            }
    return rollups


def paired_seed_list(rollups):
    s0 = set(rollups[METHODS[0]].keys())
    s1 = set(rollups[METHODS[1]].keys())
    return [s for s in SEEDS if s in s0 and s in s1]


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def percent_change(emission: float, baseline: float) -> float:
    return ((emission - baseline) / baseline) * 100.0 if baseline != 0 else np.nan


def paired_ttest(bv, ev):
    if len(bv) < 2 or len(ev) < 2:
        return np.nan, np.nan
    n = min(len(bv), len(ev))
    t, p = stats.ttest_rel(bv[:n], ev[:n])
    return float(t), float(p)


def pval_sig_label(p: float) -> str:
    if not np.isfinite(p): return 'n/a'
    if p < 0.001:          return '***'
    if p < 0.01:           return '**'
    if p < 0.05:           return '*'
    return 'ns'


# =============================================================================
# PLOT 1 — Dual wait-time
# =============================================================================

def plot_dual_wait_time(rollups, seeds):
    b_ib = [rollups['baseline']     [s]['avg_wait_time']     for s in seeds]
    e_ib = [rollups['emission_based'][s]['avg_wait_time']     for s in seeds]
    b_ft = [rollups['baseline']     [s]['avg_fulltrip_wait'] for s in seeds]
    e_ft = [rollups['emission_based'][s]['avg_fulltrip_wait'] for s in seeds]
    bim, bis = mean_std(b_ib); eim, eis = mean_std(e_ib)
    bfm, bfs = mean_std(b_ft); efm, efs = mean_std(e_ft)
    ib_pct = percent_change(eim, bim); ft_pct = percent_change(efm, bfm)
    _, ibp = paired_ttest(b_ib, e_ib); _, ftp = paired_ttest(b_ft, e_ft)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.array([0.0, 1.0]); w = 0.30
    bars1 = ax.bar(x-w/2, [bim, bfm], w, yerr=[bis, bfs], capsize=7,
                   color='lightblue',  alpha=0.88, label='Baseline')
    bars2 = ax.bar(x+w/2, [eim, efm], w, yerr=[eis, efs], capsize=7,
                   color='lightgreen', alpha=0.88, label='Emission-Based')
    for bar, val in zip(bars1, [bim, bfm]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f'{val:.1f} s', ha='center', va='bottom', fontsize=11)
    for bar, val in zip(bars2, [eim, efm]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f'{val:.1f} s', ha='center', va='bottom', fontsize=11,
                color='#2E7D32')
    ax.set_xticks(x)
    ax.set_xticklabels(['In-box wait time\n(300 m zone only)',
                        'Full-trip wait time\n(from tripinfo XML)'], fontsize=12)
    ax.set_ylabel('Average waiting time per vehicle (s)', fontsize=12)
    ax.set_title(_title('Waiting Time: Spatial Scope Comparison  (per vehicle, per simulation)'),
                 fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(bfm, efm) * 1.25)
    note = (f"In-box:    \u0394 = {ib_pct:+.1f}%  (p={ibp:.4f}, "
            f"{pval_sig_label(ibp)})\n"
            f"Full-trip: \u0394 = {ft_pct:+.1f}%  (p={ftp:.4f}, "
            f"{pval_sig_label(ftp)})")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va='top', ha='left',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.65))
    plt.tight_layout()
    _save(fig, 'dual_wait_time_comparison.png')


# =============================================================================
# PLOT 2 — CO2 savings
# =============================================================================

def plot_co2_savings_relatable(rollups, seeds):
    b_vals = [rollups['baseline']     [s]['avg_co2_per_vehicle'] for s in seeds]
    e_vals = [rollups['emission_based'][s]['avg_co2_per_vehicle'] for s in seeds]
    bm, _  = mean_std(b_vals); em, _ = mean_std(e_vals)
    _, pv  = paired_ttest(b_vals, e_vals)
    # Signed delta: positive = emission-based SAVES vs baseline; negative = emission-based INCREASES vs baseline
    smg = bm - em; sg = smg/1000; skg = sg*VEHICLES_PER_DAY/1000
    st  = skg*DAYS_PER_YEAR/1000; ekm = sg/CO2_PER_KM_CAR_G
    is_saving = smg >= 0
    direction = 'saving' if is_saving else 'increase'
    Direction = 'Saving' if is_saving else 'Increase'
    verb      = 'saved'  if is_saving else 'added'
    # Colour the headline bars green when saving, red when increasing
    agg_color1 = '#4CAF50' if is_saving else '#C62828'
    agg_color2 = '#1565C0' if is_saving else '#AD1457'
    km_color1  = '#FF9800' if is_saving else '#C62828'
    km_color2  = '#F44336' if is_saving else '#AD1457'
    km_color3  = '#9C27B0' if is_saving else '#6A1B9A'

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(_title(f'CO2 Emission {Direction}s in Context  '
                        f'(per simulation run; baseline \u2212 emission-based)'),
                 fontsize=13)
    ax1 = axes[0]
    per_seed = [b-e for b, e in zip(b_vals, e_vals)]
    ax1.bar([str(s) for s in seeds], per_seed,
            color=[_seed_color(s) for s in seeds],
            alpha=0.85, edgecolor='black', linewidth=0.6)
    ax1.axhline(smg, color='black', linestyle='--', linewidth=1.5,
                label=f'Mean = {smg:+,.0f} mg/trip')
    ax1.axhline(0, color='red', linestyle=':', linewidth=1)
    ax1.set_xlabel('Random seed')
    ax1.set_ylabel(f'CO2 change per trip (mg)\npositive = saving, negative = increase')
    ax1.set_title(f'Per-vehicle CO2 change\n(mean = {smg:+,.0f} mg/trip \u2192 {direction})',
                  fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3)
    ax2 = axes[1]
    b2 = ax2.bar(['Daily\n(kg)', 'Annual\n(tonnes)'], [skg, st],
                 color=[agg_color1, agg_color2], alpha=0.85,
                 edgecolor='black', linewidth=0.6, width=0.5)
    for bar, val, u in zip(b2, [skg, st], ['kg', 't']):
        ypos = bar.get_height()
        va = 'bottom' if ypos >= 0 else 'top'
        offset = abs(ypos)*0.02 if ypos != 0 else 0.5
        ax2.text(bar.get_x()+bar.get_width()/2,
                 ypos + (offset if ypos >= 0 else -offset),
                 f'{val:+.1f} {u}', ha='center', va=va, fontsize=12,
                 fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel(f'CO2 {verb} (positive) / added (negative)')
    ax2.set_title(f'Aggregate {direction}\n({VEHICLES_PER_DAY:,} veh/day, '
                  f'{DAYS_PER_YEAR} d/yr)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    ax3 = axes[2]
    ed = ekm*VEHICLES_PER_DAY; ey = ed*DAYS_PER_YEAR
    b3 = ax3.bar(['Per vehicle\n(km)', 'Daily\n(km)', 'Annual\n(km)'],
                 [ekm, ed, ey],
                 color=[km_color1, km_color2, km_color3],
                 alpha=0.85, edgecolor='black', linewidth=0.6, width=0.5)
    for bar, val in zip(b3, [ekm, ed, ey]):
        absval = abs(val)
        lbl = f'{val:+,.0f}' if absval < 1e6 else f'{val/1e6:+.2f}M'
        ypos = bar.get_height()
        va = 'bottom' if ypos >= 0 else 'top'
        offset = abs(ypos)*0.02 if ypos != 0 else 0.5
        ax3.text(bar.get_x()+bar.get_width()/2,
                 ypos + (offset if ypos >= 0 else -offset),
                 f'{lbl} km', ha='center', va=va, fontsize=10,
                 fontweight='bold')
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_ylabel(f'Equivalent car-km (positive = avoided, negative = extra)')
    ax3.set_title(f'Equiv. car-km {direction}\n({CO2_PER_KM_CAR_G} gCO\u2082/km)',
                  fontsize=11, pad=20)
    ax3.grid(axis='y', alpha=0.3)
    footer_color = 'lightgreen' if is_saving else '#FFCDD2'
    fig.text(0.5, 0.01,
             f"CO2 {direction} = {smg:+,.0f} mg/trip = {sg:+.2f} g/trip  "
             f"(\u0394% = {percent_change(em, bm):+.2f}%,  "
             f"p = {pv:.4f} {pval_sig_label(pv)})  |  "
             f"convention: positive = emission-based saves vs baseline",
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor=footer_color, alpha=0.7))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, 'co2_savings_relatable.png')


# =============================================================================
# PLOT 3 — Phase switch table
# =============================================================================

def plot_phase_switch_table(seeds, switches=None):
    if switches is None:
        switches = PHASE_SWITCHES
    missing = [s for s in seeds
               if s not in switches.get('baseline', {})
               or s not in switches.get('emission_based', {})]
    if missing:
        print(f"  [phase_switch_table] WARNING: missing data for seeds {missing}.")
        return
    b_sw = [switches['baseline']     [s] for s in seeds]
    e_sw = [switches['emission_based'][s] for s in seeds]
    diffs = [e-b for b, e in zip(b_sw, e_sw)]
    pcts  = [100.0*d/b if b > 0 else 0.0 for d, b in zip(diffs, b_sw)]
    rows  = [[str(s), str(b), str(e), f'{d:+d}', f'{p:+.1f}%']
             for s, b, e, d, p in zip(seeds, b_sw, e_sw, diffs, pcts)]
    rows.append(['Mean',
                 f'{np.mean(b_sw):.1f}', f'{np.mean(e_sw):.1f}',
                 f'{np.mean(diffs):+.1f}', f'{np.mean(pcts):+.1f}%'])
    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.axis('off')
    tbl = ax.table(
        cellText  = rows,
        colLabels = ['Seed',
                     'Baseline\nswitches per simulation\n(count)',
                     'Emission-Based\nswitches per simulation\n(count)',
                     '\u0394 (count\nper simulation)',
                     '\u0394% (relative\nto baseline)'],
        colLoc='center', cellLoc='center', loc='center',
        colWidths=[0.08, 0.26, 0.28, 0.19, 0.17],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor('#E0E0E0'); cell.set_text_props(weight='bold')
        elif row == len(rows):
            cell.set_facecolor('#FFF9C4'); cell.set_text_props(weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        if col == 4 and row > 0:
            try:
                val = float(rows[row-1][4].replace('%','').replace('+',''))
                cell.set_facecolor('#C8E6C9' if val < 0 else
                                   '#FFCDD2' if val > 0 else 'white')
            except ValueError:
                pass
    ax.set_title(_title('Phase Switch Analysis  (counts are per simulation run)'),
                 fontsize=13, pad=16)
    fig.text(0.5, 0.02,
             'Unit: phase switches per simulation run (one NEMA-cycle transition = one switch). '
             '\u0394 = Emission \u2212 Baseline; negative \u0394 means the emission-aware '
             'controller switches fewer times.',
             ha='center', fontsize=9, style='italic', color='#424242')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, 'phase_switch_table.png')


# =============================================================================
# PLOT 4 — V/C utilization
# =============================================================================

def plot_vc_utilization(discharge_csv_path=None, capacity_json_path=None):
    if discharge_csv_path is None:
        discharge_csv_path = os.path.join(
            'data', 'capacity_analysis', 'summary', 'discharge_by_volume.csv'
        )
    if not os.path.exists(discharge_csv_path):
        print(f"  [skip] vc_utilization: {discharge_csv_path} not found.")
        return
    cap = load_capacity_from_json(capacity_json_path)
    df  = pd.read_csv(discharge_csv_path)
    col = 'mean_avg_arrived_per_cycle' if 'mean_avg_arrived_per_cycle' in df.columns else 'avg_arrived_per_cycle'
    df['vc_pct'] = df[col] / cap * 100
    # df['vc_pct'] = df['avg_arrived_per_cycle'] / cap * 100
    vols = df['volume'].tolist(); vcs = df['vc_pct'].tolist()

    def bar_color(vc):
        if vc < 35:   return '#64B5F6'
        if vc < 55:   return '#81C784'
        if vc < 75:   return '#FFD54F'
        if vc < 90:   return '#FF8A65'
        if vc <= 100: return '#EF5350'
        return '#B71C1C'

    # Determine current volume from VOL_LABEL (e.g. 'V2900' \u2192 2900)
    _cur_vol = None
    if VOL_LABEL and re.match(r'^V\d+$', VOL_LABEL):
        try:
            _cur_vol = int(VOL_LABEL[1:])
        except ValueError:
            pass

    def _los_label(vc):
        if vc < 35:   return 'LOS A'
        if vc < 55:   return 'LOS B'
        if vc < 75:   return 'LOS C'
        if vc < 90:   return 'LOS D'
        if vc <= 100: return 'LOS E'
        return 'LOS F'

    # title suffix when a volume is highlighted
    _title_suffix = f'  \u2014  V{_cur_vol} highlighted' if _cur_vol is not None else ''

    fig, ax = plt.subplots(figsize=(32, 10))
    bars = ax.bar([str(v) for v in vols], vcs,
                  color=[bar_color(v) for v in vcs],
                  edgecolor='black', linewidth=0.5, alpha=0.9)

    # Bold border on the current-volume bar
    _highlight_idx = None
    if _cur_vol is not None and _cur_vol in vols:
        _highlight_idx = vols.index(_cur_vol)
        bars[_highlight_idx].set_edgecolor('#1a1a1a')
        bars[_highlight_idx].set_linewidth(3.0)

    for thr, lbl, ls in [(35,'LOS A/B 35%','--'),(55,'LOS B/C 55%','-.'),
                          (75,'LOS C/D 75%','--'),(90,'LOS D/E 90%','-.'),
                          (100,'Capacity 100%','-')]:
        ax.axhline(thr, linestyle=ls, linewidth=1.1,
                   color='grey' if thr < 100 else 'red', alpha=0.7)
        ax.text(len(vols)-0.5, thr+0.8, lbl, ha='right', va='bottom',
                fontsize=9, color='grey' if thr < 100 else 'red')
    _vc_stride = max(1, len(bars) // 25)  # at most ~25 value labels
    for i, (bar, vc) in enumerate(zip(bars, vcs)):
        if i % _vc_stride != 0:
            continue
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{vc:.1f}%', ha='center', va='bottom', fontsize=8,
                rotation=90)

    # Annotation arrow for the highlighted volume
    if _highlight_idx is not None:
        _hbar = bars[_highlight_idx]
        _hvc  = vcs[_highlight_idx]
        _hx   = _hbar.get_x() + _hbar.get_width() / 2
        _hy   = _hbar.get_height()
        _anno_y = _hy + max(vcs) * 0.18
        ax.annotate(
            f'V{_cur_vol}\n{_los_label(_hvc)}\nV/C = {_hvc:.0f}%',
            xy=(_hx, _hy),
            xytext=(_hx, _anno_y),
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#1a1a1a',
            arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.8),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#1a1a1a', alpha=0.85),
        )

    ax.set_xlabel('Total demand (vehicles injected into simulation)', fontsize=12)
    ax.set_ylabel('Per-cycle V/C ratio (%)', fontsize=12)
    ax.set_title(
        f'Volume-to-Capacity Utilization{_title_suffix}\n'
        f'Intersection capacity = {cap} veh/cycle  '
        f'(V/C = avg arrivals/cycle \u00f7 {cap} \u00d7 100)', fontsize=13)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#64B5F6', label='LOS A (<35%)'),
        Patch(facecolor='#81C784', label='LOS B (35-55%)'),
        Patch(facecolor='#FFD54F', label='LOS C (55-75%)'),
        Patch(facecolor='#FF8A65', label='LOS D (75-90%)'),
        Patch(facecolor='#EF5350', label='LOS E (90-100%)'),
        Patch(facecolor='#B71C1C', label='LOS F (>100%)'),
    ], loc='upper left', fontsize=10, framealpha=0.9)
    ax.tick_params(axis='x', rotation=60, labelsize=7)
    ax.set_ylim(0, max(vcs)*1.30); ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    _save(fig, 'vc_utilization.png')


# =============================================================================
# PLOT 5 — Cross-volume comparison  [NEW]
# =============================================================================

def run_cross_volume_comparison(root_data_dir: str, root_out_dir: str) -> None:
    """
    Scan root_data_dir for V#### sub-folders, load each volume's results,
    compute mean metrics, and produce a 6-panel figure showing how each
    metric changes with demand level for both methods.

    Also saves a CSV summary table for the paper.

    Output: <root_out_dir>/cross_volume/cross_volume_comparison.png
             <root_out_dir>/cross_volume/cross_volume_summary.csv
    """
    global DATA_DIR

    volume_dirs = sorted([
        d for d in os.listdir(root_data_dir)
        if re.match(r'^V\d{4}$', d) and
        os.path.isdir(os.path.join(root_data_dir, d))
    ])

    if not volume_dirs:
        print(f"  [cross_volume] No V#### sub-folders found in {root_data_dir}")
        return

    print(f"\n[cross_volume] Found {len(volume_dirs)} volumes: {volume_dirs}")

    records = []
    old_data_dir = DATA_DIR

    for vol_dir in volume_dirs:
        vol_path = os.path.join(root_data_dir, vol_dir)
        vol_int  = int(vol_dir[1:])
        DATA_DIR = vol_path   # redirect loaders

        try:
            all_results, _ = load_and_process_data()
            tripinfo_data  = load_tripinfo_data()
            rollups        = compute_seed_rollups(all_results, tripinfo_data)
            seeds          = paired_seed_list(rollups)
        except Exception as e:
            print(f"  [cross_volume] Error loading {vol_dir}: {e}")
            DATA_DIR = old_data_dir
            continue
        finally:
            DATA_DIR = old_data_dir

        if not seeds:
            print(f"  [cross_volume] No paired seeds for {vol_dir} — skipping.")
            continue

        for method in METHODS:
            for m in METRICS:
                vals = [rollups[method][s].get(m.key, 0.0) for s in seeds]
                mu, sd = mean_std(vals)
                records.append({'volume': vol_int, 'method': method,
                                 'metric': m.key, 'mean': mu, 'std': sd})

    if not records:
        print("  [cross_volume] No data collected.")
        return

    summary = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 3, figsize=(28, 14))
    axes_flat = axes.flatten()

    styles = {
        'baseline':       dict(color='steelblue',   marker='o', ls='-',
                               label='Baseline'),
        'emission_based': dict(color='forestgreen', marker='s', ls='--',
                               label='Emission-Based'),
    }

    for ax, metric in zip(axes_flat, METRICS):
        for method, style in styles.items():
            sub = summary[(summary['method'] == method) &
                          (summary['metric'] == metric.key)].sort_values('volume')
            if sub.empty:
                continue
            ax.plot(sub['volume'], sub['mean'],
                    color=style['color'], marker=style['marker'],
                    linestyle=style['ls'], linewidth=2, markersize=7,
                    label=style['label'])
            ax.fill_between(sub['volume'],
                            sub['mean'] - sub['std'],
                            sub['mean'] + sub['std'],
                            alpha=0.15, color=style['color'])
        ax.set_title(f"{metric.title} ({metric.unit})", fontsize=12)
        ax.set_xlabel('Demand volume (veh/hr)', fontsize=10)
        ax.set_ylabel(metric.unit, fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        arrow = '\u2191' if metric.higher_is_better else '\u2193'
        ax.text(0.02, 0.97, f'{arrow} better', transform=ax.transAxes,
                va='top', ha='left', fontsize=9, color='grey')

    for ax in axes_flat[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle(
        'Cross-Volume Performance: Baseline vs Emission-Based\n'
        'How does the emission benefit change with demand level?',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    out_dir = os.path.join(root_out_dir, 'cross_volume')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'cross_volume_comparison.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [cross_volume] Saved -> {out_png}")

    csv_path = os.path.join(out_dir, 'cross_volume_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"  [cross_volume] Summary CSV -> {csv_path}")


# =============================================================================
# Standard plot functions
# =============================================================================

def _add_seed_values_panel(fig, rect, metric, seeds, base_vals, emis_vals):
    dec = METRIC_DECIMALS.get(metric.key, 2)
    bmu, bsd = mean_std(base_vals); emu, esd = mean_std(emis_vals)
    axp = fig.add_axes(rect)
    axp.set_xlim(0, 1); axp.set_ylim(0, 1); axp.axis('off')
    axp.add_patch(mpatches.FancyBboxPatch(
        (0,0), 1, 1, boxstyle='round,pad=0.02',
        facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.92))

    # Dynamic line height and font so the panel fits any number of seeds
    # without overflow or clipping. All artists use clip_on=True to prevent
    # text or dots from bleeding outside the panel into the footer area.
    n   = len(seeds)
    # slots: 5 preamble rows + 2 sections × (1 header + n seeds + 1 mean)
    lh  = max(0.024, min(0.042, 0.92 / max(1, 5 + 2 * (n + 2))))
    fs  = max(7.5, min(10.0, 9.5 - max(0, n - 5) * 0.25))
    fsh = min(fs + 0.5, 10.5)

    y = 0.985
    axp.text(0.5, y, f"{metric.title} ({metric.unit})",
             ha='center', va='top', fontsize=fsh, fontweight='bold',
             clip_on=True)
    y -= lh * 1.3
    axp.text(0.5, y, "Method", ha='center', va='top',
             fontsize=fs, fontweight='bold', clip_on=True)
    y -= lh
    axp.add_patch(mpatches.Rectangle((0.08, y - lh * 0.38), 0.05, lh * 0.76,
                  facecolor='lightblue', edgecolor='black', clip_on=True))
    axp.text(0.15, y, "Baseline", ha='left', va='center',
             fontsize=fs, clip_on=True)
    y -= lh
    axp.add_patch(mpatches.Rectangle((0.08, y - lh * 0.38), 0.05, lh * 0.76,
                  facecolor='lightgreen', edgecolor='black', clip_on=True))
    axp.text(0.15, y, "Emission-Based", ha='left', va='center',
             fontsize=fs, clip_on=True)
    y -= lh
    axp.scatter([0.105], [y], s=45, marker='D', color='black', clip_on=True)
    axp.text(0.15, y, "Mean", ha='left', va='center',
             fontsize=fs, clip_on=True)
    y -= lh * 1.5

    def section(header, vals, mu, sd):
        nonlocal y
        axp.text(0.5, y, header, ha='center', va='top',
                 fontsize=fsh, fontweight='bold', clip_on=True)
        y -= lh * 1.1
        for s, v in zip(seeds, vals):
            c = _seed_color(s)
            axp.scatter([0.08], [y], s=38, color=c, edgecolors='black',
                        linewidth=0.5, alpha=0.95, zorder=5, clip_on=True)
            axp.text(0.14, y, f"Seed {s}: {v:,.{dec}f}",
                     ha='left', va='center', fontsize=fs, color=c,
                     clip_on=True)
            y -= lh
        axp.scatter([0.08], [y], s=55, marker='D', color='black', zorder=6,
                    clip_on=True)
        axp.text(0.14, y, f"Mean: {mu:,.{dec}f}  (+/- {sd:,.{dec}f})",
                 ha='left', va='center', fontsize=fs, clip_on=True)
        y -= lh * 1.1

    section("Baseline",       base_vals, bmu, bsd)
    section("Emission-Based", emis_vals, emu, esd)


def _wrap_text(s, width=150):
    return textwrap.fill(" ".join(str(s).split()), width=width)


def _interpretation_text(metric, seeds, base_vals, emis_vals):
    bm, _ = mean_std(base_vals); em, _ = mean_std(emis_vals)
    d_pct = percent_change(em, bm)
    improved = sum(1 for b, e in zip(base_vals, emis_vals)
                   if (e > b if metric.higher_is_better else e < b))
    direction = "higher is better" if metric.higher_is_better else "lower is better"
    if np.isfinite(d_pct):
        better = ("Emission-Based" if (d_pct > 0) == metric.higher_is_better
                  else "Baseline")
        return (f"Interpretation: {direction}. Mean = {bm:.3g} (Baseline) vs "
                f"{em:.3g} (Emission-Based). \u0394% = {d_pct:+.2f}%. "
                f"Emission-Based better in {improved}/{len(seeds)} seeds. "
                f"Overall: {better} performs better.")
    return (f"Interpretation: {direction}. Mean = {bm:.3g} vs {em:.3g}. "
            f"Emission-Based better in {improved}/{len(seeds)} seeds.")


def _stats_footer(fig, metric, base_vals, emis_vals, y=0.12):
    d_pct = percent_change(mean_std(emis_vals)[0], mean_std(base_vals)[0])
    _, p_val = paired_ttest(base_vals, emis_vals)
    lines = []
    if np.isfinite(d_pct):
        badge = "IMPROVEMENT" if (d_pct>0)==metric.higher_is_better else "WORSE"
        lines.append(f"\u0394% = (E vs B): {d_pct:+.2f}%  {badge}")
    if np.isfinite(p_val):
        lines.append(f"p-value: {p_val:.4f} ({pval_sig_label(p_val)})")
    if lines:
        fig.text(0.5, y, " | ".join(lines), ha='center', va='center',
                 fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='wheat',
                            alpha=0.55))


def plot_mean_std_bar(metric, seeds, base_vals, emis_vals):
    bm, bs = mean_std(base_vals); em, es = mean_std(emis_vals)
    dec = METRIC_DECIMALS.get(metric.key, 2)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(np.arange(2), [bm, em], yerr=[bs, es], capsize=8,
                  alpha=0.85, color=['lightblue', 'lightgreen'])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['Baseline', 'Emission-Based'])
    ax.set_title(_title(f"{metric.title} (Mean +/- Std)"))
    ax.set_ylabel(f"{metric.title} ({metric.unit})")
    ax.grid(True, alpha=0.3)
    for bar, m_val in zip(bars, [bm, em]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"{m_val:,.{dec}f} {metric.unit}", ha='center', va='bottom',
                fontsize=11)
    fig.subplots_adjust(bottom=0.32)
    _stats_footer(fig, metric, base_vals, emis_vals, y=0.025)
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
    _save(fig, f"single_meanstd_{metric.key}.png")


def plot_boxplot_mean(metric, seeds, base_vals, emis_vals):
    fig, ax = plt.subplots(figsize=(13.2, 8.6))
    bp = ax.boxplot([base_vals, emis_vals],
                    labels=['Baseline', 'Emission-Based'],
                    patch_artist=True, widths=0.55, showfliers=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax.set_title(_title(f"{metric.title} (Distribution)"))
    ax.set_ylabel(f"{metric.title} ({metric.unit})")
    ax.grid(True, alpha=0.3)
    offs = _fixed_x_offsets(len(seeds), span=0.16)
    for i, vals in enumerate([base_vals, emis_vals], start=1):
        for seed, v, off in zip(seeds, vals, offs):
            ax.scatter(i+off, v, s=90, color=_seed_color(seed),
                       edgecolors='black', linewidth=0.6, alpha=0.80, zorder=3)
    bm, _ = mean_std(base_vals); em, _ = mean_std(emis_vals)
    ax.scatter([1, 2], [bm, em], marker='D', s=140, color='black', zorder=4)
    fig.subplots_adjust(right=0.74, bottom=0.30)
    _add_seed_values_panel(fig, [0.76,0.10,0.23,0.88], metric, seeds,
                           base_vals, emis_vals)
    _stats_footer(fig, metric, base_vals, emis_vals, y=0.12)
    fig.text(0.5, 0.012,
             _wrap_text(_interpretation_text(metric, seeds, base_vals,
                                              emis_vals), 150),
             ha='center', va='bottom', fontsize=9, wrap=True)
    fig.tight_layout(rect=[0.03, 0.16, 0.74, 0.98])
    _save(fig, f"single_boxplot_{metric.key}.png")


def plot_paired_dots(metric, seeds, base_vals, emis_vals):
    fig, ax = plt.subplots(figsize=(15, 10.5))
    offs = _fixed_x_offsets(len(seeds), span=0.16)
    for seed, b, e, off in zip(seeds, base_vals, emis_vals, offs):
        c = _seed_color(seed)
        ax.plot([0.0+off, 1.0+off], [b, e], color=c, alpha=0.45,
                linewidth=2, zorder=2)
        ax.scatter([0.0+off], [b], s=110, color=c, edgecolors='black',
                   linewidth=0.7, alpha=0.85, zorder=3)
        ax.scatter([1.0+off], [e], s=110, color=c, edgecolors='black',
                   linewidth=0.7, alpha=0.85, zorder=3)
    bm, _ = mean_std(base_vals); em, _ = mean_std(emis_vals)
    ax.scatter([0.0, 1.0], [bm, em], marker='D', s=150, color='black', zorder=4)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(['Baseline', 'Emission-Based'])
    ax.set_title(_title(f"{metric.title} (Paired by Seed)"))
    ax.set_ylabel(f"{metric.title} ({metric.unit})")
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(right=0.74, bottom=0.30)
    _add_seed_values_panel(fig, [0.76,0.10,0.23,0.88], metric, seeds,
                           base_vals, emis_vals)
    _stats_footer(fig, metric, base_vals, emis_vals, y=0.12)
    fig.text(0.5, 0.012,
             _wrap_text(_interpretation_text(metric, seeds, base_vals,
                                              emis_vals), 150),
             ha='center', va='bottom', fontsize=9, wrap=True)
    fig.tight_layout(rect=[0.03, 0.16, 0.74, 0.98])
    _save(fig, f"single_paired_{metric.key}.png")


def create_single_comparison_figures(rollups, seeds):
    for m in METRICS:
        b = [rollups['baseline']     [s].get(m.key, 0.0) for s in seeds]
        e = [rollups['emission_based'][s].get(m.key, 0.0) for s in seeds]
        plot_mean_std_bar(m, seeds, b, e)
        plot_boxplot_mean(m, seeds, b, e)
        plot_paired_dots (m, seeds, b, e)


def create_seed_comparison_boxplots(rollups, seeds):
    for m in METRICS:
        bv = [rollups['baseline']     [s].get(m.key, 0.0) for s in seeds]
        ev = [rollups['emission_based'][s].get(m.key, 0.0) for s in seeds]
        fig, ax = plt.subplots(figsize=(15, 10.0))
        bp = ax.boxplot([bv, ev], labels=['Baseline','Emission-Based'],
                        patch_artist=True, widths=0.55, showfliers=True)
        for patch, color in zip(bp['boxes'], ['lightblue','lightgreen']):
            patch.set_facecolor(color)
        offs = _fixed_x_offsets(len(seeds), span=0.16)
        for i, vals in enumerate([bv, ev], start=1):
            for s, v, off in zip(seeds, vals, offs):
                ax.scatter(i+off, v, s=95, color=_seed_color(s),
                           edgecolors='black', linewidth=0.6, alpha=0.80,
                           zorder=3)
        bm, _ = mean_std(bv); em, _ = mean_std(ev)
        ax.scatter([1,2], [bm,em], marker='D', s=140, color='black', zorder=4)
        _, p_val = paired_ttest(bv, ev)
        ax.set_title(_title(f"{m.title} Comparison Across Seeds"))
        ax.set_ylabel(f"{m.title} ({m.unit})")
        ax.grid(True, alpha=0.3)
        fig.subplots_adjust(right=0.74, bottom=0.22)
        _add_seed_values_panel(fig, [0.76,0.10,0.23,0.88], m, seeds, bv, ev)
        if np.isfinite(p_val):
            fig.text(0.5, 0.10,
                     f"p-value (paired t-test): {p_val:.4f} "
                     f"({pval_sig_label(p_val)})",
                     ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='wheat',
                                alpha=0.55))
        fig.tight_layout(rect=[0.04, 0.18, 0.74, 0.98])
        _save(fig, f"seed_comparison_{m.key}.png")


def create_comprehensive_seed_comparison(rollups, seeds):
    if len(seeds) < 2:
        return
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]}
    )
    positions = np.arange(len(METRICS)); width = 0.35
    bms = [mean_std([rollups['baseline']     [s].get(m.key,0.) for s in seeds])[0]
           for m in METRICS]
    ems = [mean_std([rollups['emission_based'][s].get(m.key,0.) for s in seeds])[0]
           for m in METRICS]
    bss = [mean_std([rollups['baseline']     [s].get(m.key,0.) for s in seeds])[1]
           for m in METRICS]
    ess = [mean_std([rollups['emission_based'][s].get(m.key,0.) for s in seeds])[1]
           for m in METRICS]
    ax_top.bar(positions-width/2, bms, width, yerr=bss, capsize=5,
               label='Baseline',       color='lightblue', alpha=0.85)
    ax_top.bar(positions+width/2, ems, width, yerr=ess, capsize=5,
               label='Emission-Based', color='lightgreen', alpha=0.85)
    for i, (bm, em) in enumerate(zip(bms, ems)):
        dec = METRIC_DECIMALS.get(METRICS[i].key, 2)
        ax_top.text(i-width/2, bm, f'{bm:,.{dec}f}', ha='center',
                    va='bottom', fontsize=9)
        ax_top.text(i+width/2, em, f'{em:,.{dec}f}', ha='center',
                    va='bottom', fontsize=9)
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels([f"{m.title}\n({m.unit})" for m in METRICS])
    ax_top.set_ylabel('Metric Value')
    ax_top.set_title(_title('Performance Metrics Comparison (Mean +/- Std)'))
    ax_top.legend(loc='upper left'); ax_top.grid(True, alpha=0.3, axis='y')
    pct_changes = []
    for m in METRICS:
        deltas = [percent_change(
            rollups['emission_based'][s].get(m.key,0.),
            rollups['baseline']     [s].get(m.key,0.)
        ) for s in seeds]
        pct_changes.append(deltas)
    bp = ax_bot.boxplot(pct_changes, positions=positions, widths=0.6,
                        patch_artist=True)
    offs = _fixed_x_offsets(len(seeds), span=0.14)
    for i, (box, metric) in enumerate(zip(bp['boxes'], METRICS)):
        med = float(np.nanmedian(pct_changes[i]))
        box.set_facecolor(
            'lightgreen' if (med>0)==metric.higher_is_better else 'lightcoral'
        )
        for j, (seed, off) in enumerate(zip(seeds, offs)):
            if not np.isnan(pct_changes[i][j]):
                ax_bot.scatter(positions[i]+off, pct_changes[i][j], s=55,
                               color=_seed_color(seed), edgecolors='black',
                               linewidth=0.4, alpha=0.82, zorder=3)
        if not np.isnan(med):
            ax_bot.text(positions[i], med, f'{med:+.1f}%',
                        ha='center',
                        va='bottom' if med >= 0 else 'top',
                        fontsize=11, fontweight='bold')
    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels([m.title for m in METRICS])
    ax_bot.set_ylabel('Percent Change \u0394% (E vs B)')
    ax_bot.set_title('Percentage Change (\u0394% = (E \u2212 B) / B \u00d7 100)')
    ax_bot.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax_bot.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    _save(fig, 'comprehensive_seed_comparison.png')


def create_radar_chart_from_seeds(rollups, seeds):
    if len(seeds) < 2:
        return
    norm_base = []; norm_emis = []
    for m in METRICS:
        bv = mean_std([rollups['baseline']     [s].get(m.key,0.) for s in seeds])[0]
        ev = mean_std([rollups['emission_based'][s].get(m.key,0.) for s in seeds])[0]
        bs = bv if m.higher_is_better else -bv
        es = ev if m.higher_is_better else -ev
        vmin = min(bs,es); vmax = max(bs,es)
        norm_base.append((bs-vmin)/(vmax-vmin) if vmax>vmin else 0.5)
        norm_emis.append((es-vmin)/(vmax-vmin) if vmax>vmin else 0.5)
    cats   = [m.title for m in METRICS]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    norm_base += norm_base[:1]; norm_emis += norm_emis[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, norm_base, 'o-', linewidth=2, label='Baseline')
    ax.fill(angles, norm_base, alpha=0.15)
    ax.plot(angles, norm_emis, 'o-', linewidth=2, label='Emission-Based')
    ax.fill(angles, norm_emis, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats)
    ax.set_title(_title('Performance Comparison (Normalized)'), pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.05))
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, 'performance_radar_chart.png')


# =============================================================================
# Terminal summaries
# =============================================================================

def generate_statistical_summary(rollups, seeds):
    print('\n' + '='*80)
    print('STATISTICAL SUMMARY' + (f'  [{VOL_LABEL}]' if VOL_LABEL else ''))
    print('='*80)
    print(f"Seeds: {seeds}")
    for m in METRICS:
        print(f"\n--- {m.title.upper()} ({m.unit}) ---")
        bv = [rollups['baseline']     [s].get(m.key, 0.0) for s in seeds]
        ev = [rollups['emission_based'][s].get(m.key, 0.0) for s in seeds]
        bmu, bsd = mean_std(bv); emu, esd = mean_std(ev)
        print(f"baseline       | Mean: {bmu:10.3f} | Std: {bsd:8.3f} | "
              f"Min: {min(bv):8.3f} | Max: {max(bv):8.3f}")
        print(f"emission_based | Mean: {emu:10.3f} | Std: {esd:8.3f} | "
              f"Min: {min(ev):8.3f} | Max: {max(ev):8.3f}")
        _, p_val = paired_ttest(bv, ev)
        print(f"p-value: "
              f"{'n/a' if not np.isfinite(p_val) else f'{p_val:.4f} ({pval_sig_label(p_val)})'}")


def print_comprehensive_performance_summary(rollups, seeds):
    def fmt(vals, d=1):
        mu, sd = mean_std(vals)
        return f"{mu:,.{d}f} (+/- {sd:,.{d}f})"
    print('\n' + '='*85)
    print('PERFORMANCE SUMMARY' + (f'  [{VOL_LABEL}]' if VOL_LABEL else ''))
    print('='*85)
    for lbl, key in [('BASELINE','baseline'),('EMISSION_BASED','emission_based')]:
        print(f"\n--- {lbl} ---")
        print(f"Throughput:          {fmt([rollups[key][s]['throughput']          for s in seeds])} veh/h")
        print(f"In-box wait time:    {fmt([rollups[key][s]['avg_wait_time']       for s in seeds])} s")
        print(f"Full-trip wait time: {fmt([rollups[key][s]['avg_fulltrip_wait']   for s in seeds])} s")
        print(f"Avg travel time:     {fmt([rollups[key][s]['avg_travel_time']     for s in seeds])} s")
        print(f"Average speed:       {fmt([rollups[key][s]['avg_speed']           for s in seeds], d=2)} m/s")
        print(f"CO2 per vehicle:     {fmt([rollups[key][s]['avg_co2_per_vehicle'] for s in seeds], d=0)} mg")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global DATA_DIR, OUTPUT_PLOTS_DIR, VOL_LABEL

    DATA_DIR         = _ARGS.data_dir
    OUTPUT_PLOTS_DIR = _ARGS.out_dir
    VOL_LABEL        = _derive_vol_label(_ARGS.data_dir, _ARGS.vol_label)

    # Cross-volume mode
    if _ARGS.cross_volume:
        print(f"\nCross-volume mode  |  scanning: {DATA_DIR}")
        run_cross_volume_comparison(DATA_DIR, OUTPUT_PLOTS_DIR)
        return

    # Per-volume mode
    label_str = f" [{VOL_LABEL}]" if VOL_LABEL else ""
    print(f"\nAnalysing{label_str}  |  data: {DATA_DIR}")
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    all_results, raw_data = load_and_process_data()
    tripinfo               = load_tripinfo_data()

    any_data = any(d for d in all_results.values())
    if not any_data:
        print("  [WARNING] No result CSVs found in DATA_DIR. Exiting.")
        return

    # ── Phase switches ────────────────────────────────────────────────────────
    global PHASE_SWITCHES
    PHASE_SWITCHES = load_phase_switches_from_logs()

    # ── Rollups ───────────────────────────────────────────────────────────────
    rollups = compute_seed_rollups(all_results, tripinfo)
    seeds   = paired_seed_list(rollups)

    if not seeds:
        print("  [WARNING] No paired seeds found (baseline + emission_based). "
              "Check that both method folders have matching seed CSVs.")
        return

    print(f"  Paired seeds ({len(seeds)}): {seeds}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_vc_utilization()
    plot_dual_wait_time(rollups, seeds)
    plot_co2_savings_relatable(rollups, seeds)
    plot_phase_switch_table(seeds)
    create_single_comparison_figures(rollups, seeds)
    create_seed_comparison_boxplots(rollups, seeds)
    create_comprehensive_seed_comparison(rollups, seeds)
    create_radar_chart_from_seeds(rollups, seeds)

    # ── Terminal summaries ────────────────────────────────────────────────────
    generate_statistical_summary(rollups, seeds)
    print_comprehensive_performance_summary(rollups, seeds)

    print(f"\n  Done.  Plots saved to: {OUTPUT_PLOTS_DIR}")


if __name__ == '__main__':
    main()

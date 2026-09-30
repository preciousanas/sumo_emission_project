#!/usr/bin/env python3
"""
publication_figures_ext.py — extends publication_figures.py to cover the
remaining 18 plots from the "file mapping for paper 1" spec (plots 4-13,
16-23). Same publication settings apply (imported from publication_figures).

Coverage:
  Plots 4-6   threshold_search      (from threshold_summary.csv + optimal)
  Plots 7-13  per-volume boxplots / paired / dual-wait  (streamed from result_seed_*.csv)
  Plots 16-17 penetration_rate      (from penetration_summary.csv)
  Plots 18-19 composition           (from composition_summary.csv)
  Plots 20-23 threshold_x_scenarios (from pen_x_/comp_x_/optimal_threshold_matrix.csv)

All plots follow ~15 cm width, 600 DPI PNG + vector PDF, embedded fonts.
Every value is loaded from the intermediate summary CSVs — no data are
recomputed here except per-seed values for the per-volume plots, which
are streamed with the same chunked logic used by regenerate_cross_volume.py.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import the rcParams + save helper from the main module
from publication_figures import (
    FIG_W, DPI_PNG, save_both, ROOT,
    los_color, LOS_BANDS,
)


EXP = ROOT / "experiments"
DATA = ROOT / "data"

TS_DIR  = EXP / "threshold_search"
PN_DIR  = EXP / "penetration_rate"
CM_DIR  = EXP / "composition"
TX_DIR  = EXP / "threshold_x_scenarios"

TS_OUT  = TS_DIR / "visuals"
PN_OUT  = PN_DIR / "visuals"
CM_OUT  = CM_DIR / "visuals"
TX_OUT  = TX_DIR / "visuals"
XVR_OUT = ROOT / "visuals" / "cross_volume"

for d in (TS_OUT, PN_OUT, CM_OUT, TX_OUT):
    d.mkdir(parents=True, exist_ok=True)

# Consistent regime color scheme (preserved everywhere)
REGIME_COLORS = {
    "Free-flow":       "#1E88E5",
    "Near-saturation": "#43A047",
    "Breakdown onset": "#FB8C00",
    "Gridlock":        "#E53935",
}

METHOD_COLORS = {
    "baseline":       "#1565C0",
    "emission_based": "#2E7D32",
}

# ----- per-seed streaming helper (shared with regenerate_cross_volume.py) -----
CHUNK = 200_000

def _coerce_inbox(series):
    return (series.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False,
                  "yes": True, "no": False})
            .fillna(False))


_SEED_CACHE: dict = {}

def per_seed_metric(volume: int, method: str) -> dict:
    key = (volume, method)
    if key in _SEED_CACHE:
        return _SEED_CACHE[key]
    _SEED_CACHE[key] = _per_seed_metric_impl(volume, method)
    return _SEED_CACHE[key]


def _per_seed_metric_impl(volume: int, method: str) -> dict:
    """Return {seed: {'co2': float, 'wait': float, 'travel': float,
                       'speed': float, 'fulltrip_wait': float}} for one volume+method.
    Streams the result_seed_*.csv in 200k-row chunks and reads tripinfo XML
    for full-trip waits."""
    mdir = DATA / f"V{volume:04d}" / method
    seed_map = {}
    for csv_path in sorted(mdir.glob("result_seed_*.csv")):
        seed = int(csv_path.stem.split("_")[-1])
        veh_ids = set()
        veh_tmin, veh_tmax, veh_wmax = {}, {}, {}
        co2_sum = 0.0
        speed_sum, speed_n = 0.0, 0
        try:
            reader = pd.read_csv(csv_path,
                                 usecols=["sim_time", "veh_id", "speed",
                                          "wait_time", "co2_emission", "in_box"],
                                 chunksize=CHUNK, dtype={"veh_id": str},
                                 low_memory=False)
        except Exception:
            continue
        for chunk in reader:
            chunk = chunk[~chunk["sim_time"].astype(str).str.startswith("#", na=False)]
            if chunk.empty: continue
            chunk["sim_time"]     = pd.to_numeric(chunk["sim_time"],     errors="coerce")
            chunk["speed"]        = pd.to_numeric(chunk["speed"],        errors="coerce")
            chunk["wait_time"]    = pd.to_numeric(chunk["wait_time"],    errors="coerce")
            chunk["co2_emission"] = pd.to_numeric(chunk["co2_emission"], errors="coerce")
            chunk["in_box"]       = _coerce_inbox(chunk["in_box"])
            chunk = chunk.dropna(subset=["sim_time", "veh_id"])
            ib = chunk[chunk["in_box"] == True]
            if ib.empty: continue
            veh_ids.update(ib["veh_id"].unique().tolist())
            sp = ib["speed"].dropna()
            if not sp.empty:
                speed_sum += float(sp.sum()); speed_n += int(sp.size)
            co = ib["co2_emission"].dropna()
            if not co.empty: co2_sum += float(co.sum())
            grp = ib.groupby("veh_id")
            for vid, t in grp["sim_time"].min().to_dict().items():
                if t < veh_tmin.get(vid, np.inf): veh_tmin[vid] = t
            for vid, t in grp["sim_time"].max().to_dict().items():
                if t > veh_tmax.get(vid, -np.inf): veh_tmax[vid] = t
            for vid, w in grp["wait_time"].max().to_dict().items():
                try: w = float(w)
                except Exception: continue
                if np.isnan(w): continue
                if w > veh_wmax.get(vid, -np.inf): veh_wmax[vid] = w
        co2_veh = (co2_sum / len(veh_ids)) if veh_ids else 0.0
        travel_times = [veh_tmax[v] - veh_tmin[v]
                        for v in veh_tmin if v in veh_tmax]
        travel_avg = float(np.mean(travel_times)) if travel_times else 0.0
        wait_avg = float(np.mean(list(veh_wmax.values()))) if veh_wmax else 0.0
        speed_avg = (speed_sum / speed_n) if speed_n > 0 else 0.0
        # Tripinfo (full trip wait)
        tp = mdir / f"tripinfo_seed_{seed}.xml"
        ft_wait = _tripinfo_waiting(tp)
        seed_map[seed] = dict(co2=co2_veh, travel=travel_avg,
                              wait=wait_avg, speed=speed_avg,
                              fulltrip_wait=ft_wait)
    return seed_map


def _tripinfo_waiting(path: Path) -> float:
    if not path.exists(): return 0.0
    try: tree = ET.parse(path)
    except ET.ParseError: return 0.0
    waits = []
    for tr in tree.getroot().findall("tripinfo"):
        try: waits.append(float(tr.attrib.get("waitingTime", 0.0)))
        except ValueError: continue
    return float(np.mean(waits)) if waits else 0.0


# =========================================================================
# PER-VOLUME PLOTS 7-13
# =========================================================================
def _paired_ttest_p(bv, ev):
    """Two-sided p-value for a paired t-test.
    Uses scipy.stats.ttest_rel if scipy is installed (exact); falls back
    to a normal approximation via math.erf if not."""
    diffs = [float(e) - float(b) for e, b in zip(ev, bv)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    try:
        from scipy import stats
        return float(stats.ttest_rel(bv, ev).pvalue)
    except ImportError:
        import math
        mean_d = sum(diffs) / n
        var_d  = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        if var_d <= 0:
            return float("nan")
        se = math.sqrt(var_d / n)
        t  = mean_d / se
        # Normal approximation (df -> infinity) — good enough for a
        # ns / * / ** / *** label; annotate exact p in real env.
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))


def _sig_label(p):
    if not np.isfinite(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _boxplot_co2(volume: int, out: Path):
    """Boxplot of CO2/vehicle: baseline vs emission (per-seed dots).
    Restores the old plot's scientific content:
      * per-seed color coding (paired identity)
      * paired t-test p-value + significance label
      * N-of-M seed-wise win count
      * Δ% verdict (SAVING / WORSE / neutral)
    """
    b = per_seed_metric(volume, "baseline")
    e = per_seed_metric(volume, "emission_based")
    seeds = sorted(set(b) & set(e))
    b_vals = np.array([b[s]["co2"] for s in seeds]) / 1e3   # mg -> g
    e_vals = np.array([e[s]["co2"] for s in seeds]) / 1e3

    # Paired stats
    delta_pct = 100.0 * (e_vals.mean() - b_vals.mean()) / b_vals.mean()
    n_wins_em = int(np.sum(e_vals < b_vals))
    n_total   = len(seeds)
    p_val     = _paired_ttest_p(b_vals, e_vals)
    sig       = _sig_label(p_val)
    verdict   = ("SAVING" if delta_pct < 0
                 else "WORSE" if delta_pct > 0
                 else "NEUTRAL")

    # Fixed per-seed color mapping (tab10) — keyed to the seed number so
    # the same seed keeps the same color across all plots in the paper.
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.75))
    # Reserve room at the bottom for the stats banner + seed legend
    ax  = fig.add_subplot(1, 1, 1)

    bp = ax.boxplot([b_vals, e_vals], widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.0),
                    boxprops=dict(linewidth=0.7),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    showfliers=False)
    bp["boxes"][0].set_facecolor("#BBDEFB")
    bp["boxes"][1].set_facecolor("#C8E6C9")

    # Per-seed dots, colored by seed identity so a reader can pair a
    # baseline dot with its matching emission-based dot.
    rng = np.random.default_rng(volume)
    for i, vals in enumerate([b_vals, e_vals], 1):
        xj = rng.normal(i, 0.06, size=len(vals))
        for j, s in enumerate(seeds):
            ax.scatter(xj[j], vals[j], s=18, color=cmap(j % 10),
                       edgecolor="black", linewidth=0.35,
                       alpha=0.90, zorder=3)

    # Mean diamonds
    ax.scatter([1, 2], [b_vals.mean(), e_vals.mean()],
               marker="D", s=34, color="#212121",
               edgecolor="white", linewidth=0.6, zorder=4,
               label="Mean")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Emission-based"])
    ax.set_ylabel("CO$_2$ per vehicle (g)")

    ax.set_title(f"V{volume} — CO$_2$/vehicle "
                 f"(n = {n_total} seeds)")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)

    # Stats banner placed as a figure-level text, between the plot and
    # the seed legend, so it never collides with the title or the boxes.
    fig.text(0.5, 0.14,
             f"Δ = {delta_pct:+.2f} % ({verdict})   |   "
             f"p = {p_val:.4f} ({sig})   |   "
             f"Emission-based wins {n_wins_em}/{n_total} seeds",
             ha="center", va="center", fontsize=7.5, color="#212121",
             bbox=dict(facecolor="#FFF8E1", edgecolor="#E0E0E0",
                       linewidth=0.4, pad=3.0))

    # Per-seed color legend at the very bottom (compact horizontal strip)
    seed_handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=cmap(j % 10),
                               markeredgecolor="black",
                               markeredgewidth=0.35, markersize=5,
                               label=f"Seed {s}")
                    for j, s in enumerate(seeds)]
    fig.legend(handles=seed_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=5, frameon=False,
               fontsize=6.5, handletextpad=0.4, columnspacing=1.4)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.24)
    save_both(fig, out)


def _paired_co2(volume: int, out: Path):
    """Paired-seed CO2 plot: baseline<->emission with connecting lines.
    Restores the old plot's scientific content:
      * per-seed color-coded connecting lines
      * mean diamonds on both sides
      * paired t-test p-value + significance label
      * N-of-M seed-wise win count
      * Δ% verdict (SAVING / WORSE / neutral)"""
    b = per_seed_metric(volume, "baseline")
    e = per_seed_metric(volume, "emission_based")
    seeds = sorted(set(b) & set(e))
    b_vals = np.array([b[s]["co2"] for s in seeds]) / 1e3
    e_vals = np.array([e[s]["co2"] for s in seeds]) / 1e3

    # Paired stats
    delta_pct = 100.0 * (e_vals.mean() - b_vals.mean()) / b_vals.mean()
    n_wins_em = int(np.sum(e_vals < b_vals))
    n_total   = len(seeds)
    p_val     = _paired_ttest_p(b_vals, e_vals)
    sig       = _sig_label(p_val)
    verdict   = ("SAVING" if delta_pct < 0
                 else "WORSE" if delta_pct > 0
                 else "NEUTRAL")

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.72))
    ax  = fig.add_subplot(1, 1, 1)
    x_b, x_e = 1, 2
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(seeds):
        ax.plot([x_b, x_e], [b_vals[i], e_vals[i]],
                "-o", color=cmap(i % 10),
                markersize=4.0, lw=0.9,
                label=f"Seed {s}")
    # Means (black diamonds)
    ax.scatter([x_b, x_e], [b_vals.mean(), e_vals.mean()],
               marker="D", s=38, color="#212121",
               edgecolor="white", linewidth=0.7, zorder=5,
               label="Mean")
    ax.set_xticks([x_b, x_e])
    ax.set_xticklabels(["Baseline", "Emission-based"])
    ax.set_ylabel("CO$_2$ per vehicle (g)")
    ax.set_title(f"V{volume} — paired-seed CO$_2$/vehicle "
                 f"(n = {n_total} seeds)")
    ax.set_xlim(0.75, 2.4)
    # Seed legend on the right (single column)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=6.5, frameon=False, ncol=1, handlelength=1.2)

    # Stats banner below the plot area (mirrors old plot's delta box)
    fig.text(0.5, 0.075,
             f"Δ mean = {delta_pct:+.2f} % ({verdict})   |   "
             f"p = {p_val:.4f} ({sig})   |   "
             f"Emission-based wins {n_wins_em}/{n_total} seeds",
             ha="center", va="center", fontsize=7.5, color="#212121",
             bbox=dict(facecolor="#FFF8E1", edgecolor="#E0E0E0",
                       linewidth=0.4, pad=3.0))

    fig.subplots_adjust(left=0.10, right=0.82, top=0.90, bottom=0.16)
    save_both(fig, out)


def _dual_wait(volume: int, out: Path):
    """In-box wait vs full-trip wait, baseline vs emission — paired-seed bars."""
    b = per_seed_metric(volume, "baseline")
    e = per_seed_metric(volume, "emission_based")
    seeds = sorted(set(b) & set(e))
    ib_b = np.array([b[s]["wait"]           for s in seeds])
    ib_e = np.array([e[s]["wait"]           for s in seeds])
    ft_b = np.array([b[s]["fulltrip_wait"]  for s in seeds])
    ft_e = np.array([e[s]["fulltrip_wait"]  for s in seeds])

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.58))
    ax  = fig.add_subplot(1, 1, 1)
    x = np.arange(len(seeds))
    w = 0.20
    ax.bar(x - 1.5*w, ib_b, w, label="In-box, baseline",       color="#90CAF9")
    ax.bar(x - 0.5*w, ib_e, w, label="In-box, emission",       color="#1976D2")
    ax.bar(x + 0.5*w, ft_b, w, label="Full-trip, baseline",    color="#FFCC80")
    ax.bar(x + 1.5*w, ft_e, w, label="Full-trip, emission",    color="#E65100")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7.5)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Wait per vehicle (s)")
    ax.set_title(f"V{volume} — in-box vs full-trip wait, by seed")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), columnspacing=1.2, handlelength=1.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.24)
    save_both(fig, out)


def _volume_out(volume: int, name: str) -> Path:
    p = ROOT / "visuals" / f"V{volume:04d}"
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def fig_regime_matrix():
    """One 3-row x 4-column full-page figure that summarizes the four-regime
    per-volume analysis at a glance. Rows = plot type, cols = regime.
    Rows:  (1) CO2 boxplot, (2) CO2 paired-seed, (3) dual wait time
    Cols:  V1500 Free-flow | V2900 Near-sat | V3300 Breakdown | V5000 Gridlock
    Uses the same cached per_seed_metric so a fresh full run streams the
    raw CSVs exactly once per (volume, method)."""
    REGIMES = [(1500, "Free-flow"),
               (2900, "Near-saturation"),
               (3300, "Breakdown onset"),
               (5000, "Gridlock")]
    cmap = plt.get_cmap("tab10")

    # Landscape-oriented: wider than tall (previous ratio 1.45 was too tall
    # for Overleaf inclusion). Use a wider canvas at ~0.75 aspect so the
    # 3x4 matrix fits inside a single manuscript page without overflow.
    fig = plt.figure(figsize=(FIG_W * 1.35, FIG_W * 1.00))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.42,
                          left=0.11, right=0.99, top=0.90, bottom=0.19)

    all_seeds = None                       # populated on the first volume
    for col, (vol, regime) in enumerate(REGIMES):
        b = per_seed_metric(vol, "baseline")
        e = per_seed_metric(vol, "emission_based")
        seeds = sorted(set(b) & set(e))
        if all_seeds is None:
            all_seeds = seeds
        b_co2 = np.array([b[s]["co2"] for s in seeds]) / 1e3
        e_co2 = np.array([e[s]["co2"] for s in seeds]) / 1e3

        delta = 100.0 * (e_co2.mean() - b_co2.mean()) / b_co2.mean()
        pval  = _paired_ttest_p(b_co2, e_co2)
        sig   = _sig_label(pval)
        wins  = int(np.sum(e_co2 < b_co2))
        vword = "SAVING" if delta < 0 else "WORSE" if delta > 0 else "NEUTRAL"

        # ---- Row 1: boxplot ----
        ax1 = fig.add_subplot(gs[0, col])
        bp = ax1.boxplot([b_co2, e_co2], widths=0.55, patch_artist=True,
                         medianprops=dict(color="black", linewidth=0.9),
                         boxprops=dict(linewidth=0.6),
                         whiskerprops=dict(linewidth=0.5),
                         capprops=dict(linewidth=0.5),
                         showfliers=False)
        bp["boxes"][0].set_facecolor("#BBDEFB")
        bp["boxes"][1].set_facecolor("#C8E6C9")
        rng = np.random.default_rng(vol)
        for i, vals in enumerate([b_co2, e_co2], 1):
            xj = rng.normal(i, 0.05, size=len(vals))
            for j, s in enumerate(seeds):
                ax1.scatter(xj[j], vals[j], s=10, color=cmap(j % 10),
                            edgecolor="black", linewidth=0.25,
                            alpha=0.90, zorder=3)
        ax1.scatter([1, 2], [b_co2.mean(), e_co2.mean()],
                    marker="D", s=18, color="#212121",
                    edgecolor="white", linewidth=0.4, zorder=4)
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(["B", "E"], fontsize=7)
        ax1.set_title(f"V{vol}\n{regime}", fontsize=8.5)
        if col == 0:
            ax1.set_ylabel("CO$_2$/veh (g)", fontsize=7.5)

        # ---- Row 2: paired ----
        ax2 = fig.add_subplot(gs[1, col])
        for i, s in enumerate(seeds):
            ax2.plot([1, 2], [b_co2[i], e_co2[i]], "-o",
                     color=cmap(i % 10), markersize=2.4, lw=0.7)
        ax2.scatter([1, 2], [b_co2.mean(), e_co2.mean()],
                    marker="D", s=22, color="#212121",
                    edgecolor="white", linewidth=0.5, zorder=5)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["B", "E"], fontsize=7)
        ax2.set_xlim(0.75, 2.25)
        if col == 0:
            ax2.set_ylabel("CO$_2$/veh (g)", fontsize=7.5)

        # ---- Row 3: dual wait ----
        ax3 = fig.add_subplot(gs[2, col])
        ib_b = np.array([b[s]["wait"]          for s in seeds])
        ib_e = np.array([e[s]["wait"]          for s in seeds])
        ft_b = np.array([b[s]["fulltrip_wait"] for s in seeds])
        ft_e = np.array([e[s]["fulltrip_wait"] for s in seeds])
        x = np.arange(len(seeds))
        w = 0.20
        ax3.bar(x - 1.5*w, ib_b, w, color="#90CAF9",  linewidth=0.2, edgecolor="black")
        ax3.bar(x - 0.5*w, ib_e, w, color="#1976D2",  linewidth=0.2, edgecolor="black")
        ax3.bar(x + 0.5*w, ft_b, w, color="#FFCC80",  linewidth=0.2, edgecolor="black")
        ax3.bar(x + 1.5*w, ft_e, w, color="#E65100",  linewidth=0.2, edgecolor="black")
        ax3.set_xticks([])
        ax3.tick_params(axis="y", labelsize=6.5)
        if col == 0:
            ax3.set_ylabel("Wait/veh (s)", fontsize=7.5)

        # Per-column stats footer, positioned under each column so it
        # does NOT overlap the shared legends (which start at y=0.055).
        col_center_x = 0.11 + (col + 0.5) * (0.99 - 0.11) / 4
        fig.text(col_center_x, 0.155,
                 f"V{vol}: Δ = {delta:+.2f} % ({vword})\n"
                 f"p = {pval:.4f} ({sig}) · wins {wins}/{len(seeds)}",
                 ha="center", va="center", fontsize=6, color="#212121",
                 bbox=dict(facecolor="#FFF8E1", edgecolor="#E0E0E0",
                           linewidth=0.3, pad=1.5))

    # Row-label banners on the far left (x=0.02 to leave room for y-axis
    # tick labels + y-axis title, which live between 0.04 and 0.11).
    fig.text(0.020, 0.755, "CO$_2$/vehicle\n(boxplot)",
             ha="center", va="center", fontsize=7.5, color="#1F4E79",
             rotation=90, weight="bold")
    fig.text(0.020, 0.500, "CO$_2$/vehicle\n(paired seed)",
             ha="center", va="center", fontsize=7.5, color="#1F4E79",
             rotation=90, weight="bold")
    fig.text(0.020, 0.32,  "Wait/seed\n(in-box + full-trip)",
             ha="center", va="center", fontsize=7.5, color="#1F4E79",
             rotation=90, weight="bold")

    # Seed color legend (row 1 & 2 use these)
    seed_handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=cmap(j % 10),
                               markeredgecolor="black",
                               markeredgewidth=0.3, markersize=4,
                               label=f"Seed {s}")
                    for j, s in enumerate(all_seeds)]
    # Wait-plot bar-color legend
    wait_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#90CAF9", edgecolor="black",
                      linewidth=0.3, label="In-box, Baseline"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#1976D2", edgecolor="black",
                      linewidth=0.3, label="In-box, Emission"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#FFCC80", edgecolor="black",
                      linewidth=0.3, label="Full-trip, Baseline"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#E65100", edgecolor="black",
                      linewidth=0.3, label="Full-trip, Emission"),
    ]

    fig.legend(handles=seed_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.075), ncol=10, frameon=False,
               fontsize=5.6, handletextpad=0.25, columnspacing=0.75,
               title="CO$_2$ rows – seed identity",
               title_fontsize=6.0)
    fig.legend(handles=wait_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False,
               fontsize=5.6, handletextpad=0.30, columnspacing=1.0,
               title="Wait-time row – bar mapping",
               title_fontsize=6.0)

    fig.suptitle("Per-regime results matrix "
                 "(B = baseline, E = emission-aware; 10 seeds/volume)",
                 fontsize=8, y=0.985)
    save_both(fig, XVR_OUT / "regime_matrix_updated")


def fig_per_volume_all():
    """Generate the full 4-regime x 3-plot-type per-volume matrix so every
    regime (V1500 Free-flow, V2900 Near-saturation, V3300 Breakdown onset,
    V5000 Gridlock) has all three per-volume figures. Streaming is
    cached per (volume, method) so paired + dual-wait for the same volume
    only stream the raw CSVs once."""
    for v in [1500, 2900, 3300, 5000]:
        print(f"  V{v} boxplot ...")
        _boxplot_co2(v, _volume_out(v, "single_boxplot_avg_co2_per_vehicle_updated"))
        print(f"  V{v} paired ...")
        _paired_co2(v, _volume_out(v, "single_paired_avg_co2_per_vehicle_updated"))
        print(f"  V{v} dual-wait ...")
        _dual_wait(v, _volume_out(v, "dual_wait_time_comparison_updated"))


# =========================================================================
# THRESHOLD SEARCH — plots 4, 5, 6
# =========================================================================
def fig04_co2_vs_threshold_by_regime():
    """Reproduce the old plot's methodology exactly:
       * 2x2 grid, one panel per regime
       * Left y-axis: CO2 savings (%), colored line + markers
       * Right y-axis: Throughput change (%), grey triangles + dotted line
       * Red dashed horizontal line at -1% (on throughput axis)
       * Green shaded vertical bands where BOTH co2_savings > 0
         AND throughput_change > -1% (the 'sweet spot')
       * Dashed vertical line + annotation for the optimum threshold
         (max CO2 savings subject to throughput_change > -1%)
    """
    THROUGHPUT_CONSTRAINT = -1.0    # percent

    df = pd.read_csv(TS_DIR / "results" / "threshold_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 1.15),
                             sharex=False)
    # Fixed panel order so the story flows free-flow → gridlock.
    regimes = ["Free-flow", "Near-saturation", "Breakdown onset", "Gridlock"]
    vol_of = {"Free-flow": 1500, "Near-saturation": 2900,
              "Breakdown onset": 3300, "Gridlock": 5000}

    for ax, regime in zip(axes.flat, regimes):
        sub = df[df["regime"] == regime].sort_values("threshold").reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False); continue
        c   = REGIME_COLORS.get(regime, "#616161")
        x   = sub["threshold"].values.astype(float)
        co2 = sub["co2_savings_pct"].values.astype(float)
        tp  = sub["throughput_change_pct"].values.astype(float)

        # Sweet-spot bands (co2 > 0 AND throughput > -1%) — draw first so
        # they sit behind everything.
        good = (co2 > 0) & (tp > THROUGHPUT_CONSTRAINT)
        if good.any():
            idx    = np.where(good)[0]
            gaps   = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[idx[0], idx[gaps + 1]] if gaps.size else np.r_[idx[0]]
            ends   = np.r_[idx[gaps], idx[-1]]   if gaps.size else np.r_[idx[-1]]
            for s_i, e_i in zip(starts, ends):
                ax.axvspan(x[s_i], x[e_i],
                           color="#66BB6A", alpha=0.14, zorder=1)
            ax.plot([], [], color="#66BB6A", alpha=0.30, lw=6,
                    label="Sweet spot")

        # CO2 savings on left axis
        ax.plot(x, co2, "-o", color=c, markersize=2.6, lw=1.0,
                zorder=4, label="CO$_2$ savings (%)")
        ax.axhline(0, color="#616161", lw=0.4, zorder=2)
        ax.set_ylabel("CO$_2$ savings (%)", color=c)
        ax.tick_params(axis="y", labelcolor=c)

        # Throughput on right axis (grey triangles + dotted line)
        ax2 = ax.twinx()
        ax2.plot(x, tp, ":", color="#9E9E9E", lw=0.7, zorder=3)
        ax2.scatter(x, tp, marker="^", s=8, color="#616161",
                    alpha=0.75, zorder=3, label="Throughput Δ (%)")
        ax2.axhline(THROUGHPUT_CONSTRAINT, color="#D32F2F", ls="--",
                    lw=0.7, zorder=2, label="−1 % throughput limit")
        ax2.set_ylabel("Throughput change (%)", color="#616161",
                       fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#616161", labelsize=7)
        # Hide the extra spines
        ax2.spines["top"].set_visible(False)

        # Optimum threshold (max CO2 savings with throughput > -1%)
        valid_idx = np.where((tp > THROUGHPUT_CONSTRAINT) & np.isfinite(co2))[0]
        if valid_idx.size:
            i_opt = int(valid_idx[int(np.nanargmax(co2[valid_idx]))])
            ax.axvline(x[i_opt], color="#2E7D32", ls="--", lw=0.9,
                       zorder=5, label="Opt threshold")
            ymin, ymax = ax.get_ylim()
            y_ann = ymin + 0.92 * (ymax - ymin)
            xmin, xmax = ax.get_xlim()
            ha = "center"
            if x[i_opt] > xmin + 0.75 * (xmax - xmin): ha = "right"
            elif x[i_opt] < xmin + 0.25 * (xmax - xmin): ha = "left"
            ax.text(x[i_opt], y_ann,
                    f"opt = {int(x[i_opt])} mg/s\n({co2[i_opt]:+.2f} %)",
                    ha=ha, va="top", fontsize=6.5, color="#2E7D32",
                    bbox=dict(facecolor="white", edgecolor="#BDBDBD",
                              linewidth=0.3, alpha=0.9, pad=1.5))

        ax.set_title(f"V{vol_of[regime]} — {regime}", fontsize=9)
        ax.set_xlabel("Emission threshold (mg/s)")

        # Combined legend below x-axis (2 col, 4 rows)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=5.8, frameon=False,
                  ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.30),
                  handlelength=1.4, columnspacing=1.0,
                  handletextpad=0.5)

    fig.suptitle("CO$_2$ savings & throughput vs. emission threshold, by regime",
                 fontsize=9.5, y=0.985)
    fig.text(0.5, 0.945,
             "Green: CO$_2$ > 0 AND Δthroughput > −1 %  |  "
             "opt = max saving with Δthroughput > −1 %  |  10 seeds averaged",
             ha="center", fontsize=6.8, color="#616161", style="italic")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.18,
                        hspace=0.90, wspace=0.62)
    save_both(fig, TS_OUT / "co2_vs_threshold_by_regime_updated")


def fig05_stable_region_analysis():
    """Reproduce the old plot's methodology exactly:
       * Per-threshold raw CO2-savings dots
       * 5-point centered rolling average
       * ±0.3 % band around the rolling mean
       * Stable plateau shading where rolling mean >= peak - 0.3 %
       * Optimum threshold vertical line + annotation."""
    ROLL_WIN  = 5
    STAB_BAND = 0.3          # percent

    df = pd.read_csv(TS_DIR / "results" / "threshold_summary.csv")
    # sharex=False so each panel shows its own x-axis label + ticks, and
    # its legend can sit cleanly under that label.
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 1.15),
                             sharex=False)
    regimes = ["Free-flow", "Near-saturation", "Breakdown onset", "Gridlock"]

    for ax, regime in zip(axes.flat, regimes):
        sub = df[df["regime"] == regime].sort_values("threshold").reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False); continue
        c = REGIME_COLORS.get(regime, "#616161")
        x = sub["threshold"].values.astype(float)
        y = sub["co2_savings_pct"].values.astype(float)

        # Raw scatter (short label so per-panel legend fits at ncol=2)
        ax.scatter(x, y, s=8, color=c, alpha=0.55, zorder=3,
                   label="Raw values")

        # 5-point centered rolling average + ±0.3 % band
        roll = pd.Series(y).rolling(ROLL_WIN, center=True, min_periods=1).mean().values
        ax.plot(x, roll, "-", color=c, lw=1.1, zorder=4,
                label=f"Rolling avg (5-pt)")
        ax.fill_between(x, roll - STAB_BAND, roll + STAB_BAND,
                        color=c, alpha=0.18, zorder=2,
                        label=f"±{STAB_BAND:.1f} % band")

        # Stable plateau: rolling mean >= peak - 0.3 %
        if np.isfinite(roll).any():
            peak = np.nanmax(roll)
            plateau_mask = roll >= (peak - STAB_BAND)
            # Fill each contiguous plateau run
            if plateau_mask.any():
                idx = np.where(plateau_mask)[0]
                # Split into runs
                gaps  = np.where(np.diff(idx) > 1)[0]
                starts = np.r_[idx[0], idx[gaps + 1]] if gaps.size else np.r_[idx[0]]
                ends   = np.r_[idx[gaps], idx[-1]]   if gaps.size else np.r_[idx[-1]]
                for s_i, e_i in zip(starts, ends):
                    ax.axvspan(x[s_i], x[e_i], color=c, alpha=0.10, zorder=1)
                # Legend proxy (short label so per-panel legend fits at ncol=2)
                ax.plot([], [], color=c, alpha=0.25, lw=6,
                        label="Stable plateau")
            # Optimum threshold: max CO2 savings subject to throughput_change_pct > -1
            # (same convention as the original analyze_threshold_search.py)
            tp   = sub["throughput_change_pct"].values.astype(float)
            valid_idx = np.where((tp > -1.0) & np.isfinite(y))[0]
            if valid_idx.size:
                i_opt = int(valid_idx[int(np.nanargmax(y[valid_idx]))])
                ax.axvline(x[i_opt], color="#212121", lw=0.8, ls="--", zorder=5)
                # Annotation placed high (just below the panel title, well
                # above the legend at bottom-right).
                ymin, ymax = ax.get_ylim()
                y_ann = ymin + 0.92 * (ymax - ymin)
                # Nudge horizontally if the vline is near the right edge so
                # the label stays inside the axes.
                xmin, xmax = ax.get_xlim()
                x_ann = x[i_opt]
                ha = "center"
                if x_ann > xmin + 0.75 * (xmax - xmin):
                    ha = "right"
                elif x_ann < xmin + 0.25 * (xmax - xmin):
                    ha = "left"
                ax.text(x_ann, y_ann,
                        f"opt = {int(x[i_opt])} mg/s\n({y[i_opt]:+.2f} %)",
                        ha=ha, va="top",
                        fontsize=6.5, color="#212121",
                        bbox=dict(facecolor="white", edgecolor="#BDBDBD",
                                  linewidth=0.3, alpha=0.9, pad=1.5))

        ax.axhline(0, color="#616161", lw=0.5)
        ax.set_title(regime, fontsize=9)
        ax.set_ylabel("CO$_2$ savings (%)")
        ax.set_xlabel("Emission threshold (mg/s)")
        # Legend placed BELOW the x-axis label of this panel, 2 columns wide,
        # so it never covers data, never overlaps the panel title, and never
        # collides with the next row's title.
        ax.legend(fontsize=5.8, frameon=False, ncol=2,
                  loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  handlelength=1.4, columnspacing=1.2,
                  handletextpad=0.5)
    fig.suptitle("Stable-threshold regions per demand regime",
                 fontsize=9.5, y=0.985)
    # Sub-line placed low enough that the vertical gap between suptitle and
    # sub-line is clearly visible at 15 cm width.
    fig.text(0.5, 0.945,
             "Rolling mean ± 0.3 % band  |  plateau: rolling ≥ peak − 0.3 %  |  "
             "opt: max saving with Δthroughput > −1 %",
             ha="center", fontsize=6.8, color="#616161", style="italic")
    # Big hspace + big bottom margin so the per-panel legends (below each
    # x-axis) never collide with the next row's panel title or the figure
    # bottom edge. Bottom-row legend anchor is at -0.30 in axes coords, so
    # the figure bottom margin needs to be at least ~0.18 to contain it.
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.18,
                        hspace=0.90, wspace=0.28)
    save_both(fig, TS_OUT / "stable_region_analysis_updated")


def fig06_optimal_threshold_by_regime():
    df = pd.read_csv(TS_DIR / "results" / "optimal_threshold_per_regime.csv")
    if df.empty: return
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))
    ax = fig.add_subplot(1, 1, 1)
    df = df.sort_values("volume")
    colors = [REGIME_COLORS.get(r, "#616161") for r in df["regime"]]
    bars = ax.bar([f"V{int(v)}\n{r}" for v, r in zip(df["volume"], df["regime"])],
                  df["optimal_threshold"], color=colors,
                  edgecolor="black", linewidth=0.4, width=0.65)
    for b, val, pct in zip(bars, df["optimal_threshold"], df["co2_savings_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{int(val)}\n({pct:+.2f}%)",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Optimal threshold (mg/s)")
    ax.set_title("Optimal emission threshold per regime "
                 "(annotated with peak CO$_2$ savings)")
    ax.set_ylim(0, max(df["optimal_threshold"]) * 1.30)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.18)
    save_both(fig, TS_OUT / "optimal_threshold_by_regime_updated")


# =========================================================================
# PENETRATION — plots 16, 17
# =========================================================================
def fig16_co2_savings_by_penetration():
    df = pd.read_csv(PN_DIR / "results" / "penetration_summary.csv")
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.58))
    ax = fig.add_subplot(1, 1, 1)
    for regime, sub in df.groupby("regime"):
        sub = sub.sort_values("penetration_pct")
        c = REGIME_COLORS.get(regime, "#616161")
        ax.plot(sub["penetration_pct"], sub["co2_savings_pct"],
                "-o", color=c, markersize=3.2, lw=1.0, label=regime)
    ax.axhline(0, color="#616161", lw=0.6)
    ax.set_xlabel("Detector penetration (%)")
    ax.set_ylabel("CO$_2$ savings (%)")
    ax.set_title("CO$_2$ savings vs. detector penetration, by regime")
    ax.legend(fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.14)
    save_both(fig, PN_OUT / "co2_savings_by_penetration_updated")


def fig17_co2_savings_bars_by_penetration():
    df = pd.read_csv(PN_DIR / "results" / "penetration_summary.csv")
    regimes = list(df["regime"].unique())
    pens    = sorted(df["penetration_pct"].unique())
    w = 0.8 / max(1, len(regimes))
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.58))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(pens))
    for i, regime in enumerate(regimes):
        sub = df[df["regime"] == regime].set_index("penetration_pct")
        vals = [sub.loc[p, "co2_savings_pct"] if p in sub.index else 0.0
                for p in pens]
        ax.bar(x + i * w - (len(regimes) - 1) * w / 2, vals, w,
               color=REGIME_COLORS.get(regime, "#616161"),
               label=regime, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="#616161", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)}%" for p in pens])
    ax.set_xlabel("Detector penetration")
    ax.set_ylabel("CO$_2$ savings (%)")
    ax.set_title("CO$_2$ savings vs. detector penetration (bar view)")
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="best")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.14)
    save_both(fig, PN_OUT / "co2_savings_bars_by_penetration_updated")


# =========================================================================
# COMPOSITION — plots 18, 19
# =========================================================================
def fig18_co2_savings_by_composition():
    df = pd.read_csv(CM_DIR / "results" / "composition_summary.csv")
    regimes = list(df["regime"].unique())
    comps   = list(df["composition"].unique())
    w = 0.8 / max(1, len(comps))
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.58))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(regimes))
    cmap = plt.get_cmap("Set2")
    for i, comp in enumerate(comps):
        vals = []
        for r in regimes:
            sub = df[(df["regime"] == r) & (df["composition"] == comp)]
            vals.append(sub["co2_savings_pct"].iloc[0] if not sub.empty else 0.0)
        ax.bar(x + i * w - (len(comps) - 1) * w / 2, vals, w,
               color=cmap(i), label=comp,
               edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="#616161", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=8)
    ax.set_ylabel("CO$_2$ savings (%)")
    ax.set_title("CO$_2$ savings by fleet composition and regime")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.17)
    save_both(fig, CM_OUT / "co2_savings_by_composition_updated")


def fig19_absolute_co2_by_composition():
    df = pd.read_csv(CM_DIR / "results" / "composition_summary.csv")
    comps   = list(df["composition"].unique())
    regimes = list(df["regime"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_W * 0.55), sharey=True)
    cmap = plt.get_cmap("Set2")
    for ax_idx, (col, title) in enumerate(
        [("bl_co2_per_veh", "Baseline"),
         ("em_co2_per_veh", "Emission-based")]):
        ax = axes[ax_idx]
        w = 0.8 / max(1, len(comps))
        x = np.arange(len(regimes))
        for i, comp in enumerate(comps):
            vals = []
            for r in regimes:
                sub = df[(df["regime"] == r) & (df["composition"] == comp)]
                vals.append(sub[col].iloc[0] / 1e3 if not sub.empty else 0.0)
            ax.bar(x + i * w - (len(comps) - 1) * w / 2, vals, w,
                   color=cmap(i), label=comp,
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=7, rotation=15)
        ax.set_title(f"({chr(ord('a')+ax_idx)}) {title}")
        if ax_idx == 0:
            ax.set_ylabel("CO$_2$ per vehicle (g)")
        if ax_idx == 1:
            ax.legend(fontsize=6.5, frameon=False, ncol=1,
                      loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Absolute CO$_2$/vehicle by fleet composition and regime",
                 fontsize=9.5, y=0.98)
    fig.subplots_adjust(left=0.09, right=0.82, top=0.88, bottom=0.16,
                        wspace=0.15)
    save_both(fig, CM_OUT / "absolute_co2_by_composition_updated")


# =========================================================================
# THRESHOLD × SCENARIOS — plots 20, 21, 22, 23
# =========================================================================
def _heatmap(df, row_key, row_label, out: Path, title: str):
    """Generic heatmap: rows = row_key values, cols = threshold, value = CO2 savings."""
    piv = df.pivot_table(index=row_key, columns="threshold",
                         values="co2_savings_pct", aggfunc="mean")
    piv = piv.sort_index()
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))
    ax = fig.add_subplot(1, 1, 1)
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(int(c)) for c in piv.columns], fontsize=7)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(v) for v in piv.index], fontsize=7)
    ax.set_xlabel("Emission threshold (mg/s)")
    ax.set_ylabel(row_label)
    ax.set_title(title)
    # cell annotations (compact)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=5.2,
                    color="black" if abs(v) < vmax * 0.55 else "white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("CO$_2$ savings (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.10, right=1.00, top=0.90, bottom=0.16)
    save_both(fig, out)


def fig20_heatmap_pen_x_threshold():
    df = pd.read_csv(TX_DIR / "results" / "pen_x_threshold_summary.csv")
    _heatmap(df, "penetration_pct", "Detector penetration (%)",
             TX_OUT / "heatmap_pen_x_threshold_updated",
             "CO$_2$ savings — penetration × threshold (mean over regimes)")


def fig21_heatmap_comp_x_threshold():
    df = pd.read_csv(TX_DIR / "results" / "comp_x_threshold_summary.csv")
    _heatmap(df, "composition", "Fleet composition",
             TX_OUT / "heatmap_comp_x_threshold_updated",
             "CO$_2$ savings — composition × threshold (mean over regimes)")


def fig22_linechart_pen_x_threshold():
    df = pd.read_csv(TX_DIR / "results" / "pen_x_threshold_summary.csv")
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.60))
    ax = fig.add_subplot(1, 1, 1)
    pens = sorted(df["penetration_pct"].unique())
    cmap = plt.get_cmap("viridis", len(pens))
    for i, p in enumerate(pens):
        sub = (df[df["penetration_pct"] == p]
               .groupby("threshold")["co2_savings_pct"].mean()
               .reset_index().sort_values("threshold"))
        ax.plot(sub["threshold"], sub["co2_savings_pct"],
                "-o", color=cmap(i), markersize=2.5, lw=1.0,
                label=f"{int(p)}% penetration")
    ax.axhline(0, color="#616161", lw=0.6)
    ax.set_xlabel("Emission threshold (mg/s)")
    ax.set_ylabel("CO$_2$ savings (%)")
    ax.set_title("CO$_2$ savings vs. threshold, by penetration level")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="best")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.13)
    save_both(fig, TX_OUT / "linechart_pen_x_threshold_updated")


def fig23_optimal_threshold_shift():
    df = pd.read_csv(TX_DIR / "results" / "optimal_threshold_matrix.csv")
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.65))
    ax = fig.add_subplot(1, 1, 1)
    # X: scenario_value labels; Y: optimal_threshold; hue: regime
    piv = df.pivot_table(index=["scenario_type", "scenario_value"],
                          columns="regime",
                          values="optimal_threshold", aggfunc="mean")
    labels = [f"{st}\n{sv}" for st, sv in piv.index]
    x = np.arange(len(labels))
    regimes = list(piv.columns)
    w = 0.8 / max(1, len(regimes))
    for i, r in enumerate(regimes):
        vals = piv[r].values
        ax.bar(x + i * w - (len(regimes) - 1) * w / 2, vals, w,
               color=REGIME_COLORS.get(r, "#616161"), label=r,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("Optimal threshold (mg/s)")
    ax.set_title("Optimal-threshold shift across scenarios and regimes")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.24)
    save_both(fig, TX_OUT / "optimal_threshold_shift_updated")


# =========================================================================
# main
# =========================================================================
def main():
    print("[threshold_search]")
    fig04_co2_vs_threshold_by_regime()
    fig05_stable_region_analysis()
    fig06_optimal_threshold_by_regime()
    print("[per_volume]")
    fig_per_volume_all()
    print("[penetration_rate]")
    fig16_co2_savings_by_penetration()
    fig17_co2_savings_bars_by_penetration()
    print("[composition]")
    fig18_co2_savings_by_composition()
    fig19_absolute_co2_by_composition()
    print("[threshold_x_scenarios]")
    fig20_heatmap_pen_x_threshold()
    fig21_heatmap_comp_x_threshold()
    fig22_linechart_pen_x_threshold()
    fig23_optimal_threshold_shift()
    print("done.")


if __name__ == "__main__":
    main()

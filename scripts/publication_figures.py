#!/usr/bin/env python3
"""
publication_figures.py — regenerate every key manuscript figure at
publication quality per the Overleaf/Elsevier spec:

  * ~15 cm figure width (5.906 in)
  * 600-DPI PNG + true vector PDF (both outputs for every figure)
  * Embedded fonts (pdf.fonttype = 42)
  * Explicit margins (no tight_layout ambiguity)
  * No text collisions, no overlapping subplots
  * Preserves every scientific quantity from the source data — this script
    computes nothing new; it only reshapes presentation.

Data sources (all live under the project root):
  data/capacity_analysis/summary/discharge_by_volume.csv
  data/capacity_analysis/summary/discharge_by_volume_per_seed.csv
  data/capacity_analysis/summary/capacity_summary.json
  data/capacity_analysis/summary/vc_ratios.csv
  visuals/cross_volume/cross_volume_summary.csv
  visuals/cross_volume/narrative_tradeoff.csv

Outputs (all appended with _updated before extension):
  visuals/capacity_plots/01_discharge_vs_demand_updated.{png,pdf}
  visuals/capacity_plots/02_marginal_discharge_gain_updated.{png,pdf}
  visuals/capacity_plots/03a_vc_per_cycle_bars_with_los_updated.{png,pdf}
  visuals/capacity_plots/04_capacity_dashboard_updated.{png,pdf}
  visuals/capacity_plots/05_cycles_required_to_clear_updated.{png,pdf}
  visuals/cross_volume/cross_volume_comparison_updated.{png,pdf}
  visuals/cross_volume/narrative_tradeoff_updated.{png,pdf}

Run:
  python scripts/publication_figures.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Global publication settings
# ---------------------------------------------------------------------------
CM = 1 / 2.54
FIG_W = 15.0 * CM              # 5.906 in — Elsevier full text width
DPI_PNG = 600

plt.rcParams.update({
    # fonts
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  11,
    # lines / bars
    "axes.linewidth":    0.7,
    "grid.linewidth":    0.4,
    "lines.linewidth":   1.1,
    "lines.markersize":  3.5,
    "patch.linewidth":   0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # spines + grid
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linestyle":     "-",
    # save
    "savefig.dpi":        DPI_PNG,
    "savefig.bbox":       "standard",   # no auto bbox — we control margins
    "pdf.fonttype":       42,           # embed TrueType (Overleaf-safe)
    "ps.fonttype":        42,
    "pdf.compression":    9,
})


ROOT     = Path(__file__).resolve().parents[1]
SUMMARY  = ROOT / "data" / "capacity_analysis" / "summary"
CAP_JSON = SUMMARY / "capacity_summary.json"
DBV_CSV  = SUMMARY / "discharge_by_volume.csv"
DPS_CSV  = SUMMARY / "discharge_by_volume_per_seed.csv"
VCR_CSV  = SUMMARY / "vc_ratios.csv"
XVR_CSV  = ROOT / "visuals" / "cross_volume" / "cross_volume_summary.csv"
NAR_CSV  = ROOT / "visuals" / "cross_volume" / "narrative_tradeoff.csv"

CAP_OUT  = ROOT / "visuals" / "capacity_plots"
XVR_OUT  = ROOT / "visuals" / "cross_volume"
CAP_OUT.mkdir(parents=True, exist_ok=True)
XVR_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LOS proxy (preserved exactly from analyze_results / plot_capacity_analysis)
# ---------------------------------------------------------------------------
LOS_BANDS = [(35, "A", "#64B5F6"),
             (55, "B", "#81C784"),
             (75, "C", "#FFD54F"),
             (90, "D", "#FF8A65"),
             (100,"E", "#EF5350"),
             (math.inf, "F", "#B71C1C")]


def los_color(vc_pct: float) -> str:
    for thr, _lbl, col in LOS_BANDS:
        if vc_pct < thr:
            return col
    return "#B71C1C"


def los_label(vc_pct: float) -> str:
    for thr, lbl, _col in LOS_BANDS:
        if vc_pct < thr:
            return lbl
    return "F"


def save_both(fig, base_no_ext: Path):
    """Save both PNG (600 dpi raster) and PDF (vector, embedded fonts)."""
    png_path = base_no_ext.with_suffix(".png")
    pdf_path = base_no_ext.with_suffix(".pdf")
    fig.savefig(png_path, dpi=DPI_PNG)
    fig.savefig(pdf_path)                           # PDF backend, vector
    plt.close(fig)
    print(f"  wrote {png_path.name}  ({png_path.stat().st_size:,} B)  "
          f"{pdf_path.name} ({pdf_path.stat().st_size:,} B)")


# ---------------------------------------------------------------------------
# Data loaders — no derived math, straight from source
# ---------------------------------------------------------------------------
def load_capacity_context():
    with open(CAP_JSON) as f:
        cap_ctx = json.load(f)
    return cap_ctx


def load_discharge():
    df = pd.read_csv(DBV_CSV)
    df = df.dropna(subset=["volume"]).copy()
    df["volume"] = df["volume"].astype(int)
    df = df.sort_values("volume").reset_index(drop=True)
    return df


def load_per_seed():
    return pd.read_csv(DPS_CSV)


def load_vc():
    df = pd.read_csv(VCR_CSV)
    df = df.dropna(subset=["volume"]).copy()
    df["volume"] = df["volume"].astype(int)
    return df.sort_values("volume").reset_index(drop=True)


def load_cross_volume():
    return pd.read_csv(XVR_CSV)


# ---------------------------------------------------------------------------
# Tick decimation helper (keeps ALL data points, only labels a subset)
# ---------------------------------------------------------------------------
def sparse_ticks(volumes, target=8):
    n = len(volumes)
    if n <= target:
        return list(range(n)), [str(int(v)) for v in volumes]
    step = max(1, n // target)
    idx  = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx, [str(int(volumes[i])) for i in idx]


# ===========================================================================
# FIGURE 01 — Discharge vs Demand (multi-seed with error bars)
# ===========================================================================
def fig01_discharge_vs_demand():
    df = load_discharge()
    dps = load_per_seed()
    cap_ctx = load_capacity_context()
    cap = int(cap_ctx["capacity_veh_per_cycle"])
    crit = int(cap_ctx["critical_volume"])

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))
    ax = fig.add_subplot(1, 1, 1)

    vols = df["volume"].values
    # per-seed scatter for max
    for _, r in dps.iterrows():
        ax.plot(r["volume"], r["max_arrived_per_cycle"],
                marker="o", markersize=1.7, color="#1976D2",
                alpha=0.35, linewidth=0)
    # mean ± std bands
    ax.errorbar(vols, df["mean_max_arrived_per_cycle"],
                yerr=df["std_max_arrived_per_cycle"],
                fmt="-", color="#0D47A1", ecolor="#0D47A1",
                elinewidth=0.6, capsize=1.6, capthick=0.5,
                label="Max discharge / cycle (mean ± SD, 10 seeds)")
    ax.errorbar(vols, df["mean_avg_arrived_per_cycle"],
                yerr=df["std_avg_arrived_per_cycle"],
                fmt="-", color="#E65100", ecolor="#E65100",
                elinewidth=0.6, capsize=1.6, capthick=0.5,
                label="Avg discharge / cycle (mean ± SD, 10 seeds)")
    ax.axhline(cap, ls="--", lw=0.8, color="#C62828",
               label=f"Capacity = {cap} veh/cycle")
    ax.axvline(crit, ls=":", lw=0.8, color="#4A148C",
               label=f"Critical volume = {crit} veh/hr")

    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Vehicles / 90-s cycle")
    ax.set_title("Discharge vs. demand — multi-seed capacity sweep")
    ax.legend(loc="lower right", frameon=False, ncol=1)
    ax.set_xlim(vols.min() - 100, vols.max() + 100)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14)
    save_both(fig, CAP_OUT / "01_discharge_vs_demand_updated")


# ===========================================================================
# FIGURE 02 — Marginal discharge gain
# ===========================================================================
def fig02_marginal_discharge_gain():
    df = load_discharge()
    vols = df["volume"].values
    m = df["mean_max_arrived_per_cycle"].values
    d_m = np.diff(m)
    x = vols[1:]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.50))
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0, color="#616161", lw=0.6)
    ax.plot(x, d_m, "-o", color="#00695C", markersize=3,
            label="Δ mean max discharge / cycle")
    ax.fill_between(x, 0, d_m, where=(d_m >= 0),
                    color="#4DB6AC", alpha=0.25)
    ax.fill_between(x, 0, d_m, where=(d_m < 0),
                    color="#EF9A9A", alpha=0.25)

    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Δ (mean max) per 100 veh/hr step")
    ax.set_title("Marginal discharge gain — saturation signature")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(vols.min() - 100, vols.max() + 100)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14)
    save_both(fig, CAP_OUT / "02_marginal_discharge_gain_updated")


# ===========================================================================
# FIGURE 03a — Per-cycle V/C with LOS bands
#     SPEC: 50-80+ categories → split into contiguous vertical stack.
# ===========================================================================
def fig03a_vc_bars_stacked():
    df = load_vc()
    cap = int(df["capacity_veh_per_cycle"].iloc[0])
    volumes = df["volume"].values
    vcs = df["vc_percent"].values
    n = len(volumes)
    # Split into 3 contiguous vertical stacks
    third = n // 3
    ranges = [(0, third),
              (third, 2 * third),
              (2 * third, n)]

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_W * 1.05),
                             sharey=True)

    for ax, (a, b) in zip(axes, ranges):
        sub_v = volumes[a:b]
        sub_c = vcs[a:b]
        colors = [los_color(c) for c in sub_c]
        x = np.arange(len(sub_v))
        ax.bar(x, sub_c, color=colors, edgecolor="black",
               linewidth=0.35, width=0.85)
        for thr, lbl, _col in LOS_BANDS:
            if math.isinf(thr): continue
            ax.axhline(thr, color="#9E9E9E", lw=0.4, ls="--")
        ax.set_ylabel("V/C (%)")
        # Sparse ticks — every 2nd bar labeled
        step = max(1, len(sub_v) // 10)
        tick_idx = list(range(0, len(sub_v), step))
        if tick_idx[-1] != len(sub_v) - 1:
            tick_idx.append(len(sub_v) - 1)
        ax.set_xticks([x[i] for i in tick_idx])
        ax.set_xticklabels([str(sub_v[i]) for i in tick_idx],
                           rotation=0)
        ax.set_xlim(-0.6, len(sub_v) - 0.4)
        ax.set_title(f"Demand {sub_v[0]}–{sub_v[-1]} veh/hr",
                     fontsize=9, loc="left")
        ax.set_ylim(0, max(105, vcs.max() + 5))

    axes[-1].set_xlabel("Demand (veh/hr)")

    # Compact LOS legend at the top
    handles = [Patch(facecolor=col, edgecolor="black", linewidth=0.3,
                     label=f"LOS {lbl}")
               for _thr, lbl, col in LOS_BANDS]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False,
               fontsize=7.5, columnspacing=1.2, handlelength=1.2)

    fig.suptitle(
        f"Per-cycle V/C by demand (cap = {cap} veh/cycle; "
        f"LOS is a V/C proxy, not delay-based HCM LOS)",
        y=0.945, fontsize=9)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.15,
                        hspace=0.55)

    # Footer key — placed WELL below the lowest x-axis label (per spec).
    fig.text(0.5, 0.055,
             "Bands:  <60% Light   ·   60–80% Moderate   ·   "
             "80–100% Heavy   ·   >100% Saturated",
             ha="center", fontsize=7, color="#424242")
    fig.text(0.5, 0.020,
             "LOS proxy via V/C:  A <35%   ·   B 35–55%   ·   "
             "C 55–75%   ·   D 75–90%   ·   E 90–100%   ·   F >100%",
             ha="center", fontsize=7, color="#424242")

    save_both(fig, CAP_OUT / "03a_vc_per_cycle_bars_with_los_updated")


# ===========================================================================
# FIGURE 04 — Capacity dashboard, compact 2×2
# ===========================================================================
def fig04_capacity_dashboard():
    df   = load_discharge()
    dps  = load_per_seed()
    vc   = load_vc()
    ctx  = load_capacity_context()
    cap  = int(ctx["capacity_veh_per_cycle"])
    crit = int(ctx["critical_volume"])
    cycle = int(ctx["cycle_seconds"])

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.95))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.09, right=0.98, top=0.92, bottom=0.09)

    # (a) Discharge vs demand — compact
    ax1 = fig.add_subplot(gs[0, 0])
    v = df["volume"].values
    ax1.plot(v, df["mean_max_arrived_per_cycle"], "-",
             color="#0D47A1", label="max", lw=1.0)
    ax1.fill_between(v,
        df["mean_max_arrived_per_cycle"] - df["std_max_arrived_per_cycle"],
        df["mean_max_arrived_per_cycle"] + df["std_max_arrived_per_cycle"],
        color="#0D47A1", alpha=0.15)
    ax1.plot(v, df["mean_avg_arrived_per_cycle"], "-",
             color="#E65100", label="avg", lw=1.0)
    ax1.fill_between(v,
        df["mean_avg_arrived_per_cycle"] - df["std_avg_arrived_per_cycle"],
        df["mean_avg_arrived_per_cycle"] + df["std_avg_arrived_per_cycle"],
        color="#E65100", alpha=0.15)
    ax1.axhline(cap, ls="--", lw=0.7, color="#C62828")
    ax1.set_title(f"(a) Discharge/cycle — cap = {cap}")
    ax1.set_xlabel("Demand (veh/hr)")
    ax1.set_ylabel("veh / 90-s cycle")
    ax1.legend(loc="lower right", frameon=False)

    # (b) Marginal Δ max
    ax2 = fig.add_subplot(gs[0, 1])
    m = df["mean_max_arrived_per_cycle"].values
    d_m = np.diff(m)
    ax2.axhline(0, color="#616161", lw=0.6)
    ax2.plot(v[1:], d_m, "-o", color="#00695C", markersize=2.5, lw=0.9)
    ax2.set_title("(b) Δ max per +100 veh/hr")
    ax2.set_xlabel("Demand (veh/hr)")
    ax2.set_ylabel("Δ veh / cycle")

    # (c) Per-cycle V/C bars (sparse-label)
    ax3 = fig.add_subplot(gs[1, :])
    vcs = vc["vc_percent"].values
    volumes = vc["volume"].values
    colors  = [los_color(c) for c in vcs]
    x = np.arange(len(volumes))
    ax3.bar(x, vcs, color=colors, edgecolor="black", linewidth=0.25,
            width=0.85)
    for thr, _lbl, _col in LOS_BANDS:
        if math.isinf(thr): continue
        ax3.axhline(thr, color="#9E9E9E", lw=0.4, ls="--")
    idx, labs = sparse_ticks(volumes, target=10)
    ax3.set_xticks(idx)
    ax3.set_xticklabels(labs, rotation=0)
    ax3.set_xlim(-0.6, len(volumes) - 0.4)
    ax3.set_title(f"(c) Per-cycle V/C (%) with LOS proxy bands — "
                  f"critical V = {crit}", fontsize=9)
    ax3.set_xlabel("Demand (veh/hr)")
    ax3.set_ylabel("V/C (%)")
    ax3.set_ylim(0, max(105, vcs.max() + 5))

    handles = [Patch(facecolor=col, edgecolor="black", linewidth=0.3,
                     label=f"LOS {lbl}")
               for _thr, lbl, col in LOS_BANDS]
    ax3.legend(handles=handles, loc="upper left", ncol=3, frameon=False,
               fontsize=7, columnspacing=0.9, handlelength=1.0)

    fig.suptitle(
        f"Capacity dashboard — 90-s cycle, {len(volumes)} demand levels, "
        f"10 seeds  (plateau at V{crit})", fontsize=9.5, y=0.985)
    save_both(fig, CAP_OUT / "04_capacity_dashboard_updated")


# ===========================================================================
# FIGURE 05 — Cycles required to clear demand
# ===========================================================================
def fig05_cycles_required_to_clear():
    df = load_discharge()
    v = df["volume"].values
    mean_max = df["mean_max_arrived_per_cycle"].values
    # Cycles required to clear a one-hour demand at peak discharge:
    #   cycles = (volume/hr × 3600/cycle_sec) / max_per_cycle
    #   simplified: cycles = volume / (max_per_cycle × 40)   for 90 s cycle
    cycle_s = 90
    cycles_per_hour = 3600 / cycle_s
    cycles_to_clear = v / (mean_max * (3600.0 / cycle_s))  # hours needed
    # Convert to number of cycles: hours * cycles_per_hour
    cycles_needed = cycles_to_clear * cycles_per_hour

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.50))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(v, cycles_needed, "-o", color="#6A1B9A",
            markersize=2.5, lw=0.9)
    ax.axhline(cycles_per_hour, ls="--", lw=0.7, color="#C62828",
               label=f"1-hour demand horizon = {cycles_per_hour:.0f} cycles")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Cycles required to clear 1-h demand")
    ax.set_title("Cycles required to clear 1-hour demand at peak discharge")
    ax.legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14)
    save_both(fig, CAP_OUT / "05_cycles_required_to_clear_updated")


# ===========================================================================
# FIGURE X-VOL — Cross-volume comparison (baseline vs emission), 2×3
# ===========================================================================
METRIC_META = [
    ("throughput",           "Throughput",              "veh/hr",  True),
    ("avg_wait_time",        "Avg wait (in-box)",       "s",       False),
    ("avg_fulltrip_wait",    "Avg full-trip wait",      "s",       False),
    ("avg_travel_time",      "Avg trip duration",       "s",       False),
    ("avg_speed",            "Avg speed",               "m/s",     True),
    ("avg_co2_per_vehicle",  "CO$_2$/vehicle",          "mg",      False),
]


def fig_cross_volume_comparison():
    df = load_cross_volume()
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.85))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34,
                          left=0.08, right=0.985, top=0.90, bottom=0.11)

    color_b = "#1565C0"   # baseline
    color_e = "#2E7D32"   # emission-based

    for i, (key, title, unit, higher_better) in enumerate(METRIC_META):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        sub = df[df["metric"] == key]
        for meth, colour, ls in [("baseline",       color_b, "-"),
                                 ("emission_based", color_e, "--")]:
            s = sub[sub["method"] == meth].sort_values("volume")
            if s.empty: continue
            ax.plot(s["volume"], s["mean"], ls, color=colour,
                    lw=1.0, marker="o", markersize=2.5,
                    label="Baseline" if meth == "baseline" else "Emission-based")
            ax.fill_between(s["volume"],
                            s["mean"] - s["std"], s["mean"] + s["std"],
                            color=colour, alpha=0.12)
        ax.set_title(f"({chr(ord('a')+i)}) {title}")
        ax.set_xlabel("Demand (veh/hr)")
        ax.set_ylabel(unit)
        arrow = "↑" if higher_better else "↓"
        ax.text(0.02, 0.97, f"{arrow} better", transform=ax.transAxes,
                va="top", ha="left", fontsize=7, color="#616161")
        if i == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    fig.suptitle(
        "Baseline vs. emission-based cross-volume comparison "
        "(mean ± SD across seeds)",
        fontsize=9.5, y=0.975)
    save_both(fig, XVR_OUT / "cross_volume_comparison_updated")


# ===========================================================================
# FIGURE NARRATIVE — 2×2 story panel
# ===========================================================================
def _read_narrative():
    # narrative_tradeoff.csv is a two-block CSV: absolute values, then deltas
    text = NAR_CSV.read_text().splitlines()
    # find the two header lines "volume,method,regime,..." and delta header
    idx_abs = None; idx_dlt = None
    for i, line in enumerate(text):
        if line.startswith("volume,method,regime,") and idx_abs is None:
            idx_abs = i
        elif line.startswith("volume,regime,delta_"):
            idx_dlt = i
            break
    abs_block = "\n".join([text[idx_abs]] + [ln for ln in text[idx_abs+1:idx_dlt] if ln.strip()])
    dlt_block = "\n".join([text[idx_dlt]] + [ln for ln in text[idx_dlt+1:]      if ln.strip()])
    import io
    df_abs = pd.read_csv(io.StringIO(abs_block))
    df_dlt = pd.read_csv(io.StringIO(dlt_block))
    # Drop separator/comment rows (non-numeric volume)
    df_abs = df_abs[pd.to_numeric(df_abs["volume"], errors="coerce").notna()].copy()
    df_abs["volume"] = df_abs["volume"].astype(int)
    df_dlt = df_dlt[pd.to_numeric(df_dlt["volume"], errors="coerce").notna()].copy()
    df_dlt["volume"] = df_dlt["volume"].astype(int)
    return df_abs, df_dlt


def fig_narrative_tradeoff():
    df_abs, df_dlt = _read_narrative()
    vols = sorted(df_abs["volume"].unique())

    def s(metric, method):
        return np.array([
            df_abs[(df_abs["volume"] == v) & (df_abs["method"] == method)]
                  [metric].values[0]
            for v in vols
        ])

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.90))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30,
                          left=0.09, right=0.985, top=0.90, bottom=0.11)

    # (a) Wait: in-box vs full-trip
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(vols, s("mean_avg_wait_s",      "baseline"),       "o-",  color="#1565C0", markersize=2.5, lw=1.0, label="B, in-box")
    ax.plot(vols, s("mean_avg_wait_s",      "emission_based"), "o--", color="#1565C0", markersize=2.5, lw=1.0, label="E, in-box")
    ax.plot(vols, s("mean_fulltrip_wait_s", "baseline"),       "s-",  color="#EF6C00", markersize=2.5, lw=1.0, label="B, full-trip")
    ax.plot(vols, s("mean_fulltrip_wait_s", "emission_based"), "s--", color="#EF6C00", markersize=2.5, lw=1.0, label="E, full-trip")
    ax.set_title("(a) Wait: in-box vs. full-trip")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Wait per vehicle (s)")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")

    # (b) Trip duration
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(vols, s("mean_travel_s", "baseline"),       "o-",  color="#2E7D32", markersize=2.5, lw=1.0, label="Baseline")
    ax.plot(vols, s("mean_travel_s", "emission_based"), "o--", color="#C62828", markersize=2.5, lw=1.0, label="Emission")
    ax.set_title("(b) Trip duration")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Duration (s)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (c) Phase switches
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(vols, s("mean_phase_switches", "baseline"),       "o-",  color="#6A1B9A", markersize=2.5, lw=1.0, label="Baseline")
    ax.plot(vols, s("mean_phase_switches", "emission_based"), "o--", color="#8D6E63", markersize=2.5, lw=1.0, label="Emission")
    ax.set_title("(c) Phase switches / simulation")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Switches (count)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    # (d) Δ% emission vs baseline
    ax = fig.add_subplot(gs[1, 1])
    d = df_dlt.sort_values("volume")
    ax.axhline(0, color="#616161", lw=0.6)
    ax.plot(d["volume"], d["delta_co2_pct"],    "o-", color="#C62828", markersize=2.5, lw=1.0, label="ΔCO$_2$ (%)")
    ax.plot(d["volume"], d["delta_travel_pct"], "s-", color="#2E7D32", markersize=2.5, lw=1.0, label="ΔTravel (%)")
    ax.plot(d["volume"], d["delta_wait_pct"],   "^-", color="#1565C0", markersize=2.5, lw=1.0, label="ΔWait (%)")
    ax.set_title("(d) Emission vs. baseline")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Δ% (negative = emission wins)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=1)

    fig.suptitle(
        "Wait / trip-duration / switches / CO$_2$ narrative "
        "(solid = baseline; dashed = emission-based)",
        fontsize=9, y=0.975)
    save_both(fig, XVR_OUT / "narrative_tradeoff_updated")


# ===========================================================================
# main
# ===========================================================================
def main():
    print("[publication_figures] regenerating manuscript-ready figures at "
          f"{FIG_W/CM:.1f} cm width, {DPI_PNG} DPI PNG + vector PDF")
    print("[capacity_plots]")
    fig01_discharge_vs_demand()
    fig02_marginal_discharge_gain()
    fig03a_vc_bars_stacked()
    fig04_capacity_dashboard()
    fig05_cycles_required_to_clear()
    print("[cross_volume]")
    fig_cross_volume_comparison()
    fig_narrative_tradeoff()
    print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pipeline A (capacity sweep) - plots with multi-seed support.

What changed (publication-grade upgrade)
----------------------------------------
1. Reads the NEW aggregate columns produced by parse_capacity_results.py:
     mean_avg_arrived_per_cycle, std_avg_arrived_per_cycle,
     mean_max_arrived_per_cycle, std_max_arrived_per_cycle,
     n_seeds
   Falls back to old single-seed columns (avg_arrived_per_cycle,
   max_arrived_per_cycle) if present, so legacy data still plots.

2. Plot 1 (discharge vs demand) now shows:
     - per-seed scatter for max_arrived_per_cycle (translucent dots)
     - per-seed scatter for avg_arrived_per_cycle (translucent dots)
     - mean lines with error bars (+- 1 std across seeds) on top
     - horizontal capacity line unchanged
   Scatter comes from data/capacity_analysis/summary/
                       discharge_by_volume_per_seed.csv

3. Plot 4 (dashboard) discharge subplot mirrors Plot 1 in miniature
   (error bars, seed scatter).

4. The V/C that drives bars, tables, and LOS is still the standard
   demand-based V/C:
        vc_percent = mean_avg_arrived_per_cycle / capacity_per_cycle
   This matches compute_vc_ratios.py's primary "vc_percent" column.

Inputs
------
data/capacity_analysis/summary/
  discharge_by_volume.csv            (aggregate, one row per volume)
  discharge_by_volume_per_seed.csv   (long format, one row per volume,seed)
  capacity_summary.json

The per-seed CSV is optional - if it's missing, plots still render
without scatter dots.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _f(row: dict, *candidates: str, default: float | None = None) -> float:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return float(row[c])
    if default is not None:
        return default
    raise KeyError(f"None of {candidates} in row keys={list(row.keys())}")


def band_label(vc_percent: float) -> str:
    if vc_percent < 60.0:
        return "Light"
    if vc_percent < 80.0:
        return "Moderate"
    if vc_percent <= 100.0:
        return "Heavy"
    return "Saturated"


def los_from_vc_percent(vc_percent: float) -> str:
    vc = vc_percent / 100.0
    if vc < 0.35:
        return "A"
    if vc < 0.55:
        return "B"
    if vc < 0.75:
        return "C"
    if vc < 0.90:
        return "D"
    if vc <= 1.00:
        return "E"
    return "F"


def load_per_seed(per_seed_csv: Path):
    """Return dicts volume -> list[float] for avg and max per-cycle arrivals."""
    avg_by_vol: dict[int, list[float]] = defaultdict(list)
    max_by_vol: dict[int, list[float]] = defaultdict(list)
    if not per_seed_csv.exists():
        return avg_by_vol, max_by_vol
    with open(per_seed_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            v = int(row["volume"])
            avg_by_vol[v].append(float(row["avg_arrived_per_cycle"]))
            max_by_vol[v].append(float(row["max_arrived_per_cycle"]))
    return avg_by_vol, max_by_vol


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parents[1]
    summary_dir = project_root / "data" / "capacity_analysis" / "summary"
    visuals_dir = project_root / "visuals" / "capacity_plots"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    discharge_csv = summary_dir / "discharge_by_volume.csv"
    per_seed_csv = summary_dir / "discharge_by_volume_per_seed.csv"
    cap_json = summary_dir / "capacity_summary.json"

    if not discharge_csv.exists() or not cap_json.exists():
        raise RuntimeError(
            "Missing required summary files. Ensure these exist:\n"
            f"  {discharge_csv}\n"
            f"  {cap_json}\n"
            "Then rerun this script."
        )

    cap = json.loads(cap_json.read_text(encoding="utf-8"))
    capacity_per_cycle = float(cap["capacity_veh_per_cycle"])
    cycle_seconds = int(cap["cycle_seconds"])

    # ---------------------------------------------------------------------
    # Aggregate CSV
    # ---------------------------------------------------------------------
    vols: list[int] = []
    mean_max: list[float] = []
    std_max: list[float] = []
    mean_avg: list[float] = []
    std_avg: list[float] = []
    mean_total: list[float] = []
    n_seeds: list[int] = []

    with open(discharge_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vols.append(int(row["volume"]))
            mean_max.append(_f(row, "mean_max_arrived_per_cycle",
                               "max_arrived_per_cycle"))
            std_max.append(_f(row, "std_max_arrived_per_cycle", default=0.0))
            mean_avg.append(_f(row, "mean_avg_arrived_per_cycle",
                               "avg_arrived_per_cycle"))
            std_avg.append(_f(row, "std_avg_arrived_per_cycle", default=0.0))
            mean_total.append(_f(row, "mean_total_arrived",
                                 "total_arrived", default=0.0))
            n_seeds.append(int(row.get("n_seeds", 1) or 1))

    # Per-seed scatter (optional)
    avg_by_vol, max_by_vol = load_per_seed(per_seed_csv)
    have_scatter = len(max_by_vol) > 0
    max_n_seeds = max(n_seeds) if n_seeds else 1

    # ---------------------------------------------------------------------
    # Derived metrics
    # ---------------------------------------------------------------------
    vc_per_cycle_pct: list[float] = []
    bands: list[str] = []
    los: list[str] = []
    cycles_required_at_capacity: list[float] = []
    demand_to_cycle_capacity_x: list[float] = []

    for i, v in enumerate(vols):
        vc_pct = (mean_avg[i] / capacity_per_cycle) * 100.0
        vc_per_cycle_pct.append(vc_pct)
        bands.append(band_label(vc_pct))
        los.append(los_from_vc_percent(vc_pct))
        cycles_needed = float(v) / capacity_per_cycle
        cycles_required_at_capacity.append(cycles_needed)
        demand_to_cycle_capacity_x.append(cycles_needed)

    marginal_gain: list[float | None] = [None]
    marginal_pct: list[float | None] = [None]
    for i in range(1, len(mean_max)):
        dg = mean_max[i] - mean_max[i - 1]
        marginal_gain.append(dg)
        prev = mean_max[i - 1] if mean_max[i - 1] != 0 else 1.0
        marginal_pct.append(100.0 * dg / prev)

    # =========================
    # Plot 1: Discharge vs Demand (with seed scatter + error bars)
    # =========================
    plt.figure(figsize=(22, 8))

    # Per-seed scatter first, behind the lines.
    if have_scatter:
        for i, v in enumerate(vols):
            xs_max = [v] * len(max_by_vol.get(v, []))
            xs_avg = [v] * len(avg_by_vol.get(v, []))
            plt.scatter(xs_max, max_by_vol.get(v, []),
                        color="tab:blue", alpha=0.25, s=22,
                        zorder=1,
                        label="Max per seed" if i == 0 else None)
            plt.scatter(xs_avg, avg_by_vol.get(v, []),
                        color="tab:orange", alpha=0.25, s=22,
                        zorder=1,
                        label="Avg per seed" if i == 0 else None)

    # Mean +/- std error bars.
    plt.errorbar(
        vols, mean_max, yerr=std_max,
        fmt="o-", color="tab:blue", ecolor="tab:blue",
        elinewidth=1.2, capsize=4, zorder=3,
        label="Mean max discharged/cycle (±1 std)",
    )
    plt.errorbar(
        vols, mean_avg, yerr=std_avg,
        fmt="o-", color="tab:orange", ecolor="tab:orange",
        elinewidth=1.2, capsize=4, zorder=3,
        label="Mean avg discharged/cycle (±1 std)",
    )
    plt.axhline(y=capacity_per_cycle, linestyle="--", color="red",
                label=f"Capacity ≈ {capacity_per_cycle:.0f} veh/cycle")

    plt.title(
        f"Intersection Discharge per Signal Cycle (cycle={cycle_seconds}s, "
        f"n_seeds={max_n_seeds})"
    )
    plt.xlabel("Demand volume (veh/hr)")
    plt.ylabel("Vehicles per cycle")
    plt.xticks(rotation=45, ha="right", fontsize=9)

    text = (
        f"Estimated capacity ≈ {capacity_per_cycle:.0f} veh/cycle\n"
        f"Critical volume ≈ {cap.get('critical_volume', '?')} veh/hr\n"
        f"Method: {cap.get('notes', '')}\n"
        f"Aggregation: mean across {max_n_seeds} seeds; error bars = 1 std"
    )
    plt.gca().text(
        0.02, -0.22, text,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=10,
        clip_on=False,
    )

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    out1 = visuals_dir / "01_discharge_vs_demand.png"
    plt.subplots_adjust(bottom=0.18, right=0.78)
    plt.savefig(out1, dpi=240, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot 2: Marginal discharge gain (uses mean_max)
    # =========================
    plt.figure(figsize=(26, 7))
    x = vols[1:]
    y = [g for g in marginal_gain[1:]]
    plt.plot(x, y, marker="o")
    plt.axhline(y=0, linestyle="--")
    plt.title("Marginal Increase in Mean Max Discharge per Cycle "
              "(Saturation Indicator)")
    plt.xlabel("Demand volume (veh/hr)")
    plt.ylabel("Δ mean max discharged per cycle (veh/cycle)")

    _stride2 = max(1, len(vols) // 20)  # at most ~20 annotations
    for i in range(1, len(vols)):
        pct = marginal_pct[i]
        if pct is not None and marginal_gain[i] is not None:
            if (i - 1) % _stride2 != 0:
                continue
            plt.text(
                vols[i],
                marginal_gain[i] + 0.15,
                f"{pct:+.2f}%",
                ha="center", va="bottom", fontsize=9,
                rotation=45,
            )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    out2 = visuals_dir / "02_marginal_discharge_gain.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=240, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot 3a: Per-cycle V/C bars
    # =========================
    labels = [str(v) for v in vols]
    fig = plt.figure(figsize=(36, 10))
    ax = fig.add_subplot(111)

    bars = ax.bar(labels, vc_per_cycle_pct)
    ax.axhline(y=60.0, linestyle="--")
    ax.axhline(y=80.0, linestyle="--")
    ax.axhline(y=100.0, linestyle="--")

    ax.set_title("Per-Cycle Volume-to-Capacity (V/C) by Demand Level (with LOS)")
    ax.set_xlabel("Demand volume (veh/hr)")
    ax.set_ylabel("Per-cycle V/C (%)  =  (mean avg arrivals per cycle "
                  "/ capacity per cycle) × 100")

    ymax = max(vc_per_cycle_pct) if vc_per_cycle_pct else 100.0
    ax.set_ylim(0, max(120.0, ymax * 1.18))

    ax.text(
        0.015, 0.98,
        "Bands:\n<60% Light\n60-80% Moderate\n80-100% Heavy\n>100% Saturated\n\n"
        "LOS (proxy via V/C):\n"
        "A <35%\nB 35-55%\nC 55-75%\nD 75-90%\nE 90-100%\nF >100%",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
        fontsize=11,
    )

    label_font = 9 if len(bars) <= 12 else (8 if len(bars) <= 25 else 7)
    _stride3a = max(1, len(bars) // 20)  # cap at ~20 annotated bars
    y_offset = 2.0 if ymax < 120 else 6.0

    for i, b in enumerate(bars):
        if i % _stride3a != 0:
            continue
        if len(bars) > 25:
            label = f"V/C≈{vc_per_cycle_pct[i]:.1f}%\nLOS:{los[i]}"
        else:
            label = (
                f"V/C≈{vc_per_cycle_pct[i]:.1f}%\n"
                f"Band: {bands[i]}\n"
                f"LOS: {los[i]}\n"
                f"{cycles_required_at_capacity[i]:.1f} cycles @ cap"
            )
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + y_offset,
            label, ha="center", va="bottom", fontsize=label_font,
            rotation=90 if len(bars) > 25 else 0,
        )

    ax.tick_params(axis="x", rotation=45, labelsize=8)
    out3a = visuals_dir / "03a_vc_per_cycle_bars_with_los.png"
    plt.tight_layout()
    plt.savefig(out3a, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # Plot 3b: Paper-ready table (adds n_seeds + mean ± std)
    # =========================
    delta_abs = ["—"]
    delta_pct = ["—"]
    for i in range(1, len(vols)):
        da = vols[i] - vols[i - 1]
        dp = 100.0 * da / float(vols[i - 1]) if vols[i - 1] != 0 else 0.0
        delta_abs.append(f"+{da}")
        delta_pct.append(f"({dp:.1f}%)")

    table_rows = []
    for i, v in enumerate(vols):
        table_rows.append([
            f"{v}",
            f"{n_seeds[i]}",
            f"{delta_abs[i]} {delta_pct[i]}",
            f"{mean_avg[i]:.2f} ± {std_avg[i]:.2f}",
            f"{mean_max[i]:.2f} ± {std_max[i]:.2f}",
            f"{vc_per_cycle_pct[i]:.1f}%",
            bands[i],
            los[i],
            f"{cycles_required_at_capacity[i]:.1f} cycles",
        ])

    col_labels = [
        "Demand", "n_seeds", "Δ Demand vs prev",
        "Mean avg arr/cycle ± std",
        "Mean max arr/cycle ± std",
        "Per-cycle V/C", "Band", "LOS", "Cycles to clear @ cap",
    ]

    n_rows = len(table_rows) + 1
    fig_h = max(8.0, 0.35 * n_rows)
    fig_w = 28.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    ax.axis("off")

    col_widths = [0.07, 0.06, 0.13, 0.14, 0.14, 0.09, 0.08, 0.06, 0.15]

    table = ax.table(
        cellText=table_rows, colLabels=col_labels,
        colLoc="center", cellLoc="center",
        colWidths=col_widths, bbox=[0.01, 0.01, 0.98, 0.98],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.10)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=10)
            cell.set_facecolor("#e6e6e6")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "white")

    out3b_png = visuals_dir / "03b_vc_table_paper_with_los.png"
    out3b_pdf = visuals_dir / "03b_vc_table_paper_with_los.pdf"
    plt.savefig(out3b_png, dpi=300, bbox_inches="tight")
    plt.savefig(out3b_pdf, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # Plot 4: Dashboard
    # =========================
    plt.figure(figsize=(30, 14))

    ax1 = plt.subplot(2, 2, 1)
    if have_scatter:
        for v in vols:
            ax1.scatter([v] * len(max_by_vol.get(v, [])),
                        max_by_vol.get(v, []),
                        color="tab:blue", alpha=0.25, s=14, zorder=1)
            ax1.scatter([v] * len(avg_by_vol.get(v, [])),
                        avg_by_vol.get(v, []),
                        color="tab:orange", alpha=0.25, s=14, zorder=1)
    ax1.errorbar(vols, mean_max, yerr=std_max,
                 fmt="o-", color="tab:blue", ecolor="tab:blue",
                 capsize=3, elinewidth=1, zorder=3, label="Max ±1σ")
    ax1.errorbar(vols, mean_avg, yerr=std_avg,
                 fmt="o-", color="tab:orange", ecolor="tab:orange",
                 capsize=3, elinewidth=1, zorder=3, label="Avg ±1σ")
    ax1.axhline(y=capacity_per_cycle, linestyle="--", color="red",
                label="Capacity")
    ax1.set_title("Discharge vs Demand (multi-seed)")
    ax1.set_xlabel("Demand (veh/hr)")
    ax1.set_ylabel("Vehicles/cycle")
    ax1.legend(loc="best", fontsize=8)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(vols[1:], y, marker="o")
    ax2.axhline(y=0, linestyle="--")
    ax2.set_title("Marginal Discharge Gain (mean max)")
    ax2.set_xlabel("Demand (veh/hr)")
    ax2.set_ylabel("Δ mean max (veh/cycle)")

    ax3 = plt.subplot(2, 2, 3)
    ax3.bar([str(v) for v in vols], vc_per_cycle_pct)
    ax3.axhline(y=100.0, linestyle="--")
    ax3.axhline(y=80.0, linestyle="--")
    ax3.axhline(y=60.0, linestyle="--")
    ax3.set_title("Per-cycle V/C (%)")
    ax3.set_xlabel("Demand (veh/hr)")
    ax3.set_ylabel("V/C (%)")
    ax3.tick_params(axis="x", rotation=45, labelsize=7)

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    narrative = (
        "Intersection Capacity Analysis Summary\n"
        "-------------------------------------\n"
        f"Signal cycle length: {cycle_seconds} seconds\n"
        f"Estimated capacity: ~{capacity_per_cycle:.0f} veh/cycle\n"
        f"Critical volume: ~{cap.get('critical_volume', '?')} veh/hr\n"
        f"Seeds per volume: up to {max_n_seeds}\n\n"
        "Key definitions used:\n"
        "- Per-cycle V/C (%) = (mean avg arrivals per cycle / capacity) × 100\n"
        "- Error bars = ±1 std across seeds\n"
        "- Dots = per-seed values\n"
        "- LOS is an HCM-style proxy based on V/C, not delay-based HCM LOS.\n"
    )
    ax4.text(0.0, 1.0, narrative, va="top", ha="left", fontsize=10)

    out4 = visuals_dir / "04_capacity_dashboard.png"
    plt.tight_layout()
    plt.savefig(out4, dpi=240, bbox_inches="tight")
    plt.close()

    # =========================
    # Plot 5: Cycles to clear (unchanged logic)
    # =========================
    plt.figure(figsize=(22, 7))
    plt.plot(vols, cycles_required_at_capacity, marker="o")
    plt.axhline(y=1.0, linestyle="--")
    plt.title(f"Cycles Required to Fully Discharge Demand "
              f"(assuming {capacity_per_cycle:.0f} veh/cycle capacity)")
    plt.xlabel("Demand volume (veh/hr)")
    plt.ylabel("Cycles required (Demand / capacity per cycle)")

    _stride5 = max(1, len(vols) // 20)
    for i, v in enumerate(vols):
        if i % _stride5 != 0:
            continue
        plt.text(
            v, cycles_required_at_capacity[i] + 0.25,
            f"{cycles_required_at_capacity[i]:.1f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    out5 = visuals_dir / "05_cycles_required_to_clear.png"
    plt.tight_layout()
    plt.savefig(out5, dpi=240, bbox_inches="tight")
    plt.close()

    print("Plots saved:")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3a}")
    print(f"  {out3b_png}")
    print(f"  {out3b_pdf}")
    print(f"  {out4}")
    print(f"  {out5}")
    if not have_scatter:
        print()
        print("NOTE: discharge_by_volume_per_seed.csv not found — "
              "seed scatter dots were skipped. Run the updated "
              "parse_capacity_results.py to generate it.")


if __name__ == "__main__":
    main()

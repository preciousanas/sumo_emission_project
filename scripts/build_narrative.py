#!/usr/bin/env python3
"""
Build the wait-time / trip-duration / phase-switch narrative.

Reads:
  visuals/cross_volume/cross_volume_summary.csv
  data/V####/<method>/log_seed_*.txt   (for Phase switches: N)

Writes:
  visuals/cross_volume/narrative_tradeoff.csv     (tidy per-volume table)
  visuals/cross_volume/narrative_tradeoff.png     (4-panel story)

The story in one breath:
  1. In free-flow (V1000..V2400) extra demand is absorbed for free:
     wait, travel, CO2 barely move, emission-based often flips more
     frequently than baseline and still saves a little.
  2. Around V2600..V3000 the intersection starts to saturate. Here is
     where the emission controller earns the most: it delays its
     switches while a group is emitting hard, cutting CO2 by up to
     ~1% and travel-time by ~0.5..0.9 s.
  3. At V3200 the system tips over: waits/travel-times explode
     non-linearly, the queue in the box begins to back up into the
     approach lanes, and phase-switch count DROPS because both
     controllers end up holding phases for longer than the 45 s max
     just to clear the queue.
  4. Past V3600 emission-based actually becomes worse than baseline:
     its extra hold-to-finish-high-emitter cycles now block a saturated
     network. Phase-switch count keeps declining.
"""
from __future__ import annotations
import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VIS  = ROOT / "visuals" / "cross_volume"
SUMMARY_CSV = VIS / "cross_volume_summary.csv"
OUT_CSV = VIS / "narrative_tradeoff.csv"
OUT_PNG = VIS / "narrative_tradeoff.png"


def read_phase_switches():
    """Return dict: {volume: {method: [per-seed switch counts]}}"""
    out = {}
    for vdir in sorted(DATA.glob("V[0-9][0-9][0-9][0-9]")):
        vol = int(vdir.name[1:])
        out[vol] = {"baseline": [], "emission_based": []}
        for m in ("baseline", "emission_based"):
            mdir = vdir / m
            if not mdir.is_dir():
                continue
            for lf in sorted(mdir.glob("log_seed_*.txt")):
                txt = lf.read_text(errors="replace")
                mobj = re.search(r"Phase switches:\s*(\d+)", txt)
                if mobj:
                    out[vol][m].append(int(mobj.group(1)))
    return out


def summarize():
    df = pd.read_csv(SUMMARY_CSV)
    # Pivot to wide: (volume, method) -> metric columns
    wide = df.pivot_table(
        index=["volume", "method"], columns="metric",
        values=["mean", "std"])
    # wide.columns is MultiIndex of (value_level, metric). Flatten as
    # "<value_level>_<metric>" e.g. "mean_throughput".
    wide.columns = ["{}_{}".format(lvl, mt) for lvl, mt in wide.columns]
    wide = wide.reset_index()
    return wide


def classify_regime(vol, throughput, demand):
    """Return a short, human-friendly regime label."""
    # Keep simple: demand -> regime bands tuned to what the data shows.
    if vol <= 2400:   return "Free-flow"
    if vol <= 3000:   return "Near-saturation"
    if vol <= 3400:   return "Breakdown onset"
    return "Gridlock"


def main():
    wide = summarize()
    ps   = read_phase_switches()

    rows = []
    for (_, r) in wide.sort_values(["volume", "method"]).iterrows():
        v = int(r["volume"])
        m = r["method"]
        sw = ps.get(v, {}).get(m, [])
        rows.append({
            "volume":                 v,
            "method":                 m,
            "regime":                 classify_regime(v, r["mean_throughput"], v),
            "mean_throughput":        round(r["mean_throughput"], 2),
            "mean_avg_wait_s":        round(r["mean_avg_wait_time"], 3),
            "mean_fulltrip_wait_s":   round(r["mean_avg_fulltrip_wait"], 3),
            "mean_travel_s":          round(r["mean_avg_travel_time"], 3),
            "mean_speed_mps":         round(r["mean_avg_speed"], 3),
            "mean_co2_mg_per_veh":    round(r["mean_avg_co2_per_vehicle"], 1),
            "mean_phase_switches":    round(float(np.mean(sw)), 2) if sw else np.nan,
            "std_phase_switches":     round(float(np.std(sw, ddof=1)), 2) if len(sw) > 1 else 0.0,
            "n_seeds_phase_switches": len(sw),
        })
    # Build a derived "delta vs baseline at same volume" table too.
    df_rows = pd.DataFrame(rows)

    delta_records = []
    for v in sorted(df_rows["volume"].unique()):
        b = df_rows[(df_rows["volume"] == v) & (df_rows["method"] == "baseline")]
        e = df_rows[(df_rows["volume"] == v) & (df_rows["method"] == "emission_based")]
        if b.empty or e.empty:
            continue
        b = b.iloc[0]; e = e.iloc[0]
        def pct(new, old):
            return np.nan if old == 0 else 100.0 * (new - old) / old
        delta_records.append({
            "volume": v,
            "regime": b["regime"],
            "delta_wait_s":         round(e["mean_avg_wait_s"]      - b["mean_avg_wait_s"], 3),
            "delta_wait_pct":       round(pct(e["mean_avg_wait_s"],      b["mean_avg_wait_s"]), 2),
            "delta_travel_s":       round(e["mean_travel_s"]        - b["mean_travel_s"], 3),
            "delta_travel_pct":     round(pct(e["mean_travel_s"],        b["mean_travel_s"]), 2),
            "delta_co2_mg":         round(e["mean_co2_mg_per_veh"]  - b["mean_co2_mg_per_veh"], 1),
            "delta_co2_pct":        round(pct(e["mean_co2_mg_per_veh"],  b["mean_co2_mg_per_veh"]), 2),
            "delta_switches":       round(e["mean_phase_switches"]  - b["mean_phase_switches"], 2),
            "delta_switches_pct":   round(pct(e["mean_phase_switches"],  b["mean_phase_switches"]), 2),
            "emission_helps_co2":   e["mean_co2_mg_per_veh"] < b["mean_co2_mg_per_veh"],
            "emission_helps_travel": e["mean_travel_s"]      < b["mean_travel_s"],
        })
    df_delta = pd.DataFrame(delta_records)

    # Write combined narrative CSV (two blocks stacked, with a blank separator).
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# Per-volume, per-method: the WHAT (absolute numbers)"])
        writer.writerow([])
        writer.writerow(list(df_rows.columns))
        for _, r in df_rows.iterrows():
            writer.writerow(list(r.values))
        writer.writerow([])
        writer.writerow(["# Emission-based vs Baseline deltas at each volume: the SO WHAT"])
        writer.writerow([])
        writer.writerow(list(df_delta.columns))
        for _, r in df_delta.iterrows():
            writer.writerow(list(r.values))

    # -------------------- plot --------------------
    vols = sorted(df_rows["volume"].unique())

    def series(metric, method):
        vals = []
        for v in vols:
            sub = df_rows[(df_rows["volume"] == v) & (df_rows["method"] == method)]
            vals.append(sub.iloc[0][metric] if not sub.empty else np.nan)
        return np.array(vals)

    fig, axes = plt.subplots(2, 2, figsize=(24, 12))
    fig.suptitle(
        "Pipeline B narrative: how demand turns wait-time, trip-duration,\n"
        "phase-switches and CO2 against each other\n"
        "(solid = Baseline, dashed = Emission-Based, 10 seeds per point)",
        fontsize=14, y=0.99)

    # 1) Wait time
    ax = axes[0, 0]
    ax.plot(vols, series("mean_avg_wait_s",      "baseline"),       "o-",  color="#1f77b4", label="Baseline, in-box wait (s)")
    ax.plot(vols, series("mean_avg_wait_s",      "emission_based"), "o--", color="#1f77b4", label="Emission-Based, in-box wait (s)")
    ax.plot(vols, series("mean_fulltrip_wait_s", "baseline"),       "s-",  color="#ff7f0e", label="Baseline, full-trip wait (s)")
    ax.plot(vols, series("mean_fulltrip_wait_s", "emission_based"), "s--", color="#ff7f0e", label="Emission-Based, full-trip wait (s)")
    ax.set_xlabel("Demand (vehicles injected, 1000..7000 veh/hr)")
    ax.set_ylabel("Average wait per vehicle (s)")
    ax.set_title("(1) Wait time — in-box vs full-trip")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    # 2) Travel time
    ax = axes[0, 1]
    ax.plot(vols, series("mean_travel_s", "baseline"),       "o-",  color="#2ca02c", label="Baseline")
    ax.plot(vols, series("mean_travel_s", "emission_based"), "o--", color="#d62728", label="Emission-Based")
    ax.axvspan(3100, 4050, color="red", alpha=0.07,
               label="Breakdown regime (V≥3200)")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Average trip duration (s)")
    ax.set_title("(2) Trip duration — the non-linear cliff at V3200")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 3) Phase switches
    ax = axes[1, 0]
    ax.plot(vols, series("mean_phase_switches", "baseline"),       "o-",  color="#9467bd", label="Baseline")
    ax.plot(vols, series("mean_phase_switches", "emission_based"), "o--", color="#8c564b", label="Emission-Based")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Phase switches per simulation")
    ax.set_title("(3) Phase switches — emission-based holds longer,\n"
                 "and in gridlock BOTH drop")
    ax.grid(alpha=0.3); ax.legend()

    # 4) Delta plot: emission helps/hurts
    ax = axes[1, 1]
    d_co2 = df_delta["delta_co2_pct"].values
    d_trv = df_delta["delta_travel_pct"].values
    d_wait = df_delta["delta_wait_pct"].values
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(df_delta["volume"], d_co2,  "o-", color="#C62828", label="ΔCO2/veh  (%)")
    ax.plot(df_delta["volume"], d_trv,  "s-", color="#2E7D32", label="ΔTravel   (%)")
    ax.plot(df_delta["volume"], d_wait, "^-", color="#1565C0", label="ΔWait     (%)")
    ax.fill_between(df_delta["volume"], 0, d_co2,
                    where=(d_co2 < 0), interpolate=True, alpha=0.12, color="#C62828")
    ax.set_xlabel("Demand (veh/hr)")
    ax.set_ylabel("Δ% (Emission-Based vs Baseline)")
    ax.set_title("(4) Does emission-based help? — "
                 "negative = yes, positive = hurts")
    ax.grid(alpha=0.3); ax.legend()

    # Rotate x-ticks on all panels for dense x-axis
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    # 5) Annotate story beats across all panels
    for ax in axes.flat:
        ax.axvline(2400, color="gray", linestyle=":", alpha=0.6)
        ax.axvline(3000, color="gray", linestyle=":", alpha=0.6)
        ax.axvline(3400, color="gray", linestyle=":", alpha=0.6)

    fig.text(0.5, 0.01,
             "Regime bands (gray dotted):  V≤2400 free-flow · V2600-3000 near-saturation · "
             "V3200-3400 breakdown onset · V≥3600 gridlock. "
             "Units: wait/travel in seconds, CO2 in mg/vehicle, phase switches in count/simulation.",
             ha="center", fontsize=9, color="#424242", style="italic")
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98,
                        hspace=0.30, wspace=0.20)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[done] {OUT_CSV}  ({OUT_CSV.stat().st_size} bytes)")
    print(f"[done] {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

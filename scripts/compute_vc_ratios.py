#!/usr/bin/env python3
"""
Compute V/C ratios per volume from the multi-seed aggregate CSV
produced by parse_capacity_results.py.

V/C semantics (see the project notes and HCM definition of "V"):
  V = average demand arrival rate over the analysis window
  C = service capacity (veh/cycle)
  V/C = V / C  -> the standard scenario-loading ratio

The *semantically correct* V/C for Paper 1 is:
    vc_percent = mean_avg_arrived_per_cycle / capacity * 100
This is what the professor's "V2400 ~ 31.5%" sanity check was based on.
Values: undersaturated (<60%), moderate (60-85%), near-capacity (85-100%),
oversaturated (>100%).

A secondary diagnostic column, peak_vc_percent, is also written:
    peak_vc_percent = mean_max_arrived_per_cycle / capacity * 100
This is NOT a traditional V/C ratio. It tells you how full the worst
single cycle was, averaged across seeds. By construction it sits near
100% at the critical volume (because capacity is defined from that
same mean_max series), so it's really a saturation-proximity
indicator, not a loading measure.

Inputs
------
data/capacity_analysis/summary/
  discharge_by_volume.csv    (aggregate, one row per volume; written by
                              the multi-seed parse_capacity_results.py)
  capacity_summary.json

Output
------
data/capacity_analysis/summary/vc_ratios.csv
  volume, capacity_veh_per_cycle, n_seeds,
  mean_avg_arrived_per_cycle, std_avg_arrived_per_cycle,
  mean_max_arrived_per_cycle, std_max_arrived_per_cycle,
  vc_percent, peak_vc_percent
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from capacity.capacity_math import vc_ratio


def _f(row: dict, *candidates: str) -> float:
    """Read the first present column from `candidates` as float.

    Backward-compat: old (single-seed) CSVs had avg_arrived_per_cycle
    and max_arrived_per_cycle. New (multi-seed) CSVs have mean_avg_
    and mean_max_ prefixes.
    """
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return float(row[c])
    raise KeyError(f"None of {candidates} found in row keys={list(row.keys())}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    summary_dir = project_root / "data" / "capacity_analysis" / "summary"
    discharge_csv = summary_dir / "discharge_by_volume.csv"
    cap_json = summary_dir / "capacity_summary.json"

    if not discharge_csv.exists() or not cap_json.exists():
        raise RuntimeError(
            "Missing discharge_by_volume.csv or capacity_summary.json. "
            "Run parse_capacity_results.py first."
        )

    cap = json.loads(cap_json.read_text(encoding="utf-8"))
    capacity = int(cap["capacity_veh_per_cycle"])
    n_seeds_by_volume = cap.get("n_seeds_by_volume", {})

    rows = []
    with open(discharge_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vol = int(row["volume"])
            mean_avg = _f(row, "mean_avg_arrived_per_cycle",
                          "avg_arrived_per_cycle")
            std_avg = _f(row, "std_avg_arrived_per_cycle") if \
                "std_avg_arrived_per_cycle" in row else 0.0
            mean_max = _f(row, "mean_max_arrived_per_cycle",
                          "max_arrived_per_cycle")
            std_max = _f(row, "std_max_arrived_per_cycle") if \
                "std_max_arrived_per_cycle" in row else 0.0

            # n_seeds: prefer the per-row column, fall back to the json map.
            if "n_seeds" in row and row["n_seeds"] not in (None, ""):
                n_seeds = int(row["n_seeds"])
            else:
                n_seeds = int(n_seeds_by_volume.get(str(vol),
                             n_seeds_by_volume.get(vol, 1)))

            rows.append({
                "volume": vol,
                "capacity_veh_per_cycle": capacity,
                "n_seeds": n_seeds,
                "mean_avg_arrived_per_cycle": round(mean_avg, 4),
                "std_avg_arrived_per_cycle": round(std_avg, 4),
                "mean_max_arrived_per_cycle": round(mean_max, 4),
                "std_max_arrived_per_cycle": round(std_max, 4),
                # Primary V/C (standard, demand-based).
                "vc_percent": vc_ratio(mean_avg, capacity),
                # Secondary diagnostic (peak-cycle utilisation).
                "peak_vc_percent": vc_ratio(mean_max, capacity),
            })

    out_csv = summary_dir / "vc_ratios.csv"
    fieldnames = [
        "volume", "capacity_veh_per_cycle", "n_seeds",
        "mean_avg_arrived_per_cycle", "std_avg_arrived_per_cycle",
        "mean_max_arrived_per_cycle", "std_max_arrived_per_cycle",
        "vc_percent", "peak_vc_percent",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["volume"]))

    # Print a preview so you can immediately verify the values look right.
    print("V/C ratios written:")
    print(f"  {out_csv}")
    print()
    hdr = (f"  {'Demand':>7}  {'n_sd':>4}  "
           f"{'mean_avg':>9}  {'mean_max':>9}  "
           f"{'V/C %':>7}  {'peak V/C %':>10}")
    sep = (f"  {'-'*7}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*10}")
    print(hdr)
    print(sep)
    for row in sorted(rows, key=lambda x: x["volume"]):
        print(
            f"  {row['volume']:>7}  {row['n_seeds']:>4}  "
            f"{row['mean_avg_arrived_per_cycle']:>9.3f}  "
            f"{row['mean_max_arrived_per_cycle']:>9.3f}  "
            f"{row['vc_percent']:>7.2f}  "
            f"{row['peak_vc_percent']:>10.2f}"
        )
    print()
    print(f"Capacity used: {capacity} veh/cycle "
          f"(from capacity_summary.json; derived as the ceiling of "
          f"mean_max_arrived_per_cycle across the sweep).")
    print()
    print("Columns:")
    print("  vc_percent       = mean_avg_arrived_per_cycle / capacity "
          "(standard HCM V/C; demand-based loading ratio)")
    print("  peak_vc_percent  = mean_max_arrived_per_cycle / capacity "
          "(peak-cycle utilisation; diagnostic only)")
    print()
    print("Next: python scripts/plot_capacity_analysis.py")


if __name__ == "__main__":
    main()

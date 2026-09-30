#!/usr/bin/env python3
"""
Pipeline A (capacity sweep) — multi-seed parser.

What changed (Pipeline A, publication-grade upgrade)
----------------------------------------------------
The parser now expects the multi-seed folder layout produced by
run_capacity_experiment.py:

    data/capacity_analysis/volumes/V####/seed_<N>/summary.xml

For each volume V####:
  1. Discover every seed_*/summary.xml under it.
  2. Parse each seed's summary with the same Fix 2 (demand-window bound)
     and Fix 3-aware warmup logic as before.
  3. Compute per-seed avg_arrived_per_cycle and max_arrived_per_cycle.
  4. Aggregate across seeds: mean and std of each metric.
  5. Use mean(max_arrived_per_cycle) across seeds as the volume's
     discharge number for the plateau detector.

What changed (plateau-confirmation upgrade, April 2026)
-------------------------------------------------------
  6. --high-v-plateau-threshold-pct (default 0.5): tighter threshold
     applied to volumes >= --high-v-volume-cutoff (default 5000) where
     finer 100-veh/hr steps and 10 seeds lower the noise floor.
  7. --high-v-volume-cutoff: boundary between standard (1.0%) and
     tighter (0.5%) plateau detection.
  8. Both new params are recorded in capacity_summary.json.

Backward compatibility
----------------------
If a volume folder contains a flat summary.xml (old single-seed layout)
and has no seed_*/ sub-folders, the parser falls back to treating that
one file as a single seed so old data still processes cleanly.

Outputs
-------
data/capacity_analysis/summary/
  discharge_by_volume_per_seed.csv   long-format, one row per (volume, seed)
  discharge_by_volume.csv            aggregate, one row per volume
  capacity_summary.json              capacity + plateau-detection result
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from capacity.summary_parser import (
    parse_cycle_length_from_net,
    parse_summary_step_series,
    compute_arrivals_per_cycle,
)
from capacity.capacity_math import detect_capacity_plateau


def discover_seed_summaries(vdir: Path) -> list[tuple[str, Path]]:
    """
    Return [(seed_label, summary_path), ...] for a volume folder.

    Prefers the new layout vdir/seed_*/summary.xml. Falls back to a
    flat vdir/summary.xml labelled 'flat' for backward compatibility.
    """
    pairs: list[tuple[str, Path]] = []
    for seed_dir in sorted(vdir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        sp = seed_dir / "summary.xml"
        if sp.exists():
            pairs.append((seed_dir.name[len("seed_"):], sp))
    if not pairs:
        flat = vdir / "summary.xml"
        if flat.exists():
            pairs.append(("flat", flat))
    return pairs


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std. std is 0.0 when n < 2 (avoids ValueError)."""
    if not values:
        return (float("nan"), float("nan"))
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) >= 2 else 0.0
    return (m, s)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pipeline A parser — aggregates per-seed summary.xml files "
            "into discharge CSVs and detects the capacity plateau."
        )
    )
    ap.add_argument("--tl-id", type=str, default="center")
    # Fix 2 (part A - warmup): Drop the first N cycles to skip the flow-startup
    # surge. All 72 <flow> elements have begin="0", so SUMO schedules ~72
    # initial-surge vehicles at t=0 that all clear the intersection within
    # the first ~2 cycles. Those cycles are NOT representative of
    # steady-state arrivals.
    ap.add_argument("--warmup-cycles", type=int, default=3,
                    help="Drop the first N cycles to skip the flow-startup "
                         "injection surge. Default 3.")
    ap.add_argument(
        "--plateau-threshold-pct",
        type=float,
        default=1.0,
        help=(
            "Step-increase threshold (%%) for the plateau detector on "
            "low-V volumes (below --high-v-volume-cutoff). A step smaller "
            "than this is treated as a potential plateau. Default 1.0."
        ),
    )
    ap.add_argument(
        "--high-v-plateau-threshold-pct",
        type=float,
        default=0.5,
        help=(
            "Tighter step-increase threshold (%%) applied to volumes >= "
            "--high-v-volume-cutoff, where 100-veh/hr steps and 10 seeds "
            "provide lower noise. Default 0.5 (half the standard threshold)."
        ),
    )
    ap.add_argument(
        "--high-v-volume-cutoff",
        type=int,
        default=5000,
        help=(
            "Volume (veh/hr) above which --high-v-plateau-threshold-pct is "
            "used instead of --plateau-threshold-pct. Default 5000."
        ),
    )
    # Fix 2 (part B - demand-window bound): Drop cycles whose window extends
    # beyond the demand injection end. Default 3600 matches <flow end="3600">.
    ap.add_argument("--demand-end-seconds", type=float, default=3600.0,
                    help="Drop cycle bins whose window extends beyond this "
                         "demand-injection end time. Set to 0 to disable.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    net = project_root / "config" / "traffic.net.xml"
    volumes_dir = project_root / "data" / "capacity_analysis" / "volumes"
    summary_dir = project_root / "data" / "capacity_analysis" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    cycle = parse_cycle_length_from_net(net, tl_id=args.tl_id)
    demand_end = args.demand_end_seconds if args.demand_end_seconds > 0 else None

    per_seed_rows: list[dict] = []
    agg_rows: list[dict] = []
    volumes: list[int] = []
    mean_max_disc: list[float] = []

    for vdir in sorted(volumes_dir.glob("V*")):
        if not (vdir.is_dir() and vdir.name.startswith("V")
                and vdir.name[1:].isdigit()):
            continue
        vol = int(vdir.name[1:])
        pairs = discover_seed_summaries(vdir)
        if not pairs:
            print(f"  [skip] {vdir.name}: no seed_*/summary.xml or flat "
                  f"summary.xml found")
            continue

        avg_vals: list[float] = []
        max_vals: list[float] = []
        total_vals: list[int] = []
        sim_end_vals: list[float] = []

        for seed_label, summary_path in pairs:
            series = parse_summary_step_series(summary_path)
            total_arr, avg_pc, max_pc, sim_end = compute_arrivals_per_cycle(
                series,
                cycle_seconds=cycle,
                warmup_cycles=args.warmup_cycles,
                demand_end_seconds=demand_end,
            )
            per_seed_rows.append({
                "volume": vol,
                "seed": seed_label,
                "cycle_seconds": cycle,
                "sim_end_time": sim_end,
                "total_arrived": total_arr,
                "avg_arrived_per_cycle": avg_pc,
                "max_arrived_per_cycle": max_pc,
                "summary_file": str(summary_path),
            })
            avg_vals.append(avg_pc)
            max_vals.append(max_pc)
            total_vals.append(total_arr)
            sim_end_vals.append(sim_end)

        mean_avg, std_avg = _mean_std(avg_vals)
        mean_max, std_max = _mean_std(max_vals)
        mean_total = statistics.fmean(total_vals) if total_vals else float("nan")
        mean_sim_end = statistics.fmean(sim_end_vals) if sim_end_vals else float("nan")

        agg_rows.append({
            "volume": vol,
            "n_seeds": len(pairs),
            "cycle_seconds": cycle,
            "mean_sim_end_time": mean_sim_end,
            "mean_total_arrived": mean_total,
            "mean_avg_arrived_per_cycle": mean_avg,
            "std_avg_arrived_per_cycle": std_avg,
            "mean_max_arrived_per_cycle": mean_max,
            "std_max_arrived_per_cycle": std_max,
        })
        volumes.append(vol)
        mean_max_disc.append(mean_max)

    if not agg_rows:
        raise RuntimeError("No results found in " + str(volumes_dir))

    # --- per-seed long-format CSV ------------------------------------
    per_seed_csv = summary_dir / "discharge_by_volume_per_seed.csv"
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(per_seed_rows,
                           key=lambda r: (r["volume"], str(r["seed"]))))

    # --- aggregate CSV (one row per volume) --------------------------
    agg_csv = summary_dir / "discharge_by_volume.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(agg_rows, key=lambda r: r["volume"]))

    # --- plateau detection on mean(max_arrived_per_cycle) -----------
    cap = detect_capacity_plateau(
        volumes=volumes,
        max_discharges=mean_max_disc,
        plateau_threshold_pct=args.plateau_threshold_pct,
        high_v_plateau_threshold_pct=args.high_v_plateau_threshold_pct,
        high_v_volume_cutoff=args.high_v_volume_cutoff,
    )

    out_json = summary_dir / "capacity_summary.json"
    n_seeds_by_volume = {r["volume"]: r["n_seeds"] for r in agg_rows}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tl_id": args.tl_id,
            "cycle_seconds": cycle,
            "capacity_veh_per_cycle": cap.capacity_veh_per_cycle,
            "critical_volume": cap.critical_volume,
            "plateau_threshold_pct": cap.plateau_threshold_pct,
            "high_v_plateau_threshold_pct": args.high_v_plateau_threshold_pct,
            "high_v_volume_cutoff": args.high_v_volume_cutoff,
            "notes": cap.notes,
            "warmup_cycles": args.warmup_cycles,
            "demand_end_seconds": args.demand_end_seconds,
            "aggregation": (
                "mean across seeds of max_arrived_per_cycle; "
                "see discharge_by_volume_per_seed.csv for per-seed detail"
            ),
            "n_seeds_by_volume": n_seeds_by_volume,
        }, f, indent=2)

    print("Parsed discharge/cycle metrics:")
    print("  " + str(per_seed_csv) + "  (per-seed long format)")
    print("  " + str(agg_csv) + "  (aggregate, mean+std across seeds)")
    print("Capacity summary:")
    print("  " + str(out_json))
    print()
    print("Next: python scripts/compute_vc_ratios.py")


if __name__ == "__main__":
    main()

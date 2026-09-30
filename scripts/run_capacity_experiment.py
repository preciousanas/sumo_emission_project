#!/usr/bin/env python3
"""
Pipeline A (capacity sweep) — uniform multi-seed runner.

Volume range  : V1000–V7000 in 100-veh/hr steps (61 volumes).
Seeds per vol : 10 seeds [42, 10, 25, 50, 75, 13, 17, 37, 5, 77] — identical
                for every volume so comparisons across the full range are
                fully balanced.
Total sims    : 61 volumes × 10 seeds = 610 simulations.

Folder structure produced
-------------------------
data/capacity_analysis/volumes/
  V1000/
    traffic_V1000.rou.xml        (scaled route file — seed-invariant)
    traffic_V1000.sumocfg        (SUMO config — seed-invariant)
    seed_42/   seed_10/   seed_25/  seed_50/  seed_75/
    seed_13/   seed_17/   seed_37/  seed_5/   seed_77/
      summary.xml  tripinfo.xml  statistics.xml
  V1100/ ... V7000/   (same layout)

Usage
-----
# Full sweep (default — 610 simulations):
    python scripts/run_capacity_experiment.py

# Re-run specific volumes only:
    python scripts/run_capacity_experiment.py --volumes 5000 5100 5200

# Custom seeds:
    python scripts/run_capacity_experiment.py --seeds 42 10 25

# Custom SUMO binary:
    python scripts/run_capacity_experiment.py --sumo-binary sumo-gui
"""
from __future__ import annotations
import argparse
import subprocess
import time
from pathlib import Path

from capacity.route_scaler import scale_route_file
from capacity.sumo_config_builder import build_sumocfg_for_run

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_VOLUMES = list(range(1000, 7001, 100))          # 61 volumes
_DEFAULT_SEEDS   = [42, 10, 25, 50, 75, 13, 17, 37, 5, 77]  # 10 seeds


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Pipeline A — capacity sweep. "
            "All volumes use the same seed list for a fully balanced design. "
            "Default: 61 volumes (V1000–V7000, 100-veh/hr steps) × 10 seeds "
            "= 610 simulations."
        )
    )
    ap.add_argument(
        "--volumes",
        nargs="+",
        type=int,
        default=_DEFAULT_VOLUMES,
        help=(
            "Demand volumes (veh/hr) to sweep. "
            "Default: V1000–V7000 in 100-veh/hr steps (61 volumes). "
            "Override with a space-separated list to re-run specific volumes."
        ),
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=_DEFAULT_SEEDS,
        help=(
            "Random seeds applied to every volume. "
            "Default: [42, 10, 25, 50, 75, 13, 17, 37, 5, 77] (10 seeds)."
        ),
    )
    ap.add_argument("--sumo-binary", type=str, default="sumo")
    ap.add_argument("--begin", type=int, default=0)
    ap.add_argument(
        "--end",
        type=int,
        default=7200,
        help=(
            "Simulation end time (s). Default 7200 gives delayed insertions "
            "at high V/C extra time after the demand window closes at t=3600. "
            "The analysis window is bounded to [0, 3600] by summary_parser.py."
        ),
    )
    ap.add_argument("--step-length", type=float, default=1.0)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_dir   = project_root / "config"
    data_dir     = project_root / "data" / "capacity_analysis"
    volumes_dir  = data_dir / "volumes"

    net          = config_dir / "traffic.net.xml"
    base_rou     = config_dir / "traffic.rou.xml"
    base_sumocfg = config_dir / "traffic.sumocfg"

    grand_total = len(args.volumes) * len(args.seeds)
    grand_count = 0
    grand_start = time.time()

    print(
        f"\nPipeline A (capacity) — {len(args.volumes)} volumes × "
        f"{len(args.seeds)} seeds = {grand_total} simulations\n"
        f"  Volumes : V{min(args.volumes):04d}–V{max(args.volumes):04d} "
        f"(100-veh/hr steps)\n"
        f"  Seeds   : {args.seeds}\n"
    )

    for v in args.volumes:
        vol_label = f"V{v:04d}"
        out_dir   = volumes_dir / vol_label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Scaled route file and sumocfg are SEED-INVARIANT — build once per volume.
        scaled_rou = out_dir / f"traffic_{vol_label}.rou.xml"
        run_cfg    = out_dir / f"traffic_{vol_label}.sumocfg"

        rep = scale_route_file(base_rou, scaled_rou, v)
        build_sumocfg_for_run(
            base_sumocfg_path=base_sumocfg,
            output_sumocfg_path=run_cfg,
            net_file_path=net,
            route_file_path=scaled_rou,
            begin=args.begin,
            end=args.end,
            step_length=args.step_length,
        )

        print(
            f"\n[CAPACITY] {vol_label}  "
            f"base_est={rep.base_estimated_vehicles:.1f}, "
            f"factor={rep.scale_factor:.3f} ({rep.mode})"
        )

        for seed in args.seeds:
            grand_count += 1
            seed_dir = out_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            summary_out  = seed_dir / "summary.xml"
            tripinfo_out = seed_dir / "tripinfo.xml"
            stats_out    = seed_dir / "statistics.xml"

            cmd = [
                args.sumo_binary,
                "-c", str(run_cfg),
                "--seed", str(seed),
                "--no-step-log", "true",
                "--summary-output",   str(summary_out),
                "--tripinfo-output",  str(tripinfo_out),
                "--statistic-output", str(stats_out),
            ]

            print(
                f"  [{grand_count:3d}/{grand_total}] "
                f"{vol_label} seed={seed}  -> seed_{seed}/"
            )
            subprocess.run(cmd, check=True)

    elapsed = time.time() - grand_start
    print(f"\nPipeline A complete: {grand_total} simulations in {elapsed:.1f}s")
    print("Next: python scripts/parse_capacity_results.py")


if __name__ == "__main__":
    main()

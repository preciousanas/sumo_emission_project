"""
Standalone cross-volume summary regenerator.

Reads per-seed result CSVs and tripinfo XMLs directly, streams the large
result CSVs in chunks, and rebuilds visuals/cross_volume/cross_volume_summary.csv.

Usage:
    python scripts/regenerate_cross_volume.py
    python scripts/regenerate_cross_volume.py --only 1000,1200
    python scripts/regenerate_cross_volume.py --only 2800,3000,3200 --append
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METRICS = [
    ("throughput",          "veh/h"),
    ("avg_wait_time",       "s"),
    ("avg_fulltrip_wait",   "s"),
    ("avg_travel_time",     "s"),
    ("avg_speed",           "m/s"),
    ("avg_co2_per_vehicle", "mg"),
]

CHUNKSIZE = 200_000


def _coerce_inbox(series):
    return (series.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False,
                  "yes": True, "no": False})
            .fillna(False))


def process_seed_csv(csv_path: Path) -> Dict[str, float]:
    veh_ids_global = set()
    sim_t_min = np.inf
    sim_t_max = -np.inf
    inbox_veh_ids = set()
    inbox_veh_tmin: Dict[str, float] = {}
    inbox_veh_tmax: Dict[str, float] = {}
    inbox_veh_wmax: Dict[str, float] = {}
    inbox_speed_sum = 0.0
    inbox_speed_n = 0
    inbox_co2_sum = 0.0
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=["sim_time", "veh_id", "speed", "wait_time",
                     "co2_emission", "in_box"],
            chunksize=CHUNKSIZE,
            dtype={"veh_id": str},
            low_memory=False,
        )
    except Exception as e:
        print(f"    [warn] cannot open {csv_path.name}: {e}")
        return _zero_row()
    for chunk in reader:
        chunk = chunk[~chunk["sim_time"].astype(str).str.startswith("#", na=False)]
        if chunk.empty:
            continue
        chunk["sim_time"]     = pd.to_numeric(chunk["sim_time"],     errors="coerce")
        chunk["speed"]        = pd.to_numeric(chunk["speed"],        errors="coerce")
        chunk["wait_time"]    = pd.to_numeric(chunk["wait_time"],    errors="coerce")
        chunk["co2_emission"] = pd.to_numeric(chunk["co2_emission"], errors="coerce")
        chunk["in_box"]       = _coerce_inbox(chunk["in_box"])
        chunk = chunk.dropna(subset=["sim_time", "veh_id"])
        if chunk.empty:
            continue
        veh_ids_global.update(chunk["veh_id"].unique().tolist())
        smin = float(chunk["sim_time"].min())
        smax = float(chunk["sim_time"].max())
        if smin < sim_t_min: sim_t_min = smin
        if smax > sim_t_max: sim_t_max = smax
        ib = chunk[chunk["in_box"] == True]
        if ib.empty:
            continue
        inbox_veh_ids.update(ib["veh_id"].unique().tolist())
        sp = ib["speed"].dropna()
        if not sp.empty:
            inbox_speed_sum += float(sp.sum())
            inbox_speed_n   += int(sp.size)
        co = ib["co2_emission"].dropna()
        if not co.empty:
            inbox_co2_sum += float(co.sum())
        grp = ib.groupby("veh_id")
        tmin_c = grp["sim_time"].min().to_dict()
        tmax_c = grp["sim_time"].max().to_dict()
        wmax_c = grp["wait_time"].max().to_dict()
        for vid, t in tmin_c.items():
            prev = inbox_veh_tmin.get(vid, np.inf)
            if t < prev: inbox_veh_tmin[vid] = t
        for vid, t in tmax_c.items():
            prev = inbox_veh_tmax.get(vid, -np.inf)
            if t > prev: inbox_veh_tmax[vid] = t
        for vid, w in wmax_c.items():
            try:
                w = float(w)
            except Exception:
                continue
            if np.isnan(w):
                continue
            prev = inbox_veh_wmax.get(vid, -np.inf)
            if w > prev: inbox_veh_wmax[vid] = w
    dur_h = (sim_t_max - sim_t_min) / 3600.0 if sim_t_max > sim_t_min else 0.0
    throughput = (len(veh_ids_global) / dur_h) if dur_h > 0 else 0.0
    avg_co2 = (inbox_co2_sum / len(inbox_veh_ids)) if inbox_veh_ids else 0.0
    avg_wait = float(np.mean(list(inbox_veh_wmax.values()))) if inbox_veh_wmax else 0.0
    travel_times = [inbox_veh_tmax[v] - inbox_veh_tmin[v]
                    for v in inbox_veh_tmin if v in inbox_veh_tmax]
    avg_travel = float(np.mean(travel_times)) if travel_times else 0.0
    avg_speed = (inbox_speed_sum / inbox_speed_n) if inbox_speed_n > 0 else 0.0
    return {
        "throughput":          throughput,
        "avg_wait_time":       avg_wait,
        "avg_travel_time":     avg_travel,
        "avg_speed":           avg_speed,
        "avg_co2_per_vehicle": avg_co2,
    }


def _zero_row():
    return {"throughput": 0.0, "avg_wait_time": 0.0, "avg_travel_time": 0.0,
            "avg_speed": 0.0, "avg_co2_per_vehicle": 0.0}


def read_tripinfo_waiting(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return 0.0
    waits = []
    for tr in tree.getroot().findall("tripinfo"):
        a = tr.attrib
        try:
            waits.append(float(a.get("waitingTime", 0.0)))
        except ValueError:
            continue
    return float(np.mean(waits)) if waits else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of volumes (e.g. 1000,1200). Default: all.")
    parser.add_argument("--append", action="store_true",
                        help="Append/update rows in existing CSV instead of overwrite.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data"
    out_dir = root / "visuals" / "cross_volume"
    out_dir.mkdir(parents=True, exist_ok=True)
    only = None
    if args.only:
        only = set(int(x.strip()) for x in args.only.split(",") if x.strip())
    volume_dirs = sorted([
        d for d in os.listdir(data_root)
        if re.match(r"^V\d{4}$", d) and (data_root / d).is_dir()
        and (only is None or int(d[1:]) in only)
    ])
    if not volume_dirs:
        print(f"[ERROR] No V#### folders match filter.")
        return 1
    print(f"[regenerate_cross_volume] Processing: {volume_dirs}")
    records: List[Dict] = []
    for vol_dir in volume_dirs:
        vol_int = int(vol_dir[1:])
        vpath = data_root / vol_dir
        print(f"\n  Volume {vol_int} ...")
        for method in ("baseline", "emission_based"):
            mdir = vpath / method
            if not mdir.exists():
                print(f"    [skip] {mdir} missing")
                continue
            seed_csvs = sorted(mdir.glob("result_seed_*.csv"))
            if not seed_csvs:
                print(f"    [skip] no result_seed_*.csv in {mdir}")
                continue
            per_seed: Dict[str, List[float]] = {k: [] for k, _ in METRICS}
            for sc in seed_csvs:
                m = re.match(r"result_seed_(\d+)\.csv", sc.name)
                if not m:
                    continue
                seed = int(m.group(1))
                tp = mdir / f"tripinfo_seed_{seed}.xml"
                row = process_seed_csv(sc)
                ft_wait = read_tripinfo_waiting(tp)
                per_seed["throughput"].append(row["throughput"])
                per_seed["avg_wait_time"].append(row["avg_wait_time"])
                per_seed["avg_fulltrip_wait"].append(ft_wait)
                per_seed["avg_travel_time"].append(row["avg_travel_time"])
                per_seed["avg_speed"].append(row["avg_speed"])
                per_seed["avg_co2_per_vehicle"].append(row["avg_co2_per_vehicle"])
            for k, _unit in METRICS:
                vals = per_seed[k]
                mu = float(np.mean(vals)) if vals else 0.0
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                records.append({"volume": vol_int, "method": method,
                                "metric": k, "mean": mu, "std": sd})
            print(f"    [{method}] seeds={len(seed_csvs)}  "
                  f"CO2/veh={np.mean(per_seed['avg_co2_per_vehicle']):.0f} mg  "
                  f"travel={np.mean(per_seed['avg_travel_time']):.1f} s  "
                  f"wait(in-box)={np.mean(per_seed['avg_wait_time']):.1f} s")
    csv_path = out_dir / "cross_volume_summary.csv"
    df_new = pd.DataFrame(records, columns=["volume", "method", "metric", "mean", "std"])
    if args.append and csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path)
        except Exception:
            df_old = pd.DataFrame(columns=df_new.columns)
        if not df_new.empty:
            df_old = df_old[~df_old["volume"].isin(df_new["volume"].unique())]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out = df_out.sort_values(["volume", "method", "metric"]).reset_index(drop=True)
    df_out.to_csv(csv_path, index=False)
    print(f"\n[done] Wrote {csv_path}  ({csv_path.stat().st_size} bytes, "
          f"{len(df_out)} rows across {df_out['volume'].nunique()} volumes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

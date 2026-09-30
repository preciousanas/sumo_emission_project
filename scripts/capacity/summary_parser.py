from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional


@dataclass
class DischargeMetrics:
    volume: int
    cycle_seconds: int
    sim_end_time: float
    total_arrived: int
    max_arrived_per_cycle: int
    avg_arrived_per_cycle: float
    notes: str


def parse_cycle_length_from_net(net_path: Path, tl_id: str = "center") -> int:
    """
    Reads the static TLS program cycle length from traffic.net.xml.
    For 'center', your file shows 42+3+42+3 = 90 seconds.
    """
    tree = ET.parse(net_path)
    root = tree.getroot()
    tls = root.find(f".//tlLogic[@id='{tl_id}']")
    if tls is None:
        raise ValueError(f"Could not find tlLogic id='{tl_id}' in {net_path}")

    phases = tls.findall("phase")
    if not phases:
        raise ValueError(f"tlLogic id='{tl_id}' has no <phase> entries.")

    cycle = 0
    for p in phases:
        cycle += int(float(p.attrib["duration"]))
    return cycle


def parse_summary_step_series(summary_path: Path) -> List[Tuple[float, int]]:
    """
    Parse SUMO summary.xml and return list of (time, arrived_total) per step.
    SUMO summary output typically has: <step time="..." arrived="...">.
    """
    tree = ET.parse(summary_path)
    root = tree.getroot()

    series = []
    for step in root.findall(".//step"):
        t = float(step.attrib.get("time", "0"))
        arrived = int(float(step.attrib.get("arrived", "0")))
        series.append((t, arrived))

    if not series:
        raise ValueError(f"No <step> elements found in {summary_path}")
    return series


def compute_arrivals_per_cycle(
    step_series: List[Tuple[float, int]],
    cycle_seconds: int,
    warmup_cycles: int = 0,
    demand_end_seconds: Optional[float] = None,
) -> Tuple[int, float, int, float]:
    """
    Given step-wise cumulative 'arrived', compute:
      - total_arrived
      - avg_arrived_per_cycle
      - max_arrived_per_cycle
      - sim_end_time

    Fix 2 (demand-window bounding):
        If demand_end_seconds is provided, ONLY cycle bins whose entire
        window [c*cycle_seconds, (c+1)*cycle_seconds) lies within the
        active demand injection window [0, demand_end_seconds) are retained.
        This removes the sim-end cooldown artifact where leftover in-transit
        vehicles all arrive in the final cycles after demand injection ends,
        which previously pegged max_arrived_per_cycle at ~51 across V0200-V0800.
    """
    deltas = []
    prev_arr = step_series[0][1]
    for (t, arr) in step_series[1:]:
        deltas.append((t, max(0, arr - prev_arr)))
        prev_arr = arr

    sim_end = step_series[-1][0]
    total_arrived = step_series[-1][1]

    bins: Dict[int, int] = {}
    for t, d in deltas:
        c = int(t // cycle_seconds)
        bins[c] = bins.get(c, 0) + d

    if warmup_cycles > 0:
        bins = {k: v for k, v in bins.items() if k >= warmup_cycles}

    if demand_end_seconds is not None and demand_end_seconds > 0:
        max_full_cycles = int(demand_end_seconds // cycle_seconds)
        bins = {k: v for k, v in bins.items() if k < max_full_cycles}

    if not bins:
        return total_arrived, 0.0, 0, sim_end

    per_cycle = list(bins.values())
    max_pc = max(per_cycle)
    avg_pc = sum(per_cycle) / float(len(per_cycle))
    return total_arrived, avg_pc, max_pc, sim_end

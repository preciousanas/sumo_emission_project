from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class CapacityResult:
    capacity_veh_per_cycle: int
    critical_volume: int
    method: str
    plateau_threshold_pct: float
    notes: str


def detect_capacity_plateau(
    volumes,
    max_discharges,
    plateau_threshold_pct: float = 1.0,
    gate_fraction_of_ceiling: float = 0.60,
    confirm_steps: int = 2,
    proximity_to_max_pct: float = 10.0,
    high_v_plateau_threshold_pct: float = 0.5,
    high_v_volume_cutoff: int = 5000,
) -> CapacityResult:
    """
    Detect the demand volume at which increasing demand no longer meaningfully
    increases per-cycle discharge (i.e. the intersection is saturating).

    Fix 4 (noise-aware plateau, April 2026):
        1. proximity_to_max_pct (default 10%): the candidate plateau max
           must be within proximity_to_max_pct of the GLOBAL observed max.
        2. The confirm window is extended to scan ALL subsequent steps.
           If ANY later step rises above the effective threshold AND crosses
           the proximity band, the candidate is rejected as a dip.

    Fix 5 (volume-dependent threshold, April 2026):
        3. high_v_plateau_threshold_pct (default 0.5%): a tighter threshold
           applied to steps where the volume >= high_v_volume_cutoff (default
           5000 veh/hr). This is the range where 100-veh/hr steps and 10 seeds
           are used, so the noise floor is low enough to distinguish a genuine
           0.5% plateau from random variation.
           Steps below the cutoff use plateau_threshold_pct (1.0%).

    Capacity is taken as the maximum observed discharge per cycle across all
    volumes. critical_volume is the first volume where the confirmed plateau
    begins AND where the curve is close enough to the global max to qualify.

    Parameters
    ----------
    volumes : list[int]
        Demand volumes in ascending order.
    max_discharges : list[float]
        Mean-across-seeds max arrived per cycle, one entry per volume.
    plateau_threshold_pct : float
        Step-increase threshold (%) for low-V volumes (< high_v_volume_cutoff).
        Default 1.0.
    gate_fraction_of_ceiling : float
        Ignore steps below this fraction of the global max. Default 0.60.
    proximity_to_max_pct : float
        Candidate discharge must be within this % of the global max. Default 10.0.
    high_v_plateau_threshold_pct : float
        Tighter threshold (%) for volumes >= high_v_volume_cutoff. Default 0.5.
    high_v_volume_cutoff : int
        Volume (veh/hr) above which the tighter threshold is active. Default 5000.
    """
    if len(volumes) != len(max_discharges) or not volumes:
        raise ValueError("volumes and max_discharges must be same non-empty length")

    cap = max(max_discharges)
    crit = volumes[-1]
    gate = gate_fraction_of_ceiling * float(cap)
    proximity_floor = (1.0 - proximity_to_max_pct / 100.0) * float(cap)
    notes = (
        "No plateau detected above "
        f"{int(100 * gate_fraction_of_ceiling)}%-of-ceiling gate; "
        f"using max observed discharge as capacity "
        f"(proximity-to-max threshold: {proximity_to_max_pct:.1f}%)."
    )

    def _effective_threshold(vol: int) -> float:
        """Return the plateau threshold appropriate for this volume."""
        if vol >= high_v_volume_cutoff:
            return high_v_plateau_threshold_pct
        return plateau_threshold_pct

    n = len(volumes)
    for i in range(1, n):
        prev = max_discharges[i - 1]
        cur  = max_discharges[i]
        if prev <= 0:
            continue
        if cur < gate:
            continue
        if cur < proximity_floor:
            continue

        threshold_i = _effective_threshold(volumes[i])
        inc_pct = 100.0 * (cur - prev) / float(prev)
        if inc_pct >= threshold_i:
            continue

        confirmed = True
        reject_reason = ""
        for j in range(i + 1, n):
            pj = max_discharges[j - 1]
            cj = max_discharges[j]
            if pj <= 0:
                continue
            if cj < gate:
                continue
            threshold_j = _effective_threshold(volumes[j])
            inc_j = 100.0 * (cj - pj) / float(pj)
            if inc_j >= threshold_j and cj >= proximity_floor:
                confirmed = False
                reject_reason = (
                    f" rejected: later step {volumes[j-1]}->{volumes[j]} rose "
                    f"{inc_j:+.2f}% (>= {threshold_j}%) to "
                    f"{cj:.2f} (>= proximity floor {proximity_floor:.2f})."
                )
                break

        if confirmed:
            crit = volumes[i]
            which = "high-V threshold" if volumes[i] >= high_v_volume_cutoff else "standard threshold"
            notes = (
                f"Plateau confirmed at volume={crit}: step increase "
                f"{inc_pct:.3f}% < {threshold_i}% ({which}), "
                f"current value {cur:.2f} within {proximity_to_max_pct:.1f}% "
                f"of global max {cap} (proximity floor {proximity_floor:.2f}), "
                f"and no later recovery above threshold."
            )
            break
        else:
            notes = (
                "No plateau confirmed; last candidate was "
                f"volume={volumes[i]} (inc={inc_pct:+.3f}%, "
                f"threshold={_effective_threshold(volumes[i])}%).{reject_reason}"
            )

    return CapacityResult(
        capacity_veh_per_cycle=int(cap),
        critical_volume=int(crit),
        method=(
            "plateau(max discharge per cycle, gated + confirmed + "
            "proximity-bounded + volume-dependent threshold)"
        ),
        plateau_threshold_pct=float(plateau_threshold_pct),
        notes=notes,
    )


def vc_ratio(volume: int, capacity: int) -> float:
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    return 100.0 * float(volume) / float(capacity)

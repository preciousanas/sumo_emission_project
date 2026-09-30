#!/usr/bin/env python3
"""
welch_ttest.py -- Statistical significance testing for CO2 savings.

Runs both Welch's t-test (unequal-variance) and a paired t-test
comparing per-seed in-box CO2/vehicle between baseline and emission_based
controllers, for all four representative traffic volumes.

Tests are run at two thresholds per volume:
  1. thr=50 mg/s  -- default reference threshold (data/ folder)
  2. Optimal threshold per regime -- from fine-grid threshold search

Metric: in-box CO2 (in_box==True rows) / total vehicle count.
This is consistent with the main analysis pipeline.

Output saved to:
  experiments/threshold_search/results/welch_ttest_results.txt
"""

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT / 'data'
EXP_ROOT     = PROJECT_ROOT / 'experiments' / 'threshold_search'
OUT_FILE     = EXP_ROOT / 'results' / 'welch_ttest_results.txt'

VOLUMES = [1500, 2900, 3300, 5000]
REGIME_LABELS = {
    1500: 'Free-flow',
    2900: 'Near-saturation',
    3300: 'Breakdown onset',
    5000: 'Gridlock',
}
OPTIMAL_THR = {1500: 41, 2900: 42, 3300: 135, 5000: 138}


# ---------------------------------------------------------------------------
# Regularised incomplete beta -- Numerical Recipes (betacf + betai)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for betainc, via modified Lentz (Numerical Recipes)."""
    TINY, EPS, MAX_IT = 1e-30, 3e-7, 200
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY: d = TINY
    d = 1.0 / d
    h = d
    for m in range(1, MAX_IT + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d;  d = TINY if abs(d) < TINY else d;  d = 1.0 / d
        c = 1.0 + aa / c;  c = TINY if abs(c) < TINY else c
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d;  d = TINY if abs(d) < TINY else d;  d = 1.0 / d
        c = 1.0 + aa / c;  c = TINY if abs(c) < TINY else c
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    # Use symmetry relation for better convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - betainc(b, a, 1.0 - x)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    return front * _betacf(a, b, x)


def t_pvalue(t_stat: float, df: float) -> float:
    """Two-sided p-value for t-distribution with given degrees of freedom."""
    # I_x(df/2, 0.5) with x = df/(df+t²) gives the two-sided p-value directly
    x = df / (df + t_stat * t_stat)
    return betainc(df / 2.0, 0.5, x)


# ---------------------------------------------------------------------------
# CSV parsing  --  in-box CO2
# ---------------------------------------------------------------------------

def parse_seed_csv(csv_path: Path):
    try:
        vehicle_ids = set()
        co2_sum = 0.0
        with open(csv_path, newline='', errors='replace') as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                if header is None:
                    header = [c.strip() for c in row]
                    continue
                if len(row) < len(header):
                    continue
                d = dict(zip(header, row))
                veh = d.get('veh_id', '')
                if not veh or veh.startswith('#'):
                    continue
                vehicle_ids.add(veh)
                if d.get('in_box', '').strip().lower() == 'true':
                    co2_sum += float(d.get('co2_emission', 0))
        n = len(vehicle_ids)
        return co2_sum / n if n > 0 else None
    except Exception as e:
        print(f"  [WARN] {csv_path.name}: {e}", file=sys.stderr)
        return None


def load_per_seed_co2(folder: Path) -> dict:
    results = {}
    if not folder.exists():
        return results
    for f in sorted(folder.glob('result_seed_*.csv')):
        m = re.search(r'result_seed_(\d+)\.csv', f.name)
        if not m:
            continue
        val = parse_seed_csv(f)
        if val is not None:
            results[int(m.group(1))] = val
    return results


def load_em_at_thr(vol: int, thr: int) -> dict:
    lbl = f'V{vol:04d}'
    if thr == 50:
        folder = DATA_ROOT / lbl / 'emission_based'
    else:
        folder = EXP_ROOT / 'data' / f'thr_{thr:03d}' / lbl / 'emission_based'
    return load_per_seed_co2(folder)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    na, nb   = len(a), len(b)
    ma, mb   = float(np.mean(a)), float(np.mean(b))
    s2a, s2b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(s2a / na + s2b / nb)
    if se == 0:
        return {'t': 0.0, 'df': 0.0, 'p': 1.0}
    t     = (ma - mb) / se
    num   = (s2a / na + s2b / nb) ** 2
    denom = (s2a / na) ** 2 / (na - 1) + (s2b / nb) ** 2 / (nb - 1)
    df    = num / denom if denom > 0 else 1.0
    return {'t': t, 'df': df, 'p': t_pvalue(t, df)}


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    d  = a - b
    n  = len(d)
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(n)
    if se == 0:
        return {'t': 0.0, 'df': n - 1, 'p': 1.0}
    t  = md / se
    df = n - 1
    return {'t': t, 'df': df, 'p': t_pvalue(t, df)}


def sig_label(p):
    if   p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    else:           return "n.s."


def effect_label(d):
    ad = abs(d)
    if   ad >= 0.8: return "large"
    elif ad >= 0.5: return "medium"
    elif ad >= 0.2: return "small"
    else:           return "negligible"


# ---------------------------------------------------------------------------
# Per-threshold test block builder
# ---------------------------------------------------------------------------

def _test_block(thr: int, tag: str, bl_seeds: dict, em_seeds: dict) -> list:
    common = sorted(set(bl_seeds) & set(em_seeds))
    out = []
    if len(common) < 2:
        out.append(f"  [{tag}]  --  only {len(common)} common seeds, skipping.")
        return out

    bl      = np.array([bl_seeds[s] for s in common])
    em      = np.array([em_seeds[s] for s in common])
    savings = (bl - em) / bl * 100.0

    w   = welch_test(bl, em)
    pt  = paired_test(bl, em)
    cd  = cohen_d(bl, em)

    mean_sav = float(np.mean(savings))
    std_sav  = float(np.std(savings, ddof=1))

    out += [
        "",
        f"  -- {tag} --",
        f"  Seeds: {common}  (n={len(common)})",
        "",
        f"  Mean in-box CO2/veh  --  Baseline:       {np.mean(bl):>13,.1f} mg  +/-  {np.std(bl, ddof=1):,.1f}",
        f"  Mean in-box CO2/veh  --  Emission-based: {np.mean(em):>13,.1f} mg  +/-  {np.std(em, ddof=1):,.1f}",
        "",
        "  Per-seed CO2 savings (baseline - emission) / baseline x 100:",
    ]
    for s, sv in zip(common, savings):
        out.append(f"    seed {s:3d}: {sv:+.4f}%")

    out += [
        "",
        f"  Mean savings: {mean_sav:+.4f}%  +/-  {std_sav:.4f}%",
        "",
        "  -- Welch t-test (unequal variance, unpaired) --",
        f"     t = {w['t']:+.4f},  df = {w['df']:.2f},  p = {w['p']:.6f}  {sig_label(w['p'])}",
        f"     Cohen's d = {cd:+.4f}  ({effect_label(cd)} effect)",
        "",
        "  -- Paired t-test (matched seeds -- more powerful) --",
        f"     t = {pt['t']:+.4f},  df = {pt['df']},  p = {pt['p']:.6f}  {sig_label(pt['p'])}",
        "",
    ]

    ref_p = pt['p']
    if ref_p < 0.05:
        out += [
            f"  STATISTICALLY SIGNIFICANT (paired p={ref_p:.4f}).",
            f"  Emission-based controller reduces in-box CO2 by",
            f"  {abs(mean_sav):.4f}% on average -- unlikely due to chance.",
        ]
    else:
        out += [
            f"  NOT statistically significant (paired p={ref_p:.4f}).",
            f"  Mean savings of {abs(mean_sav):.4f}% is within noise across {len(common)} seeds.",
        ]

    if thr != 50:
        out += [
            "",
            f"  CAUTION: optimal threshold (thr={thr}) was selected from the same dataset.",
            "  This test may be optimistic (selection bias). The thr=50 result above",
            "  is the pre-specified unbiased reference.",
        ]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEP  = '=' * 68
    DASH = '-' * 68

    all_lines = [
        SEP,
        "  Welch t-test + Paired t-test -- CO2 Savings Significance",
        "  Controller: emission-based vs gap-based baseline",
        "  Tested at : thr=50 mg/s (reference) AND optimal threshold per regime",
        "  Metric    : in-box CO2 per total vehicle (mg)",
        "              (in_box==True rows; consistent with main analysis pipeline)",
        "  Optimal thresholds (fine-grid, throughput-constrained):",
        "    V1500=41 (Free-flow)  V2900=42 (Near-sat)  V3300=135 (Breakdown)  V5000=122 (Gridlock)",
        "  betainc: Numerical Recipes Lentz CF (no scipy required)",
        SEP,
    ]

    for vol in VOLUMES:
        regime = REGIME_LABELS[vol]
        lbl    = f'V{vol:04d}'
        opt    = OPTIMAL_THR[vol]

        all_lines += ["", DASH, f"  {lbl}  --  {regime}", DASH]

        bl_seeds = load_per_seed_co2(DATA_ROOT / lbl / 'baseline')
        if not bl_seeds:
            all_lines.append(f"  [WARN] No baseline data for {lbl} -- skipping.")
            continue

        em50 = load_em_at_thr(vol, 50)
        all_lines += _test_block(50, 'thr=50 mg/s  (default reference)', bl_seeds, em50)

        if opt != 50:
            em_opt = load_em_at_thr(vol, opt)
            all_lines += _test_block(opt, f'thr={opt} mg/s  (regime optimal)', bl_seeds, em_opt)

    all_lines += [
        "",
        SEP,
        "  Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05",
        "  Cohen's d: negligible <0.2 | small 0.2-0.5 | medium 0.5-0.8 | large >=0.8",
        "  Paired test preferred: seeds are matched (same RNG seed for both conditions).",
        "  For optimal-threshold results, treat significance with caution --",
        "  selecting the threshold from the same data inflates apparent significance.",
        SEP,
    ]

    text = '\n'.join(all_lines) + '\n'
    for line in all_lines:
        print(line)

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n  Saved --> experiments/threshold_search/results/welch_ttest_results.txt")


if __name__ == '__main__':
    main()

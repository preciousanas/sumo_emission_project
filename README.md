# Emission-Based Actuated Traffic Signal Control (EBAC)

**A Constrained Exhaustive Grid Search Across Traffic Regimes Using SUMO Simulation.**

This repository accompanies the paper:

> Anavberokhai, P., Bikdash, M., Comert, G., Gokaraju, B., & Mwakalonge, J. L.
> Emission-Based Actuated Traffic Signal Control: A Constrained Exhaustive Grid Search Across
> Traffic Regimes Using SUMO Simulation.
> Manuscript submitted to *Transportation Research Interdisciplinary Perspectives* (TRIP), 2026.

## Summary

We introduce **Emission-Based Actuated Control (EBAC)**, a drop-in replacement for the
gap-acceptance switch criterion of a standard actuated signal controller (SAC). EBAC
switches phases when the aggregate CO₂ emission rate on the opposing (red) approaches
exceeds that on the current (green) approaches by more than a tunable
`EMISSION_THRESHOLD` (mg/s). Using SUMO with the HBEFA3 emission model, we performed a
constrained exhaustive grid search over 50 emission thresholds (10–300 mg/s) at four
traffic regimes with 10 random seeds each (2,000 simulations), and further ran V2X
penetration and fleet-composition sensitivity experiments (4,320 additional
simulations).

**Headline results (at the throughput-constrained optimum):**

| Regime | Volume | v/c (%) | Opt. thr (mg/s) | ΔCO₂ | ΔTput | ΔWait |
|---|---:|---:|---:|---:|---:|---:|
| Free-flow       | 1,500 | 38.9 | 41  | +0.37% | 0.00% | −4.5% |
| Near-saturation | 2,900 | 74.7 | 42  | +1.95% | 0.00% | −8.7% |
| Breakdown onset | 3,300 | 82.9 | 135 | +3.01% | 0.00% | −13.6% |
| Gridlock        | 5,000 | 91.1 | 138 | +0.20% | 0.00% | −15.0% |

A universal threshold of **44 mg/s** delivers positive CO₂ savings in three of four
regimes without per-site calibration.

## Repository Structure

```
sumo_emission_project/
├── README.md                     ← this file
├── LICENSE                       ← MIT license
├── CITATION.cff                  ← machine-readable citation
├── requirements.txt              ← Python package pins
├── .gitignore
├── paper1_TRIP.tex               ← manuscript (Elsevier elsarticle format)
├── paper1_main_updated(1).tex    ← manuscript (single-column preprint format)
├── references.bib                ← BibTeX for TRIP submission
│
├── config/                       ← SUMO configuration
│   ├── traffic.net.xml           ← SUMO network (four-way intersection)
│   ├── traffic.rou.xml           ← Route file (baseline fleet composition)
│   ├── traffic.rou.light.xml     ← Light fleet composition
│   ├── traffic.rou.mixed.xml     ← Mixed fleet composition
│   └── traffic.rou.heavy.xml     ← Heavy fleet composition
│
├── scripts/                      ← Simulation + analysis code
│   ├── run_capacity_experiment.py     ← Pipeline A driver (capacity sweep)
│   ├── run_experiment.py              ← Pipeline B driver (baseline vs EBAC)
│   ├── run_simulation.py              ← Single-run TraCI loop
│   ├── analyze_all_volumes.py         ← Post-process all V####/
│   ├── analyze_results.py             ← Per-volume plots + statistics
│   ├── parse_capacity_results.py      ← Capacity plateau detection
│   ├── compute_vc_ratios.py           ← v/c derivation
│   ├── plot_capacity_analysis.py      ← Capacity dashboard
│   ├── build_narrative.py             ← Wait/travel/switch narrative figure
│   ├── regenerate_cross_volume.py     ← Streaming cross-volume CSV rebuild
│   ├── publication_figures.py         ← Publication-quality figures (capacity + xvol)
│   ├── publication_figures_ext.py     ← Publication-quality figures (per-volume + experiments)
│   ├── capacity/                      ← Capacity math + summary parser
│   └── experiments/                   ← Sensitivity experiments (threshold, penetration, composition)
│
├── data/                         ← Raw per-seed simulation outputs
│   ├── capacity_analysis/        ← Pipeline A output (71 volumes × 10 seeds)
│   ├── V0500/ … V7500/           ← Pipeline B output (baseline + emission_based per volume)
│   │   ├── baseline/
│   │   │   ├── result_seed_*.csv      ← Per-second telemetry per seed
│   │   │   ├── tripinfo_seed_*.xml    ← Per-vehicle trip summaries
│   │   │   └── log_seed_*.txt         ← Phase switch counts
│   │   └── emission_based/            ← Same structure under EBAC
│
├── experiments/                  ← Sensitivity experiments
│   ├── threshold_search/         ← 50 thresholds × 4 regimes × 10 seeds
│   ├── penetration_rate/         ← 5 V2X penetration rates × 4 regimes × 10 seeds
│   ├── composition/              ← 3 fleet compositions × 4 regimes × 10 seeds
│   └── threshold_x_scenarios/    ← 11 thresholds × (penetration ∪ composition) × 4 regimes
│       ├── results/              ← Aggregated summary CSVs
│       └── visuals/              ← Publication figures (.png + .pdf)
│
└── visuals/                      ← Figures generated for the paper
    ├── capacity_plots/           ← Fig 1–4 (discharge, v/c bars, dashboard)
    ├── cross_volume/             ← Cross-volume + narrative + regime matrix
    ├── V1500/ V2900/ V3300/ V5000/  ← Per-regime plots (boxplot, paired, dual-wait)
    └── network/                  ← Screenshots of the SUMO intersection
```

## Reproducing the Paper End-to-End

### Prerequisites

- **Python 3.10+**
- **SUMO 1.19+** with TraCI Python bindings (`pip install sumolib traci` or use SUMO's bundled Python).
- Python packages: see `requirements.txt`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Confirm SUMO is on your `PATH`:

```bash
sumo --version
```

### Full pipeline (fresh reproduction)

```bash
# ==== Pipeline A: capacity sweep ====
python scripts/run_capacity_experiment.py            # 71 volumes × 10 seeds (~30 min on 8 cores)
python scripts/parse_capacity_results.py             # → data/capacity_analysis/summary/
python scripts/compute_vc_ratios.py

# ==== Pipeline B: baseline vs EBAC ====
python scripts/run_experiment.py                     # 71 volumes × 10 seeds × 2 controllers (~4 h)
python scripts/analyze_all_volumes.py                # → visuals/cross_volume/cross_volume_summary.csv
python scripts/build_narrative.py                    # → visuals/cross_volume/narrative_tradeoff.*

# ==== Sensitivity experiments ====
python scripts/experiments/run_threshold_search.py           # 50 thr × 4 vol × 10 seeds (~1.5 h)
python scripts/experiments/analyze_threshold_search.py       # → experiments/threshold_search/results/
python scripts/experiments/run_penetration_experiment.py     # 5 pen × 4 vol × 10 seeds
python scripts/experiments/analyze_penetration.py            # → experiments/penetration_rate/results/
python scripts/experiments/run_composition_experiment.py     # 3 comp × 4 vol × 10 seeds
python scripts/experiments/analyze_composition.py            # → experiments/composition/results/
python scripts/experiments/run_threshold_x_scenarios.py      # 11 thr × 8 scenarios × 10 seeds
python scripts/experiments/analyze_threshold_x_scenarios.py  # → experiments/threshold_x_scenarios/results/

# ==== Publication figures (regenerates every figure in the paper) ====
python scripts/publication_figures.py                # Capacity + cross-volume figures
python scripts/publication_figures_ext.py            # Per-volume + experiment figures + regime matrix
```

### Figure-only regeneration (uses already-committed summary CSVs)

If the raw per-seed CSVs are already in place, only the last two commands are required
to reproduce every figure in the paper. Estimated wall-clock time: ~4 minutes.

### Compiling the manuscript

```bash
# TRIP submission (Elsevier elsarticle)
pdflatex paper1_TRIP.tex
bibtex   paper1_TRIP
pdflatex paper1_TRIP.tex
pdflatex paper1_TRIP.tex
```

## Data Format

### Per-seed telemetry CSV (`result_seed_<N>.csv`)

One row per (simulation second, vehicle). Columns:

| Column | Type | Description |
|---|---|---|
| `sim_time` | float | Simulation time (s). |
| `veh_id` | str | Unique vehicle identifier (SUMO auto-generated). |
| `speed` | float | Instantaneous speed (m/s). |
| `wait_time` | float | Cumulative wait since last motion (s). |
| `co2_emission` | float | Instantaneous CO₂ rate (mg/s). |
| `in_box` | bool | True if vehicle inside the 300 m × 300 m detection zone. |
| `phase_duration` | float | Current signal phase duration (s). |

### Aggregated cross-volume CSV (`cross_volume_summary.csv`)

One row per (volume, method, metric): `mean` and `std` computed across seeds for
throughput, avg_wait_time (in-box), avg_fulltrip_wait, avg_travel_time, avg_speed, and
avg_co2_per_vehicle.

### Experiment summary CSVs

| File | Cells |
|---|---|
| `experiments/threshold_search/results/threshold_summary.csv`         | (volume, threshold, co2_savings_pct, throughput_change_pct, wait_change_pct) |
| `experiments/threshold_search/results/optimal_threshold_per_regime.csv` | (volume, regime, optimal_threshold, co2_savings_pct) |
| `experiments/penetration_rate/results/penetration_summary.csv`       | (volume, penetration_pct, co2_savings_pct, ...) |
| `experiments/composition/results/composition_summary.csv`            | (volume, composition, bl_co2_per_veh, em_co2_per_veh, co2_savings_pct, ...) |
| `experiments/threshold_x_scenarios/results/pen_x_threshold_summary.csv`  | (volume, penetration_pct, threshold, co2_savings_pct, ...) |
| `experiments/threshold_x_scenarios/results/comp_x_threshold_summary.csv` | (volume, composition, threshold, co2_savings_pct, ...) |

## Random Seeds

All experiments use ten independent seeds:
`{5, 10, 13, 17, 25, 37, 42, 50, 75, 77}`.
Seed identity is preserved across every figure that shows per-seed data.

## Regime Definitions

The four traffic regimes are defined by the v/c ratio derived from the Pipeline A
capacity sweep (capacity = 97 veh/cycle at 90 s cycle):

| Regime | Volume (veh/hr) | v/c (%) | HCM-proxy LOS |
|---|---:|---:|:-:|
| Free-flow       | 1,500 | 38.9 | B |
| Near-saturation | 2,900 | 74.7 | C |
| Breakdown onset | 3,300 | 82.9 | D |
| Gridlock        | 5,000 | 91.1 | E |

## Citation

Please cite the paper as:

```bibtex
@article{anavberokhai2026ebac,
  author  = {Anavberokhai, Precious and Bikdash, Marwan and Comert, Gurcan and Gokaraju, Balakrishna and Mwakalonge, Judith L.},
  title   = {Emission-Based Actuated Traffic Signal Control:
             A Constrained Exhaustive Grid Search Across Traffic Regimes Using SUMO Simulation},
  journal = {Transportation Research Interdisciplinary Perspectives},
  year    = {2026},
  note    = {Manuscript submitted for publication.}
}
```

A machine-readable citation is also provided in `CITATION.cff`.

## License

- **Source code:** MIT License (see `LICENSE`).
- **Data (per-seed CSVs, aggregated CSVs, figures):** Creative Commons Attribution 4.0
  International (CC BY 4.0).

## Contact

**Corresponding author:** Precious Anavberokhai, `panavberokhai@aggies.ncat.edu`,
Department of Computational Data Science and Engineering, North Carolina Agricultural
and Technical State University.

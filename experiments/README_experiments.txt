==============================================================
EXPERIMENTS — Overview
==============================================================

This folder contains four sensitivity experiments that build on
the main Pipeline A + B simulation results in data/ and visuals/.
None of these experiments modify any original files.

FOLDER STRUCTURE
----------------
experiments/
  threshold_search/       Task 1  — Optimal emission threshold search
  penetration_rate/       Task 2  — Partial sensor coverage (V2X penetration)
  composition/            Task 3  — Vehicle fleet composition sensitivity
  threshold_x_scenarios/  Task 4  — Threshold grid × all scenarios

SCRIPTS (all in scripts/experiments/)
--------------------------------------
run_simulation_threshold.py     Wrapper: adds --emission-threshold to run_simulation
run_simulation_ext.py           Wrapper: adds --penetration-rate to run_simulation
run_threshold_search.py         Task 1 runner
analyze_threshold_search.py     Task 1 analyser
run_penetration_experiment.py   Task 2 runner
analyze_penetration.py          Task 2 analyser
run_composition_experiment.py   Task 3 runner
analyze_composition.py          Task 3 analyser
run_threshold_x_scenarios.py    Task 4 runner
analyze_threshold_x_scenarios.py Task 4 analyser

RUN ORDER
---------
Step 1: Task 1 — Run coarse threshold grid
    python scripts/experiments/run_threshold_search.py --phase coarse

Step 2: Task 1 — Analyse coarse results
    python scripts/experiments/analyze_threshold_search.py
    OUTPUT: experiments/threshold_search/results/coarse_analysis_report.txt
            (auto-saved; contains optimal-threshold table and the exact
             --fine-thresholds command to paste into Step 3)

Step 3: Task 1 — Run fine threshold grid
    python scripts/experiments/run_threshold_search.py --phase fine --fine-thresholds <T1 T2 ...>
    NOTE: copy the pre-filled command from coarse_analysis_report.txt —
          no manual calculation required.

Step 4: Task 1 — Re-analyse full grid (coarse + fine combined)
    python scripts/experiments/analyze_threshold_search.py --phase fine
    OUTPUT: experiments/threshold_search/results/fine_analysis_report.txt
            (auto-saved; contains final optimal threshold per regime)

Step 5: Task 2 — Run penetration rate experiment
    python scripts/experiments/run_penetration_experiment.py

Step 6: Task 2 — Analyse penetration results
    python scripts/experiments/analyze_penetration.py

Step 7: Task 3 — Run vehicle composition experiment
    python scripts/experiments/run_composition_experiment.py

Step 8: Task 3 — Analyse composition results
    python scripts/experiments/analyze_composition.py

Step 9: Task 4 — Run threshold x scenario cross-experiment
    python scripts/experiments/run_threshold_x_scenarios.py
    REQUIRES: Steps 5-8 complete

Step 10: Task 4 — Analyse cross-scenario results
    python scripts/experiments/analyze_threshold_x_scenarios.py

REPRESENTATIVE VOLUMES
-----------------------
V1500  Free-flow        (V500-V2400 regime)
V2900  Near-saturation  (V2500-V3000 regime)
V3300  Breakdown onset  (V3100-V3400 regime)
V5000  Gridlock         (V3500-V7500 regime)

THRESHOLD COARSE GRID
-----------------------
[10, 25, 50, 75, 100, 150, 200, 300] mg/s
(thr=50 is reused from existing data/ — not re-run)

KEY DESIGN DECISIONS
--------------------
- All existing data/ results are READ but never modified
- New simulations write only to experiments/ subfolders
- Analysis scripts auto-discover available results
- Run SUMO headless (--nogui) for speed (GUI available via --gui flag)
- 10 seeds per run (same as main pipeline)
==============================================================

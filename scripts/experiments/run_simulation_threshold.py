#!/usr/bin/env python3
"""
run_simulation_threshold.py — Thin wrapper around run_simulation.py.

Accepts all standard run_simulation.py arguments PLUS:
  --emission-threshold FLOAT   Override EMISSION_THRESHOLD (default 50.0 mg/s)

The original run_simulation.py is NEVER modified. This wrapper patches the
module-level constant at import time before any simulation logic executes.

Called by run_threshold_search.py and run_threshold_x_scenarios.py.
Not intended for direct user invocation (use run_threshold_search.py instead).
"""

import sys
import os
import optparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate run_simulation in the parent scripts/ directory
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
SCRIPTS_DIR  = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Parse arguments — superset of run_simulation's options
# ---------------------------------------------------------------------------
opt_parser = optparse.OptionParser()
opt_parser.add_option('--seed',               type='int',    default=42)
opt_parser.add_option('--method',             type='string', default='emission_based')
opt_parser.add_option('--nogui',              action='store_true', default=False)
opt_parser.add_option('--output',             type='string', default='simulation_data.csv')
opt_parser.add_option('--route',              type='string', default=None)
opt_parser.add_option('--emission-threshold', type='float',  default=50.0,
                      dest='emission_threshold',
                      help='Override EMISSION_THRESHOLD in mg/s (default 50.0).')
options, _ = opt_parser.parse_args()

# ---------------------------------------------------------------------------
# Import run_simulation and patch the threshold BEFORE calling run_simulation
# ---------------------------------------------------------------------------
import run_simulation as rs

rs.EMISSION_THRESHOLD = options.emission_threshold

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
rs.run_simulation(
    seed        = options.seed,
    method      = options.method,
    use_gui     = not options.nogui,
    output_file = options.output,
    route_file  = options.route,
)

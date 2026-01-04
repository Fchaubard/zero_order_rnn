#!/bin/bash

# Generate convergence grid from training runs
# Shows argmin iterations over all LRs and sat_alpha for each (solver, perturbations, scale) combination

echo "Capturing logs from screen sessions..."
LOGFILE=$(mktemp)
./scripts/capture_screen_logs.sh 2>&1 | tr -d '\r' > "$LOGFILE"

python3 - "$LOGFILE" << 'PYEOF'
import sys
import re
import os
import json
import glob
from collections import defaultdict

logfile = sys.argv[1]
results_dir = './results_15SPSA_run'

with open(logfile, 'r', errors='ignore') as f:
    content = f.read()

# Model scale to param count mapping
SCALE_TO_PARAMS = {
    1: 304357,
    2: 1132901,
    4: 4362853,
    8: 17114213,
    16: 67782757,
    32: 269783141,
    64: 1076437093,
    128: 4300357733
}

PARAM_LABELS = {
    1: "304K",
    2: "1.1M",
    4: "4.4M",
    8: "17M",
    16: "68M",
    32: "270M",
    64: "1.1B",
    128: "4.3B"
}

# Split by Session:
sessions = re.split(r'-{10,}\s*\n\s*Session:', content)

# Store results: (solver, perturbations, scale) -> list of (iters, lr, sat_alpha) for converged runs
results = defaultdict(list)

# Map from hidden_size to scale
HIDDEN_TO_SCALE = {128: 1, 256: 2, 512: 4, 1024: 8, 2048: 16, 4096: 32, 8192: 64, 16384: 128}

# First, parse JSON result files to get sat_alpha (more reliable than logs)
json_runs = {}  # Map screen_name_prefix -> sat_alpha
for json_path in glob.glob(os.path.join(results_dir, '*.json')):
    try:
        with open(json_path) as f:
            data = json.load(f)
        args = data.get('args', {})
        solver = args.get('solver', '')
        pert = args.get('num_perturbations', 0)
        hidden = args.get('hidden_size', 0)
        scale = HIDDEN_TO_SCALE.get(hidden, 0)
        lr = args.get('learning_rate', 0)
        sat_alpha = args.get('saturating_alpha', None)
        status = data.get('status', '')
        iters = data.get('iters', 0)
        final_loss = data.get('final_loss', 0)

        # Skip "training" status from JSON - those are incomplete runs
        if status == 'training':
            continue

        if solver and scale:
            results[(solver, pert, scale)].append({
                'iters': iters,
                'lr': str(lr),
                'sat_alpha': str(sat_alpha) if sat_alpha is not None else '?',
                'final_loss': final_loss,
                'status': 'converged' if status in ('success', 'converged') else 'diverged' if status == 'diverged' else 'completed',
                'source': 'json'
            })
    except Exception as e:
        pass  # Skip invalid JSON files

# Then parse screen logs for still-training runs (JSON files handle completed runs)
for session in sessions[1:]:
    first_line = session.split('\n')[0] if session else ''

    # Extract solver from session name
    solver_match = re.search(r'SOLV_([^_]+)', first_line)
    if not solver_match:
        continue
    solver = solver_match.group(1)

    # Extract scale from session name
    scale_match = re.search(r'scale(\d+)', first_line)
    if not scale_match:
        continue
    scale = int(scale_match.group(1))

    # Extract perturbations from session name
    pert_match = re.search(r'pert_(\d+)', first_line)
    if not pert_match:
        continue
    perturbations = int(pert_match.group(1))

    # Extract LR from session name
    lr_match = re.search(r'_lr([0-9.e-]+)', first_line)
    lr = lr_match.group(1) if lr_match else "?"

    # Skip completed/diverged runs - these are handled by JSON parsing
    if re.search(r"status (converged|success|diverged)", session):
        continue

    # Check for still-training runs (showing Iteration progress)
    iter_match = re.search(r"Iteration \d+/(\d+)", session)
    if iter_match:
        max_iters = int(iter_match.group(1))
        iter_matches = re.findall(r"Iteration (\d+)/" + str(max_iters) + r".*?Train Loss: ([0-9.]+)", session)
        if iter_matches:
            last_match = iter_matches[-1]
            iters = int(last_match[0])
            final_loss = float(last_match[1])

            if iters < max_iters:
                # Still training - sat_alpha not available from screen logs
                # but we can at least track progress
                results[(solver, perturbations, scale)].append({
                    'iters': iters,
                    'lr': lr,
                    'sat_alpha': '?',  # Not available in screen logs
                    'final_loss': final_loss,
                    'status': 'training',
                    'source': 'screen'
                })

# Define row order
ROW_ORDER = [
    ("BPTT", None),
    ("1SPSA", 8),
    ("1SPSA", 96),
    ("1SPSA", 512),
    ("1SPSA", 1024),
    ("1.5-SPSA", 8),
    ("1.5-SPSA", 96),
    ("1.5-SPSA", 512),
    ("1.5-SPSA", 1024),
]

# Column order (scales)
SCALES = [1, 2, 4, 8, 16, 32, 64, 128]

# Build grid
grid = {}
best_configs = {}  # Store the best config for each cell

for (solver, perturbations, scale), runs in results.items():
    if not runs:
        continue

    # Find best run (minimum iters for converged/completed, or best progress for training)
    converged_runs = [r for r in runs if r['status'] in ('converged', 'completed')]

    if converged_runs:
        best = min(converged_runs, key=lambda x: x['iters'])
        grid[(solver, perturbations, scale)] = best['iters']
        best_configs[(solver, perturbations, scale)] = best
    else:
        # Show best training progress with asterisk
        training_runs = [r for r in runs if r['status'] == 'training']
        if training_runs:
            best = max(training_runs, key=lambda x: x['iters'])
            grid[(solver, perturbations, scale)] = f"{best['iters']}*"
            best_configs[(solver, perturbations, scale)] = best

# Print header
print()
print("=" * 120)
print("CONVERGENCE GRID - Minimum iterations to converge (argmin over LR, sat_alpha)")
print("* = still training, showing current iteration")
print("=" * 120)
print()

# Print column headers
header = f"{'Solver':<30}"
for scale in SCALES:
    header += f" | {PARAM_LABELS[scale]:>8}"
print(header)
print("-" * 120)

# Print rows
for solver, pert in ROW_ORDER:
    if pert is None:
        row_label = solver
    else:
        row_label = f"{solver} @ {pert} pert/step"

    row = f"{row_label:<30}"
    for scale in SCALES:
        key = (solver, pert, scale) if pert else (solver, None, scale)

        # For BPTT, check without perturbations
        if solver == "BPTT":
            key = ("BPTT", None, scale)
            # Also try common variations
            for try_pert in [None, 1, 8, 96]:
                if ("BPTT", try_pert, scale) in grid:
                    key = ("BPTT", try_pert, scale)
                    break

        if key in grid:
            val = grid[key]
            row += f" | {str(val):>8}"
        else:
            row += f" | {'-':>8}"

    print(row)

print("-" * 120)
print()

# Print detailed best configs
print("=" * 120)
print("BEST CONFIGURATIONS (LR, sat_alpha) for each cell:")
print("=" * 120)
for (solver, pert, scale), config in sorted(best_configs.items()):
    if pert:
        label = f"{solver}@{pert}, scale={scale}"
    else:
        label = f"{solver}, scale={scale}"
    lr = config.get('lr', '?')
    sat = config.get('sat_alpha', '?')
    iters = config.get('iters', '?')
    status = config.get('status', '?')
    loss = config.get('final_loss', '?')
    print(f"  {label:<35}: iters={iters}, lr={lr}, sat_alpha={sat}, status={status}, loss={loss}")

print()
PYEOF

rm -f "$LOGFILE"

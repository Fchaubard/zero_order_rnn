#!/bin/bash

# Summarize training runs from screen sessions
# Parses output and creates a table sorted by iterations

echo "Capturing logs from screen sessions..."
LOGFILE=$(mktemp)
./scripts/capture_screen_logs.sh 2>&1 | tr -d '\r' > "$LOGFILE"

TMPFILE=$(mktemp)

# Use python for reliable parsing
python3 - "$LOGFILE" "$TMPFILE" << 'PYEOF'
import sys
import re

logfile = sys.argv[1]
outfile = sys.argv[2]

with open(logfile, 'r', errors='ignore') as f:
    content = f.read()

# Split by "Session:" to get individual session blocks
sessions = re.split(r'-{10,}\s*\n\s*Session:', content)

results = []

for session in sessions[1:]:  # Skip first split
    solver = None
    lr = None
    sat_alpha = None
    final_loss = None
    train_time = None
    status = None
    iters = None
    screen_name = None

    # Extract from session name line (first line of session block)
    # Format: 1524240.copy__DNC_pert_8_scale32_SOLV_1.5-SPSA_lr0.05_overfit_true_cx100_
    first_line = session.split('\n')[0] if session else ''

    # Extract full screen name (everything before any whitespace on first line)
    match = re.search(r'^(\S+)', first_line.strip())
    if match:
        screen_name = match.group(1)

    # Extract solver from session name (SOLV_xxx)
    match = re.search(r'SOLV_([^_]+)', first_line)
    if match:
        solver = match.group(1)

    # Extract learning rate from session name (lr0.xxx) or Namespace
    match = re.search(r'_lr([0-9.]+)', first_line)
    if match:
        lr = match.group(1)
    else:
        # Try to get LR from Namespace output (for binary LR search runs)
        match = re.search(r"learning_rate=([0-9.e+-]+)", session)
        if match:
            try:
                lr_val = float(match.group(1))
                lr = f"{lr_val:.2e}"
            except:
                lr = match.group(1)
        # Check if this is a binlr run
        if '_binlr_' in first_line:
            lr = lr if lr else "binlr"

    # Extract saturating_alpha from Namespace or estimate from log
    match = re.search(r"saturating_alpha=([0-9.]+)", session)
    if match:
        sat_alpha = match.group(1)
    else:
        # Try to get from session name if available
        match = re.search(r'saturating_alpha[=:]?\s*([0-9.]+)', session)
        if match:
            sat_alpha = match.group(1)

    # Check for diverged
    match = re.search(r"status diverged iters (\d+)", session)
    if match:
        status = "diverged"
        iters = int(match.group(1))

        # Get final loss
        loss_match = re.search(r"Final loss \(after training\): ([0-9.]+)", session)
        if loss_match:
            final_loss = loss_match.group(1)

        # Get training time
        time_match = re.search(r"Total training time: ([0-9.]+)", session)
        if time_match:
            train_time = time_match.group(1)

    # Check for converged/success
    elif re.search(r"status (converged|success)", session):
        match = re.search(r"status (converged|success) iters (\d+)", session)
        if match:
            status = "converged"
            iters = int(match.group(2))

            loss_match = re.search(r"Final loss \(after training\): ([0-9.]+)", session)
            if loss_match:
                final_loss = loss_match.group(1)

            time_match = re.search(r"Total training time: ([0-9.]+)", session)
            if time_match:
                train_time = time_match.group(1)

    # Check if still training or completed max iterations
    # Handle both 2000 and 5000 max_iters
    iter_match = re.search(r"Iteration \d+/(\d+)", session)
    if iter_match:
        max_iters = int(iter_match.group(1))
        # Get the last iteration
        iter_matches = re.findall(r"Iteration (\d+)/" + str(max_iters) + r".*?Train Loss: ([0-9.]+)", session)
        if iter_matches:
            last_match = iter_matches[-1]
            iters = int(last_match[0])
            final_loss = last_match[1]
            train_time = "-"
            # Check if completed (reached max iterations)
            if iters >= max_iters:
                status = "completed"
            else:
                status = "still_training"

    # Only add if we have key fields (solver required, lr defaults to binlr if missing)
    if solver and status:
        results.append({
            'iters': iters if iters is not None else 0,
            'solver': solver,
            'lr': lr or 'binlr',
            'sat_alpha': sat_alpha or '-',
            'final_loss': final_loss or '-',
            'train_time': train_time or '-',
            'status': status,
            'screen_name': screen_name or '-'
        })

# Sort by iterations
results.sort(key=lambda x: x['iters'])

# Write to output file
with open(outfile, 'w') as f:
    for r in results:
        f.write(f"{r['iters']:06d}|{r['solver']}|{r['lr']}|{r['sat_alpha']}|{r['final_loss']}|{r['train_time']}|{r['status']}|{r['screen_name']}\n")
PYEOF

echo ""
echo "========================================================================================================================================"
echo "                                                    TRAINING RUNS SUMMARY"
echo "========================================================================================================================================"
printf "%-8s | %-10s | %-8s | %-12s | %-10s | %-15s | %s\n" "Iters" "Solver" "LR" "Final Loss" "Time(s)" "Status" "Screen Name"
echo "----------------------------------------------------------------------------------------------------------------------------------------"

# Display sorted results
while IFS='|' read -r iters solver lr sat_alpha final_loss train_time status screen_name; do
    # Remove leading zeros from iters
    iters_clean=$(echo "$iters" | sed 's/^0*//' | sed 's/^$/0/')
    printf "%-8s | %-10s | %-8s | %-12s | %-10s | %-15s | %s\n" \
        "$iters_clean" "$solver" "$lr" "$final_loss" "$train_time" "$status" "$screen_name"
done < "$TMPFILE"

echo "----------------------------------------------------------------------------------------------------------------------------------------"

# Summary counts
total=$(wc -l < "$TMPFILE" 2>/dev/null | tr -d ' \n')
diverged=$(grep -c "diverged" "$TMPFILE" 2>/dev/null | tr -d '\n' || echo 0)
converged=$(grep -c "converged" "$TMPFILE" 2>/dev/null | tr -d '\n' || echo 0)
completed=$(grep -c "completed" "$TMPFILE" 2>/dev/null | tr -d '\n' || echo 0)
still_training=$(grep -c "still_training" "$TMPFILE" 2>/dev/null | tr -d '\n' || echo 0)

echo ""
echo "Summary: Total=$total | Completed=$completed | Converged=$converged | Diverged=$diverged | Still Training=$still_training"
echo ""

rm -f "$TMPFILE" "$LOGFILE" 2>/dev/null

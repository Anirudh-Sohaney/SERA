#!/bin/bash
# Simple Training Queue for Oracle Server
# Runs experiments sequentially with RAM safety

set -e

# Configuration
MAX_RAM=95
WARNING_RAM=85
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p logs

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_ram() {
    free | grep Mem | awk '{print int($3/$2 * 100)}'
}

wait_for_ram() {
    local target=70
    local max_wait=120
    local waited=0
    
    while [ $(check_ram) -ge "$target" ] && [ "$waited" -lt "$max_wait" ]; do
        log "RAM at $(check_ram)%, waiting..."
        sleep 10
        waited=$((waited + 10))
    done
    
    [ "$waited" -lt "$max_wait" ]
}

run_experiment() {
    local name="$1"
    local script="$2"
    local args="$3"
    
    log "Starting: $name"
    log "RAM before: $(check_ram)%"
    
    if [ $(check_ram) -ge "$WARNING_RAM" ]; then
        log "RAM too high, waiting..."
        wait_for_ram || { log "Cannot proceed"; return 1; }
    fi
    
    # Run with timeout
    timeout 7200 python3 $script $args 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}
    
    if [ $status -eq 0 ]; then
        log "Completed: $name"
    else
        log "Failed: $name (exit code $status)"
    fi
    
    log "RAM after: $(check_ram)%"
    return $status
}

# Main
log "Training Queue Started"
log "Max RAM: ${MAX_RAM}%, Warning: ${WARNING_RAM}%"
log "=========================================="

# System info
log "System:"
free -h
echo ""

# Track results
declare -A results

# Experiment 1: E6-A with lightweight training
if run_experiment "E6-A-Lightweight" "train_lightweight.py" \
    "--output-dir checkpoints/experiment_e6_lightweight \
     --batch-size 8 \
     --gradient-accumulation 8 \
     --epochs 2"; then
    results["E6-A"]="SUCCESS"
else
    results["E6-A"]="FAILED"
fi

# Wait between experiments
log "Waiting 30s between experiments..."
sleep 30

# Experiment 2: E6-B (20% augmentation) - only if E6-A succeeded
if [ "${results["E6-A"]}" = "SUCCESS" ]; then
    # Generate 20% augmentation
    log "Generating 20% augmentation..."
    python3 src/data/augmentation_v2.py --ratio 0.20 \
        --output data/processed/e6_augmented_20pct.jsonl \
        --manifest data/processed/e6_manifest_20pct.json 2>&1 | tee -a "$LOG_FILE"
    
    if run_experiment "E6-B-Lightweight" "train_lightweight.py" \
        "--output-dir checkpoints/experiment_e6_lightweight_20pct \
         --augmented-data data/processed/e6_augmented_20pct.jsonl \
         --batch-size 8 \
         --gradient-accumulation 8 \
         --epochs 2"; then
        results["E6-B"]="SUCCESS"
    else
        results["E6-B"]="FAILED"
    fi
fi

# Summary
log "=========================================="
log "Training Complete"
log "=========================================="
for exp in "${!results[@]}"; do
    log "$exp: ${results[$exp]}"
done

log "Final RAM: $(check_ram)%"
log "Finished at $(date)"

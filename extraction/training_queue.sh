#!/bin/bash
# E6 Training Queue with RAM Safety Controls
# Runs all experiments sequentially with automatic RAM monitoring
# If RAM exceeds 95%, training pauses and reduces batch size

set -e

# Configuration
MAX_RAM_PERCENT=95
WARNING_RAM_PERCENT=85
LOG_DIR="logs/training_queue"
CHECKPOINT_DIR="checkpoints"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_queue_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check RAM usage
check_ram() {
    local ram_percent=$(free | grep Mem | awk '{print $3/$2 * 100.0}' | cut -d'.' -f1)
    echo "$ram_percent"
}

# Function to check if training should continue
should_continue() {
    local ram_percent=$(check_ram)
    
    if [ "$ram_percent" -ge "$MAX_RAM_PERCENT" ]; then
        log "CRITICAL: RAM usage at ${ram_percent}% (>= ${MAX_RAM_PERCENT}%)"
        log "Stopping training to prevent system instability"
        return 1
    elif [ "$ram_percent" -ge "$WARNING_RAM_PERCENT" ]; then
        log "WARNING: RAM usage at ${ram_percent}% (>= ${WARNING_RAM_PERCENT}%)"
        log "Reducing batch size and slowing down training"
        return 2
    fi
    return 0
}

# Function to wait for RAM to normalize
wait_for_ram() {
    local target_percent=70
    local max_wait=300  # 5 minutes
    local waited=0
    
    while [ $(check_ram) -ge "$target_percent" ] && [ "$waited" -lt "$max_wait" ]; do
        log "Waiting for RAM to drop below ${target_percent}%... (current: $(check_ram)%)"
        sleep 10
        waited=$((waited + 10))
    done
    
    if [ "$waited" -ge "$max_wait" ]; then
        log "Timeout waiting for RAM to normalize"
        return 1
    fi
    return 0
}

# Function to run training with adaptive batch size
run_training() {
    local experiment_name="$1"
    local script="$2"
    local base_args="$3"
    local batch_size="$4"
    local epochs="$5"
    
    log "=========================================="
    log "Starting experiment: ${experiment_name}"
    log "Base batch size: ${batch_size}"
    log "Epochs: ${epochs}"
    log "=========================================="
    
    # Check initial RAM
    local initial_ram=$(check_ram)
    log "Initial RAM usage: ${initial_ram}%"
    
    if [ "$initial_ram" -ge "$WARNING_RAM_PERCENT" ]; then
        log "High initial RAM, waiting for normalization..."
        if ! wait_for_ram; then
            log "Failed to normalize RAM, skipping experiment"
            return 1
        fi
    fi
    
    # Adaptive batch size based on RAM
    local current_batch_size=$batch_size
    local ram_check_interval=50  # Check every 50 steps
    
    # Run training with monitoring
    python3 "$script" $base_args \
        --batch-size "$current_batch_size" \
        --epochs "$epochs" \
        2>&1 | while IFS= read -r line; do
            echo "$line" | tee -a "$LOG_FILE"
            
            # Check RAM periodically
            if echo "$line" | grep -q "step"; then
                local step=$(echo "$line" | grep -o "step [0-9]*" | awk '{print $2}')
                if [ -n "$step" ] && [ $((step % ram_check_interval)) -eq 0 ]; then
                    local current_ram=$(check_ram)
                    
                    if [ "$current_ram" -ge "$MAX_RAM_PERCENT" ]; then
                        log "CRITICAL: RAM at ${current_ram}% during training"
                        log "Emergency stop triggered"
                        kill -INT $$
                        return 1
                    elif [ "$current_ram" -ge "$WARNING_RAM_PERCENT" ]; then
                        log "WARNING: RAM at ${current_ram}%, reducing batch size"
                        current_batch_size=$((current_batch_size / 2))
                        if [ "$current_batch_size" -lt 4 ]; then
                            current_batch_size=4
                        fi
                        log "New batch size: ${current_batch_size}"
                    fi
                fi
            fi
        done
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "Experiment ${experiment_name} completed successfully"
    else
        log "Experiment ${experiment_name} failed with exit code ${exit_code}"
    fi
    
    return $exit_code
}

# Function to evaluate model
evaluate_model() {
    local checkpoint_dir="$1"
    local eval_name="$2"
    
    log "Evaluating model: ${eval_name}"
    
    python3 -c "
import sys, json, logging, torch, numpy as np
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from src.models.token_classifier import create_model
from src.data.alignment import build_bio_labels
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def extract_spans_from_bio(tag_sequence, offset_mapping, prompt):
    spans = []
    start = None
    for i, tag in enumerate(tag_sequence):
        if tag == 1:  # B
            if start is not None:
                spans.append({'start': start, 'end': end, 'text': prompt[start:end]})
            start = offset_mapping[i][0]
            end = offset_mapping[i][1]
        elif tag == 0:  # O
            if start is not None:
                spans.append({'start': start, 'end': end, 'text': prompt[start:end]})
                start = None
        elif tag == 2:  # I
            if start is not None:
                end = offset_mapping[i][1]
    if start is not None:
        spans.append({'start': start, 'end': end, 'text': prompt[start:end]})
    return spans

class EvalDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = []
        for ex in examples:
            prompt = ex.get('prompt', '')
            spans = ex.get('spans', [])
            if not prompt or not spans:
                continue
            encoded = build_bio_labels(prompt, spans, tokenizer, max_length)
            self.examples.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': encoded['labels'],
                'prompt': prompt,
                'spans': spans,
                'offset_mapping': encoded['offset_mapping'],
            })
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'input_ids': torch.tensor(ex['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(ex['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(ex['labels'], dtype=torch.long),
        }

model = create_model('google/bert_uncased_L-6_H-512_A-8', num_labels=3, dropout=0.1, freeze_embeddings=True)
state = torch.load('${checkpoint_dir}/best/model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-6_H-512_A-8')

with open('data/evaluation/evaluation_suite.json') as f:
    eval_data = json.load(f)

def evaluate_subset(model, tokenizer, examples, max_length=128):
    ds = EvalDataset(examples, tokenizer, max_length=max_length)
    if len(ds) == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0, 'count': 0}
    loader = DataLoader(ds, batch_size=64)
    tp = fp = fn = 0
    for batch in loader:
        with torch.no_grad():
            out = model(batch['input_ids'], batch['attention_mask'])
        preds = torch.argmax(out['logits'], dim=-1).numpy()
        labels = batch['labels'].numpy()
        masks = batch['attention_mask'].numpy()
        for p, l, m in zip(preds, labels, masks):
            act = m == 1
            pp = p[act] >= 1
            ll = l[act] >= 1
            tp += int((pp & ll).sum())
            fp += int((pp & ~ll).sum())
            fn += int((~pp & ll).sum())
    prec = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    return {'f1': f1, 'precision': prec, 'recall': rec, 'count': len(ds)}

results = {}
results['random_test'] = evaluate_subset(model, tokenizer, eval_data['random_test']['examples'])
results['context_positive'] = evaluate_subset(model, tokenizer, eval_data['context_reversal']['positive_examples'])
results['unseen_vocab'] = evaluate_subset(model, tokenizer, eval_data['unseen_vocabulary']['unseen_examples'])

print(json.dumps(results, indent=2))

with open('${checkpoint_dir}/full_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
" 2>&1 | tee -a "$LOG_FILE"
}

# Main training queue
log "Starting E6 Training Queue"
log "RAM Safety: Max ${MAX_RAM_PERCENT}%, Warning ${WARNING_RAM_PERCENT}%"
log "=========================================="

# Check system resources
log "System Resources:"
free -h
echo ""

# Track success/failure
declare -A experiment_results

# Experiment 1: E6-A (10% augmentation)
log "Experiment 1: E6-A (10% augmentation)"
if run_training "E6-A" "train_e6.py" \
    "--output-dir checkpoints/experiment_e6_targeted_fp" \
    "16" "2"; then
    experiment_results["E6-A"]="SUCCESS"
    evaluate_model "checkpoints/experiment_e6_targeted_fp" "E6-A"
else
    experiment_results["E6-A"]="FAILED"
    log "E6-A failed, continuing with next experiment"
fi

# Check RAM before next experiment
if ! wait_for_ram; then
    log "Cannot proceed due to high RAM usage"
    exit 1
fi

# Experiment 2: E6-B (20% augmentation) - only if E6-A succeeded
if [ "${experiment_results["E6-A"]}" = "SUCCESS" ]; then
    log "Experiment 2: E6-B (20% augmentation)"
    
    # Generate 20% augmentation
    python3 src/data/augmentation_v2.py --ratio 0.20 \
        --output data/processed/e6_targeted_augmented_20.jsonl \
        --manifest data/processed/e6_manifest_20.json
    
    if run_training "E6-B" "train_e6.py" \
        "--output-dir checkpoints/experiment_e6_targeted_fp_20" \
        "--augmented-data data/processed/e6_targeted_augmented_20.jsonl" \
        "16" "2"; then
        experiment_results["E6-B"]="SUCCESS"
        evaluate_model "checkpoints/experiment_e6_targeted_fp_20" "E6-B"
    else
        experiment_results["E6-B"]="FAILED"
        log "E6-B failed"
    fi
    
    # Wait for RAM
    if ! wait_for_ram; then
        log "Cannot proceed due to high RAM usage"
        exit 1
    fi
fi

# Summary
log "=========================================="
log "Training Queue Complete"
log "=========================================="
log "Results:"
for experiment in "${!experiment_results[@]}"; do
    log "  ${experiment}: ${experiment_results[$experiment]}"
done

# Check final RAM
final_ram=$(check_ram)
log "Final RAM usage: ${final_ram}%"

log "Training queue finished at $(date)"

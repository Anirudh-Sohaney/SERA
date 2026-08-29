#!/bin/bash
# Oracle Autonomous Training Queue
# RAM limit: 95% (21.85GB of 23GB)
# Trades speed for memory safety

set -e
cd ~/sera_models/extraction

LOGDIR=logs/oracle_$(date +%Y%m%d_%H%M%S)
mkdir -p $LOGDIR
LOG=${LOGDIR}/training.log
RESULTS=${LOGDIR}/results.json

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG; }

check_ram() {
    free | grep Mem | awk '{print int($3/$2 * 100)}'
}

# Pre-flight RAM check
RAM=$(check_ram)
log "Initial RAM: ${RAM}%"
if [ $RAM -ge 90 ]; then
    log "RAM too high at start (${RAM}%). Waiting 60s..."
    sleep 60
    RAM=$(check_ram)
    if [ $RAM -ge 90 ]; then
        log "CRITICAL: RAM still ${RAM}%. Aborting."
        exit 1
    fi
fi

# Install any missing deps
pip3 install --quiet torchcrf 2>/dev/null || true

echo '{' > $RESULTS
echo '  "experiments": [' >> $RESULTS
FIRST=true

run_exp() {
    local NAME="$1"
    local SCRIPT="$2"
    local ARGS="$3"
    local DIR="$4"

    log "========================================"
    log "START: ${NAME}"
    log "RAM before: $(check_ram)%"
    log "========================================"

    # Check RAM
    local R=$(check_ram)
    if [ $R -ge 85 ]; then
        log "RAM at ${R}%, waiting for cooldown..."
        for i in $(seq 1 12); do
            sleep 10
            R=$(check_ram)
            if [ $R -lt 75 ]; then break; fi
        done
        R=$(check_ram)
        if [ $R -ge 90 ]; then
            log "CRITICAL: RAM ${R}%, skipping ${NAME}"
            return 1
        fi
    fi

    # Run with timeout (2 hours max per experiment)
    timeout 7200 python3 $SCRIPT $ARGS 2>&1 | tee -a $LOG
    local EXIT=${PIPESTATUS[0]}

    log "END: ${NAME} (exit=${EXIT})"
    log "RAM after: $(check_ram)%"

    if [ $FIRST = true ]; then FIRST=false; else echo ',' >> $RESULTS; fi
    echo "    {\"name\": \"${NAME}\", \"exit\": ${EXIT}}" >> $RESULTS

    # Cooldown between experiments
    log "Cooldown 30s..."
    sleep 30
    return $EXIT
}

# ===========================
# EXPERIMENT QUEUE
# ===========================

# E6-A: Targeted FP augmentation (10%)
run_exp "E6-A-Lightweight" "train_lightweight.py" \
    "--output-dir checkpoints/oracle_e6a \
     --batch-size 8 \
     --gradient-accumulation 8 \
     --epochs 2 \
     --max-val 300" \
    "checkpoints/oracle_e6a"
E6A_OK=$?

# E6-B: 20% augmentation (only if E6-A succeeded)
if [ "$E6A_OK" = "0" ]; then
    log "E6-A succeeded, generating 20% augmentation..."
    python3 src/data/augmentation_v2.py --ratio 0.20 \
        --output data/processed/e6_augmented_20pct.jsonl \
        --manifest data/processed/e6_manifest_20pct.json 2>&1 | tee -a $LOG

    run_exp "E6-B-Lightweight" "train_lightweight.py" \
        "--output-dir checkpoints/oracle_e6b \
         --augmented-data data/processed/e6_augmented_20pct.jsonl \
         --batch-size 8 \
         --gradient-accumulation 8 \
         --epochs 2 \
         --max-val 300" \
        "checkpoints/oracle_e6b"
fi

# ===========================
# EVALUATION
# ===========================

log "========================================"
log "RUNNING EVALUATIONS"
log "========================================"

eval_model() {
    local DIR="$1"
    local NAME="$2"
    if [ -f "${DIR}/best/model.pt" ]; then
        log "Evaluating ${NAME}..."
        python3 -c "
import sys, json, torch, numpy as np
sys.path.insert(0, '.')
from src.models.token_classifier import create_model
from src.data.alignment import build_bio_labels
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class EvalDS(Dataset):
    def __init__(self, exs, tok, ml=128):
        self.data = []
        for e in exs:
            p, s = e.get('prompt',''), e.get('spans',[])
            if not p: continue
            enc = build_bio_labels(p, s, tok, ml)
            self.data.append({k: torch.tensor(v, dtype=torch.long) if isinstance(v, list) else v for k,v in enc.items() if k in ['input_ids','attention_mask','labels']})
            self.data[-1]['prompt'] = p
            self.data[-1]['spans'] = s
            self.data[-1]['offset_mapping'] = enc['offset_mapping']
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {k:v for k,v in self.data[i].items() if k in ['input_ids','attention_mask','labels']}

model = create_model('google/bert_uncased_L-6_H-512_A-8', num_labels=3, dropout=0.1, freeze_embeddings=True)
model.load_state_dict(torch.load('${DIR}/best/model.pt', map_location='cpu', weights_only=True))
model.eval()
tok = AutoTokenizer.from_pretrained('google/bert_uncased_L-6_H-512_A-8')
with open('data/evaluation/evaluation_suite.json') as f: ed = json.load(f)
def ev(exs):
    ds = EvalDS(exs, tok)
    if not ds: return {'f1':0,'p':0,'r':0,'n':0}
    tp=fp=fn=0
    for b in DataLoader(ds, batch_size=64):
        with torch.no_grad(): out = model(b['input_ids'], b['attention_mask'])
        pr = torch.argmax(out['logits'],-1).numpy(); lb = b['labels'].numpy(); m = b['attention_mask'].numpy()
        for p,l,ms in zip(pr,lb,m):
            a=ms==1; pp=p[a]>=1; ll=l[a]>=1
            tp+=int((pp&ll).sum()); fp+=int((pp&~ll).sum()); fn+=int((~pp&ll).sum())
    p_=tp/(tp+fp+1e-8); r_=tp/(tp+fn+1e-8); f_=2*p_*r_/(p_+r_+1e-8)
    return {'f1':round(f_,4),'p':round(p_,4),'r':round(r_,4),'n':len(ds)}
r = {}
r['test'] = ev(ed['random_test']['examples'])
r['ctx_pos'] = ev(ed['context_reversal']['positive_examples'])
r['unseen'] = ev(ed['unseen_vocabulary']['unseen_examples'])
print(json.dumps(r, indent=2))
with open('${DIR}/eval_results.json','w') as f: json.dump(r,f,indent=2)
" 2>&1 | tee -a $LOG
    fi
}

eval_model "checkpoints/oracle_e6a" "E6-A"
eval_model "checkpoints/oracle_e6b" "E6-B"

# ===========================
# SUMMARY
# ===========================

echo '  ]' >> $RESULTS
echo '}' >> $RESULTS

log "========================================"
log "ALL TRAINING COMPLETE"
log "========================================"
log "Results saved to: $RESULTS"
log "Final RAM: $(check_ram)%"
log "Finished: $(date)"

cat $RESULTS

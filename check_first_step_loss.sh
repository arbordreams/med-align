#!/bin/bash
# Quick script to check first step loss from training runs on Lambda

RUN_DIR="/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334"

echo "Checking first step loss from training runs..."
echo ""

# Check embedding warmup
WARMUP_DIR="${RUN_DIR}/embedding_warmup"
if [ -d "$WARMUP_DIR" ]; then
    echo "=== EMBEDDING WARMUP ==="
    for ckpt_dir in "${WARMUP_DIR}"/checkpoint-*; do
        if [ -d "$ckpt_dir" ]; then
            trainer_state="${ckpt_dir}/trainer_state.json"
            if [ -f "$trainer_state" ]; then
                ckpt_name=$(basename "$ckpt_dir")
                python3 << PYEOF
import json
import sys
try:
    with open('${trainer_state}', 'r') as f:
        state = json.load(f)
    if 'log_history' in state and len(state['log_history']) > 0:
        first_log = state['log_history'][0]
        step = first_log.get('step', 'N/A')
        loss = first_log.get('loss', None)
        if loss is not None:
            print(f"Checkpoint: ${ckpt_name}")
            print(f"  First step: {step}, Loss: {loss:.6f}")
        else:
            print(f"Checkpoint: ${ckpt_name}")
            print(f"  First step: {step}, Loss: Not found")
            print(f"  Available keys: {list(first_log.keys())}")
    else:
        print(f"Checkpoint: ${ckpt_name} - No log_history found")
except Exception as e:
    print(f"Error reading ${trainer_state}: {e}")
PYEOF
            fi
        fi
    done
    echo ""
else
    echo "Embedding warmup directory not found"
    echo ""
fi

# Check vocab adaptation Stage 1
STAGE1_DIR="${RUN_DIR}/vocab_adaptation/stage1_embed_only"
if [ -d "$STAGE1_DIR" ]; then
    echo "=== VOCAB ADAPTATION STAGE 1 (Embeddings-only) ==="
    for ckpt_dir in "${STAGE1_DIR}"/checkpoint-*; do
        if [ -d "$ckpt_dir" ]; then
            trainer_state="${ckpt_dir}/trainer_state.json"
            if [ -f "$trainer_state" ]; then
                ckpt_name=$(basename "$ckpt_dir")
                python3 << PYEOF
import json
import sys
try:
    with open('${trainer_state}', 'r') as f:
        state = json.load(f)
    if 'log_history' in state and len(state['log_history']) > 0:
        first_log = state['log_history'][0]
        step = first_log.get('step', 'N/A')
        loss = first_log.get('loss', None)
        if loss is not None:
            print(f"Checkpoint: ${ckpt_name}")
            print(f"  First step: {step}, Loss: {loss:.6f}")
        else:
            print(f"Checkpoint: ${ckpt_name}")
            print(f"  First step: {step}, Loss: Not found")
    else:
        print(f"Checkpoint: ${ckpt_name} - No log_history found")
except Exception as e:
    print(f"Error reading ${trainer_state}: {e}")
PYEOF
            fi
        fi
    done
    echo ""
else
    echo "Vocab adaptation Stage 1 directory not found"
    echo ""
fi

# Check vocab adaptation Stage 2
STAGE2_DIR="${RUN_DIR}/vocab_adaptation/stage2_full"
if [ -d "$STAGE2_DIR" ]; then
    echo "=== VOCAB ADAPTATION STAGE 2 (Full model) ==="
    for ckpt_dir in "${STAGE2_DIR}"/checkpoint-*; do
        if [ -d "$ckpt_dir" ]; then
            trainer_state="${ckpt_dir}/trainer_state.json"
            if [ -f "$trainer_state" ]; then
                ckpt_name=$(basename "$ckpt_dir")
                python3 << PYEOF
import json
import sys
try:
    with open('${trainer_state}', 'r') as f:
        state = json.load(f)
    if 'log_history' in state and len(state['log_history']) > 0:
        first_log = state['log_history'][0]
        step = first_log.get('step', 'N/A')
        loss = first_log.get('loss', None)
        if loss is not None:
            print(f"Checkpoint: ${ckpt_name}")
            print(f"  First step: {step}, Loss: {loss:.6f}")
        else:
            print(f"Checkpoint: ${ckpt_name}")
            print(f"  First step: {step}, Loss: Not found")
    else:
        print(f"Checkpoint: ${ckpt_name} - No log_history found")
except Exception as e:
    print(f"Error reading ${trainer_state}: {e}")
PYEOF
            fi
        fi
    done
    echo ""
else
    echo "Vocab adaptation Stage 2 directory not found"
    echo ""
fi


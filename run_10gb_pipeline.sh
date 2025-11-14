#!/usr/bin/env bash
# Non-interruptible pipeline runner for 10GB corpus
# This script will continue running even if SSH disconnects
# Usage: nohup bash run_10gb_pipeline.sh > pipeline.log 2>&1 &

set -e  # Exit on error, but we'll handle errors gracefully

# Set working directory
cd /lambda/nfs/med-align/med-align || exit 1

# Log file with timestamp
LOG_DIR="/lambda/nfs/med-align/tokenizer_adapt/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log "ERROR: $1"
    log "Pipeline failed. Check logs at: $LOG_FILE"
    exit 1
}

# Start logging
log "=========================================="
log "Starting 10GB Pipeline Run"
log "=========================================="
log "Log file: $LOG_FILE"
log "Working directory: $(pwd)"

# Step 1: Pull latest code
log "Step 1: Pulling latest code..."
git fetch origin || error_exit "Failed to fetch from origin"
git checkout research-config-optimization || error_exit "Failed to checkout branch"
git pull origin research-config-optimization || error_exit "Failed to pull latest changes"
log "✓ Code updated successfully"

# Step 2: Read config and build corpus
log "Step 2: Reading config and calculating byte budget..."
SIZE_GB=$(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/research_optimal.yaml')); print(int(cfg['corpus']['size_gb'] * 1024 * 1024 * 1024))") || error_exit "Failed to read config"
SIZE_GB_DISPLAY=$(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/research_optimal.yaml')); print(cfg['corpus']['size_gb'])") || error_exit "Failed to read config"
log "✓ Config loaded: ${SIZE_GB_DISPLAY} GB (${SIZE_GB} bytes)"

# Step 3: Set environment variables
log "Step 3: Setting environment variables..."
export MAIN_DIR="/lambda/nfs/med-align/med-align"
export MEDICAL_CORPUS_DIR="/lambda/nfs/med-align/corpora/pubmed_10gb"
export MEDICAL_BYTE_BUDGET="$SIZE_GB"
export MEDICAL_DEDUP=1
export MEDICAL_HASH_NAME="sha256"
log "✓ Environment variables set"

# Step 4: Build corpus
log "Step 4: Building 10GB corpus (this may take a while)..."
if [ -f "$MEDICAL_CORPUS_DIR/aggregated/medical_corpus.jsonl" ]; then
    log "⚠ Corpus already exists, skipping build. Delete $MEDICAL_CORPUS_DIR to rebuild."
else
    bash script/build_medical_corpus.sh >> "$LOG_FILE" 2>&1 || error_exit "Corpus building failed"
    log "✓ Corpus built successfully"
fi

CORPUS_PATH="$MEDICAL_CORPUS_DIR/aggregated/medical_corpus.jsonl"
if [ ! -f "$CORPUS_PATH" ]; then
    error_exit "Corpus file not found: $CORPUS_PATH"
fi
log "✓ Corpus file verified: $CORPUS_PATH"

# Step 5: Run pipeline
log "Step 5: Running medical pipeline..."
log "This will take several hours. Pipeline output will be logged."
python3 script/run_medical_pipeline.py \
  --config configs/research_optimal.yaml \
  --input "$CORPUS_PATH" \
  --run-root /lambda/nfs/med-align/tokenizer_adapt >> "$LOG_FILE" 2>&1 || error_exit "Pipeline execution failed"

log "=========================================="
log "Pipeline completed successfully!"
log "Check results in: /lambda/nfs/med-align/tokenizer_adapt/"
log "Full log: $LOG_FILE"
log "=========================================="


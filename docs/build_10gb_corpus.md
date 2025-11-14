# Building 10GB Corpus from Scratch

## Step 1: Pull Latest Code

```bash
cd /lambda/nfs/med-align/med-align
git fetch origin
git checkout research-config-optimization
git pull origin research-config-optimization
```

## Step 2: Build 10GB Corpus (Reading from Config)

```bash
cd /lambda/nfs/med-align/med-align

# Read size_gb from config and convert to bytes
SIZE_GB=$(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/research_optimal.yaml')); print(int(cfg['corpus']['size_gb'] * 1024 * 1024 * 1024))")
echo "Building corpus with byte budget: $SIZE_GB bytes ($(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/research_optimal.yaml')); print(cfg['corpus']['size_gb'])") GB)"

# Set environment variables from config
export MAIN_DIR="/lambda/nfs/med-align/med-align"
export MEDICAL_CORPUS_DIR="/lambda/nfs/med-align/corpora/pubmed_10gb"
export MEDICAL_BYTE_BUDGET="$SIZE_GB"
export MEDICAL_DEDUP=1  # From config: deduplicate: true
export MEDICAL_HASH_NAME="sha256"  # From config: hash_name: sha256

# Build the corpus
bash script/build_medical_corpus.sh
```

## Step 3: Run Pipeline with Built Corpus

```bash
cd /lambda/nfs/med-align/med-align

# Get the corpus path
CORPUS_PATH="/lambda/nfs/med-align/corpora/pubmed_10gb/aggregated/medical_corpus.jsonl"

# Run pipeline with research_optimal config
python3 script/run_medical_pipeline.py \
  --config configs/research_optimal.yaml \
  --input "$CORPUS_PATH" \
  --run-root /lambda/nfs/med-align/tokenizer_adapt
```

## All-in-One Command

```bash
cd /lambda/nfs/med-align/med-align && \
git fetch origin && \
git checkout research-config-optimization && \
git pull origin research-config-optimization && \
SIZE_GB=$(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/research_optimal.yaml')); print(int(cfg['corpus']['size_gb'] * 1024 * 1024 * 1024))") && \
export MAIN_DIR="/lambda/nfs/med-align/med-align" && \
export MEDICAL_CORPUS_DIR="/lambda/nfs/med-align/corpora/pubmed_10gb" && \
export MEDICAL_BYTE_BUDGET="$SIZE_GB" && \
export MEDICAL_DEDUP=1 && \
export MEDICAL_HASH_NAME="sha256" && \
bash script/build_medical_corpus.sh && \
python3 script/run_medical_pipeline.py \
  --config configs/research_optimal.yaml \
  --input /lambda/nfs/med-align/corpora/pubmed_10gb/aggregated/medical_corpus.jsonl \
  --run-root /lambda/nfs/med-align/tokenizer_adapt
```

## Notes

- The byte budget is calculated dynamically from `configs/research_optimal.yaml` (`size_gb: 10.0`)
- Corpus will be saved to `/lambda/nfs/med-align/corpora/pubmed_10gb/`
- Deduplication and hash settings match the config
- The pipeline will use the full 10GB corpus as configured


# TokAlign: Efficient Vocabulary Adaptation via Token Alignment

The code for the ACL 2025 conference paper "**TokAlign: Efficient Vocabulary Adaptation via Token Alignment**".

## Overview
We propose an efficient method named TokAlign to replace the vocabulary of LLM from the token co-occurrences view, and further transfer the token-level knowledge between models. It first aligns the source vocabulary to the target one by learning a one-to-one mapping matrix for token IDs. Model parameters, including embeddings, are rearranged and progressively fine-tuned for the new vocabulary. The following figure illustrates the method of TokAlign:

![](figure/method.png)

## How to run?

### Environment prerequisites

- Python **3.12.x** (the pipeline is tested against CPython 3.12.3)
- CUDA **12.8** drivers + runtime (Torch 2.8.0 CUDA 12.8 wheels)
- An NVIDIA GPU with at least 40 GB RAM for the full medical alignment

**Recommended GPU configurations:**
- **1x GH200 (96GB)**: ARM64 + H100, 64 vCPUs, 432 GiB RAM — **Recommended for cost efficiency** ($1.49/hr, 1.5-2x faster than H100)
- **1x H100 (80GB PCIe)**: x86_64, 26 vCPUs, 200 GiB RAM — Standard configuration ($2.49/hr)
- **8x H100 (80GB SXM5)**: x86_64, 208 vCPUs, 1800 GiB RAM — For large-scale multi-GPU training ($23.92/hr)

See [GH200 ARM64 Support](docs/GH200_ARM64_SUPPORT.md) for detailed information on GH200 setup and performance comparisons.

Quick sanity checks on a fresh machine:

```
python --version        # Expect 3.12.x
nvidia-smi              # Expect CUDA Version: 12.8
```

### Set up a virtual environment
```
conda create -n tokalign python=3.12
conda activate tokalign

# Install PyTorch first. Use the CUDA 12.8 wheel on Linux GPU hosts:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# On macOS or CPU-only environments, omit the CUDA index:
# pip install torch==2.8.0

pip install -r requirements.txt
```

`requirements.txt` pins modern PyPI builds compatible with Python 3.12, including the precompiled `fasttext-wheel>=0.9.2` distribution required for the FastText embedding backend.

The medical pipeline is tested with **Python 3.12**, **CUDA 12.8**, and **PyTorch 2.8.0**. Confirm your interpreter with `python --version` before running the installers.

Linux-only extras (`deepspeed`, `bitsandbytes`) are guarded by environment markers and will be skipped automatically on unsupported platforms. `flash-attn` is installed best-effort by the helper script and is optional for the medical pipeline.

### RunPod quickstart

On a fresh RunPod Torch 2.8 (CUDA 12.8) image, the entire setup and a tiny end-to-end run can be launched with:

```
git clone https://github.com/your-org/align-medical.git && cd align-medical && chmod +x install_deps.sh && ./install_deps.sh && bash script/quickstart_runpod.sh
```

### Prepare tokenized data (parallel mode)

1. Download and merge multilingual, code, and math data, e.g., [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), [the-stack](https://huggingface.co/datasets/bigcode/the-stack) and [proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) from HuggingFace. We provide a small corpus in the "./data/pretrain-corpus" directory for example. 

2. Tokenize corpus and prepare files of GloVe vector training and evaluation
```
# Replace the path with your corpus and tokenizers' path
vim script/convert2glove_corpus.sh 
bash script/convert2glove_corpus.sh 
```

### Medical monolingual pipeline

The medical adaptation ingests JSONL shards where each line is `{"text": ...}` and
reuses the same text for both source and target tokenizers.

Medical term extraction is handled by `src/medical_terms.py`. It supports:

- Frequency-based and TF-IDF-based term mining.
- Adaptive thresholds that select a minimum frequency close to the desired `top_k`.
- Heuristic quality filtering and scoring of candidate medical terms.

Key config fields (see `docs/config_schema.md` and YAML presets such as
`configs/research.yaml` or `configs/tokalign_paper_optimal.yaml`):

- `term_mining.top_k`: maximum number of terms to keep.
- `term_mining.min_frequency`: minimum raw frequency for a term to be considered.
- `term_mining.use_tfidf`: if true, rank by TF-IDF scores instead of pure frequency.
- `term_mining.use_adaptive_thresholds`: if true (default), compute an adaptive
  frequency threshold (percentile / mean+std / log-scaled) to approximate `top_k`.
- `term_mining.quality_filter`: if true (default), apply heuristic quality checks
  (length, alphabetic content, optional regex patterns).
- `term_mining.min_quality_score`: minimum composite quality score in `[0, 1]`
  required for inclusion. The score combines frequency (log-scaled), TF-IDF, and
  token characteristics.
- `term_mining.medical_patterns`: optional list of regex patterns; when set, mined
  terms must match at least one pattern.

The mining function returns both the selected terms and metadata:

- `terms`: list of strings, the mined medical terms.
- `metadata`: dictionary with statistics such as:
  - `total_docs`, `total_terms`
  - `filtered_by_threshold`, `filtered_by_quality`
  - `adaptive_thresholds` (per-method thresholds and the chosen one)
  - `final_term_count`
  - `quality_score_stats` (min/max/mean/std)
  - `max_tfidf`

The Python runner (`script/run_medical_pipeline.py`) writes two files under
each run directory:

- `corpus/medical_terms.txt`: newline-separated list of mined medical terms.
- `corpus/medical_terms.txt.metadata.json`: JSON metadata matching the schema above.

The standalone CLI in `src/medical_terms.py` exposes the same functionality:

```bash
python -m src.medical_terms mine \
  --corpus /path/to/medical_corpus.jsonl \
  --output /tmp/medical_terms.txt \
  --top-k 2000 \
  --min-count 10 \
  --use-tfidf \
  --use-adaptive-thresholds \
  --quality-filter \
  --min-quality-score 0.3 \
  --medical-patterns "(?i)hypertension" \
  --medical-patterns "(?i)diabetes"
```

This writes `/tmp/medical_terms.txt` and `/tmp/medical_terms.txt.metadata.json`.

#### End-to-end runner

For automation, the Python runner sequences the entire medical pipeline with
retry-aware stages and evaluation hooks:

```
# 1. Aggregate/limit raw shards (e.g. 50 MB smoke corpus)
python script/aggregate_corpus.py \
  --input /abs/path/to/raw_pubmed/*.jsonl \
  --output /tmp/pubmed_smoke.jsonl \
  --byte-budget $((50 * 1024 * 1024)) \
  --deduplicate

# 2. Run the pipeline on the prepared corpus
python script/run_medical_pipeline.py \
  --config configs/ultra_quick_demo.yaml \
  --input /tmp/pubmed_smoke.jsonl \
  --source-tokenizer BioMistral/BioMistral-7B \
  --target-tokenizer mistralai/Mistral-7B-v0.3 \
  --source-model BioMistral/BioMistral-7B \
  --embedding-backend fasttext \
  --run-root /lambda/nfs/med-align/tokenizer_adapt
```

Logs are written under `runs/logs/` and a roll-up summary can be found in
`runs/tokenizer_adapt/<timestamp>/pipeline_summary.json`.

### Configuration (YAML presets)

You can drive the entire pipeline via YAML configs. Presets live under `configs/`:

- `configs/ultra_quick_demo.yaml` — ultra-fast demo (~5MB corpus, very fast settings). Intended for smoke tests/CI only; the adaptation stages run for just a few steps and the resulting metrics are **not** representative.
- `configs/research.yaml` — quality-first defaults (5GB, TF-IDF, 2000 pivots)


Examples:

```
# Use a preset
python script/run_medical_pipeline.py --config configs/research.yaml \
  --input /abs/path/to/corpus.jsonl

# Override config at the CLI (CLI wins over YAML)
python script/run_medical_pipeline.py --config configs/research.yaml \
  --pivot-count 3000 \
  --input /abs/path/to/corpus.jsonl

# Show the final merged config without running
python script/run_medical_pipeline.py --config configs/research.yaml \
  --input /abs/path/to/corpus.jsonl --show-config
```

In the research shell runner, you can also pass a config file:

```
CONFIG_FILE=configs/research.yaml RUN_ROOT=/tmp/custom_root \
bash script/run_medical_pipeline_research.sh
```

See `docs/config_schema.md` for the full schema and `configs/examples/custom_example.yaml`
for environment variable substitution patterns.

### Modes: Smoke test vs Research

- Smoke test (default quickstart): small corpus (~5MB), fast config for sanity checks.
- Research mode (overnight, quality-first): larger corpus cap (default 5GB), more terms, more pivots, longer embedding training, tuned for H100 or GH200.

Run research-grade pipeline:

```
bash script/run_medical_pipeline_research.sh
```

Key research defaults:
- Byte budget: 5 GiB
- Term mining: top-k=2000, min-frequency=3, TF-IDF enabled
- FastText: epochs=30, minCount=1, lr=0.05, dim=300, threads=12 (per worker, 2 workers = 24 threads total on H100)
- Alignment: pivot_count=2000
- Tokenization: workers=24
- Evaluation: perplexity and QA enabled (QA fail-soft if dataset/split missing)

**Hardware-specific configurations:**

The research script auto-detects your hardware and optimizes thread counts:
- **GH200 (64 vCPUs)**: Auto-scales to 64 threads for OMP/MKL, 32 threads per FastText worker
- **H100 PCIe (26 vCPUs)**: Uses 24 threads for OMP/MKL, 12 threads per FastText worker

Recommended env (auto-configured by research script):
```
export OMP_NUM_THREADS=24  # H100 PCIe; GH200 uses 64
export MKL_NUM_THREADS=24  # H100 PCIe; GH200 uses 64
export TOKENIZERS_PARALLELISM=true
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
```

**Performance comparison:**
- **1x GH200**: ~3-5 hours per run, $4.47-7.45 per run (1.5-2x faster, 3-5x cheaper)
- **1x H100 PCIe**: ~6-10 hours per run, $14.94-24.90 per run

See [GH200 ARM64 Support](docs/GH200_ARM64_SUPPORT.md) for detailed performance benchmarks and setup instructions.

#### Curated medical corpus builder

To avoid parsing mismatches, TokAlign ships with a dataset registry and builder
that fetches vetted Hugging Face corpora, normalizes them to JSONL, and
produces an aggregated shard compatible with the medical pipeline.

Supported dataset slugs (see `src/medical_corpus_registry.py` for details):

- `pubmed_abstract` — `uiyunkim-hub/pubmed-abstract` (PubMed abstracts, Public Domain)

Usage:

```bash
export MAIN_DIR=/abs/path/to/TokAlign
export HF_TOKEN=hf_xxxxxxxx          # required for gated datasets like MIMIC
cd $MAIN_DIR

# Optional filters:
# export MEDICAL_DATASETS="pubmed_abstracts biomed_articles"
# export MEDICAL_MAX_SAMPLES=100000
bash script/build_medical_corpus.sh
```

The builder writes a manifest and aggregated corpus under
`runs/corpora/default_medical/`. To feed TokAlign medical mode:

```bash
export MEDICAL_INPUTS="$MAIN_DIR/runs/corpora/default_medical/aggregated/medical_corpus.jsonl"
```

The manifest (`manifest.json`) records dataset checksums, licenses, and any
skipped resources so runs remain auditable. Adjust `MEDICAL_CORPUS_DIR` to
change the output location, and use `MEDICAL_DEDUP=0` if you want to inspect the
raw merged shards without deduplication.

### Train GloVe vectors and obtain token alignment matrix

```
git clone https://github.com/stanfordnlp/GloVe.git
# Train GloVe vectors for source vocabulary and target vocabulary
bash script/token_align.sh
```

### Evaluation of one-to-one token alignment matrix learned
```
# Change the path to the alignment matrix path for evaluation, and choose an evaluation method (BLEU-1 or Bert-score).
vim script/eval_align.sh
bash script/eval_align.sh
```

### Initialize the model weight with the token alignment matrix

```
# Modify the path of alignment matrix
vim script/init_model.sh 
bash script/init_model.sh 
```

### Evaluation hooks

The medical pipeline ships with a lightweight perplexity evaluator that accepts
Hugging Face datasets. When datasets are not yet available the script records a
placeholder requesting the required benchmark names. Dataset specs follow the
format `dataset_id[config]:split` where both `[config]` and `:split` are optional.

Examples:

- `uiyunkim-hub/pubmed-abstract:train`
- `pubmed_qa[pqa_labeled]:validation`

```
python src/eval_medical.py \
  --model runs/tokenizer_adapt/<timestamp>/adapted_model \
  --tokenizer runs/tokenizer_adapt/<timestamp>/tokenizers/target \
  --dataset medical_benchmark:test \
  --output runs/tokenizer_adapt/<timestamp>/evaluation.json
```

MedMCQA accuracy and coverage metrics:

- The evaluator uses `openlifescienceai/medmcqa` (validation split) and fails fast if the dataset is unavailable.
- Results compare the adapted model against the baseline Mistral-7B-v0.3 by default and include tokenization and alignment coverage metrics.

```
python -m src.eval_medical \
  --model runs/tokenizer_adapt/<timestamp>/vocab_adaptation/stage2_full/checkpoint-<steps> \
  --tokenizer runs/tokenizer_adapt/<timestamp>/vocab_adaptation/stage2_full/checkpoint-<steps> \
  --dataset "uiyunkim-hub/pubmed-abstract:train" \
  --max-samples 1000 \
  --run-medmcqa \
  --baseline-model mistralai/Mistral-7B-v0.3 \
  --medmcqa-split validation \
  --output runs/tokenizer_adapt/<timestamp>/metrics/medical_eval.json
- Coverage metrics compare the augmented tokenizer against the baseline to highlight the effect of vocabulary adaptation.
- The `dataset` flag accepts the `[config]:split` syntax described above. Omit `[config]` for datasets without sub-configs.
### Tokenizer diagnostics

Need to confirm that mined medical terms produce meaningful tokens before
running the full pipeline? Use the lightweight helper to compare baseline versus
augmented tokenizers:

```
python script/tokenizer_term_diagnostics.py \
  --baseline mistralai/Mistral-7B-v0.3 \
  --candidate runs/tokenizer_adapt/<timestamp>/tokenizers/target \
  --terms runs/tokenizer_adapt/<timestamp>/corpus/medical_terms.txt \
  --limit 25
```

The script prints token sequences for each term and reports the average length /
single-token ratio for both tokenizers, making it easy to sanity-check tokenizer
augmentation without rerunning the rest of the pipeline.
```

### Expected gains with medical alignment

By mining domain-specific terminology and mirroring the corpus across both
tokenizers, TokAlign accelerates convergence on medical language modelling
tasks. The adaptation is designed to reduce the number of OOV fragments and to
retain the safety guard-rails of the original model while improving recall on
high-frequency terms from the medical corpus.

### Vocabulary Adaptation
```
# First tokenize the training dataset used for vocabulary adaptation
vim script/tokenize_dataset.sh
bash script/tokenize_dataset.sh

# Replace some paths and hyper-parameters with yours, and start the vocabulary adaptation process
vim script/vocab_adaptation.sh
bash script/vocab_adaptation.sh
```

## 📎 Models

We open-source the following models:

| **Name**                     | **LLaMA3 Tokenizer** | **Qwen2 Tokenizer** | **Gemma Tokenizer** |
|------------------------------|:--------------------:|:-------------------:|:-------------------:|
| TokAlign                     |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-1b-LLaMA3-Tokenizer)     |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-1b-Qwen2-Tokenizer)    |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-1b-Gemma-Tokenizer)    |
| + Token-level Distill         |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-1b-Distill-LLaMA-3-8b)     |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-1b-Distill-Qwen-2-7b)    |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-1b-Distill-Gemma-7b)    |


Table 1. Models from $Pythia_{1b}$

| **Name**                     | **LLaMA3 Tokenizer** | **Qwen2 Tokenizer** | **Gemma Tokenizer** |
|------------------------------|:--------------------:|:-------------------:|:-------------------:|
| TokAlign                     |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-6.9b-LLaMA3-Tokenizer)     |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-6.9b-Qwen2-Tokenizer)    |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-6.9b-Gemma-Tokenizer)    |
| + Token-level Distill         |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-6.9b-Distill-LLaMA3-8b)     |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-6.9b-Distill-Qwen-2-7b)    |     [🤗](https://huggingface.co/chongli17/TokAlign-Pythia-6.9b-Distill-Gemma-7b)    |


Table 2. Models from $Pythia_{6.9b}$

## How to cite our paper?
```
@inproceedings{li-etal-2025-TokAlign,
  author    = {Chong Li and
               Jiajun Zhang and
               Chengqing Zong},
  title = "TokAlign: Efficient Vocabulary Adaptation via Token Alignment",
  booktitle = "Proceedings of the 63nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
}
```

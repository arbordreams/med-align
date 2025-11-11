<!-- f11e1783-e84f-48f6-8df8-7df3ad9c1ba5 6f9ed0c8-9a0a-46d5-b754-74adace32964 -->
# Research‑Grade, Crash‑Resistant Overnight Adaptation (1GB)

### Goals

- Run Step 1 (vocabulary adaptation) end‑to‑end overnight on a 1GB corpus
- Separate smoke test vs research mode
- Maximize alignment quality while ensuring stability on H100 (80GB GPU, 26 vCPUs, 200GB RAM)
- Keep evaluation fail‑soft; no crashes on missing datasets

### Preflight Validations (non‑functional guards)

- Verify environment:
- Python 3.12.x; Torch 2.8.0 CUDA 12.8; flash‑attn 2.8.3 present
- `nvidia-smi` shows H100; free GPU memory ≥ 72GB
- Disk space ≥ 150GB free on `/lambda/nfs/.../runs` and HF cache
- Set environment for throughput & stability:
- `export OMP_NUM_THREADS=24`, `export MKL_NUM_THREADS=24`, `export TOKENIZERS_PARALLELISM=true`
- `export HF_DATASETS_CACHE=/lambda/nfs/.../.cache/huggingface`
- `export NVIDIA_TF32_OVERRIDE=1` (enable TF32 on Hopper)
- `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`
- Enable retry/fail‑soft already in runner (`_retry`), keep QA fail‑soft

### Separate Modes

- Smoke test (existing quickstart): small corpus, fast defaults
- Research mode (new flag): large corpus, quality‑first parameters

### Code Changes (minimal, targeted)

1) `script/run_medical_pipeline.py`

- Add `--research-mode` flag
- When set, override to research defaults:
- `--byte-budget=1073741824` (1GB)
- `--term-top-k=2000`, `--min-term-frequency=3`, `--use-tfidf`
- `--pivot-count=1000`, `--tokenizer-workers=24`
- `--max-eval-samples=1000`
- Pass FastText params to pipeline: `--fasttext-epochs`, `--fasttext-mincount`, `--fasttext-lr`

2) `src/medical_pipeline.py`

- Extend `train_embeddings_and_align` signature and FastText call:
- Params: `fasttext_epochs:int`, `fasttext_mincount:int`, `fasttext_lr:float`, `thread:int`
- Call: `fasttext.train_unsupervised(input=..., model="skipgram", dim=300, epoch=fasttext_epochs, minCount=fasttext_mincount, lr=fasttext_lr, thread=thread)`
- Wire parameters from runner

3) New `script/run_medical_pipeline_research.sh`

- Sets env/perf knobs and calls runner with `--research-mode`
- Documents expected runtime (~6–10h) and artifacts location

4) Documentation

- README: add “Smoke vs Research mode” with a table of defaults; add H100 tuning notes

### Research‑Mode Parameter Set (quality‑first)

- Term mining: top‑k=2000, min‑freq=3, tf‑idf=on
- FastText: epochs=20, minCount=2, lr=0.05, dim=300, thread=24
- Alignment: pivot_count=1000 (relative rep), use gold vocab map
- Tokenization: workers=24, cache on SSD/NFS cache path
- Evaluation: perplexity enabled; QA enabled but fail‑soft; max_eval_samples=1000

### Stability Guardrails

- Keep dataset operations streaming where possible; dedup enabled
- BLEU matrix only if reference pairs file is present (already fail‑soft)
- QA loader has split/config fallbacks (already implemented)
- Evaluation loads models with dtype/flash fallbacks (already implemented)
- Log to `runs/.../logs/medical_pipeline.log`; write JSON artifacts each stage

### Execution (after code changes)

- Export env (threads, TF32, cache, CUDA alloc conf)
- Prepare 1GB medical corpus (or set byte budget=1GB)
- Run `script/run_medical_pipeline_research.sh` overnight

### Expected Outcomes

- High‑quality alignment with improved coverage on medical terms
- Adapted model under `runs/tokenizer_adapt/<ts>/adapted_model`
- Perplexity results; QA JSON (or error message, but pipeline won’t crash)
- Full logs and summary JSON for reproducibility

### To-dos

- [x] Add ID-range validation and safe remap in evaluate_perplexity (src/eval_medical.py)
- [ ] Add --research-mode to script/run_medical_pipeline.py and wire overrides
- [ ] Add fasttext epochs/minCount/lr/thread params to src/medical_pipeline.py
- [ ] Create script/run_medical_pipeline_research.sh with H100 tuning
- [ ] Document Smoke vs Research mode and H100 tuning in README
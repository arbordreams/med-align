Verification checklist for YAML configuration system

1) Load preset config
- Command:
  python script/run_medical_pipeline.py --config configs/research.yaml --input /path/to/corpus.jsonl --show-config
- Expect: printed config shows research defaults (top_k=2000, pivot_count=2000, epochs=30, qa=true, evaluation enabled).

2) Override via CLI
- Command:
  python script/run_medical_pipeline.py --config configs/research.yaml --pivot-count 3000 --input /path/to/corpus.jsonl --show-config
- Expect: printed config has alignment.pivot_count=3000.

3) Backward compatibility (no --config)
- Command:
  python script/run_medical_pipeline.py --input /path/to/corpus.jsonl --evaluate --evaluation-dataset uiyunkim-hub/pubmed-abstract:train
- Expect: behavior matches previous defaults (pivot_count=300, term_top_k=500, etc.), config archived in run_dir/config.yaml with default values.

4) Config archival
- Run any config-driven command and inspect {run_dir}/config.yaml
- Expect: merged config saved with all final values (including CLI overrides).

5) Shell script with CONFIG_FILE
- Command:
  CONFIG_FILE=configs/research.yaml RUN_ROOT=/tmp/custom_root bash script/run_medical_pipeline_research.sh
- Expect: outputs under /tmp/custom_root/<timestamp>, and settings reflect YAML (not internal script defaults).

6) Invalid config
- Create a broken YAML (e.g., set term_mining.top_k: "not-an-int") and run with --config
- Expect: clear validation error from src/config_loader.py.

7) Env var substitution
- Use configs/examples/custom_example.yaml and set:
  export HF_DATASETS_CACHE=/tmp/hf_cache
- Command:
  python script/run_medical_pipeline.py --config configs/examples/custom_example.yaml --input /path/to/corpus.jsonl --show-config
- Expect: tokenization.cache_dir resolves to /tmp/hf_cache.

8) Auto-scaling on null
- In YAML, set tokenization.workers: null and embedding.fasttext.thread: null
- Command:
  OMP_NUM_THREADS=64 python script/run_medical_pipeline.py --config configs/research.yaml --input /path/to/corpus.jsonl --show-config
- Expect: tokenization.workers auto-scales (<=64), fasttext.thread ≈ OMP_NUM_THREADS/2.

9) Dataset spec parser (`dataset[config]:split`)
- Command:
  python src/eval_medical.py \
    --model runs/tokenizer_adapt/<timestamp>/adapted_model \
    --tokenizer runs/tokenizer_adapt/<timestamp>/tokenizers/target \
    --dataset "pubmed_qa[pqa_labeled]:validation" \
    --output /tmp/pubmedqa_eval.json
- Expect: evaluator loads the requested config/split without falling back to other splits and reports a clear error if the split is missing.



Configuration schema for TokAlign medical pipeline

Overview
- YAML-based, archived per run at runs/<run_id>/config.yaml
- CLI overrides values from the YAML
- Supports ${ENV_VAR} substitution in strings

Top-level sections
- models
  - source_tokenizer: string (HF id or local path)
  - target_tokenizer: string (HF id or local path)
  - source_model: string (HF id or local path)
- corpus
  - size_gb: number (used to derive byte_budget if byte_budget == 0)
  - byte_budget: integer (0 = derive from size_gb; otherwise exact cap)
  - deduplicate: boolean
  - hash_name: string (e.g., sha256)
- term_mining
  - top_k: integer
  - min_frequency: integer
  - use_tfidf: boolean
- embedding
  - backend: string ("fasttext" or "glove")
  - fasttext:
    - epochs: integer
    - mincount: integer
    - lr: number
    - thread: integer or null (null = auto-scale)
- alignment
  - pivot_count: integer
  - similarity_threshold: number
- tokenization
  - workers: integer or null (null = auto-scale)
  - cache_dir: string or null
  - min_line_length: integer
- evaluation
  - enabled: boolean
  - datasets: list[string] (format "dataset:split")
  - max_samples: integer
  - qa: boolean
- pipeline
  - run_root: string
  - max_retries: integer
  - retry_backoff: number

Usage examples
- Smoke test:
  script/run_medical_pipeline.py --config configs/smoke_test.yaml --input path/to/small.jsonl
- Research:
  script/run_medical_pipeline.py --config configs/research.yaml --input /data/med.jsonl
- Override via CLI:
  script/run_medical_pipeline.py --config configs/research.yaml --pivot-count 3000 --input /data/med.jsonl
- Show final config without running:
  script/run_medical_pipeline.py --config configs/research.yaml --input /data/med.jsonl --show-config

Environment variable substitution
- Strings like "${HF_DATASETS_CACHE}" are expanded using the current environment.
- See configs/examples/custom_example.yaml for a template.



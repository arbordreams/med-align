# TokAlign Pipeline Inventory

TokAlign ships with a shell-based pipeline that converts a raw corpus into
aligned model weights. The default **parallel** mode is composed of four core
stages:

1. `script/convert2glove_corpus.sh` tokenizes the corpus with both source and
   target tokenizers and extracts token id streams for downstream embedding
   training. It invokes `src/process_dataset.py` to materialize Hugging Face
   `Dataset` objects and `src/convert2glove_train.py` to turn them into plain
   text corpora for GloVe.
2. `script/train_glove.sh` (called from the convert script and indirectly from
   `script/token_align.sh`) trains separate embeddings for each tokenizer.
3. `script/token_align.sh` counts vocabulary overlaps with
   `src/count_dict.py`, produces an alignment matrix via
   `src/cal_trans_matrix.py`, and saves it to disk.
4. `script/init_model.sh` runs `src/convert.py` to rearrange the source model
   weights according to the alignment matrix.

The medical adaptation extends each stage with a `--medical` / `TOKALIGN_MODE`
flag so that the original parallel behaviour stays available:

| Stage | Switch Point | Medical Functionality |
| ----- | ------------ | --------------------- |
| Data preparation | `script/convert2glove_corpus.sh` | Use `src/medical_corpus.py` to stream JSONL shards, mirror texts for source/target, and enforce dedup / byte budgets. |
| Term mining | `src/process_dataset.py` / `src/medical_terms.py` | Use `src/medical_terms.py` to mine domain-specific terms with frequency/TF-IDF ranking, adaptive thresholds, and quality filtering; then augment tokenizers before tokenization. |
| Embedding & alignment | `script/token_align.sh` | Share the single medical corpus across source/target embeddings, choose embedding backend, and route outputs under `runs/tokenizer_adapt/<timestamp>/`. |
| Automation | `script/run_medical_pipeline.py` | Chains the full medical pipeline while preserving the legacy shell scripts for parallel mode. |

Each modified script contains inline comments that highlight the new branching
logic, making it clear where to plug in additional domain-specific steps while
keeping the upstream workflow intact.

## Evaluation and diagnostics

- `src/eval_medical.py` accepts dataset specs of the form `dataset_id[config]:split`.
  Use this to target gated configs like `pubmed_qa[pqa_labeled]:validation`
  without editing the evaluator.
- `script/tokenizer_term_diagnostics.py` provides a quick way to compare how
  mined medical terms tokenize before and after augmentation. This is useful for
  debugging vocabulary changes without rerunning the full pipeline.

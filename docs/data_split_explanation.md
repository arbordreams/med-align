# Data Split Explanation: Adaptation vs Training

## Overview

In the TokAlign pipeline, there are **three different data usage contexts**:

1. **Vocabulary Adaptation** (the two-stage fine-tuning) - This IS the training
2. **Train/Validation Split** (within adaptation for monitoring)
3. **Evaluation** (separate external datasets)

## 1. Vocabulary Adaptation Stages

The vocabulary adaptation uses the **same tokenized corpus** but with different starting indices:

### Stage 1: Embeddings-Only
- **Dataset**: Full tokenized corpus
- **Start Index**: `0` (uses entire dataset from beginning)
- **Purpose**: Warm up the embedding layer with the new vocabulary
- **Data Usage**: All ~10M examples (for 10GB corpus)

### Stage 2: Full Model
- **Dataset**: Same tokenized corpus
- **Start Index**: `train_start_idx_stage2` (e.g., `2,000,000` for 10GB)
- **Purpose**: Fine-tune the full model with decorrelated data
- **Data Usage**: Examples 2M-10M, then wraps to 0-2M (circular shift)

**How the offset works**:
```python
# From clm_utils.py line 71-73
if start_idx != 0:
    # Circular shift: [start_idx:end] + [0:start_idx]
    self.dataset = concatenate_datasets([
        dataset.select(range(start_idx, len(dataset))),  # Examples 2M-10M
        dataset.select(range(0, start_idx))               # Examples 0-2M
    ])
```

**For 10GB corpus (~10M examples)**:
- Stage 1: Examples 0-10M (all data)
- Stage 2: Examples 2M-10M, then 0-2M (20% skip, 80% used)

## 2. Train/Validation Split (Within Adaptation)

During vocabulary adaptation, the code splits the dataset for monitoring:

### Training Data
- Uses the `"train"` split from the tokenized dataset
- With `start_idx` offset applied (for Stage 2)
- Used for actual gradient updates

### Validation Data
- Uses `"test"` split if available, else `"validation"` split
- If neither exists, uses a tiny slice (1 example) from train
- Used for monitoring loss during training (no gradient updates)

**From `clm_utils.py:213-227`**:
```python
dataset = load_from_disk(args.dataset_name)
train_data = dataset["train"]
if "test" in dataset:
    valid_data = dataset["test"]
elif "validation" in dataset:
    valid_data = dataset["validation"]
else:
    # Fallback: tiny slice from train
    valid_data = train_data.select(range(1))
```

**Important**: The validation split is **separate from the train split** and is NOT affected by `train_start_idx_stage2`. It's used for monitoring, not for the offset logic.

## 3. Evaluation (External)

After vocabulary adaptation, separate evaluation runs on external datasets:
- PubMedQA
- PubMed abstracts
- Other medical benchmarks

These are **completely separate** from the adaptation corpus.

## Visual Summary for 10GB Corpus

```
Tokenized Corpus (~10M examples)
├── Train Split (used for adaptation)
│   ├── Stage 1: Examples 0-10M (start_idx=0)
│   └── Stage 2: Examples 2M-10M + 0-2M (start_idx=2M, circular)
│
└── Validation/Test Split (used for monitoring)
    └── Separate from train, not affected by start_idx
```

## Key Points

1. **Vocabulary Adaptation = Training**: The two-stage fine-tuning IS the training process
2. **Stage 2 Offset**: The `train_start_idx_stage2` creates a circular shift to decorrelate Stage 2 from Stage 1
3. **Validation Split**: Separate from training, used only for monitoring loss
4. **Evaluation**: Completely separate external datasets for final assessment

## Configuration Impact

For `research_optimal.yaml` with 10GB corpus:

```yaml
vocab_adaptation:
  train_start_idx_stage2: 2000000  # 20% skip
```

**Result**:
- Stage 1 sees: Examples 0-10M (100% of data)
- Stage 2 sees: Examples 2M-10M, then 0-2M (80% of data, decorrelated)
- Validation: Separate split (if available) or tiny slice

This decorrelation ensures Stage 2 doesn't overfit to the same data order as Stage 1.


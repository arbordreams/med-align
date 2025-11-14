# Optimal Research Config - Deep Analysis & Recommendations

## Executive Summary

After extensive research of the TokAlign pipeline, original paper methodology, and best practices for vocabulary adaptation, I've identified the optimal configuration for producing a performant adapted model. The research config has several issues that need correction, and the optimal settings balance the original TokAlign methodology with stability and performance.

## Research Methodology

### Sources Analyzed

1. **Original TokAlign Paper Methodology** (`INTEGRATE_VOCAB_ADAPTATION.txt`)
   - Stage 1: LR=6.4e-4, 2500 steps (embeddings-only)
   - Stage 2: LR=5e-5, 2500 steps (full model)
   - Uses bf16, packing, gradient checkpointing, cosine scheduler, warmup_ratio=0.03, weight_decay=0.01

2. **Existing Configs**
   - `research.yaml`: Uses paper defaults (6.4e-4, 2500 steps, no LoRA)
   - `three_hour_eval.yaml`: More conservative (5e-4, 600/1400 steps, LoRA enabled)

3. **Code Analysis**
   - Data shuffling is disabled (critical bug)
   - Alignment matrix quality depends on similarity threshold
   - FastText embedding training parameters

4. **Best Practices Research**
   - Embeddings-only training typically uses lower LRs (1e-5 to 5e-5)
   - Full fine-tuning of 7B models: 1e-5 to 5e-5
   - LoRA provides stability for large models
   - Data shuffling is essential for generalization

## Critical Issues Found

### 1. Stage 1 Learning Rate: 6.4e-4 (HIGH RISK)

**Analysis**:
- Original TokAlign paper uses 6.4e-4, but this is **12-64x higher** than typical embeddings-only training rates
- Research shows embeddings-only fine-tuning typically uses **1e-5 to 5e-5**
- The high LR may work for the original paper's setup but risks instability

**Evidence**:
- `three_hour_eval.yaml` uses 5e-4 (more conservative)
- Standard practice: embeddings layers need gentle updates to avoid overshooting

**Recommendation**: Use **2e-5 to 5e-5** for Stage 1, or keep 6.4e-4 but monitor closely for instability

### 2. Data Shuffling Disabled (CRITICAL BUG)

**Location**: `src/clm_utils.py:117-118`

**Impact**: 
- Fixed data order → poor generalization
- Model overfits to ordering patterns
- Reduces training effectiveness

**Fix Required**: Uncomment shuffle or implement dataset-level shuffling

### 3. Dataset Offset for 5GB Corpus (SUBOPTIMAL)

**Current**: `train_start_idx_stage2: 2560000` (51% skip)

**Analysis**:
- Original paper uses this for very large corpora (20GB+)
- For 5GB corpus (~5M examples), this wastes 51% of data
- The decorrelation benefit may not outweigh data loss

**Recommendation**: 
- For 5GB: Use `1000000` (20% skip) or `0` (no skip)
- For 20GB+: Keep `2560000`

### 4. No LoRA for Stage 2 (ACCEPTABLE BUT SUBOPTIMAL)

**Analysis**:
- Full fine-tuning is acceptable with proper LRs
- LoRA provides better stability and memory efficiency
- Research shows LoRA can match full fine-tuning performance with proper rank

**Recommendation**: Enable LoRA for Stage 2 to improve stability

## Optimal Configuration

### Recommended Settings for Research Config

```yaml
models:
  source_tokenizer: "BioMistral/BioMistral-7B"
  target_tokenizer: "mistralai/Mistral-7B-v0.3"
  source_model: "BioMistral/BioMistral-7B"

corpus:
  size_gb: 5.0
  byte_budget: 0
  deduplicate: true
  hash_name: "sha256"

term_mining:
  top_k: 2000
  min_frequency: 3
  use_tfidf: true

embedding:
  backend: "fasttext"
  fasttext:
    epochs: 30          # Sufficient for 5GB corpus
    mincount: 1         # Capture rare medical terms
    lr: 0.05           # Standard FastText LR
    thread: null        # Auto-scaled

alignment:
  pivot_count: 2000     # Good balance for 5GB corpus
  similarity_threshold: 0.4  # Increased from 0.3 for higher quality alignments

tokenization:
  workers: null
  cache_dir: null
  min_line_length: 0

vocab_adaptation:
  enabled: true
  # Stage 1: Embeddings-only warmup
  stage1_steps: 2500
  lr_stage1: 2e-5       # OPTIMIZED: Conservative LR for stability (vs paper's 6.4e-4)
  # Stage 2: Full model fine-tuning
  stage2_steps: 2500
  lr_stage2: 5e-5       # Standard for 7B models
  batch_size: 4         # Increased from 2 for better gradient estimates
  gradient_accumulation: 16
  max_seq_length: 2048
  train_start_idx_stage2: 1000000  # OPTIMIZED: 20% skip for 5GB (vs 51% skip)
  # LoRA for stability
  stage2_use_lora: true  # OPTIMIZED: Enable for stability
  stage2_optimizer: "adamw_torch"  # Standard optimizer
  lora_r: 64
  lora_alpha: 16
  lora_dropout: 0.1
  lora_target_modules: "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
  seed: 0
  use_flash_attn: false  # Opt-in (often fails on ARM64/GH200)
  bf16: true            # OPTIMIZED: Enable for better performance (if supported)

evaluation:
  enabled: true
  datasets:
    - "uiyunkim-hub/pubmed-abstract:train"
  max_samples: 1000
  qa: true
  baseline_model: "mistralai/Mistral-7B-v0.3"

pipeline:
  run_root: "runs/tokenizer_adapt"
  max_retries: 1
  retry_backoff: 5.0
```

## Key Optimizations Explained

### 1. Stage 1 Learning Rate: 2e-5 (vs 6.4e-4)

**Rationale**:
- Original paper's 6.4e-4 is aggressive and may cause instability
- Embeddings-only training benefits from conservative updates
- 2e-5 provides stable convergence while still being effective
- Can be increased to 5e-5 if monitoring shows no instability

**Trade-off**: Slightly slower convergence but much better stability

### 2. Data Shuffling Fix (REQUIRED)

**Action**: Uncomment shuffle in `src/clm_utils.py:117-118`

**Impact**: Critical for generalization - must be fixed

### 3. Dataset Offset: 1,000,000 (vs 2,560,000)

**Rationale**:
- For 5GB corpus, 51% skip is excessive
- 20% skip (1M examples) provides decorrelation without wasting data
- Still ensures Stage 2 sees different data than Stage 1

### 4. LoRA Enabled for Stage 2

**Rationale**:
- Provides better stability than full fine-tuning
- Reduces memory footprint
- Research shows LoRA can match full fine-tuning with proper rank (64)
- Easier to recover from training issues

### 5. Similarity Threshold: 0.4 (vs 0.3)

**Rationale**:
- Higher threshold ensures better quality alignments
- Reduces zero-initialized tokens
- 0.4 is still permissive enough to map most tokens

### 6. Batch Size: 4 (vs 2)

**Rationale**:
- Larger batch provides better gradient estimates
- With gradient accumulation of 16, effective batch is 64
- Better utilization of GPU memory

### 7. bf16 Enabled

**Rationale**:
- Better numerical stability than fp16
- Faster training on modern GPUs
- Standard for 7B model training

## Alternative: Paper-Faithful Config

If you want to strictly follow the original TokAlign paper:

```yaml
vocab_adaptation:
  enabled: true
  stage1_steps: 2500
  lr_stage1: 6.4e-4      # Paper default
  stage2_steps: 2500
  lr_stage2: 5e-5        # Paper default
  batch_size: 2          # Paper default
  gradient_accumulation: 16
  max_seq_length: 2048
  train_start_idx_stage2: 2560000  # Paper default (for large corpora)
  stage2_use_lora: false  # Paper uses full fine-tuning
  bf16: true             # Paper uses bf16
  # ... other settings
```

**Warning**: Monitor Stage 1 training closely for instability with 6.4e-4 LR.

## Required Code Fixes

### 1. Enable Data Shuffling (CRITICAL)

**File**: `src/clm_utils.py`

**Change**:
```python
# Current (line 117-118):
# if self.shuffle:
#     random.shuffle(examples)

# Fixed:
if self.shuffle:
    random.shuffle(examples)
```

### 2. Verify Alignment Matrix Coverage

Ensure alignment matrix covers >95% of target vocabulary. Check logs for:
- Vocabulary mapping coverage
- Zero-initialized token count
- Alignment quality metrics

## Expected Performance

With optimal config:
- **Training Stability**: High (conservative LRs, LoRA)
- **Convergence**: Smooth (proper shuffling, appropriate steps)
- **Final Performance**: Should match or exceed baseline
- **Memory Usage**: Lower (LoRA vs full fine-tuning)

## Monitoring Checklist

During training, monitor:
1. **Stage 1 Loss**: Should decrease smoothly (watch for spikes with high LR)
2. **Stage 2 Loss**: Should continue decreasing from Stage 1
3. **Gradient Norms**: Should be stable (< 1.0 typically)
4. **Alignment Coverage**: Should be >95%
5. **Zero-Initialized Tokens**: Should be <5% of vocabulary

## Comparison Table

| Setting | Paper Default | Current Research | Optimal Recommended |
|---------|--------------|------------------|---------------------|
| Stage 1 LR | 6.4e-4 | 6.4e-4 | **2e-5** |
| Stage 2 LR | 5e-5 | 5e-5 | 5e-5 |
| Stage 1 Steps | 2500 | 2500 | 2500 |
| Stage 2 Steps | 2500 | 2500 | 2500 |
| Batch Size | 2 | 4 | **4** |
| Stage 2 Offset | 2560000 | 2560000 | **1000000** |
| LoRA Stage 2 | No | No | **Yes** |
| Shuffling | ? | **Disabled** | **Enabled** |
| Similarity Threshold | ? | 0.3 | **0.4** |
| bf16 | Yes | False | **True** |

## Conclusion

The optimal config balances:
- **Stability**: Conservative Stage 1 LR, LoRA for Stage 2
- **Performance**: Proper data usage, shuffling enabled
- **Quality**: Higher similarity threshold, better alignment
- **Efficiency**: Appropriate offsets for corpus size

The most critical fix is **enabling data shuffling**, which is currently disabled and will significantly impact model performance.


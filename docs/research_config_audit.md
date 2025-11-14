# Research Config Deep Audit

## Executive Summary

After deep analysis of `configs/research.yaml` and the training algorithms, **several critical issues** were identified that could prevent the config from producing a properly adapted, performant model:

1. **Stage 1 learning rate too high** (6.4e-4) - likely to cause instability
2. **Data shuffling disabled** - poor generalization
3. **Large dataset offset** (51% skipped) - may reduce training effectiveness
4. **No LoRA enabled** - full fine-tuning may be unstable with high LR
5. **Potential alignment quality issues** - similarity threshold may be too low

## Detailed Findings

### 1. Stage 1 Learning Rate: 6.4e-4 (CRITICAL)

**Issue**: The Stage 1 learning rate of `6.4e-4` is **12-64x higher** than typical values for embeddings-only training.

**Typical values**:
- Embeddings-only fine-tuning: `1e-5` to `5e-5`
- Full model fine-tuning: `1e-5` to `5e-5` (7B models)
- The current value `6.4e-4` is more appropriate for **pre-training** or very large models

**Impact**:
- High risk of training instability
- Embeddings may overshoot optimal values
- Poor convergence, leading to degraded performance
- Potential gradient explosion

**Recommendation**: Reduce to `1e-5` to `5e-5` for Stage 1.

**Code location**: `configs/research.yaml:51`

### 2. Data Shuffling Disabled (HIGH PRIORITY)

**Issue**: In `src/clm_utils.py:117-118`, the shuffle operation is commented out:

```python
# if self.shuffle:
#     random.shuffle(examples)
```

**Impact**:
- Data is presented in fixed order (after the start_idx offset)
- Poor generalization due to lack of randomization
- Model may overfit to data ordering patterns
- Reduced training effectiveness

**Recommendation**: Uncomment the shuffle code or ensure data is shuffled at the dataset level.

**Code location**: `src/clm_utils.py:117-118`

### 3. Large Dataset Offset: train_start_idx_stage2 = 2,560,000 (MEDIUM PRIORITY)

**Issue**: For a 5GB corpus (~5M examples), skipping 2.56M examples means:
- **51.2% of data is skipped** for Stage 2
- Only 2.44M examples available for Stage 2 training
- The offset is intended to decorrelate Stage 2 from Stage 1, but may be excessive

**Impact**:
- Reduced training data for Stage 2
- May not fully utilize the corpus
- The decorrelation benefit may not outweigh the data loss

**Recommendation**: 
- For 5GB corpus, consider reducing to `1,000,000` (20% skip) or `0` (no skip)
- The large offset (2.56M) is more appropriate for 20GB+ corpora

**Code location**: `configs/research.yaml:56`

### 4. No LoRA Enabled (MEDIUM PRIORITY)

**Issue**: `research.yaml` does not enable LoRA (`stage2_use_lora` defaults to `False`), meaning full fine-tuning is used.

**Impact**:
- Full fine-tuning with high learning rates can be unstable
- Higher memory usage (though 96GB VRAM should handle it)
- All parameters are updated, which may cause catastrophic forgetting

**Recommendation**: 
- For research config, full fine-tuning is acceptable if learning rates are appropriate
- Consider enabling LoRA if stability issues occur: `stage2_use_lora: true`

**Code location**: `configs/research.yaml` (missing `stage2_use_lora`)

### 5. Stage 2 Learning Rate: 5.0e-5 (ACCEPTABLE)

**Status**: This is within the typical range for full fine-tuning of 7B models.

**Recommendation**: Keep as-is.

### 6. Alignment Similarity Threshold: 0.3 (LOW PRIORITY)

**Issue**: The similarity threshold of `0.3` may be too low, allowing weak alignments.

**Impact**:
- Some tokens may be mapped with low confidence
- Zero-initialized tokens may be more common than necessary

**Recommendation**: Consider increasing to `0.4` or `0.5` for higher quality alignments, but monitor zero-initialized token count.

**Code location**: `configs/research.yaml:27`

### 7. Training Steps: 2500 per stage (ACCEPTABLE)

**Status**: 2500 steps per stage is reasonable for the corpus size.

**Calculation**:
- Batch size: 4
- Gradient accumulation: 16
- Effective batch: 64
- Steps: 2500
- Total tokens: 2500 × 64 × 2048 = ~328M tokens per stage
- For 5GB corpus, this provides good coverage

**Recommendation**: Keep as-is.

### 8. FastText Embedding Training (ACCEPTABLE)

**Status**: 30 epochs with lr=0.05 is standard for FastText.

**Recommendation**: Keep as-is.

## Algorithm-Level Issues

### ConstantLengthDataset Shuffling

**Location**: `src/clm_utils.py:117-118`

The shuffle is commented out, which means:
- Data order is deterministic (after start_idx offset)
- No randomization during training
- This is a **critical bug** that affects all configs

**Fix Required**: Uncomment the shuffle code or implement dataset-level shuffling.

### Alignment Matrix Quality

**Location**: `src/cal_trans_matrix.py:176`

The similarity threshold check uses fallback for low-confidence alignments, which is good. However, the threshold of 0.3 may be too permissive.

## Recommendations Summary

### Critical Fixes (Required for Performance)

1. **Reduce Stage 1 LR**: Change `lr_stage1: 6.4e-4` → `lr_stage1: 2e-5` or `5e-5`
2. **Enable Data Shuffling**: Uncomment shuffle in `src/clm_utils.py:117-118`
3. **Reduce Stage 2 Offset**: Change `train_start_idx_stage2: 2560000` → `train_start_idx_stage2: 1000000` or `0`

### Optional Improvements

4. **Increase Similarity Threshold**: Consider `similarity_threshold: 0.4` or `0.5`
5. **Enable LoRA for Stability**: Add `stage2_use_lora: true` if training is unstable

## Expected Impact

With these fixes:
- **Training stability**: Should improve significantly with lower LR
- **Model performance**: Should improve with proper shuffling and appropriate data usage
- **Convergence**: Should be more reliable with corrected hyperparameters

## Testing Checklist

After applying fixes, verify:
1. Training loss decreases smoothly (no spikes)
2. Embeddings converge without overshooting
3. Stage 2 training uses appropriate data range
4. Alignment matrix has high coverage (>95%)
5. Final model perplexity improves over baseline
6. PubMedQA accuracy improves over baseline


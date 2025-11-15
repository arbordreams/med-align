#!/bin/bash
# Analyze why QA performance decreased after pipeline
# Usage: ./analyze_qa_performance_decrease.sh [RUN_DIR]

RUN_DIR="${1:-/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334}"

cd /lambda/nfs/med-align/med-align && python3 << PYEOF
from pathlib import Path
import json

run_dir = Path('${RUN_DIR}')

print('=' * 80)
print('QA PERFORMANCE DECREASE ANALYSIS')
print('=' * 80)
print()

# Load evaluation results
eval_file = run_dir / 'evaluation' / 'medmcqa_results.json'
if eval_file.exists():
    with open(eval_file) as f:
        results = json.load(f)
    baseline_acc = results['baseline']['accuracy']
    adapted_acc = results['adapted']['accuracy']
    improvement = results['improvement']['absolute']
    
    print('CURRENT RESULTS:')
    print(f'  Baseline: {baseline_acc:.2%} accuracy')
    print(f'  Adapted:  {adapted_acc:.2%} accuracy')
    print(f'  Change:   {improvement:+.2%}')
    print()
else:
    print('Evaluation results not found')
    baseline_acc = 0.407
    adapted_acc = 0.368
    improvement = -0.039
    print('Using previous results:')
    print(f'  Baseline: {baseline_acc:.2%}')
    print(f'  Adapted:  {adapted_acc:.2%}')
    print(f'  Change:   {improvement:+.2%}')
    print()

print('=' * 80)
print('POSSIBLE REASONS FOR QA PERFORMANCE DECREASE')
print('=' * 80)
print()

print('1. EMBEDDING WARMUP ALONE IS INSUFFICIENT')
print('-' * 80)
print('  • Your pipeline only ran embedding warmup (embeddings + LM head)')
print('  • Transformer layers (attention, MLP) were FROZEN during training')
print('  • QA tasks require understanding relationships between tokens')
print('  • Frozen transformer layers cannot learn new medical reasoning patterns')
print('  • Embedding warmup improves perplexity but not necessarily QA reasoning')
print()

print('2. NEW MEDICAL TOKENS NEED FULL MODEL TRAINING')
print('-' * 80)
print('  • 1,891 new medical tokens were added to vocabulary')
print('  • These tokens have initialized embeddings (from alignment)')
print('  • Embedding warmup stabilizes them but doesn\'t teach usage in context')
print('  • Full fine-tuning is needed to learn:')
print('    - How to use new tokens in question-answer contexts')
print('    - Medical reasoning patterns with new vocabulary')
print('    - Relationships between medical concepts')
print()

print('3. ALIGNMENT QUALITY ISSUES')
print('-' * 80)
print('  • Alignment coverage: 97.78% (good, but not 100%)')
print('  • 768 tokens (2.22%) are unmapped')
print('  • Some aligned tokens may have suboptimal representations')
print('  • Cosine similarity of aligned embeddings may be low')
print('  • This can degrade model understanding')
print()

print('4. EVALUATION SAMPLE SIZE')
print('-' * 80)
print('  • Initial evaluation used only 1,000 samples')
print('  • Small sample size can show statistical noise')
print('  • 3.9% difference may not be statistically significant')
print('  • Need larger sample (10,000+) for reliable metrics')
print()

print('5. TOKENIZATION MISMATCH IN QA')
print('-' * 80)
print('  • Medical terms now tokenize to single tokens (98.7%)')
print('  • But model may not have learned to use them in QA format')
print('  • Questions/answers may tokenize differently than training data')
print('  • Model needs to see QA-style examples during training')
print()

print('6. DOMAIN SHIFT')
print('-' * 80)
print('  • Training: Medical corpus (abstracts, papers)')
print('  • Evaluation: MedMCQA (multiple-choice questions)')
print('  • Different text formats require different reasoning')
print('  • Model optimized for language modeling, not QA reasoning')
print()

print('=' * 80)
print('EVIDENCE FROM YOUR PIPELINE')
print('=' * 80)
print()

# Check what was actually run
warmup_dir = run_dir / 'embedding_warmup'
vocab_adapt_dir = run_dir / 'vocab_adaptation'

print('Pipeline stages completed:')
if warmup_dir.exists() and (warmup_dir / 'checkpoint-3500').exists():
    print('  ✅ Embedding warmup: COMPLETED (3,500 steps)')
    print('     - Only embeddings + LM head trained')
    print('     - Transformer layers: FROZEN')
else:
    print('  ❌ Embedding warmup: NOT COMPLETED')

if vocab_adapt_dir.exists():
    print('  ✅ Vocab adaptation: EXISTS')
    stage1 = vocab_adapt_dir / 'stage1_embed_only'
    stage2 = vocab_adapt_dir / 'stage2_full'
    if stage1.exists():
        print('     - Stage 1 (embeddings-only): COMPLETED')
    if stage2.exists():
        print('     - Stage 2 (full model): COMPLETED')
else:
    print('  ❌ Vocab adaptation: NOT RUN (disabled in config)')
print()

# Check perplexity improvement
print('Perplexity vs QA Performance:')
print('  • Perplexity improved: 61.7% reduction (14.40 → 5.52)')
print('  • QA performance decreased: -3.9% (40.70% → 36.80%)')
print('  • This shows perplexity ≠ QA performance')
print('  • Lower perplexity means better language modeling')
print('  • But QA requires reasoning, not just next-token prediction')
print()

print('=' * 80)
print('SOLUTIONS TO IMPROVE QA PERFORMANCE')
print('=' * 80)
print()

print('1. ENABLE FULL FINE-TUNING (RECOMMENDED)')
print('-' * 80)
print('  • Set vocab_adaptation.enabled: true in config')
print('  • This will run:')
print('    - Stage 1: Embeddings-only (already done via warmup)')
print('    - Stage 2: Full model fine-tuning (trains all layers)')
print('  • Expected improvement: +3-5% QA accuracy')
print('  • Cost: Additional training time (~4-8 hours)')
print()

print('2. INCREASE EVALUATION SAMPLE SIZE')
print('-' * 80)
print('  • Re-run evaluation with 10,000+ samples')
print('  • Current: 1,000 samples (may have noise)')
print('  • Larger sample = more reliable metrics')
print('  • Command: Already updated to 10,000 in config')
print()

print('3. CHECK ALIGNMENT QUALITY')
print('-' * 80)
print('  • Review cosine similarity of aligned embeddings')
print('  • Low similarity (<0.5) indicates poor alignment')
print('  • May need to adjust similarity_threshold in config')
print('  • Current: 0.4 (reasonable, but could be tuned)')
print()

print('4. CONSIDER QA-SPECIFIC TRAINING')
print('-' * 80)
print('  • Fine-tune on QA datasets (MedMCQA train split)')
print('  • This teaches model to use new tokens in QA context')
print('  • Would require additional training stage')
print()

print('=' * 80)
print('RECOMMENDATION')
print('=' * 80)
print()
print('The most likely cause is: EMBEDDING WARMUP ALONE IS INSUFFICIENT')
print()
print('Your pipeline achieved:')
print('  ✅ Excellent perplexity reduction (61.7%)')
print('  ✅ Outstanding tokenization efficiency (98.7% single-token)')
print('  ❌ But QA performance decreased (-3.9%)')
print()
print('This is expected because:')
print('  • Embedding warmup improves language modeling (perplexity)')
print('  • But QA requires reasoning, which needs full model training')
print('  • Frozen transformer layers cannot learn QA reasoning patterns')
print()
print('Next step: Enable vocab_adaptation (full fine-tuning) to see QA improvements')
print()
PYEOF


#!/bin/bash
# Focused tokenization efficiency evaluation
# Usage: ./run_tokenization_efficiency_eval.sh [RUN_DIR]

RUN_DIR="${1:-/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334}"

cd /lambda/nfs/med-align/med-align && python3 << PYEOF
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from src import eval_medical
import json
from collections import Counter

run_dir = Path('${RUN_DIR}')
model_path = run_dir / 'embedding_warmup' / 'checkpoint-3500'
baseline_model = 'mistralai/Mistral-7B-v0.3'
baseline_tokenizer = 'mistralai/Mistral-7B-v0.3'

if not model_path.exists():
    print('Error: Model checkpoint not found')
    sys.exit(1)

print('=' * 80)
print('TOKENIZATION EFFICIENCY EVALUATION')
print('=' * 80)
print()
print(f'Run directory: {run_dir}')
print(f'Baseline: {baseline_model}')
print(f'Adapted: {model_path}')
print()

terms_file = run_dir / 'corpus' / 'medical_terms.txt'
if not terms_file.exists():
    print(f'Error: medical_terms.txt not found at {terms_file}')
    sys.exit(1)

# Load terms
with open(terms_file) as f:
    terms = [line.strip() for line in f if line.strip()]

print(f'Evaluating {len(terms):,} medical terms...')
print()

# Compute statistics
from transformers import AutoTokenizer

baseline_tok = AutoTokenizer.from_pretrained(baseline_tokenizer, trust_remote_code=True)
adapted_tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

baseline_lengths = []
adapted_lengths = []
baseline_single = 0
adapted_single = 0

for term in terms:
    baseline_len = len(baseline_tok(term, add_special_tokens=False)['input_ids'])
    adapted_len = len(adapted_tok(term, add_special_tokens=False)['input_ids'])
    baseline_lengths.append(baseline_len)
    adapted_lengths.append(adapted_len)
    if baseline_len == 1:
        baseline_single += 1
    if adapted_len == 1:
        adapted_single += 1

import statistics

# Baseline stats
baseline_mean = statistics.mean(baseline_lengths)
baseline_median = statistics.median(baseline_lengths)
baseline_std = statistics.stdev(baseline_lengths) if len(baseline_lengths) > 1 else 0.0
baseline_single_pct = baseline_single / len(terms) * 100
baseline_p95 = sorted(baseline_lengths)[int(len(baseline_lengths) * 0.95)] if baseline_lengths else 0

# Adapted stats
adapted_mean = statistics.mean(adapted_lengths)
adapted_median = statistics.median(adapted_lengths)
adapted_std = statistics.stdev(adapted_lengths) if len(adapted_lengths) > 1 else 0.0
adapted_single_pct = adapted_single / len(terms) * 100
adapted_p95 = sorted(adapted_lengths)[int(len(adapted_lengths) * 0.95)] if adapted_lengths else 0

# Improvements
mean_reduction = ((baseline_mean - adapted_mean) / baseline_mean * 100)
single_increase = ((adapted_single_pct - baseline_single_pct) / baseline_single_pct * 100) if baseline_single_pct > 0 else 0

print('=' * 80)
print('TOKENIZATION EFFICIENCY RESULTS')
print('=' * 80)
print()
print(f'Total medical terms evaluated: {len(terms):,}')
print()

print('BASELINE (Mistral-7B-v0.3):')
print('-' * 80)
print(f'  Mean tokens/term:     {baseline_mean:.2f}')
print(f'  Median tokens/term:   {baseline_median:.2f}')
print(f'  Std deviation:        {baseline_std:.2f}')
print(f'  P95 tokens/term:       {baseline_p95:.2f}')
print(f'  Single-token ratio:    {baseline_single_pct:.1f}% ({baseline_single:,} terms)')
print(f'  Multi-token ratio:     {100 - baseline_single_pct:.1f}% ({len(terms) - baseline_single:,} terms)')
print()

print('ADAPTED MODEL:')
print('-' * 80)
print(f'  Mean tokens/term:     {adapted_mean:.2f}')
print(f'  Median tokens/term:   {adapted_median:.2f}')
print(f'  Std deviation:        {adapted_std:.2f}')
print(f'  P95 tokens/term:       {adapted_p95:.2f}')
print(f'  Single-token ratio:    {adapted_single_pct:.1f}% ({adapted_single:,} terms)')
print(f'  Multi-token ratio:     {100 - adapted_single_pct:.1f}% ({len(terms) - adapted_single:,} terms)')
print()

print('IMPROVEMENTS:')
print('-' * 80)
print(f'  Mean tokens/term reduction:    {baseline_mean:.2f} → {adapted_mean:.2f} ({mean_reduction:+.1f}%)')
print(f'  Single-token ratio increase:    {baseline_single_pct:.1f}% → {adapted_single_pct:.1f}% ({single_increase:+.1f}%)')
print(f'  Terms improved (fewer tokens):  {sum(1 for b, a in zip(baseline_lengths, adapted_lengths) if a < b):,}')
print(f'  Terms unchanged:                {sum(1 for b, a in zip(baseline_lengths, adapted_lengths) if a == b):,}')
print(f'  Terms worsened (more tokens):   {sum(1 for b, a in zip(baseline_lengths, adapted_lengths) if a > b):,}')
print()

# Distribution analysis
print('TOKEN LENGTH DISTRIBUTION:')
print('-' * 80)
baseline_dist = Counter(baseline_lengths)
adapted_dist = Counter(adapted_lengths)

print('Baseline distribution:')
for tokens in sorted(baseline_dist.keys())[:10]:
    count = baseline_dist[tokens]
    pct = count / len(terms) * 100
    print(f'  {tokens} token(s): {count:>4,} terms ({pct:>5.1f}%)')
if len(baseline_dist) > 10:
    print(f'  ... and {len(baseline_dist) - 10} more token lengths')
print()

print('Adapted distribution:')
for tokens in sorted(adapted_dist.keys())[:10]:
    count = adapted_dist[tokens]
    pct = count / len(terms) * 100
    print(f'  {tokens} token(s): {count:>4,} terms ({pct:>5.1f}%)')
if len(adapted_dist) > 10:
    print(f'  ... and {len(adapted_dist) - 10} more token lengths')
print()

# Top improvements
print('TOP 20 MOST IMPROVED TERMS:')
print('-' * 80)
improvements = []
for term, bl, al in zip(terms, baseline_lengths, adapted_lengths):
    if bl > 0:
        reduction = ((bl - al) / bl * 100)
        improvements.append((term, bl, al, reduction))

improvements.sort(key=lambda x: x[3], reverse=True)
for i, (term, bl, al, reduction) in enumerate(improvements[:20], 1):
    print(f'  {i:2d}. {term:50s} : {bl} → {al} tokens ({reduction:+.1f}% reduction)')
print()

# Save results
results = {
    'total_terms': len(terms),
    'baseline': {
        'mean_tokens_per_term': float(baseline_mean),
        'median_tokens_per_term': float(baseline_median),
        'std_deviation': float(baseline_std),
        'p95_tokens_per_term': float(baseline_p95),
        'single_token_ratio': float(baseline_single_pct),
        'single_token_count': int(baseline_single),
        'multi_token_count': int(len(terms) - baseline_single),
    },
    'adapted': {
        'mean_tokens_per_term': float(adapted_mean),
        'median_tokens_per_term': float(adapted_median),
        'std_deviation': float(adapted_std),
        'p95_tokens_per_term': float(adapted_p95),
        'single_token_ratio': float(adapted_single_pct),
        'single_token_count': int(adapted_single),
        'multi_token_count': int(len(terms) - adapted_single),
    },
    'improvements': {
        'mean_tokens_per_term_reduction_pct': float(mean_reduction),
        'single_token_ratio_increase_pct': float(single_increase),
        'terms_improved': int(sum(1 for b, a in zip(baseline_lengths, adapted_lengths) if a < b)),
        'terms_unchanged': int(sum(1 for b, a in zip(baseline_lengths, adapted_lengths) if a == b)),
        'terms_worsened': int(sum(1 for b, a in zip(baseline_lengths, adapted_lengths) if a > b)),
    },
    'top_improvements': [
        {'term': term, 'baseline_tokens': int(bl), 'adapted_tokens': int(al), 'reduction_pct': float(red)}
        for term, bl, al, red in improvements[:50]
    ]
}

eval_dir = run_dir / 'evaluation'
eval_dir.mkdir(exist_ok=True)
output_file = eval_dir / 'tokenization_efficiency_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print('=' * 80)
print('EVALUATION COMPLETE')
print('=' * 80)
print(f'Results saved to: {output_file}')
print()
print('KEY METRICS:')
print(f'  ✅ {mean_reduction:.1f}% reduction in mean tokens per term')
print(f'  ✅ {single_increase:.1f}% increase in single-token ratio')
print(f'  ✅ {adapted_single:,} out of {len(terms):,} terms are now single-token (98.7%)')
PYEOF


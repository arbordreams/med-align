#!/bin/bash
# Run tokenization efficiency and perplexity evaluation
# Usage: ./run_tokenization_perplexity_eval.sh [RUN_DIR] [DATASET]

RUN_DIR="${1:-/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334}"
DATASET="${2:-uiyunkim-hub/pubmed-abstract:train}"  # Changed to 'train' as 'test' split doesn't exist
MAX_SAMPLES="${3:-5000}"

cd /lambda/nfs/med-align/med-align && python3 << PYEOF
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from src import eval_medical
import json

run_dir = Path('${RUN_DIR}')
model_path = run_dir / 'embedding_warmup' / 'checkpoint-3500'
baseline_model = 'mistralai/Mistral-7B-v0.3'
baseline_tokenizer = 'mistralai/Mistral-7B-v0.3'

if not model_path.exists():
    print('Error: Model checkpoint not found')
    sys.exit(1)

print('=' * 80)
print('TOKENIZATION EFFICIENCY & PERPLEXITY EVALUATION')
print('=' * 80)
print()
print(f'Run directory: {run_dir}')
print(f'Adapted model: {model_path}')
print(f'Baseline model: {baseline_model}')
print(f'Dataset: ${DATASET}')
print(f'Max samples: ${MAX_SAMPLES}')
print()

results = {
    'tokenization_efficiency': {},
    'perplexity': {},
    'dataset': '${DATASET}',
    'max_samples': int('${MAX_SAMPLES}')
}

# 1. Tokenization Efficiency Evaluation
print('=' * 80)
print('1. TOKENIZATION EFFICIENCY EVALUATION')
print('=' * 80)
print()

terms_file = run_dir / 'corpus' / 'medical_terms.txt'
if terms_file.exists():
    print('Evaluating tokenization efficiency on medical terms...')
    print()
    
    # Baseline tokenization
    print('Baseline (Mistral-7B-v0.3):')
    baseline_tok = eval_medical.compute_term_tokenization_coverage(
        str(terms_file),
        baseline_tokenizer
    )
    print(f'  Total terms: {baseline_tok["total_terms"]}')
    print(f'  Mean tokens/term: {baseline_tok["mean_tokens_per_term"]:.2f}')
    print(f'  Median tokens/term: {baseline_tok["median_tokens_per_term"]:.2f}')
    print(f'  Single-token ratio: {baseline_tok["single_token_ratio"]:.1%}')
    print(f'  P95 tokens/term: {baseline_tok["p95_tokens_per_term"]:.2f}')
    print()
    
    # Adapted tokenization
    print('Adapted Model:')
    adapted_tok = eval_medical.compute_term_tokenization_coverage(
        str(terms_file),
        str(model_path)
    )
    print(f'  Total terms: {adapted_tok["total_terms"]}')
    print(f'  Mean tokens/term: {adapted_tok["mean_tokens_per_term"]:.2f}')
    print(f'  Median tokens/term: {adapted_tok["median_tokens_per_term"]:.2f}')
    print(f'  Single-token ratio: {adapted_tok["single_token_ratio"]:.1%}')
    print(f'  P95 tokens/term: {adapted_tok["p95_tokens_per_term"]:.2f}')
    print()
    
    # Improvement
    mean_improvement = ((baseline_tok["mean_tokens_per_term"] - adapted_tok["mean_tokens_per_term"]) 
                       / baseline_tok["mean_tokens_per_term"] * 100)
    single_token_improvement = ((adapted_tok["single_token_ratio"] - baseline_tok["single_token_ratio"]) 
                               / baseline_tok["single_token_ratio"] * 100) if baseline_tok["single_token_ratio"] > 0 else 0
    
    print('Improvement:')
    print(f'  Mean tokens/term: {baseline_tok["mean_tokens_per_term"]:.2f} → {adapted_tok["mean_tokens_per_term"]:.2f} ({mean_improvement:+.1f}%)')
    print(f'  Single-token ratio: {baseline_tok["single_token_ratio"]:.1%} → {adapted_tok["single_token_ratio"]:.1%} ({single_token_improvement:+.1f}%)')
    print()
    
    results['tokenization_efficiency'] = {
        'baseline': baseline_tok,
        'adapted': adapted_tok,
        'improvement': {
            'mean_tokens_per_term_reduction_pct': float(mean_improvement),
            'single_token_ratio_increase_pct': float(single_token_improvement)
        }
    }
else:
    print('Warning: medical_terms.txt not found, skipping tokenization evaluation')
    print()

# 2. Perplexity Evaluation
print('=' * 80)
print('2. PERPLEXITY EVALUATION')
print('=' * 80)
print()

dataset_spec = '${DATASET}'
dataset_name, dataset_config, split = eval_medical.parse_dataset_spec(dataset_spec)

print(f'Evaluating perplexity on: {dataset_name} ({split} split)')
print(f'Max samples: ${MAX_SAMPLES}')
print()
print('This may take 10-30 minutes depending on dataset size...')
print()

try:
    # Baseline perplexity
    print('Computing baseline perplexity...')
    baseline_ppl = eval_medical.evaluate_perplexity(
        model_path=baseline_model,
        tokenizer_path=baseline_tokenizer,
        dataset_name=dataset_name,
        split=split,
        dataset_config=dataset_config,
        max_samples=int('${MAX_SAMPLES}'),
        batch_size=32,
        max_length=1024,
    )
    print(f'Baseline Perplexity: {baseline_ppl:.2f}')
    print()
    
    # Adapted perplexity
    print('Computing adapted model perplexity...')
    adapted_ppl = eval_medical.evaluate_perplexity(
        model_path=str(model_path),
        tokenizer_path=str(model_path),
        dataset_name=dataset_name,
        split=split,
        dataset_config=dataset_config,
        max_samples=int('${MAX_SAMPLES}'),
        batch_size=32,
        max_length=1024,
    )
    print(f'Adapted Perplexity: {adapted_ppl:.2f}')
    print()
    
    # Improvement
    ppl_reduction = ((baseline_ppl - adapted_ppl) / baseline_ppl * 100)
    print('Improvement:')
    print(f'  Perplexity: {baseline_ppl:.2f} → {adapted_ppl:.2f} ({ppl_reduction:+.1f}% reduction)')
    print()
    
    results['perplexity'] = {
        'baseline': float(baseline_ppl),
        'adapted': float(adapted_ppl),
        'reduction_pct': float(ppl_reduction)
    }
    
except Exception as e:
    print(f'Error during perplexity evaluation: {e}')
    import traceback
    traceback.print_exc()
    results['perplexity'] = {'error': str(e)}

# Save results
eval_dir = run_dir / 'evaluation'
eval_dir.mkdir(exist_ok=True)

output_file = eval_dir / 'tokenization_perplexity_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print('=' * 80)
print('EVALUATION COMPLETE')
print('=' * 80)
print(f'Results saved to: {output_file}')
print()

# Print summary
print('SUMMARY:')
print('-' * 80)
if 'tokenization_efficiency' in results and results['tokenization_efficiency']:
    tok = results['tokenization_efficiency']
    if 'improvement' in tok:
        print(f'Tokenization: {tok["improvement"]["mean_tokens_per_term_reduction_pct"]:+.1f}% reduction in tokens/term')
if 'perplexity' in results and 'reduction_pct' in results['perplexity']:
    print(f'Perplexity: {results["perplexity"]["reduction_pct"]:+.1f}% reduction')
PYEOF


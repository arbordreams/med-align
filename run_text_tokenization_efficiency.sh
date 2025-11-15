#!/bin/bash
# Evaluate tokenization efficiency on actual text (tokens per same text)
# Usage: ./run_text_tokenization_efficiency.sh [RUN_DIR] [NUM_SAMPLES]

RUN_DIR="${1:-/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334}"
NUM_SAMPLES="${2:-100000}"  # Increased to 100,000 for better statistical accuracy

export RUN_DIR NUM_SAMPLES

cd /lambda/nfs/med-align/med-align && python3 << PYEOF
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from transformers import AutoTokenizer
from datasets import load_from_disk
import json
import statistics
import random
import os

run_dir = Path(os.environ.get('RUN_DIR', '/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334'))
num_samples = int(os.environ.get('NUM_SAMPLES', '100000'))
model_path = run_dir / 'embedding_warmup' / 'checkpoint-3500'
baseline_tokenizer = 'mistralai/Mistral-7B-v0.3'

if not model_path.exists():
    print('Error: Model checkpoint not found')
    sys.exit(1)

print('=' * 80)
print('TOKENIZATION EFFICIENCY ON TEXT (TOKENS PER SAME TEXT)')
print('=' * 80)
print()
print(f'Run directory: {run_dir}')
print(f'Baseline: {baseline_tokenizer}')
print(f'Adapted: {model_path}')
print(f'Number of samples: {num_samples:,}')
print()

# Load tokenizers
print('Loading tokenizers...')
baseline_tok = AutoTokenizer.from_pretrained(baseline_tokenizer, trust_remote_code=True)
adapted_tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
print('Tokenizers loaded.')
print()

# Load dataset
target_dataset_path = run_dir / 'datasets' / 'target'
if not target_dataset_path.exists():
    print(f'Error: Target dataset not found at {target_dataset_path}')
    sys.exit(1)

print('Loading dataset...')
target_ds = load_from_disk(str(target_dataset_path))
print(f'Dataset loaded: {len(target_ds["train"]):,} examples')
print()

# Get original text from corpus (before tokenization)
corpus_file = run_dir / 'corpus' / 'medical_corpus.jsonl'
if corpus_file.exists():
    print('Reading text from corpus file...')
    texts = []
    with open(corpus_file, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                doc = json.loads(line.strip())
                # Extract text
                text = ''
                if isinstance(doc, dict):
                    for field in ['text', 'content', 'abstract', 'body', 'title']:
                        if field in doc and isinstance(doc[field], str):
                            text = doc[field]
                            break
                    if not text:
                        text = ' '.join(str(v) for v in doc.values() if isinstance(v, str))
                elif isinstance(doc, str):
                    text = doc
                
                if text and len(text.strip()) > 10:  # Minimum length
                    texts.append(text.strip())
            except:
                continue
    
    if not texts:
        print('Warning: Could not extract text from corpus, using tokenized dataset instead')
        texts = None
    else:
        print(f'Extracted {len(texts):,} text samples from corpus')
else:
    print('Corpus file not found, using tokenized dataset (will decode tokens to text)')
    texts = None

# If we don't have original text, decode from tokenized dataset
if texts is None:
    print('Decoding text from tokenized dataset...')
    texts = []
    sample_indices = random.sample(range(len(target_ds['train'])), min(num_samples, len(target_ds['train'])))
    for idx in sample_indices:
        try:
            # Decode using adapted tokenizer (should be close to original)
            input_ids = target_ds['train'][idx]['input_ids']
            text = adapted_tok.decode(input_ids, skip_special_tokens=True)
            if text and len(text.strip()) > 10:
                texts.append(text.strip())
        except:
            continue
    print(f'Decoded {len(texts):,} text samples')
print()

if not texts:
    print('Error: No text samples available')
    sys.exit(1)

print('=' * 80)
print('TOKENIZING TEXT SAMPLES')
print('=' * 80)
print(f'Processing {len(texts):,} text samples...')
print()

baseline_lengths = []
adapted_lengths = []
char_lengths = []
byte_lengths = []

for i, text in enumerate(texts):
    if (i + 1) % 10000 == 0 or (i + 1) == len(texts):
        print(f'  Processed {i + 1:,} / {len(texts):,} samples ({((i+1)/len(texts)*100):.1f}%)...')
    
    # Tokenize with both tokenizers
    baseline_tokens = baseline_tok(text, add_special_tokens=False)['input_ids']
    adapted_tokens = adapted_tok(text, add_special_tokens=False)['input_ids']
    
    baseline_len = len(baseline_tokens)
    adapted_len = len(adapted_tokens)
    
    baseline_lengths.append(baseline_len)
    adapted_lengths.append(adapted_len)
    char_lengths.append(len(text))
    byte_lengths.append(len(text.encode('utf-8')))

print()
print('=' * 80)
print('TOKENIZATION EFFICIENCY RESULTS')
print('=' * 80)
print()

# Calculate statistics
baseline_mean = statistics.mean(baseline_lengths)
adapted_mean = statistics.mean(adapted_lengths)
baseline_median = statistics.median(baseline_lengths)
adapted_median = statistics.median(adapted_lengths)
baseline_std = statistics.stdev(baseline_lengths) if len(baseline_lengths) > 1 else 0.0
adapted_std = statistics.stdev(adapted_lengths) if len(adapted_lengths) > 1 else 0.0

mean_char_length = statistics.mean(char_lengths)
mean_byte_length = statistics.mean(byte_lengths)

# Improvement
token_reduction = ((baseline_mean - adapted_mean) / baseline_mean * 100)
tokens_saved_per_text = baseline_mean - adapted_mean
total_tokens_saved = tokens_saved_per_text * len(texts)

print(f'Text samples analyzed: {len(texts):,}')
print(f'Average text length: {mean_char_length:.0f} characters, {mean_byte_length:.0f} bytes')
print()

print('BASELINE (Mistral-7B-v0.3):')
print('-' * 80)
print(f'  Mean tokens per text:     {baseline_mean:.2f}')
print(f'  Median tokens per text:   {baseline_median:.2f}')
print(f'  Std deviation:            {baseline_std:.2f}')
print(f'  Min: {min(baseline_lengths)}, Max: {max(baseline_lengths)}')
print(f'  Tokens per 1000 chars:    {baseline_mean / mean_char_length * 1000:.1f}')
print(f'  Tokens per 1KB:           {baseline_mean / mean_byte_length * 1024:.1f}')
print()

print('ADAPTED MODEL:')
print('-' * 80)
print(f'  Mean tokens per text:     {adapted_mean:.2f}')
print(f'  Median tokens per text:   {adapted_median:.2f}')
print(f'  Std deviation:            {adapted_std:.2f}')
print(f'  Min: {min(adapted_lengths)}, Max: {max(adapted_lengths)}')
print(f'  Tokens per 1000 chars:    {adapted_mean / mean_char_length * 1000:.1f}')
print(f'  Tokens per 1KB:           {adapted_mean / mean_byte_length * 1024:.1f}')
print()

print('IMPROVEMENT:')
print('-' * 80)
print(f'  Token reduction per text: {baseline_mean:.2f} → {adapted_mean:.2f} ({token_reduction:+.2f}%)')
print(f'  Tokens saved per text:     {tokens_saved_per_text:.2f} tokens')
print(f'  Total tokens saved:       {total_tokens_saved:,.0f} tokens ({total_tokens_saved/1e6:.2f}M tokens)')
print(f'  Tokens per 1000 chars:    {baseline_mean / mean_char_length * 1000:.1f} → {adapted_mean / mean_char_length * 1000:.1f} ({((baseline_mean - adapted_mean) / mean_char_length * 1000):+.1f})')
print(f'  Tokens per 1KB:            {baseline_mean / mean_byte_length * 1024:.1f} → {adapted_mean / mean_byte_length * 1024:.1f} ({((baseline_mean - adapted_mean) / mean_byte_length * 1024):+.1f})')
print()

# Distribution
from collections import Counter
print('TOKEN COUNT DISTRIBUTION:')
print('-' * 80)
print('Baseline token count ranges:')
baseline_ranges = {
    '1-50': sum(1 for x in baseline_lengths if 1 <= x <= 50),
    '51-100': sum(1 for x in baseline_lengths if 51 <= x <= 100),
    '101-200': sum(1 for x in baseline_lengths if 101 <= x <= 200),
    '201-500': sum(1 for x in baseline_lengths if 201 <= x <= 500),
    '501-1000': sum(1 for x in baseline_lengths if 501 <= x <= 1000),
    '1000+': sum(1 for x in baseline_lengths if x > 1000),
}
for range_name, count in baseline_ranges.items():
    pct = count / len(baseline_lengths) * 100
    print(f'  {range_name:>10} tokens: {count:>5,} texts ({pct:>5.1f}%)')

print()
print('Adapted token count ranges:')
adapted_ranges = {
    '1-50': sum(1 for x in adapted_lengths if 1 <= x <= 50),
    '51-100': sum(1 for x in adapted_lengths if 51 <= x <= 100),
    '101-200': sum(1 for x in adapted_lengths if 101 <= x <= 200),
    '201-500': sum(1 for x in adapted_lengths if 201 <= x <= 500),
    '501-1000': sum(1 for x in adapted_lengths if 501 <= x <= 1000),
    '1000+': sum(1 for x in adapted_lengths if x > 1000),
}
for range_name, count in adapted_ranges.items():
    pct = count / len(adapted_lengths) * 100
    print(f'  {range_name:>10} tokens: {count:>5,} texts ({pct:>5.1f}%)')
print()

# Save results
results = {
    'num_samples': len(texts),
    'baseline': {
        'mean_tokens_per_text': float(baseline_mean),
        'median_tokens_per_text': float(baseline_median),
        'std_deviation': float(baseline_std),
        'min_tokens': int(min(baseline_lengths)),
        'max_tokens': int(max(baseline_lengths)),
        'tokens_per_1000_chars': float(baseline_mean / mean_char_length * 1000),
        'tokens_per_1kb': float(baseline_mean / mean_byte_length * 1024),
    },
    'adapted': {
        'mean_tokens_per_text': float(adapted_mean),
        'median_tokens_per_text': float(adapted_median),
        'std_deviation': float(adapted_std),
        'min_tokens': int(min(adapted_lengths)),
        'max_tokens': int(max(adapted_lengths)),
        'tokens_per_1000_chars': float(adapted_mean / mean_char_length * 1000),
        'tokens_per_1kb': float(adapted_mean / mean_byte_length * 1024),
    },
    'improvements': {
        'token_reduction_pct': float(token_reduction),
        'tokens_saved_per_text': float(tokens_saved_per_text),
        'total_tokens_saved': float(total_tokens_saved),
        'tokens_saved_per_1000_chars': float((baseline_mean - adapted_mean) / mean_char_length * 1000),
        'tokens_saved_per_1kb': float((baseline_mean - adapted_mean) / mean_byte_length * 1024),
    }
}

eval_dir = run_dir / 'evaluation'
eval_dir.mkdir(exist_ok=True)
output_file = eval_dir / 'text_tokenization_efficiency_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print('=' * 80)
print('EVALUATION COMPLETE')
print('=' * 80)
print(f'Results saved to: {output_file}')
print()
print('KEY METRIC:')
print(f'  ✅ {token_reduction:.2f}% fewer tokens needed for the same text')
print(f'  ✅ Saves {tokens_saved_per_text:.2f} tokens per text on average')
PYEOF


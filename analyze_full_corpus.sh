#!/bin/bash
# Full Corpus Tokenization Efficiency Analysis
# Run this on Lambda to analyze the entire corpus
#
# Usage:
#   ./analyze_full_corpus.sh                                    # Uses default run dir, 32 workers
#   RUN_DIR=/path/to/run ./analyze_full_corpus.sh               # Custom run directory
#   NUM_WORKERS=16 ./analyze_full_corpus.sh                     # Custom worker count
#   RUN_DIR=/path/to/run NUM_WORKERS=16 ./analyze_full_corpus.sh # Both custom
#
# The script uses multiprocessing to parallelize:
#   - Batch processing of corpus examples
#   - Medical term tokenization
#   - Token length distribution analysis
#
# Efficiency notes:
#   - Uses unified worker initializer (loads all datasets/tokenizers once per worker)
#   - Single pool reused across all three phases (workers initialize once, not three times)
#   - Datasets are memory-mapped (efficient OS-level sharing)
#   - Reduces initialization from 96 loads (32×3) to 32 loads (32×1) with 32 workers

cd /lambda/nfs/med-align/med-align && python3 << 'PYEOF'
from transformers import AutoTokenizer
from datasets import load_from_disk
from pathlib import Path
from collections import Counter
import json
import statistics
import time
import sys
import os
import random
import re
from multiprocessing import Pool
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Configure run directory (default: latest tokalign_paper_optimal run, can be overridden via RUN_DIR env var)
default_run_dir = '/lambda/nfs/med-align/tokenizer_adapt/tokalign_paper_optimal_20251114-114334'
run_dir_str = os.environ.get('RUN_DIR', default_run_dir)
run_dir = Path(run_dir_str)

# Validate run directory exists and is complete
if not run_dir.exists():
    print(f'ERROR: Run directory does not exist: {run_dir}')
    print('Please set RUN_DIR environment variable to a valid run directory.')
    sys.exit(1)

required_paths = [
    run_dir / 'tokenizers' / 'source',
    run_dir / 'tokenizers' / 'target',
    run_dir / 'datasets' / 'source',
    run_dir / 'datasets' / 'target',
    run_dir / 'corpus' / 'medical_terms.txt',
]

missing_paths = [p for p in required_paths if not p.exists()]
if missing_paths:
    print(f'ERROR: Run directory is incomplete. Missing paths:')
    for p in missing_paths:
        print(f'  - {p}')
    print('Please ensure the pipeline has completed successfully.')
    sys.exit(1)

# Configure number of workers (default: 32, can be overridden via NUM_WORKERS env var)
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '32'))

def print_progress_bar(current, total, width=50, extra_info=''):
    """Print a simple text progress bar with optional extra info."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = '=' * filled + '-' * (width - filled)
    # Combine progress bar and extra info in a single write to avoid terminal conflicts
    line = f'\r  [{bar}] {percent*100:5.1f}%{extra_info}'
    sys.stdout.write(line)
    sys.stdout.flush()

def calculate_percentile(data, percentile):
    """
    Calculate percentile with proper interpolation (linear method).
    Uses numpy if available, otherwise implements interpolation manually.
    """
    if not data:
        return 0.0
    if HAS_NUMPY:
        return float(np.percentile(data, percentile, method='linear'))
    # Manual interpolation (linear method, same as numpy)
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return float(sorted_data[0])
    # Linear interpolation between closest ranks
    rank = (percentile / 100.0) * (n - 1)
    lower_rank = int(rank)
    upper_rank = min(lower_rank + 1, n - 1)
    weight = rank - lower_rank
    return float(sorted_data[lower_rank] * (1 - weight) + sorted_data[upper_rank] * weight)

# Global variables for worker processes (loaded once per worker via unified initializer)
_worker_state = {
    'source_tok': None,
    'target_tok': None,
    'baseline_tok': None,
    'source_ds': None,
    'target_ds': None,
    'run_dir': None,
}

# Unified worker initializer - loads all datasets/tokenizers once per worker
def init_worker(run_dir_str):
    """Unified worker initializer - loads all datasets/tokenizers once per worker."""
    import os
    worker_id = os.getpid()
    print(f"[Worker {worker_id}] Loading all datasets and tokenizers (one-time initialization)...", flush=True)
    
    run_dir_local = Path(run_dir_str)
    
    # Load tokenizers (shared across all task types)
    _worker_state['source_tok'] = AutoTokenizer.from_pretrained(
        str(run_dir_local / 'tokenizers' / 'source'), trust_remote_code=True
    )
    _worker_state['target_tok'] = AutoTokenizer.from_pretrained(
        str(run_dir_local / 'tokenizers' / 'target'), trust_remote_code=True
    )
    _worker_state['baseline_tok'] = AutoTokenizer.from_pretrained(
        'mistralai/Mistral-7B-v0.3', trust_remote_code=True
    )
    
    # Load datasets (memory-mapped, efficient)
    _worker_state['source_ds'] = load_from_disk(str(run_dir_local / 'datasets' / 'source'))
    _worker_state['target_ds'] = load_from_disk(str(run_dir_local / 'datasets' / 'target'))
    _worker_state['run_dir'] = run_dir_str
    
    print(f"[Worker {worker_id}] Worker fully initialized (all datasets/tokenizers loaded)", flush=True)

# Worker function for processing a batch of examples
def process_batch(args):
    """Process a batch of examples and return token lengths and token counters."""
    batch_indices, sample_every = args
    
    # Use pre-loaded datasets and tokenizers from unified worker state
    source_tok = _worker_state['source_tok']
    target_tok = _worker_state['target_tok']
    source_ds = _worker_state['source_ds']
    target_ds = _worker_state['target_ds']
    
    source_lengths = []
    target_lengths = []
    source_token_counter = Counter()
    target_token_counter = Counter()
    
    batch_indices_list = list(batch_indices)
    
    # Sample tokens for frequency analysis (random sampling to avoid systematic bias)
    # Sample approximately 1/sample_every of the batch
    sample_count = max(1, len(batch_indices_list) // sample_every)
    if len(batch_indices_list) > sample_count:
        sample_indices = random.sample(batch_indices_list, sample_count)
    else:
        sample_indices = batch_indices_list
    
    # Cache input_ids for sampled examples to avoid double dataset access
    sampled_source_ids = {}
    sampled_target_ids = {}
    
    # Get token lengths and cache input_ids for sampled examples
    for idx in batch_indices_list:
        source_ids = source_ds['train'][idx]['input_ids']
        target_ids = target_ds['train'][idx]['input_ids']
        source_lengths.append(len(source_ids))
        target_lengths.append(len(target_ids))
        
        # Cache for sampled examples to avoid re-accessing dataset
        if idx in sample_indices:
            sampled_source_ids[idx] = source_ids
            sampled_target_ids[idx] = target_ids
    
    # Now use cached input_ids for token conversion
    for idx in sample_indices:
        source_tokens = source_tok.convert_ids_to_tokens(sampled_source_ids[idx])
        target_tokens = target_tok.convert_ids_to_tokens(sampled_target_ids[idx])
        source_token_counter.update(source_tokens)
        target_token_counter.update(target_tokens)
    
    return {
        'source_lengths': source_lengths,
        'target_lengths': target_lengths,
        'source_token_counter': dict(source_token_counter),
        'target_token_counter': dict(target_token_counter),
    }

# Worker function for medical terms tokenization
def tokenize_medical_term(term):
    """Tokenize a single medical term."""
    # Use pre-loaded tokenizers from unified worker state
    baseline_tok = _worker_state['baseline_tok']
    target_tok = _worker_state['target_tok']
    
    baseline_len = len(baseline_tok(term, add_special_tokens=False)['input_ids'])
    # Cache tokenization result to avoid double computation
    target_tokenized = target_tok(term, add_special_tokens=False)['input_ids']
    target_len = len(target_tokenized)
    target_tokens = target_tok.convert_ids_to_tokens(target_tokenized)
    
    return {
        'term': term,
        'baseline_len': baseline_len,
        'target_len': target_len,
        'target_tokens': target_tokens,
    }

# Worker function for token length distribution analysis
def analyze_token_lengths(idx):
    """Analyze token lengths for a single example."""
    # Use pre-loaded datasets and tokenizers from unified worker state
    source_tok = _worker_state['source_tok']
    target_tok = _worker_state['target_tok']
    source_ds = _worker_state['source_ds']
    target_ds = _worker_state['target_ds']
    
    source_tokens = source_tok.convert_ids_to_tokens(source_ds['train'][idx]['input_ids'])
    target_tokens = target_tok.convert_ids_to_tokens(target_ds['train'][idx]['input_ids'])
    
    return {
        'source_byte_lengths': [len(token.encode('utf-8')) for token in source_tokens],
        'target_byte_lengths': [len(token.encode('utf-8')) for token in target_tokens],
        'source_char_lengths': [len(token) for token in source_tokens],
        'target_char_lengths': [len(token) for token in target_tokens],
    }

print('='*80)
print('TOKENIZATION EFFICIENCY ANALYSIS (FULL CORPUS)')
print('='*80)
print()

# Load tokenizers
baseline_tok = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3', trust_remote_code=True)
source_tok = AutoTokenizer.from_pretrained(str(run_dir / 'tokenizers' / 'source'), trust_remote_code=True)
target_tok = AutoTokenizer.from_pretrained(str(run_dir / 'tokenizers' / 'target'), trust_remote_code=True)

# Load datasets
source_ds = load_from_disk(str(run_dir / 'datasets' / 'source'))
target_ds = load_from_disk(str(run_dir / 'datasets' / 'target'))

total_examples = len(source_ds['train'])

print(f'Analyzing ENTIRE corpus: {total_examples:,} examples')
print(f'Using {NUM_WORKERS} worker processes for parallel processing')
print('(This will take 10-20 minutes with multiprocessing)')
print()

start_time = time.time()
random.seed(42)  # Set seed for reproducible random sampling

# Process in batches to show progress and manage memory
batch_size = 100000
run_dir_str = str(run_dir)
sample_every = 20  # Sample every 20th example for token frequency analysis

# Load medical terms early (needed for phase 2)
terms_file = run_dir / 'corpus' / 'medical_terms.txt'
with open(terms_file) as f:
    terms = [line.strip() for line in f if line.strip()]

# Prepare batch arguments for parallel processing
batch_args = []
for i in range(0, total_examples, batch_size):
    end_idx = min(i + batch_size, total_examples)
    batch_indices = range(i, end_idx)
    batch_args.append((batch_indices, sample_every))

# Process batches in parallel
source_lengths = []
target_lengths = []
source_token_counter = Counter()
target_token_counter = Counter()

processed_batches = 0
total_batches = len(batch_args)

print(f'Initializing {NUM_WORKERS} worker processes (unified pool, loads all datasets/tokenizers once)...')
print('(Workers will be reused across all three analysis phases)')
print()

# Create single pool that will be reused for all three phases
with Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(run_dir_str,)) as pool:
    # ===== PHASE 1: Batch processing =====
    print('Phase 1: Processing corpus batches...')
    for result in pool.imap(process_batch, batch_args):
        source_lengths.extend(result['source_lengths'])
        target_lengths.extend(result['target_lengths'])
        source_token_counter.update(result['source_token_counter'])
        target_token_counter.update(result['target_token_counter'])
        
        processed_batches += 1
        processed_examples = min(processed_batches * batch_size, total_examples)
        
        elapsed = time.time() - start_time
        rate = processed_examples / elapsed if elapsed > 0 else 0
        remaining = (total_examples - processed_examples) / rate if rate > 0 else 0
        
        # Progress bar + detailed stats (combined to avoid terminal escape sequence issues)
        extra_info = f'  {processed_examples:,} / {total_examples:,} | Elapsed: {elapsed/60:.1f}m | Rate: {rate:.0f} ex/s | ETA: {remaining/60:.1f}m'
        print_progress_bar(processed_examples, total_examples, extra_info=extra_info)
    
    print()
    print()
    print('✅ Phase 1 complete')
    print()
    
    # ===== PHASE 2: Medical terms (reusing same pool) =====
    print(f'Phase 2: Tokenizing {len(terms):,} medical terms (reusing initialized workers)...')
    
    baseline_lengths = []
    target_lengths_medical = []
    term_results = pool.map(tokenize_medical_term, terms)
    for result in term_results:
        baseline_lengths.append(result['baseline_len'])
        target_lengths_medical.append(result['target_len'])
    
    print('✅ Phase 2 complete')
    print()
    
    # ===== PHASE 3: Token length distribution (reusing same pool) =====
    sample_size = min(100000, total_examples)
    if total_examples > 0 and sample_size > 0:
        # Random sampling to avoid systematic bias (e.g., if corpus has patterns)
        sample_indices = random.sample(range(total_examples), sample_size)
    else:
        sample_indices = []
    
    print(f'Phase 3: Analyzing token lengths from {len(sample_indices):,} sample examples (reusing initialized workers)...')
    
    source_token_byte_lengths = []
    target_token_byte_lengths = []
    source_token_char_lengths = []
    target_token_char_lengths = []
    
    length_results = pool.map(analyze_token_lengths, sample_indices)
    for result in length_results:
        source_token_byte_lengths.extend(result['source_byte_lengths'])
        target_token_byte_lengths.extend(result['target_byte_lengths'])
        source_token_char_lengths.extend(result['source_char_lengths'])
        target_token_char_lengths.extend(result['target_char_lengths'])
    
    print('✅ Phase 3 complete')
    print()

# Pool is closed here - all three phases completed
print('✅ All analysis phases complete')
print()

source_avg = statistics.mean(source_lengths)
target_avg = statistics.mean(target_lengths)
source_median = statistics.median(source_lengths)
target_median = statistics.median(target_lengths)
source_std = statistics.stdev(source_lengths) if len(source_lengths) > 1 else 0.0
target_std = statistics.stdev(target_lengths) if len(target_lengths) > 1 else 0.0
# Use proper percentile calculation with interpolation (not truncation)
source_p25 = calculate_percentile(source_lengths, 25)
source_p75 = calculate_percentile(source_lengths, 75)
target_p25 = calculate_percentile(target_lengths, 25)
target_p75 = calculate_percentile(target_lengths, 75)

improvement_pct = ((source_avg - target_avg) / source_avg) * 100
total_tokens_saved = (source_avg - target_avg) * total_examples

print('='*80)
print('VOCABULARY STATISTICS')
print('='*80)
print()
print(f'Baseline (Mistral-7B-v0.3): {len(baseline_tok):,} tokens')
print(f'Source (BioMistral): {len(source_tok):,} tokens')
print(f'Target (Mistral + adapted): {len(target_tok):,} tokens')
print(f'  Vocabulary growth: {len(target_tok) - len(baseline_tok):+,} tokens ({((len(target_tok) - len(baseline_tok)) / len(baseline_tok) * 100):+.2f}%)')
print()

# Token overlap analysis
source_vocab_set = set(source_tok.get_vocab().keys())
target_vocab_set = set(target_tok.get_vocab().keys())
baseline_vocab_set = set(baseline_tok.get_vocab().keys())

source_target_overlap = len(source_vocab_set & target_vocab_set)
target_new_tokens = len(target_vocab_set - baseline_vocab_set)
target_unique_tokens = len(target_vocab_set - source_vocab_set)

print('VOCABULARY OVERLAP:')
print(f'  Source ∩ Target: {source_target_overlap:,} tokens ({source_target_overlap/len(target_vocab_set)*100:.1f}% of target)')
print(f'  Target new tokens (not in baseline): {target_new_tokens:,} tokens')
print(f'  Target unique tokens (not in source): {target_unique_tokens:,} tokens')
print()

print('='*80)
print('CORPUS-WIDE RESULTS (FULL CORPUS)')
print('='*80)
print()
print(f'  Total Examples Analyzed: {total_examples:,}')
print(f'  Source: {source_avg:.2f} tokens/example')
print(f'    Median: {source_median:.2f}, Std: {source_std:.2f}')
print(f'    IQR: {source_p25:.0f} - {source_p75:.0f}')
print(f'    Min: {min(source_lengths):.0f}, Max: {max(source_lengths):.0f}')
print(f'  Target: {target_avg:.2f} tokens/example')
print(f'    Median: {target_median:.2f}, Std: {target_std:.2f}')
print(f'    IQR: {target_p25:.0f} - {target_p75:.0f}')
print(f'    Min: {min(target_lengths):.0f}, Max: {max(target_lengths):.0f}')
print(f'  Improvement: {improvement_pct:+.2f}%')
print(f'  Total tokens saved: {total_tokens_saved:,.0f} tokens')
print(f'  Total tokens saved: {total_tokens_saved / 1e9:.2f} billion tokens')
print()

# Single-token ratio analysis
source_single_token = sum(1 for l in source_lengths if l == 1) / len(source_lengths)
target_single_token = sum(1 for l in target_lengths if l == 1) / len(target_lengths)

print('SINGLE-TOKEN RATIO:')
print(f'  Source: {source_single_token:.2%} of examples are single-token')
print(f'  Target: {target_single_token:.2%} of examples are single-token')
print(f'  Improvement: {(target_single_token - source_single_token):+.2%}')
print()

# Most common tokens
print('='*80)
print('MOST COMMON TOKENS (TOP 100)')
print('='*80)
print()

# Cache sum() to avoid repeated O(n) operations
source_token_total = sum(source_token_counter.values())
target_token_total = sum(target_token_counter.values())

print('TOP 100 SOURCE TOKENS (BioMistral):')
print('-'*80)
for i, (token, count) in enumerate(source_token_counter.most_common(100), 1):
    pct = (count / source_token_total) * 100  # Use cached sum
    print(f'  {i:3d}. {token:30s} : {count:>12,} occurrences ({pct:5.2f}%)')
print()

print('TOP 100 TARGET TOKENS (Mistral + adapted):')
print('-'*80)
for i, (token, count) in enumerate(target_token_counter.most_common(100), 1):
    pct = (count / target_token_total) * 100  # Use cached sum
    print(f'  {i:3d}. {token:30s} : {count:>12,} occurrences ({pct:5.2f}%)')
print()

# Cache most_common() results to avoid repeated expensive computations
source_all_rankings = list(source_token_counter.most_common())
target_all_rankings = list(target_token_counter.most_common())
source_rankings = {token: rank for rank, (token, _) in enumerate(source_all_rankings, 1)}
target_rankings = {token: rank for rank, (token, _) in enumerate(target_all_rankings, 1)}

# Find tokens that appear in target but not in source top 100 (using cached rankings)
source_top100 = set(token for token, _ in source_all_rankings[:100])
target_top100 = set(token for token, _ in target_all_rankings[:100])
new_in_target = target_top100 - source_top100

if new_in_target:
    print('NEW TOKENS IN TARGET TOP 100 (not in source top 100):')
    print('-'*80)
    for token in sorted(new_in_target, key=lambda t: target_token_counter[t], reverse=True):
        target_count = target_token_counter[token]
        source_count = source_token_counter.get(token, 0)
        target_rank = target_rankings.get(token, None)
        print(f'  {token:30s} : Target={target_count:>12,} (rank #{target_rank}), Source={source_count:>12,}')
    print()

# Tokens that dropped out of top 100
dropped_from_source = source_top100 - target_top100
if dropped_from_source:
    print('TOKENS IN SOURCE TOP 100 (dropped from target top 100):')
    print('-'*80)
    for token in sorted(dropped_from_source, key=lambda t: source_token_counter[t], reverse=True)[:20]:
        source_count = source_token_counter[token]
        target_count = target_token_counter.get(token, 0)
        source_rank = source_rankings.get(token, None)
        target_rank = target_rankings.get(token, None) if target_count > 0 else None
        rank_str = f'rank #{target_rank}' if target_rank else 'not in top 100'
        print(f'  {token:30s} : Source={source_count:>12,} (rank #{source_rank}), Target={target_count:>12,} ({rank_str})')
    print()

# Alignment matrix - using same pattern as compute_alignment_coverage in eval_medical.py
vocab_mapping_path = run_dir / 'alignment' / 'vocab_mapping.json'
if vocab_mapping_path.exists():
    try:
        with open(vocab_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        
        target_vocab_size = len(target_tok)
        mapped = 0
        
        if isinstance(mapping_data, dict):
            if "mapping" in mapping_data and isinstance(mapping_data["mapping"], dict):
                mapped = len([k for k, v in mapping_data["mapping"].items() if v is not None])
            elif "target_to_source" in mapping_data and isinstance(mapping_data["target_to_source"], dict):
                mapped = len([k for k, v in mapping_data["target_to_source"].items() if v is not None])
            else:
                mapped = len([k for k, v in mapping_data.items() if str(k).isdigit() and v is not None])
        elif isinstance(mapping_data, list):
            mapped = len(mapping_data)
        
        coverage = (mapped / target_vocab_size) * 100 if target_vocab_size > 0 else 0
        
        print('='*80)
        print('ALIGNMENT MATRIX')
        print('='*80)
        print(f'  Coverage: {coverage:.2f}% ({mapped:,}/{target_vocab_size:,} tokens)')
        print(f'  Unmapped tokens: {target_vocab_size - mapped:,} ({100 - coverage:.2f}%)')
        print()
    except Exception as e:
        print('='*80)
        print('ALIGNMENT MATRIX')
        print('='*80)
        print(f'  Warning: Failed to parse vocab_mapping.json ({e})')
        print()

# ===== ALIGNMENT EMBEDDING SIMILARITY (COSINE SIMILARITY) =====
print('='*80)
print('ALIGNMENT EMBEDDING SIMILARITY (COSINE SIMILARITY)')
print('='*80)
print()

# Try to find embedding files (FastText or GloVe)
source_emb_file = None
target_emb_file = None
embedding_type = None

for emb_type in ['fasttext', 'glove']:
    source_candidate = run_dir / 'alignment' / f'source_vec.{emb_type}.txt'
    target_candidate = run_dir / 'alignment' / f'target_vec.{emb_type}.txt'
    if source_candidate.exists() and target_candidate.exists():
        source_emb_file = source_candidate
        target_emb_file = target_candidate
        embedding_type = emb_type
        break

if not source_emb_file or not target_emb_file:
    print('Warning: Embedding files not found (checked for .fasttext.txt and .glove.txt)')
    print('Skipping cosine similarity analysis.')
    print()
elif not HAS_NUMPY:
    print('Warning: NumPy is required for cosine similarity analysis but is not available.')
    print('Skipping cosine similarity analysis.')
    print()
else:
    print(f'Embedding files:')
    print(f'  Source: {source_emb_file}')
    print(f'  Target: {target_emb_file}')
    print(f'  Type: {embedding_type.upper()}')
    print()
    
    # Load embeddings into dictionaries (token_name -> vector)
    print('Loading source embeddings...')
    source_embeddings = {}
    try:
        with open(source_emb_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 50000 == 0:
                    print(f'  Loaded {line_num:,} source embeddings...')
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                token_name = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vector) > 0:
                        source_embeddings[token_name] = vector
                except (ValueError, IndexError):
                    continue
        print(f'  Loaded {len(source_embeddings):,} source embeddings')
    except Exception as e:
        print(f'  Error loading source embeddings: {e}')
        source_embeddings = {}
    
    print('Loading target embeddings...')
    target_embeddings = {}
    try:
        with open(target_emb_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 50000 == 0:
                    print(f'  Loaded {line_num:,} target embeddings...')
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                token_name = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vector) > 0:
                        target_embeddings[token_name] = vector
                except (ValueError, IndexError):
                    continue
        print(f'  Loaded {len(target_embeddings):,} target embeddings')
    except Exception as e:
        print(f'  Error loading target embeddings: {e}')
        target_embeddings = {}
    
    print()
    
    if not source_embeddings or not target_embeddings:
        print('Warning: Could not load embeddings. Skipping cosine similarity analysis.')
        print()
    else:
        # Load alignment mapping
        alignment_pairs = []  # List of (target_token_name, source_token_name) pairs
        
        if vocab_mapping_path.exists():
            try:
                with open(vocab_mapping_path, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
                
                # Extract target -> source mappings
                target_to_source = {}
                if isinstance(mapping_data, dict):
                    if "mapping" in mapping_data and isinstance(mapping_data["mapping"], dict):
                        target_to_source = mapping_data["mapping"]
                    elif "target_to_source" in mapping_data and isinstance(mapping_data["target_to_source"], dict):
                        target_to_source = mapping_data["target_to_source"]
                    else:
                        # Direct key-value format
                        for k, v in mapping_data.items():
                            if str(k).isdigit() and v is not None:
                                target_to_source[int(k)] = v
                elif isinstance(mapping_data, list):
                    # List format: index is target_id, value is source_id
                    for target_id, source_id in enumerate(mapping_data):
                        if source_id is not None:
                            target_to_source[target_id] = source_id
                
                # Convert token IDs to token names and match with embeddings
                # Embedding files store raw token strings, so we need to match tokenizer token names to embedding keys
                print('Converting token IDs to token names and matching with embeddings...')
                
                # Get vocabularies (maps token string -> token ID)
                target_vocab = target_tok.get_vocab()
                source_vocab = source_tok.get_vocab()
                
                # Create reverse mapping: token ID -> token string (from vocab)
                target_id_to_string = {v: k for k, v in target_vocab.items()}
                source_id_to_string = {v: k for k, v in source_vocab.items()}
                
                # Also get token names via convert_ids_to_tokens (may include SentencePiece prefixes)
                # We'll try both formats for matching
                
                matched_count = 0
                for target_id, source_id in target_to_source.items():
                    try:
                        target_id_int = int(target_id)
                        source_id_int = int(source_id)
                        
                        # Get token string from vocab (raw format, likely matches embedding keys)
                        target_token_string = target_id_to_string.get(target_id_int)
                        source_token_string = source_id_to_string.get(source_id_int)
                        
                        # Also get token name from tokenizer (may have prefixes)
                        target_token_name = target_tok.convert_ids_to_tokens([target_id_int])[0]
                        source_token_name = source_tok.convert_ids_to_tokens([source_id_int])[0]
                        
                        # Try to find matching embedding keys
                        target_emb_name = None
                        source_emb_name = None
                        
                        # Strategy 1: Try vocab string (raw format)
                        if target_token_string and target_token_string in target_embeddings:
                            target_emb_name = target_token_string
                        elif target_token_name in target_embeddings:
                            target_emb_name = target_token_name
                        else:
                            # Strategy 2: Try normalized (remove SentencePiece prefix)
                            target_normalized = target_token_name.replace('▁', ' ').strip() if target_token_name else None
                            if target_normalized and target_normalized in target_embeddings:
                                target_emb_name = target_normalized
                            elif target_token_string:
                                # Strategy 3: Try exact match with any embedding key
                                for emb_key in target_embeddings.keys():
                                    if emb_key == target_token_string or emb_key == target_token_name:
                                        target_emb_name = emb_key
                                        break
                                    # Try normalized match
                                    emb_normalized = emb_key.replace('▁', ' ').strip()
                                    if emb_normalized == target_normalized:
                                        target_emb_name = emb_key
                                        break
                        
                        if source_token_string and source_token_string in source_embeddings:
                            source_emb_name = source_token_string
                        elif source_token_name in source_embeddings:
                            source_emb_name = source_token_name
                        else:
                            source_normalized = source_token_name.replace('▁', ' ').strip() if source_token_name else None
                            if source_normalized and source_normalized in source_embeddings:
                                source_emb_name = source_normalized
                            elif source_token_string:
                                for emb_key in source_embeddings.keys():
                                    if emb_key == source_token_string or emb_key == source_token_name:
                                        source_emb_name = emb_key
                                        break
                                    emb_normalized = emb_key.replace('▁', ' ').strip()
                                    if emb_normalized == source_normalized:
                                        source_emb_name = emb_key
                                        break
                        
                        # If we found matches, add to alignment pairs
                        if target_emb_name and source_emb_name:
                            alignment_pairs.append((target_emb_name, source_emb_name))
                            matched_count += 1
                    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                        continue
                
                if matched_count > 0:
                    print(f'  Matched {matched_count:,} token pairs using multiple matching strategies')
                
                # Calculate alignment coverage for tokens with embeddings
                total_aligned = len(target_to_source)
                aligned_with_embeddings = len(alignment_pairs)
                alignment_coverage = (aligned_with_embeddings / total_aligned * 100) if total_aligned > 0 else 0
                
                print(f'  Found {aligned_with_embeddings:,} aligned token pairs with embeddings')
                print(f'  Aligned tokens: {aligned_with_embeddings:,} / {total_aligned:,} ({alignment_coverage:.2f}%)')
                print()
                
            except Exception as e:
                print(f'  Warning: Failed to load alignment mapping ({e})')
                alignment_pairs = []
        else:
            print('  Warning: vocab_mapping.json not found')
            alignment_pairs = []
        
        if not alignment_pairs:
            print('No aligned token pairs found. Skipping cosine similarity analysis.')
            print()
        else:
            # Compute cosine similarity for all aligned pairs
            print('Computing cosine similarities...')
            similarities = []
            similarity_pairs = []  # Store (target_name, source_name, similarity) for ranking
            
            for target_name, source_name in alignment_pairs:
                target_vec = target_embeddings[target_name]
                source_vec = source_embeddings[source_name]
                
                # Normalize vectors and compute cosine similarity
                target_norm = np.linalg.norm(target_vec)
                source_norm = np.linalg.norm(source_vec)
                
                # Skip zero vectors
                if target_norm == 0 or source_norm == 0:
                    continue
                
                # Normalize vectors
                target_vec_norm = target_vec / target_norm
                source_vec_norm = source_vec / source_norm
                
                # Compute cosine similarity: dot product of normalized vectors
                cos_sim = np.dot(target_vec_norm, source_vec_norm)
                similarities.append(float(cos_sim))
                similarity_pairs.append((target_name, source_name, float(cos_sim)))
            
            if not similarities:
                print('  Warning: No valid similarity computations (all vectors were zero?)')
                print()
            else:
                print(f'  Computed {len(similarities):,} cosine similarities')
                print()
                
                # Compute statistics
                avg_sim = statistics.mean(similarities)
                median_sim = statistics.median(similarities)
                std_sim = statistics.stdev(similarities) if len(similarities) > 1 else 0.0
                min_sim = min(similarities)
                max_sim = max(similarities)
                p25_sim = calculate_percentile(similarities, 25)
                p75_sim = calculate_percentile(similarities, 75)
                
                # Count by similarity thresholds
                high_sim = sum(1 for s in similarities if s > 0.8)
                medium_sim = sum(1 for s in similarities if 0.5 <= s <= 0.8)
                low_sim = sum(1 for s in similarities if s < 0.5)
                
                # Output statistics
                print('COSINE SIMILARITY STATISTICS:')
                print(f'  Average similarity: {avg_sim:.3f}')
                print(f'  Median similarity: {median_sim:.3f}')
                print(f'  Std deviation: {std_sim:.3f}')
                print(f'  Min: {min_sim:.3f}, Max: {max_sim:.3f}')
                print(f'  IQR: {p25_sim:.3f} - {p75_sim:.3f}')
                print()
                
                print('SIMILARITY DISTRIBUTION:')
                print(f'  High similarity (>0.8): {high_sim:,} tokens ({high_sim/len(similarities)*100:.1f}%)')
                print(f'  Medium similarity (0.5-0.8): {medium_sim:,} tokens ({medium_sim/len(similarities)*100:.1f}%)')
                print(f'  Low similarity (<0.5): {low_sim:,} tokens ({low_sim/len(similarities)*100:.1f}%)')
                print()
                
                # Top 20 best-aligned tokens
                similarity_pairs.sort(key=lambda x: x[2], reverse=True)
                print('TOP 20 BEST-ALIGNED TOKENS (Highest Cosine Similarity):')
                print('-'*80)
                for i, (target_name, source_name, sim) in enumerate(similarity_pairs[:20], 1):
                    # Clean token names for display (remove special characters)
                    target_display = target_name.replace('▁', ' ').strip()
                    source_display = source_name.replace('▁', ' ').strip()
                    print(f'  {i:2d}. {target_display:30s} (target) <-> {source_display:30s} (source) : {sim:.3f} similarity')
                print()
                
                # Top 20 worst-aligned tokens
                similarity_pairs.sort(key=lambda x: x[2])  # Sort ascending
                print('TOP 20 WORST-ALIGNED TOKENS (Lowest Cosine Similarity):')
                print('-'*80)
                for i, (target_name, source_name, sim) in enumerate(similarity_pairs[:20], 1):
                    # Clean token names for display
                    target_display = target_name.replace('▁', ' ').strip()
                    source_display = source_name.replace('▁', ' ').strip()
                    print(f'  {i:2d}. {target_display:30s} (target) <-> {source_display:30s} (source) : {sim:.3f} similarity')
                print()

# Medical terms statistics (using results from phase 2)
baseline_avg = statistics.mean(baseline_lengths)
target_avg_medical = statistics.mean(target_lengths_medical)
baseline_single = sum(1 for l in baseline_lengths if l == 1) / len(baseline_lengths)
target_single = sum(1 for l in target_lengths_medical if l == 1) / len(target_lengths_medical)

print('='*80)
print('MEDICAL TERMS ANALYSIS (ALL 2,000)')
print('='*80)
print()
print(f'  Baseline: {baseline_avg:.2f} tokens/term, {baseline_single:.1%} single-token')
print(f'  Target: {target_avg_medical:.2f} tokens/term, {target_single:.1%} single-token')
print(f'  Improvement: {((baseline_avg - target_avg_medical)/baseline_avg*100):+.1f}%')
print()

# Find terms with biggest improvement (using results from parallel processing)
term_improvements = []
for result in term_results:
    term = result['term']
    baseline_len = result['baseline_len']
    target_len = result['target_len']
    if baseline_len > 0:
        improvement = ((baseline_len - target_len) / baseline_len) * 100
        term_improvements.append((term, baseline_len, target_len, improvement))

term_improvements.sort(key=lambda x: x[3], reverse=True)

print('TOP 20 MOST IMPROVED MEDICAL TERMS:')
print('-'*80)
for i, (term, bl, tl, imp) in enumerate(term_improvements[:20], 1):
    print(f'  {i:2d}. {term:40s} : {bl} → {tl} tokens ({imp:+.1f}% improvement)')
print()

# Most common medical term tokens
print('='*80)
print('MOST COMMON MEDICAL TERM TOKENS (TOP 100)')
print('='*80)
print()

# Count tokens from medical terms (using results from parallel processing)
medical_token_counter = Counter()
for result in term_results:
    medical_token_counter.update(result['target_tokens'])

# Cache sum() to avoid repeated O(n) operations
medical_token_total = sum(medical_token_counter.values())

print('TOP 100 TOKENS FROM MEDICAL TERMS:')
print('-'*80)
for i, (token, count) in enumerate(medical_token_counter.most_common(100), 1):
    pct = (count / medical_token_total) * 100  # Use cached sum
    print(f'  {i:3d}. {token:30s} : {count:>8,} occurrences ({pct:5.2f}%)')
print()

# Token length distribution (using results from phase 3)
print('='*80)
print('TOKEN LENGTH DISTRIBUTION')
print('='*80)
print()

print('BYTE LENGTH PER TOKEN (sample):')
print(f'  Source: mean={statistics.mean(source_token_byte_lengths):.1f}, median={statistics.median(source_token_byte_lengths):.1f}')
print(f'  Target: mean={statistics.mean(target_token_byte_lengths):.1f}, median={statistics.median(target_token_byte_lengths):.1f}')
print()

print('CHARACTER LENGTH PER TOKEN (sample):')
print(f'  Source: mean={statistics.mean(source_token_char_lengths):.1f}, median={statistics.median(source_token_char_lengths):.1f}')
print(f'  Target: mean={statistics.mean(target_token_char_lengths):.1f}, median={statistics.median(target_token_char_lengths):.1f}')
print()

total_time = time.time() - start_time
print('='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print(f'Total analysis time: {total_time/60:.1f} minutes')
print(f'Examples processed: {total_examples:,}')
print(f'Processing rate: {total_examples/total_time:.0f} examples/second')
print()
print('='*80)
PYEOF


#!/usr/bin/env python3
"""Test script to verify which medical datasets actually exist on Hugging Face."""

from datasets import load_dataset

# Known medical datasets to test
test_datasets = [
    # Medical QA datasets
    ("medalpaca/medical_meadow_mediqa", None, "Medical Meadow MEDIQA"),
    ("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", "PubMed QA"),
    ("GBaker/MedQA-USMLE-4-options", None, "MedQA USMLE"),
    
    # Clinical text datasets  
    ("mteb/mteb_medicalqa", None, "Medical QA MTEB"),
    ("medalpaca/medical_meadow_health_advice", None, "Medical Health Advice"),
    
    # Biomedical corpora
    ("ncbi/pubmed-qa", None, "NCBI PubMed QA"),
    ("allenai/scirepeval", "cite_prediction_new", "SciRepEval"),
    
    # Alternative medical datasets
    ("BI55/MedText", None, "MedText"),
    ("keivalya/MedQuad-MedicalQnADataset", None, "MedQuad"),
]

print("Testing Hugging Face medical datasets...\n")
working_datasets = []
failed_datasets = []

for dataset_id, config, name in test_datasets:
    try:
        print(f"Testing {name} ({dataset_id})...", end=" ")
        if config:
            ds = load_dataset(dataset_id, config, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_id, split="train", streaming=True)
        
        # Try to fetch first example
        first = next(iter(ds))
        print(f"✓ OK - Fields: {list(first.keys())[:5]}")
        working_datasets.append((dataset_id, config, name, list(first.keys())))
    except Exception as e:
        print(f"✗ FAILED - {str(e)[:60]}")
        failed_datasets.append((dataset_id, config, name, str(e)))

print(f"\n{'='*80}")
print(f"WORKING DATASETS ({len(working_datasets)}):")
print(f"{'='*80}\n")

for dataset_id, config, name, fields in working_datasets:
    config_str = f", config='{config}'" if config else ""
    print(f"  {name}")
    print(f"    ID: {dataset_id}{config_str}")
    print(f"    Fields: {', '.join(fields[:5])}")
    print()

print(f"\n{'='*80}")
print(f"FAILED DATASETS ({len(failed_datasets)}):")
print(f"{'='*80}\n")

for dataset_id, config, name, error in failed_datasets:
    print(f"  {name} ({dataset_id}): {error[:60]}")


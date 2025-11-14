"""
Evaluation hooks for the medical TokAlign pipeline.

The CLI can run perplexity-style evaluations on Hugging Face datasets or emit a
placeholder log indicating which datasets need to be provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Sequence, Dict, Any, Tuple, Optional
import statistics as _stats

import datasets
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASETS: List[str] = ["uiyunkim-hub/pubmed-abstract:train"]

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model_path: str,
    tokenizer_path: str,
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
    batch_size: int = 32,
    max_length: int = 1024,
) -> float:
    """
    Compute a simple perplexity metric over the requested dataset split.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")

    # Enable TF32 for improved throughput on Hopper (H100)
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    try:
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    # Load model with robustness and validate with a dummy forward pass to catch device-side asserts early.
    def _load_and_smoke_test(torch_dtype: torch.dtype, use_flash: bool) -> AutoModelForCausalLM | None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if use_flash and hasattr(cfg, "attn_implementation"):
                setattr(cfg, "attn_implementation", "flash_attention_2")
            model_local = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg if use_flash else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            try:
                model_local = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except Exception:
                return None
        try:
            model_local.to("cuda").eval()
            # Dummy forward to detect device-side asserts early
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype if torch_dtype != torch.float32 else torch.float32):
                test_ids = tokenizer("hello world", return_tensors="pt", truncation=True, max_length=16)["input_ids"]
                # Map any OOB ids to eos as safety
                embed_size = model_local.get_input_embeddings().num_embeddings
                oob_mask = (test_ids < 0) | (test_ids >= embed_size)
                if oob_mask.any():
                    test_ids = test_ids.masked_fill(oob_mask, eos_id)
                test_ids = test_ids.to("cuda", non_blocking=True)
                _ = model_local(test_ids, labels=test_ids)
            return model_local
        except Exception:
            try:
                del model_local  # type: ignore[unreachable]
            except Exception:
                pass
            torch.cuda.empty_cache()
            return None

    # Try progressively less aggressive configs: bf16+flash_attn2 -> fp16 no-flash -> fp32 no-flash
    # Prefer high-accuracy FP32 first on GH200; fall back to BF16/FP16 only if needed
    model = (
        _load_and_smoke_test(torch.float32, use_flash=False)
        or _load_and_smoke_test(torch.bfloat16, use_flash=True)
        or _load_and_smoke_test(torch.float16, use_flash=False)
    )
    if model is None:
        raise RuntimeError("Failed to load model on CUDA for evaluation after multiple fallbacks.")
    # Pick the active dtype from parameters (first param tensor dtype)
    first_param = next(model.parameters())
    active_dtype = first_param.dtype

    # Use streaming to avoid downloading entire large datasets. Fall back if split is missing.
    def _try_load(name: str, sp: str):
        return datasets.load_dataset(name, split=sp, streaming=True)

    ds = None
    try:
        ds = _try_load(dataset_name, split)
    except Exception:
        # Fallback order: validation -> train
        for fallback_split in ("validation", "train"):
            try:
                logger.warning(
                    "Requested split '%s' unavailable for %s; falling back to '%s'.",
                    split,
                    dataset_name,
                    fallback_split,
                )
                ds = _try_load(dataset_name, fallback_split)
                split = fallback_split
                break
            except Exception:
                continue
        if ds is None:
            raise

    losses: List[float] = []
    count = 0
    batch_texts: List[str] = []
    # Allow adaptive downshifting on OOM if batch_size <= 0 is provided or OOM occurs.
    current_bs = batch_size if batch_size and batch_size > 0 else 64
    last_oom_error = None  # store last CUDA OOM to re-raise with context when needed

    # Safety: ensure token ids are within embedding size; warn once
    embed_size = model.get_input_embeddings().num_embeddings
    oob_warned = False

    def _process_batch(texts: List[str]) -> bool:
        nonlocal losses, count, current_bs, last_oom_error, oob_warned
        if not texts:
            return True
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # OOB id guard: map any ids outside embedding range to eos_id
        ids = enc["input_ids"]
        oob_mask = (ids < 0) | (ids >= embed_size)
        if oob_mask.any():
            if not oob_warned:
                logger.warning(
                    "Detected %d out-of-bound token ids; mapping to eos_id=%s. "
                    "Check tokenizer/model vocab alignment.",
                    int(oob_mask.sum().item()),
                    str(eos_id),
                )
                oob_warned = True
            ids = ids.masked_fill(oob_mask, eos_id)
            enc["input_ids"] = ids
        enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=active_dtype if active_dtype != torch.float32 else torch.float32):
                out = model(**enc, labels=enc["input_ids"])
                losses.append(float(out.loss.detach().cpu()))
            count += len(texts)
            return True
        except torch.cuda.OutOfMemoryError as e:
            last_oom_error = e
            torch.cuda.empty_cache()
            # Downshift micro-batch and signal failure to retry with smaller chunk.
            current_bs = max(1, current_bs // 2)
            return False
    for example in ds:
        if max_samples and count >= max_samples:
            break
        text = example.get("text") or example.get("document") or example.get("abstract") or ""
        if not text:
            continue
        batch_texts.append(text)
        if len(batch_texts) >= current_bs:
            # Try to process; if OOM, reduce current_bs and retry with a smaller slice.
            while len(batch_texts) >= current_bs:
                if _process_batch(batch_texts[:current_bs]):
                    batch_texts = batch_texts[current_bs:]
                else:
                    # If we dropped to bs=1 and still OOM, give up early with a clear error.
                    if current_bs <= 1:
                        raise RuntimeError(
                            "CUDA out of memory during evaluation at batch size 1; "
                            "reduce --batch-size or --max-length and retry."
                        ) from last_oom_error

    # Flush remaining
    if batch_texts and (not max_samples or count < max_samples):
        # Process any remaining texts in micro-batches of current_bs
        while batch_texts:
            bs = min(current_bs, len(batch_texts))
            if _process_batch(batch_texts[:bs]):
                batch_texts = batch_texts[bs:]
            else:
                if current_bs <= 1:
                    raise RuntimeError(
                        "CUDA out of memory while flushing remaining texts at batch size 1; "
                        "reduce --batch-size or --max-length and retry."
                    ) from last_oom_error

    if not losses:
        return float("nan")

    avg_loss = sum(losses) / len(losses)
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    return perplexity


def _score_candidates_logprob_sum(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidates: Sequence[str],
    dtype: torch.dtype,
    max_length: int = 256,
) -> Dict[str, float]:
    """
    Score each candidate by the (approximate) sum of log-probs over its tokens
    given the prompt. We approximate per-token log-prob by scaling the sequence
    loss with the candidate token count, which is robust and efficient.
    """
    scores: Dict[str, float] = {}
    # Tokenize prompt once
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_ids = enc_prompt["input_ids"][0]
    prompt_len = prompt_ids.shape[0]
    for cand in candidates:
        full = tokenizer(prompt + " " + cand, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = full["input_ids"][0]
        labels_ids = input_ids.clone()
        # Ignore prompt tokens; only supervise candidate tokens
        labels_ids[:prompt_len] = -100
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=dtype if dtype != torch.float32 else torch.float32
        ):
            out = model(
                input_ids=full["input_ids"].to("cuda"),
                labels=labels_ids.unsqueeze(0).to("cuda"),
            )
        # loss is mean NLL over candidate tokens; approximate sum by mean * count
        cand_len = int((labels_ids != -100).sum().item()) or 1
        scores[cand] = -float(out.loss.detach().cpu()) * cand_len
    return scores


def _load_model_with_fallbacks(model_path: str, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    """
    Load model with progressive fallbacks to avoid CUDA device-side asserts.
    """
    def _load_and_smoke_test(torch_dtype: torch.dtype, use_flash: bool) -> AutoModelForCausalLM | None:
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if use_flash and hasattr(cfg, "attn_implementation"):
                setattr(cfg, "attn_implementation", "flash_attention_2")
            model_local = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg if use_flash else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            try:
                model_local = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
            except Exception:
                return None
        try:
            model_local.to("cuda").eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype if torch_dtype != torch.float32 else torch.float32):
                _ = model_local(**tokenizer("hello world", return_tensors="pt", truncation=True, max_length=8).to("cuda"))
            return model_local
        except Exception:
            try:
                del model_local  # type: ignore
            except Exception:
                pass
            torch.cuda.empty_cache()
            return None

    model = (
        _load_and_smoke_test(torch.bfloat16, use_flash=True)
        or _load_and_smoke_test(torch.float16, use_flash=False)
        or _load_and_smoke_test(torch.float32, use_flash=False)
    )
    if model is None:
        raise RuntimeError("Failed to load model on CUDA for evaluation after multiple fallbacks.")
    return model


def _try_load_pubmedqa_bigbio(split: str) -> Optional[datasets.IterableDataset]:
    """
    Try to load BigBio PubMedQA 'source' labeled split.
    Returns streaming IterableDataset or None on failure.
    """
    try:
        ds = datasets.load_dataset(
            "bigbio/pubmed_qa",
            "pubmed_qa_labeled_fold0_source",
            split=split,
            streaming=True,
        )
        logger.info("Loaded bigbio/pubmed_qa (source) split=%s (streaming).", split)
        return ds
    except Exception as e:
        logger.warning("BigBio pubmed_qa source split=%s unavailable: %s", split, e)
        try:
            ds = datasets.load_dataset(
                "bigbio/pubmed_qa",
                "pubmed_qa_labeled_fold0",
                split=split,
                streaming=True,
            )
            logger.info("Loaded bigbio/pubmed_qa (fold0) split=%s (streaming).", split)
            return ds
        except Exception as e2:
            logger.warning("BigBio pubmed_qa fold0 split=%s unavailable: %s", split, e2)
    return None


def _try_load_pubmedqa_standard(split: str) -> Tuple[datasets.IterableDataset, str]:
    """
    Load standard pubmed_qa with a reasonable fallback to train if requested split is missing.
    Returns (dataset, resolved_split).
    """
    cfg = "pqa_labeled"
    try:
        ds = datasets.load_dataset("pubmed_qa", cfg, split=split, streaming=True)
        logger.info("Loaded pubmed_qa:%s split=%s (streaming).", cfg, split)
        return ds, split
    except Exception as e:
        logger.warning("pubmed_qa:%s split=%s unavailable (%s); falling back to train.", cfg, split, e)
        ds = datasets.load_dataset("pubmed_qa", cfg, split="train", streaming=True)
        logger.info("Loaded pubmed_qa:%s split=train (streaming).", cfg)
        return ds, "train"


def _extract_pubmedqa_fields(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Robust extraction for both BigBio and standard schemas.
    Returns (question, context_text, gold_label).
    """
    if "QUESTION" in example:
        question = str(example.get("QUESTION") or "")
        contexts = example.get("CONTEXTS") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        ctx = " ".join([str(c) for c in contexts if isinstance(c, (str,))])
        label = str(example.get("final_decision") or "").strip().lower()
        return question, ctx, label
    question = str(example.get("question") or "")
    ctx_field = example.get("context")
    if isinstance(ctx_field, dict):
        ctx_list = ctx_field.get("contexts") or []
        if not isinstance(ctx_list, list):
            ctx_list = [str(ctx_list)]
        ctx = " ".join([str(c) for c in ctx_list if isinstance(c, (str,))])
    elif isinstance(ctx_field, list):
        ctx = " ".join([str(c) for c in ctx_field])
    else:
        ctx = str(ctx_field or "")
    label = str(example.get("final_decision") or "").strip().lower()
    return question, ctx, label


def evaluate_pubmedqa(
    *,
    model_path: str,
    tokenizer_path: str,
    split: str = "test",
    max_samples: int | None = 200,
    dataset_preference: str = "auto",
) -> dict:
    """
    Zero-shot PubMedQA accuracy using conditional likelihood over {yes,no,maybe}.

    Uses the official PubMedQA dataset from bigbio/pubmed_qa with the
    pubmed_qa_labeled_fold0_source config.
    Only uses the requested split (default: test) for evaluation.
    No fallbacks to train/validation and no alternate datasets.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model_with_fallbacks(model_path, tokenizer)
    dtype = next(model.parameters()).dtype

    # Preferred BigBio; fallback to standard train split if BigBio unavailable in this environment.
    ds: Optional[datasets.IterableDataset] = None
    resolved_split = split
    if dataset_preference in ("auto", "bigbio"):
        ds = _try_load_pubmedqa_bigbio(split)
    if ds is None:
        ds, resolved_split = _try_load_pubmedqa_standard(split)
    labels = ["yes", "no", "maybe"]
    correct = 0
    total = 0
    per_class = {k: {"correct": 0, "total": 0} for k in labels}
    def _score_answer(prompt: str, answer: str) -> float:
        # Compute negative loss over answer tokens only
        full = f"{prompt} {answer}"
        enc_full = tokenizer(full, return_tensors="pt", truncation=True, max_length=256)
        enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc_full["input_ids"][0]
        cutoff = enc_prompt["input_ids"].shape[1]
        labels_ids = input_ids.clone()
        labels_ids[:cutoff] = -100  # ignore prompt tokens
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype if dtype != torch.float32 else torch.float32):
            out = model(input_ids=enc_full["input_ids"].to("cuda"), labels=labels_ids.unsqueeze(0).to("cuda"))
        return -float(out.loss.detach().cpu())

    for example in ds:  # type: ignore[union-attr]
        if max_samples and total >= max_samples:
            break
        question, ctx, label = _extract_pubmedqa_fields(example)
        if label not in labels or not question:
            continue
        prompt = f"Question: {question}\nContext: {ctx}\nAnswer:" if ctx else f"Question: {question}\nAnswer:"
        scores = {ans: _score_answer(prompt, ans) for ans in labels}
        pred = max(scores, key=scores.get)
        total += 1
        per_class[label]["total"] += 1
        if pred == label:
            correct += 1
            per_class[label]["correct"] += 1
    acc = float(correct) / total if total else float("nan")
    return {
        "accuracy": acc,
        "total": total,
        "split": resolved_split,
        "per_class": {
            k: {
                "accuracy": (v["correct"] / v["total"]) if v["total"] else float("nan"),
                "total": v["total"],
            }
            for k, v in per_class.items()
        },
    }


# ---------- New: Alternative classification-style evaluations ----------

def _load_bioasq_yesno(split: str = "test") -> Optional[datasets.IterableDataset]:
    """
    Try several BigBio/BioASQ variants and return a streaming dataset of yes/no QA.
    Returns None if none are available in the current environment.
    """
    trials: List[Tuple[str, Optional[str]]] = [
        ("bigbio/bioasq", "yesno"),
        ("bigbio/bioasq", None),
        ("bigbio/bioasq_qa", None),
    ]
    for ds_name, cfg in trials:
        try:
            if cfg:
                ds = datasets.load_dataset(ds_name, cfg, split=split, streaming=True)
            else:
                ds = datasets.load_dataset(ds_name, split=split, streaming=True)
            logger.info("Loaded %s%s split=%s (streaming).", ds_name, f":{cfg}" if cfg else "", split)
            return ds
        except Exception as e:
            logger.warning("BioASQ trial %s:%s failed: %s", ds_name, cfg, e)
            continue
    return None


def _extract_bioasq_yesno_fields(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract (question, label) from BioASQ-like schema.
    Returns empty strings when missing.
    """
    # BigBio variants differ; try common fields.
    q = str(example.get("question") or example.get("body") or example.get("query") or "")
    # Labels can be 'yes'/'no' or boolean; normalize to lowercase strings.
    label = example.get("answer") or example.get("yesno") or example.get("label")
    if isinstance(label, bool):
        label = "yes" if label else "no"
    label = str(label or "").strip().lower()
    return q, label


def evaluate_yesno(
    *,
    model_path: str,
    tokenizer_path: str,
    max_samples: int | None = 2000,
    dataset: str = "bioasq",
    score_method: str = "logprob_sum",
) -> Dict[str, Any]:
    """
    Deterministic yes/no evaluation using label log-likelihood scoring.
    If BioASQ is unavailable, falls back to PubMedQA filtered to {yes,no}.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model_with_fallbacks(model_path, tokenizer)
    dtype = next(model.parameters()).dtype

    labels = ["yes", "no"]
    per_class = {k: {"correct": 0, "total": 0} for k in labels}
    total = 0
    correct = 0

    ds: Optional[datasets.IterableDataset] = None
    resolved = "test"
    if dataset == "bioasq":
        ds = _load_bioasq_yesno(split="test")
        resolved = "test"
    if ds is None:
        # Fall back to PubMedQA train and filter {yes,no}
        pub_ds, resolved = _try_load_pubmedqa_standard("test")
        ds = pub_ds

    def _score(prompt: str) -> str:
        if score_method == "logprob_sum":
            sc = _score_candidates_logprob_sum(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=labels, dtype=dtype
            )
            return max(sc, key=sc.get)
        else:
            # Fallback to mean-loss (tie-break equivalent)
            sc = _score_candidates_logprob_sum(
                model=model, tokenizer=tokenizer, prompt=prompt, candidates=labels, dtype=dtype
            )
            return max(sc, key=sc.get)

    for ex in ds:  # type: ignore[union-attr]
        if max_samples and total >= max_samples:
            break
        q, y = _extract_bioasq_yesno_fields(ex)
        if not q or y not in labels:
            # Try PubMedQA extraction if BioASQ fields missing
            try:
                q2, ctx2, y2 = _extract_pubmedqa_fields(ex)
                if y2 in labels and q2:
                    q = q2 if not ctx2 else f"{q2}\nContext: {ctx2}"
                    y = y2
                else:
                    continue
            except Exception:
                continue
        prompt = f"Question: {q}\nAnswer:"
        pred = _score(prompt)
        total += 1
        per_class[y]["total"] += 1
        if pred == y:
            correct += 1
            per_class[y]["correct"] += 1

    acc = float(correct) / total if total else float("nan")
    return {
        "accuracy": acc,
        "total": total,
        "split": resolved,
        "per_class": {k: {"accuracy": (v["correct"] / v["total"]) if v["total"] else float("nan"), "total": v["total"]}
                      for k, v in per_class.items()},
        "labels": labels,
    }


def _try_load_medmcqa(split: str = "validation") -> Optional[datasets.IterableDataset]:
    """
    Try several MedMCQA variants; return streaming dataset or None.
    """
    trials: List[Tuple[str, Optional[str]]] = [
        ("openlifescienceai/medmcqa", None),
        ("medmcqa", None),
        ("AI4Medicine/MedMCQA", None),
    ]
    for name, cfg in trials:
        try:
            if cfg:
                ds = datasets.load_dataset(name, cfg, split=split, streaming=True)
            else:
                ds = datasets.load_dataset(name, split=split, streaming=True)
            logger.info("Loaded %s%s split=%s (streaming).", name, f":{cfg}" if cfg else "", split)
            return ds
        except Exception as e:
            logger.warning("MedMCQA trial %s:%s failed: %s", name, cfg, e)
    return None


def _extract_medmcqa_fields(example: Dict[str, Any]) -> Tuple[str, List[str], Optional[int]]:
    """
    Normalize MedMCQA example into (question, [A,B,C,D], correct_index or None)
    """
    q = str(example.get("question") or example.get("Q") or "")
    # Common option keys: opa/opb/opc/opd or A/B/C/D
    opts = []
    for k in ("opa", "opb", "opc", "opd"):
        if k in example:
            opts = [str(example.get("opa") or ""), str(example.get("opb") or ""), str(example.get("opc") or ""), str(example.get("opd") or "")]
            break
    if not opts:
        opts = [str(example.get("A") or ""), str(example.get("B") or ""), str(example.get("C") or ""), str(example.get("D") or "")]
    # Answer may be letter or index
    ans = example.get("cop") or example.get("answer") or example.get("label")
    idx: Optional[int] = None
    if isinstance(ans, str):
        s = ans.strip().lower()
        map_ltr = {"a": 0, "b": 1, "c": 2, "d": 3}
        idx = map_ltr.get(s)
        if idx is None:
            # If answer equals the option text, match
            try:
                idx = opts.index(ans)
            except Exception:
                idx = None
    elif isinstance(ans, int):
        idx = ans if 0 <= ans <= 3 else None
    return q, opts, idx


def evaluate_mcq(
    *,
    model_path: str,
    tokenizer_path: str,
    max_samples: int | None = 2000,
    dataset: str = "medmcqa",
    score_method: str = "logprob_sum",
) -> Dict[str, Any]:
    """
    Multiple-choice (A–D) evaluation with deterministic choice scoring.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for eval")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _load_model_with_fallbacks(model_path, tok)
    dtype = next(model.parameters()).dtype

    ds = _try_load_medmcqa(split="validation") or _try_load_medmcqa(split="test") or _try_load_medmcqa(split="train")
    if ds is None:
        raise RuntimeError("Unable to load MedMCQA dataset in this environment.")

    labels = ["A", "B", "C", "D"]
    per_class = {k: {"correct": 0, "total": 0} for k in labels}
    cm = [[0, 0, 0, 0] for _ in range(4)]
    total = 0
    correct = 0

    def _score(prompt: str, options: List[str]) -> int:
        # Score option letter prefixed to normalize style
        candidates = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]
        sc = _score_candidates_logprob_sum(
            model=model, tokenizer=tok, prompt=prompt, candidates=candidates, dtype=dtype
        )
        # pick best by score
        best = max(range(4), key=lambda i: sc[candidates[i]])
        return best

    for ex in ds:  # type: ignore[union-attr]
        if max_samples and total >= max_samples:
            break
        q, options, gold_idx = _extract_medmcqa_fields(ex)
        if not q or len(options) != 4 or gold_idx is None:
            continue
        prompt = f"Question: {q}\nChoices: (A) {options[0]} (B) {options[1]} (C) {options[2]} (D) {options[3]}\nAnswer:"
        pred_idx = _score(prompt, options)
        total += 1
        cm[gold_idx][pred_idx] += 1
        per_class[labels[gold_idx]]["total"] += 1
        if pred_idx == gold_idx:
            correct += 1
            per_class[labels[gold_idx]]["correct"] += 1

    acc = float(correct) / total if total else float("nan")
    per_class_acc = {
        k: {"accuracy": (v["correct"] / v["total"]) if v["total"] else float("nan"), "total": v["total"]}
        for k, v in per_class.items()
    }
    return {
        "accuracy": acc,
        "total": total,
        "labels": labels,
        "per_class": per_class_acc,
        "confusion_matrix": cm,
    }


def compute_term_tokenization_coverage(terms_path: str, tokenizer_path: str) -> Dict[str, Any]:
    """
    Compute tokenization coverage for mined medical terms.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lengths: List[int] = []
    total = 0
    with open(terms_path, "r", encoding="utf-8") as fp:
        for line in fp:
            term = line.strip()
            if not term:
                continue
            ids = tok(term, add_special_tokens=False)["input_ids"]
            piece_len = len(ids) if isinstance(ids, list) else int(ids.shape[-1])
            lengths.append(piece_len)
            total += 1
    if not total:
        return {
            "total_terms": 0,
            "single_token_ratio": float("nan"),
            "mean_tokens_per_term": float("nan"),
            "median_tokens_per_term": float("nan"),
            "p95_tokens_per_term": float("nan"),
        }
    single = sum(1 for l in lengths if l == 1)
    return {
        "total_terms": total,
        "single_token_ratio": single / total,
        "mean_tokens_per_term": float(_stats.mean(lengths)),
        "median_tokens_per_term": float(_stats.median(lengths)),
        "p95_tokens_per_term": float(_stats.quantiles(lengths, n=20)[-1]) if len(lengths) >= 20 else max(lengths),
    }


def compute_alignment_coverage(
    *,
    vocab_mapping_path: str,
    target_tokenizer_path: str,
) -> Dict[str, Any]:
    """
    Estimate alignment coverage using vocab_mapping.json (target→source coverage).
    """
    tok = AutoTokenizer.from_pretrained(target_tokenizer_path, trust_remote_code=True)
    target_vocab = int(getattr(tok, "vocab_size", None) or len(tok))
    mapped = 0
    try:
        with open(vocab_mapping_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            if "mapping" in data and isinstance(data["mapping"], dict):
                mapped = len([k for k, v in data["mapping"].items() if v is not None])
            elif "target_to_source" in data and isinstance(data["target_to_source"], dict):
                mapped = len([k for k, v in data["target_to_source"].items() if v is not None])
            else:
                mapped = len([k for k, v in data.items() if str(k).isdigit() and v is not None])
        elif isinstance(data, list):
            mapped = len(data)
    except Exception as e:
        logger.warning("Failed to parse vocab_mapping.json (%s); alignment coverage unknown.", e)
        return {"target_vocab": target_vocab, "mapped": None, "coverage": None}
    coverage = mapped / target_vocab if target_vocab > 0 else None
    return {"target_vocab": target_vocab, "mapped": mapped, "coverage": coverage}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medical TokAlign models.")
    parser.add_argument("--model", required=True, help="Path to adapted model directory.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path used during evaluation.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Hugging Face dataset name (optionally dataset:split). Can be supplied multiple times.",
    )
    parser.add_argument("--output", required=True, help="Path to JSON file with evaluation results.")
    # Perplexity/general knobs
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate per dataset (0 = all samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts per forward pass.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    # PubMedQA options
    parser.add_argument(
        "--run-pubmedqa",
        action="store_true",
        help="Run PubMedQA accuracy in addition to perplexity.",
    )
    parser.add_argument(
        "--baseline-model",
        default="mistralai/Mistral-7B-v0.3",
        help="Baseline model for PubMedQA (only used if --run-pubmedqa).",
    )
    parser.add_argument(
        "--pubmedqa-dataset",
        default="auto",
        choices=["auto", "bigbio", "standard"],
        help="Dataset preference for PubMedQA.",
    )
    # New: Classification-style tasks
    parser.add_argument(
        "--eval-task",
        choices=["ynm", "mcq"],
        help="Alternative classification task to run (yes/no/maybe or multiple-choice).",
    )
    parser.add_argument(
        "--eval-dataset",
        choices=["bioasq", "pubmedqa", "medmcqa"],
        help="Dataset to use for the alternative evaluation.",
    )
    parser.add_argument(
        "--score-method",
        default="logprob_sum",
        choices=["logprob_sum", "loss_mean"],
        help="Scoring method for classification-style evaluations.",
    )
    parser.add_argument(
        "--log-missing",
        action="store_true",
        help="If set and no dataset is provided, write a placeholder log describing required evaluations.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    os.makedirs(Path(args.output).parent, exist_ok=True)

    datasets_to_evaluate: List[str] = args.dataset if args.dataset else DEFAULT_DATASETS

    if not datasets_to_evaluate:
        placeholder = {
            "status": "missing_datasets",
            "message": (
                "No medical evaluation datasets were provided and no defaults are configured. "
                "Supply --dataset flags with Hugging Face IDs."
            ),
        }
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(placeholder, fp, indent=2)
        if args.log_missing:
            logger.warning(placeholder["message"])
        return

    results: Dict[str, Any] = {}
    for dataset_item in datasets_to_evaluate:
        if ":" in dataset_item:
            dataset_name, split = dataset_item.split(":", maxsplit=1)
        else:
            dataset_name, split = dataset_item, "test"
        logger.info("Evaluating dataset %s split %s.", dataset_name, split)
        perplexity = evaluate_perplexity(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            dataset_name=dataset_name,
            split=split,
            max_samples=None if args.max_samples <= 0 else args.max_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        results[f"{dataset_name}:{split}"] = {"perplexity": perplexity}

    if args.run_pubmedqa:
        try:
            baseline = evaluate_pubmedqa(
                model_path=args.baseline_model,
                tokenizer_path=args.baseline_model,
                split="test",
                max_samples=None if args.max_samples <= 0 else args.max_samples,
                dataset_preference=args.pubmedqa_dataset,
            )
            adapted = evaluate_pubmedqa(
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                split="test",
                max_samples=None if args.max_samples <= 0 else args.max_samples,
                dataset_preference=args.pubmedqa_dataset,
            )
            results["pubmedqa"] = {
                "baseline": baseline,
                "adapted": adapted,
                "delta": {"accuracy": adapted["accuracy"] - baseline["accuracy"]},
            }
        except Exception as qa_exc:
            logger.warning("PubMedQA evaluation failed: %s", qa_exc)
            results["pubmedqa"] = {"status": "error", "message": str(qa_exc)}

    # New: Alternative evaluations (classification-style)
    if args.eval_task:
        try:
            if args.eval_task == "ynm":
                # yes/no (optionally maybe if dataset provides)
                # Baseline
                baseline = evaluate_yesno(
                    model_path=args.baseline_model,
                    tokenizer_path=args.baseline_model,
                    max_samples=None if args.max_samples <= 0 else args.max_samples,
                    dataset=args.eval_dataset or "bioasq",
                    score_method=args.score_method,
                )
                adapted = evaluate_yesno(
                    model_path=args.model,
                    tokenizer_path=args.tokenizer,
                    max_samples=None if args.max_samples <= 0 else args.max_samples,
                    dataset=args.eval_dataset or "bioasq",
                    score_method=args.score_method,
                )
                results["ynm"] = {
                    "baseline": baseline,
                    "adapted": adapted,
                    "delta": {"accuracy": adapted["accuracy"] - baseline["accuracy"]},
                }
            elif args.eval_task == "mcq":
                baseline = evaluate_mcq(
                    model_path=args.baseline_model,
                    tokenizer_path=args.baseline_model,
                    max_samples=None if args.max_samples <= 0 else args.max_samples,
                    dataset=args.eval_dataset or "medmcqa",
                    score_method=args.score_method,
                )
                adapted = evaluate_mcq(
                    model_path=args.model,
                    tokenizer_path=args.tokenizer,
                    max_samples=None if args.max_samples <= 0 else args.max_samples,
                    dataset=args.eval_dataset or "medmcqa",
                    score_method=args.score_method,
                )
                results["mcq"] = {
                    "baseline": baseline,
                    "adapted": adapted,
                    "delta": {"accuracy": adapted["accuracy"] - baseline["accuracy"]},
                }
        except Exception as alt_exc:  # pragma: no cover
            logger.warning("Alternative eval '%s' failed: %s", args.eval_task, alt_exc)
            results[args.eval_task] = {"status": "error", "message": str(alt_exc)}

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Evaluation results written to %s.", args.output)


if __name__ == "__main__":
    main()


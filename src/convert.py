import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import random
import argparse

_EMBED_DICT = {
    "gpt_neox": "gpt_neox.embed_in.weight",
    "llama": "model.embed_tokens.weight",
    "mistral": "model.embed_tokens.weight",
}

_LMHEAD_DICT = {
    "gpt_neox": "embed_out.weight",
    "llama": "lm_head.weight",
    "mistral": "lm_head.weight",
}

def trans2switch(
    trans_path="./log/gemma2pythia/glove.json",
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2gemma",
    tgt_tok_path="./data/gemma-2b",
    random_shuffle=-1,
):
    # Prefer full precision on GPU when available (96GB VRAM fits 7B comfortably).
    use_cuda = torch.cuda.is_available()
    preferred_dtype = torch.float32 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else "cpu"
    try:
        src_model = AutoModelForCausalLM.from_pretrained(
            src_clm_path,
            torch_dtype=preferred_dtype,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    except Exception:
        # Memory/driver fallback: bf16 on GPU, then fp16, then fp32 CPU as last resort
        try:
            src_model = AutoModelForCausalLM.from_pretrained(
                src_clm_path,
                torch_dtype=torch.bfloat16 if use_cuda else torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        except Exception:
            try:
                src_model = AutoModelForCausalLM.from_pretrained(
                    src_clm_path,
                    torch_dtype=torch.float16 if use_cuda else torch.float16,
                    trust_remote_code=True,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
            except Exception:
                src_model = AutoModelForCausalLM.from_pretrained(
                    src_clm_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map="cpu",
                )
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)

    # Load trans matrix
    with open(trans_path, "r") as f:
        trans = json.load(f)
    
    with torch.no_grad():
        src_params = dict(src_model.named_parameters())

        src_embed = src_params[_EMBED_DICT[src_model.config.model_type]]
        src_lm_head = src_params[_LMHEAD_DICT[src_model.config.model_type]]

        assert src_embed.shape[0] == src_lm_head.shape[0]

        hid_dim = src_embed.shape[1]

        src_len = src_embed.shape[0]
        tgt_len = len(tgt_tok)
        mapping_entries = len(trans) if isinstance(trans, dict) else len(list(trans))
        if mapping_entries != tgt_len:
            print(
                f"[convert] Alignment matrix entries={mapping_entries} differ from target tokenizer length={tgt_len}. "
                "Missing rows will be zero-initialized."
            )

        dtype = src_embed.dtype
        tgt_embed = torch.zeros((tgt_len, hid_dim), dtype=dtype)
        tgt_lm_head = torch.zeros((tgt_len, hid_dim), dtype=dtype)

        missing_targets = []

        for i in range(tgt_len):
            tj_value = trans.get(f"{i}")
            if isinstance(tj_value, (list, tuple)):
                tj_value = tj_value[0] if len(tj_value) > 0 else None
            try:
                tj = int(tj_value) if tj_value is not None else -1
            except (TypeError, ValueError):
                tj = -1
            if random_shuffle > 0 and random.random() < random_shuffle:
                tj = random.randint(0, src_len - 1)

            if tj < 0 or tj >= src_len:
                missing_targets.append(i)
                tgt_embed[i] = torch.zeros(hid_dim, dtype=dtype)
                tgt_lm_head[i] = torch.zeros(hid_dim, dtype=dtype)
                continue

            tgt_embed[i] = src_embed[tj]
            tgt_lm_head[i] = src_lm_head[tj]
            if i > 0 and i % 50000 == 0:
                print(f"[convert] Processed {i}/{tgt_len} tokens...")

        if missing_targets:
            print(
                f"[convert] Initialized {len(missing_targets)} target tokens with zeros because they mapped "
                f"outside the source vocabulary (max index {src_len - 1}). First few: {missing_targets[:10]}"
            )
        else:
            print("[convert] All target tokens mapped to source embeddings.")

        print(f"[convert] Finalizing model with target vocab size {tgt_len}...")
        # Resize token embeddings first, then copy weights directly to modules
        src_model.resize_token_embeddings(tgt_len)
        # Copy into the resized modules to avoid state_dict shape mismatches
        src_model.get_input_embeddings().weight.data.copy_(tgt_embed.to(src_model.get_input_embeddings().weight.dtype))
        lm_head = getattr(src_model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight.data.copy_(tgt_lm_head.to(lm_head.weight.dtype))
        else:
            # Fallback for uncommon heads
            state = src_model.state_dict()
            state[_LMHEAD_DICT[src_model.config.model_type]] = tgt_lm_head.to(state[_LMHEAD_DICT[src_model.config.model_type]].dtype)
            src_model.load_state_dict(state, strict=False)
        src_model.save_pretrained(tgt_clm_path)
        tgt_tok.save_pretrained(tgt_clm_path)
        summary = {
            "source_vocab": src_len,
            "target_vocab": tgt_len,
            "mapped": tgt_len - len(missing_targets),
            "zero_initialized": len(missing_targets),
        }
        summary_path = Path(tgt_clm_path) / "tokalign_alignment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[convert] Alignment summary written to {summary_path}")

def random_permute(
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2gemma",
    tgt_tok_path="./data/gemma-2b",
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
    )
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)

    with torch.no_grad():
        src_params = dict(src_model.named_parameters())

        src_embed = src_params[_EMBED_DICT[src_model.config.model_type]]
        src_lm_head = src_params[_LMHEAD_DICT[src_model.config.model_type]]

        assert src_embed.shape[0] == src_lm_head.shape[0]

        src_len, hid_dim = src_embed.shape[0], src_embed.shape[1]

        tgt_len = len(tgt_tok)

        dtype = src_embed.dtype
        tgt_embed = torch.zeros((tgt_len, hid_dim), dtype=dtype)
        tgt_lm_head = torch.zeros((tgt_len, hid_dim), dtype=dtype)

        for i in range(tgt_len):
            tj = random.randint(0, src_len - 1)
            tgt_embed[i] = src_embed[tj]
            tgt_lm_head[i] = src_lm_head[tj]

        src_model.resize_token_embeddings(len(tgt_tok))

        src_model.get_input_embeddings().weight.data.copy_(tgt_embed.to(src_model.get_input_embeddings().weight.dtype))
        lm_head = getattr(src_model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight.data.copy_(tgt_lm_head.to(lm_head.weight.dtype))
        else:
            state = src_model.state_dict()
            state[_LMHEAD_DICT[src_model.config.model_type]] = tgt_lm_head.to(state[_LMHEAD_DICT[src_model.config.model_type]].dtype)
            src_model.load_state_dict(state, strict=False)
        src_model.save_pretrained(tgt_clm_path)
        tgt_tok.save_pretrained(tgt_clm_path)

def random_initial_all(
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2gemma",
    tgt_tok_path="./data/gemma-2b",
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
    )
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)
    
    with torch.no_grad():
        src_params = dict(src_model.named_parameters())

        src_embed = src_params[_EMBED_DICT[src_model.config.model_type]]
        src_lm_head = src_params[_LMHEAD_DICT[src_model.config.model_type]]

        assert src_embed.shape[0] == src_lm_head.shape[0]

        src_len, hid_dim = src_embed.shape[0], src_embed.shape[1]
        dtype = src_embed.dtype

        tgt_len = len(tgt_tok)

        src_model.resize_token_embeddings(src_len + tgt_len)

        # After first resize, take the newly-added rows as initial values for target vocab
        new_embed = src_model.get_input_embeddings().weight.data[src_len:].clone()
        new_lm_head = getattr(src_model, "lm_head", None)
        if new_lm_head is not None and hasattr(new_lm_head, "weight"):
            new_lm = new_lm_head.weight.data[src_len:].clone()
        else:
            # Fallback to state_dict if head is not a direct attribute
            state = src_model.state_dict()
            new_lm = state[_LMHEAD_DICT[src_model.config.model_type]][src_len:].clone()

        src_model.resize_token_embeddings(tgt_len)

        # Copy the newly created slices into the correctly-sized resized modules
        src_model.get_input_embeddings().weight.data.copy_(new_embed.to(src_model.get_input_embeddings().weight.dtype))
        if hasattr(src_model, "lm_head") and hasattr(src_model.lm_head, "weight"):
            src_model.lm_head.weight.data.copy_(new_lm.to(src_model.lm_head.weight.dtype))
        else:
            state_after = src_model.state_dict()
            state_after[_LMHEAD_DICT[src_model.config.model_type]] = new_lm.to(state_after[_LMHEAD_DICT[src_model.config.model_type]].dtype)
            src_model.load_state_dict(state_after, strict=False)
        src_model.save_pretrained(tgt_clm_path)
        tgt_tok.save_pretrained(tgt_clm_path)


def random_initial_aug(
    src_clm_path="./data/pythia-1b",
    tgt_clm_path="./data/pythia-1b2gemma",
    tgt_tok_path="./data/gemma-2b",
    seed=0,
):
    random.seed(seed)
    set_seed(seed)

    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
    )
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path,  trust_remote_code=True)

    with torch.no_grad():
        src_model.resize_token_embeddings(len(tgt_tok))
        src_model.save_pretrained(tgt_clm_path)
        tgt_tok.save_pretrained(tgt_clm_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--one2one-matrix-path", type=str, default="./data/pythia2gemma/glove.json")
    parser.add_argument("-s", "--source-model-path", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("-t", "--target-tokenizer-path", type=str, default="google/gemma-2b")
    parser.add_argument("-o", "--output-model-path", type=str, default="./data/pythia2gemma/glove")
    parser.add_argument("-r", "--random-shuffle-percentage", type=float, default=-1, help="The percentage of token pairs that are randomly shuffled rather than map to the target.")

    args = parser.parse_args()

    trans2switch(
        trans_path=args.one2one_matrix_path,
        src_clm_path=args.source_model_path,
        tgt_clm_path=args.output_model_path,
        tgt_tok_path=args.target_tokenizer_path,
        random_shuffle=args.random_shuffle_percentage
    )

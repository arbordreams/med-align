import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
import json
import torch
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
    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path,
        dtype=torch.float32,
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
        tgt_len = len(list(trans.keys()))

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

        if missing_targets:
            print(
                f"[convert] Initialized {len(missing_targets)} target tokens with zeros because they mapped "
                f"outside the source vocabulary (max index {src_len - 1}). First few: {missing_targets[:10]}"
            )

        src_model.resize_token_embeddings(tgt_len)

        src_params[_EMBED_DICT[src_model.config.model_type]] = tgt_embed.to(dtype)
        src_params[_LMHEAD_DICT[src_model.config.model_type]] = tgt_lm_head.to(dtype)

        src_model.load_state_dict(src_params)
        src_model.save_pretrained(tgt_clm_path)
        tgt_tok.save_pretrained(tgt_clm_path)

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
        dtype=torch.float32,
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

        src_params[_EMBED_DICT[src_model.config.model_type]] = tgt_embed.to(dtype)
        src_params[_LMHEAD_DICT[src_model.config.model_type]] = tgt_lm_head.to(dtype)

        src_model.load_state_dict(src_params)
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
        dtype=torch.float32,
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

        resized_params = dict(src_model.named_parameters())

        src_params[_EMBED_DICT[src_model.config.model_type]] = resized_params[_EMBED_DICT[src_model.config.model_type]][src_len:].to(dtype)
        src_params[_LMHEAD_DICT[src_model.config.model_type]] = resized_params[_LMHEAD_DICT[src_model.config.model_type]][src_len:].to(dtype)

        src_model.resize_token_embeddings(tgt_len)

        src_model.load_state_dict(src_params)
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
        dtype=torch.float32,
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

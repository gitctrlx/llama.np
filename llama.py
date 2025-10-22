import math
import sys
import time
from typing import Optional

import numpy as np

from tokenizer import Tokenizer
from config import LlamaConfig


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))


def swiglu(x: np.ndarray, gate: np.ndarray) -> np.ndarray:
    swish = gate * (1 / (1 + np.exp(-gate)))
    return x * swish


def rotate_half(x: np.ndarray) -> np.ndarray:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos = np.expand_dims(cos, axis=(0, 2))  # (1, seq_len, 1, head_dim)
    sin = np.expand_dims(sin, axis=(0, 2))
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return x
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    return np.repeat(x[:, :, None, :, :], n_rep, axis=2).reshape(
        bsz, seqlen, n_kv_heads * n_rep, head_dim
    )


class LlamaRMSNorm:
    def __init__(self, weight: np.ndarray, eps: float):
        self.weight = weight
        self.variance_epsilon = eps

    def __call__(self, hidden_states: np.ndarray) -> np.ndarray:
        variance = (
            np.mean(hidden_states**2, axis=-1, keepdims=True) + self.variance_epsilon
        )
        hidden_states = hidden_states / np.sqrt(variance)
        return hidden_states * self.weight


class LlamaRotaryEmbedding:
    def __init__(self, config: LlamaConfig):
        head_dim = config.head_dim
        max_seq_len = config.max_position_embeddings
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling

        if rope_scaling and rope_scaling["type"] == "linear":
            scaling_factor = rope_scaling.get("factor", 1.0)
            max_seq_len = int(max_seq_len * scaling_factor)
        elif rope_scaling and rope_scaling["type"] == "dynamic":
            # Dynamic scaling: Simplified placeholder; implement runtime adjustment if needed
            pass

        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = np.cos(emb).astype(np.float32)
        self.sin_cached = np.sin(emb).astype(np.float32)


class LlamaMLP:
    def __init__(
        self,
        gate_proj_weight: np.ndarray,
        up_proj_weight: np.ndarray,
        down_proj_weight: np.ndarray,
        hidden_act: str,
    ):
        self.gate_proj = gate_proj_weight.T
        self.up_proj = up_proj_weight.T
        self.down_proj = down_proj_weight.T
        self.hidden_act = hidden_act

    def __call__(self, x: np.ndarray) -> np.ndarray:
        gate = x @ self.gate_proj
        if self.hidden_act == "swiglu":
            gate = swiglu(x, gate)
        elif self.hidden_act == "silu":
            gate = silu(gate)
        else:
            raise ValueError(f"Unsupported hidden_act: {self.hidden_act}")
        up = x @ self.up_proj
        ff = gate * up
        return ff @ self.down_proj


class LlamaAttention:
    def __init__(
        self,
        config: LlamaConfig,
        q_proj_weight: np.ndarray,
        k_proj_weight: np.ndarray,
        v_proj_weight: np.ndarray,
        o_proj_weight: np.ndarray,
    ):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = q_proj_weight.T
        self.k_proj = k_proj_weight.T
        self.v_proj = v_proj_weight.T
        self.o_proj = o_proj_weight.T

        self.cache_k = np.zeros(
            (
                config.max_batch_size,
                config.max_position_embeddings,
                self.num_key_value_heads,
                self.head_dim,
            )
        )
        self.cache_v = np.zeros(
            (
                config.max_batch_size,
                config.max_position_embeddings,
                self.num_key_value_heads,
                self.head_dim,
            )
        )

    def __call__(
        self,
        hidden_states: np.ndarray,
        start_pos: int,
        cos: np.ndarray,
        sin: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        bsz, seqlen, _ = hidden_states.shape

        xq = hidden_states @ self.q_proj
        xk = hidden_states @ self.k_proj
        xv = hidden_states @ self.v_proj

        xq = xq.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        xq = xq.transpose(0, 2, 1, 3)  # (bsz, num_heads, seqlen, head_dim)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        scores = np.matmul(xq, keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask[None, None, :, :]

        attn_weights = softmax(scores)
        attn_output = np.matmul(attn_weights, values)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return attn_output @ self.o_proj


class LlamaDecoderLayer:
    def __init__(
        self, config: LlamaConfig, layer_idx: int, weights: dict[str, np.ndarray]
    ):
        prefix = f"model.layers.{layer_idx}."
        self.self_attn = LlamaAttention(
            config,
            weights[f"{prefix}self_attn.q_proj.weight"],
            weights[f"{prefix}self_attn.k_proj.weight"],
            weights[f"{prefix}self_attn.v_proj.weight"],
            weights[f"{prefix}self_attn.o_proj.weight"],
        )
        self.mlp = LlamaMLP(
            weights[f"{prefix}mlp.gate_proj.weight"],
            weights[f"{prefix}mlp.up_proj.weight"],
            weights[f"{prefix}mlp.down_proj.weight"],
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(
            weights[f"{prefix}input_layernorm.weight"], eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            weights[f"{prefix}post_attention_layernorm.weight"], eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: np.ndarray,
        start_pos: int,
        cos: np.ndarray,
        sin: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, start_pos, cos, sin, mask)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states


class LlamaModel:
    def __init__(self, config: LlamaConfig, weights: dict[str, np.ndarray]):
        self.config = config
        self.embed_tokens = weights["model.embed_tokens.weight"]
        self.layers = [
            LlamaDecoderLayer(config, layer_idx, weights)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = LlamaRMSNorm(weights["model.norm.weight"], eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(self, tokens: np.ndarray, start_pos: int) -> np.ndarray:
        bsz, seqlen = tokens.shape
        hidden_states = self.embed_tokens[tokens]

        cos = self.rotary_emb.cos_cached[start_pos : start_pos + seqlen]
        sin = self.rotary_emb.sin_cached[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = np.zeros((seqlen, seqlen), dtype=hidden_states.dtype)
            mask[np.triu_indices(seqlen, k=1)] = -np.inf
            mask = np.hstack(
                [np.zeros((seqlen, start_pos), dtype=hidden_states.dtype), mask]
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, start_pos, cos, sin, mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM:
    def __init__(self, model_path: str, config: LlamaConfig):
        self.config = config
        weights = np.load(model_path)
        self.model = LlamaModel(config, weights)
        self.lm_head = weights["lm_head.weight"].T
        del weights

    def forward(self, tokens: np.ndarray, start_pos: int) -> np.ndarray:
        hidden_states = self.model.forward(tokens, start_pos)
        return hidden_states[:, -1, :] @ self.lm_head

    def generate(self, tokens: np.ndarray, max_new_tokens: int) -> np.ndarray:
        _, seqlen = tokens.shape
        for cur_pos in range(seqlen, seqlen + max_new_tokens):
            if cur_pos == seqlen:
                logits = self.forward(tokens, 0)
            else:
                logits = self.forward(next_token[:, None], cur_pos - 1)
            next_token = np.argmax(logits, axis=-1)
            yield next_token[:, None]


if __name__ == "__main__":
    # Example: Use LLaMA-1 7B config
    config = get_llama3_8b_config()
    tokenizer = Tokenizer("./tokenizer.np")
    model = LlamaForCausalLM(
        "./stories15M.npz", config
    )  # Replace with actual model path

    prompt = sys.argv[1] if len(sys.argv) > 1 else "I have a dream"
    print(f"\n{prompt}", end="", flush=True)

    input_tokens = np.array([tokenizer.encode(prompt)])
    start_time = time.time()
    generated_len = input_tokens.shape[1]

    for token_id in model.generate(
        input_tokens, 50
    ):  # Hardcoded max_new_tokens for example
        generated_len += 1
        tid = int(token_id[0, 0])
        if tid in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        decoded = tokenizer.decode(tid)
        print(decoded, end="", flush=True)

    elapsed = time.time() - start_time
    print(
        f"\n\nToken count: {generated_len}, elapsed: {elapsed:.2f}s, {generated_len / elapsed:.0f} tokens/s"
    )

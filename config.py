from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaConfig:
    vocab_size: int = 32000  # LLaMA-1/2: 32000, LLaMA-3: 128256
    hidden_size: int = 4096  # LLaMA-7B: 4096, LLaMA-13B: 5120, LLaMA-70B: 8192
    intermediate_size: int = 11008  # LLaMA-7B: 11008, LLaMA-13B: 13824
    num_hidden_layers: int = 32  # LLaMA-7B: 32, LLaMA-13B: 40
    num_attention_heads: int = 32  # LLaMA-7B: 32, LLaMA-13B: 40
    num_key_value_heads: Optional[int] = None  # GQA for LLaMA-3, e.g., 8 for 70B
    head_dim: Optional[int] = None  # Default: hidden_size // num_attention_heads
    max_position_embeddings: int = 2048  # LLaMA-1: 2048, LLaMA-2: 4096, LLaMA-3: 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0  # LLaMA-3 may use higher values (e.g., 500000.0)
    max_batch_size: int = 1
    hidden_act: str = "swiglu"  # Support "silu" or "swiglu" for LLaMA-1/2/3
    attention_dropout: float = 0.0  # For inference, typically 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None
    rope_scaling: Optional[dict] = None  # For LLaMA-3 long sequences

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # Validate rope_scaling if provided
        if self.rope_scaling:
            assert isinstance(self.rope_scaling, dict), "rope_scaling must be a dict"
            assert "type" in self.rope_scaling, "rope_scaling must specify 'type'"
            assert self.rope_scaling["type"] in [
                "linear",
                "dynamic",
            ], "rope_scaling type must be 'linear' or 'dynamic'"


# Predefined configurations for LLaMA-1/2/3
def get_llama1_7b_config():
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # MHA
        max_position_embeddings=2048,
        hidden_act="swiglu",
        rope_theta=10000.0,
    )


def get_llama2_7b_config():
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # MHA or GQA in variants
        max_position_embeddings=4096,
        hidden_act="swiglu",
        rope_theta=10000.0,
    )


def get_llama3_8b_config():
    return LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        max_position_embeddings=8192,
        hidden_act="swiglu",
        rope_theta=500000.0,
        rope_scaling={"type": "linear", "factor": 1.0},
    )

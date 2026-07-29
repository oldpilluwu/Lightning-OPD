"""Qwen3-4B training model for PyTorch 2.9/XLA and NxD Core.

It follows the Hugging Face Qwen3 architecture while replacing the large
linear/embedding layers with NxD tensor-parallel layers and using the Neuron
NKI flash-attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)


def _normal_init(std: float, tensor: torch.Tensor) -> torch.Tensor:
    return nn.init.normal_(tensor, mean=0.0, std=std)


def validate_qwen3_4b_base_config(config: Qwen3Config) -> None:
    expected = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "hidden_act": "silu",
        "rope_theta": 1_000_000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "use_sliding_window": False,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        raise ValueError(
            f"Model config does not match Qwen/Qwen3-4B-Base: {mismatches}"
        )
    if config.rope_scaling is not None:
        raise ValueError("Qwen3-4B-Base reproduction expects rope_scaling=None")


class Qwen3RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype,
        sequence_parallel_enabled: bool,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        setattr(
            self.weight,
            "sequence_parallel_enabled",
            sequence_parallel_enabled,
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        frequencies = torch.matmul(inv_freq, positions).transpose(1, 2)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embeddings.cos().to(hidden_states.dtype),
            embeddings.sin().to(hidden_states.dtype),
        )


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        if config.num_key_value_heads % tp_size:
            raise ValueError("num_key_value_heads must be divisible by TP size")

        self.head_dim = config.head_dim
        self.local_query_heads = config.num_attention_heads // tp_size
        self.local_kv_heads = config.num_key_value_heads // tp_size
        self.local_kv_groups = self.local_query_heads // self.local_kv_heads
        self.sequence_parallel_enabled = config.sequence_parallel_enabled
        self.scaling = self.head_dim**-0.5
        init_method = partial(_normal_init, config.initializer_range)

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.q_norm = Qwen3RMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            config.torch_dtype,
            self.sequence_parallel_enabled,
        )
        self.k_norm = Qwen3RMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            config.torch_dtype,
            self.sequence_parallel_enabled,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.sequence_parallel_enabled:
            local_sequence, batch, _ = hidden_states.shape
            sequence = local_sequence * get_tensor_model_parallel_size()
            query = self.q_proj(hidden_states).view(
                sequence, batch, self.local_query_heads, self.head_dim
            )
            key = self.k_proj(hidden_states).view(
                sequence, batch, self.local_kv_heads, self.head_dim
            )
            value = self.v_proj(hidden_states).view(
                sequence, batch, self.local_kv_heads, self.head_dim
            )
            query = self.q_norm(query).permute(1, 2, 0, 3)
            key = self.k_norm(key).permute(1, 2, 0, 3)
            value = value.permute(1, 2, 0, 3)
        else:
            batch, sequence, _ = hidden_states.shape
            query = self.q_proj(hidden_states).view(
                batch, sequence, self.local_query_heads, self.head_dim
            )
            key = self.k_proj(hidden_states).view(
                batch, sequence, self.local_kv_heads, self.head_dim
            )
            value = self.v_proj(hidden_states).view(
                batch, sequence, self.local_kv_heads, self.head_dim
            )
            query = self.q_norm(query).permute(0, 2, 1, 3)
            key = self.k_norm(key).permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

        query, key = apply_rotary_pos_emb(
            query, key, position_embeddings[0], position_embeddings[1]
        )
        key = repeat_kv(key, self.local_kv_groups)
        value = repeat_kv(value, self.local_kv_groups)

        # The Trn2 NKI kernel consumes Q/K/V as [B, H, D, S] and returns
        # [B, H, S, D]. It requires sequence lengths in multiples of 2048.
        query = query.permute(0, 1, 3, 2)
        key = key.permute(0, 1, 3, 2)
        value = value.permute(0, 1, 3, 2)
        attention = nki_flash_attn_func(
            query,
            key,
            value,
            lnc=2,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=True,
            mixed_precision=True,
            transpose_nki_inputs=True,
        )
        expected_attention_shape = (
            batch,
            self.local_query_heads,
            sequence,
            self.head_dim,
        )
        if attention.shape != expected_attention_shape:
            raise ValueError(
                "NKI flash attention returned "
                f"{attention.shape}; expected {expected_attention_shape}"
            )
        if self.sequence_parallel_enabled:
            attention = attention.permute(2, 0, 1, 3).reshape(
                sequence,
                batch,
                self.local_query_heads * self.head_dim,
            )
        else:
            attention = attention.transpose(1, 2).reshape(
                batch,
                sequence,
                self.local_query_heads * self.head_dim,
            )
        return self.o_proj(attention)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        init_method = partial(_normal_init, config.initializer_range)
        sequence_parallel_enabled = config.sequence_parallel_enabled
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.activation(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.torch_dtype,
            config.sequence_parallel_enabled,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.torch_dtype,
            config.sequence_parallel_enabled,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        init_method = partial(_normal_init, config.initializer_range)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            init_method=init_method,
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            config.torch_dtype,
            config.sequence_parallel_enabled,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)


@dataclass
class Qwen3CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        validate_qwen3_4b_base_config(config)
        config.torch_dtype = torch.bfloat16
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=partial(_normal_init, config.initializer_range),
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Qwen3CausalLMOutput:
        del attention_mask  # Packed examples are fixed-length and causal.
        hidden_states = self.model(input_ids, position_ids)
        logits = self.lm_head(hidden_states)
        if self.config.sequence_parallel_enabled:
            logits = logits.transpose(0, 1).contiguous()
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
            flat_labels = shift_labels.view(-1).to(flat_logits.device)
            token_loss = parallel_cross_entropy(flat_logits, flat_labels)
            valid = (flat_labels != -100).to(token_loss.dtype)
            loss = (token_loss * valid).sum() / valid.sum().clamp_min(1.0)
        return Qwen3CausalLMOutput(loss=loss, logits=logits)

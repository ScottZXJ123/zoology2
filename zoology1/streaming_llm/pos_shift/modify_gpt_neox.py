import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.gpt_neox.modeling_gpt_neox import (
  apply_rotary_pos_emb,
  rotate_half,
  GPTNeoXAttention,
)
import types
import transformers
from transformers.models.gpt_neox.modeling_gpt_neox import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


__all__ = ["enable_gpt_neox_pos_shift_attention"]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def gpt_neox_pos_shift_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    has_layer_past = layer_past is not None

    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size]
    #   --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)]
    #   --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.config.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]
    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]

    # Get rotary embeddings from kwargs
    cos, sin = kwargs["position_embeddings"]

    # Apply rotary embeddings to query and key
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_rot = (query_rot * cos) + (rotate_half(query_rot) * sin)
    key_rot = (key_rot * cos) + (rotate_half(key_rot) * sin)

    query = torch.cat((query_rot, query_pass), dim=-1)
    key = torch.cat((key_rot, key_pass), dim=-1)

    # Cache QKV values
    if layer_past is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_ndims, "cache_position": kwargs["cache_position"]}
        key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

    # Compute attention
    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    attn_output, attn_weights = attention_interface(
        self,
        query,
        key,
        value,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
    )

    # Reshape outputs
    input_shape = hidden_states.shape[:-1]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    outputs = (attn_output, layer_past)
    if output_attentions:
        outputs = outputs + (attn_weights,)

    return outputs


def enable_gpt_neox_pos_shift_attention(model):
  for name, module in reversed(model._modules.items()):
    if len(list(module.children())) > 0:
      enable_gpt_neox_pos_shift_attention(
        module,
      )

    if isinstance(module, GPTNeoXAttention):
      module.forward = types.MethodType(
        gpt_neox_pos_shift_attention_forward, module
      ) 
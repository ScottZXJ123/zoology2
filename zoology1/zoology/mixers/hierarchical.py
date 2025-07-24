# file: zoology/mixers/hierarchical.py (最终正确版本)

import inspect
from typing import Type

import torch
import torch.nn as nn
from einops import rearrange

from zoology.config import ModuleConfig
from zoology.utils import import_from_str


class HierarchicalMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        levels: int = 1,
        mixer_config: ModuleConfig = None,
        layer_idx: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.levels = levels

        if mixer_config is None:
            raise ValueError("mixer_config must be provided to HierarchicalMixer")

        base_mixer_cls: Type[nn.Module] = import_from_str(mixer_config.name)
        base_mixer_signature = inspect.signature(base_mixer_cls.__init__)
        accepts_l_max = 'l_max' in base_mixer_signature.parameters

        self.mixers = nn.ModuleList()
        num_mixers = (2**levels) - 1
        for i in range(num_mixers):
            mixer_kwargs = {
                "d_model": d_model,
                "layer_idx": layer_idx,
                **mixer_config.kwargs
            }
            if accepts_l_max:
                mixer_kwargs["l_max"] = l_max
            
            # Use the class and kwargs to instantiate, not the config's instantiate method
            self.mixers.append(base_mixer_cls(**mixer_kwargs))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        x: (B, L, D)
        """
        # 初始输入，用于最终的残差连接
        initial_input = x

        # 逐层处理
        for level in range(self.levels):
            # num_chunks: 1, 2, 4, ...
            num_chunks = 2**level
            # 获取当前层级对应的 mixer 的索引范围
            start_mixer_idx = (2**level) - 1
            
            chunk_len = x.shape[1] // num_chunks
            if x.shape[1] % num_chunks != 0:
                raise ValueError(f"Sequence length {x.shape[1]} must be divisible by num_chunks {num_chunks} at level {level}")

            # 1. 切分成块
            # (B, L, D) -> (B, num_chunks, chunk_len, D)
            chunks = rearrange(x, 'b (c l) d -> b c l d', c=num_chunks, l=chunk_len)
            
            output_chunks = []
            for i in range(num_chunks):
                # 2. 对每个块独立应用mixer
                chunk_input = chunks[:, i, :, :]
                mixer = self.mixers[start_mixer_idx + i]
                
                # mixer的输出
                output_chunk = mixer(chunk_input)
                output_chunks.append(output_chunk)
            
            # 3. 拼接结果
            # List of (B, chunk_len, D) -> (B, num_chunks, chunk_len, D)
            level_output = torch.stack(output_chunks, dim=1) 
            
            # (B, num_chunks, chunk_len, D) -> (B, L, D)
            level_output = rearrange(level_output, 'b c l d -> b (c l) d')
            
            # 4. 残差连接：将当前层的输出加到当前层的输入上
            x = x + level_output

        # The TransformerBlock is responsible for the final residual connection.
        # We subtract the initial input to isolate the "change" made by the mixer,
        # which prevents a double residual issue.
        return x - initial_input
        
    def state_size(self, sequence_length: int):
        # 估算状态大小，简单地将所有混合器的状态大小相加
        total_state_size = 0
        for mixer in self.mixers:
            if hasattr(mixer, 'state_size'):
                # 注意：每个mixer处理的序列长度是不同的
                # 为了简化，我们假设所有mixer都基于l_max，这通常是上界
                total_state_size += mixer.state_size(sequence_length=self.l_max)
        return total_state_size
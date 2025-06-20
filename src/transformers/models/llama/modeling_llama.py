# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask

from ...integrations import use_kernel_forward_from_hub


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


@use_kernel_forward_from_hub("RMSNorm")
# LlamaRMSNorm 是一个实现均方根层归一化 (Root Mean Square Layer Normalization) 的模块。
# 这是一个 Hugging Face 特有的装饰器，用于性能优化。
# 它的作用是：如果 Hugging Face Hub 上存在一个为 RMSNorm 编写的、预编译的、
# 高度优化的“核函数”（例如用 Triton 或 CUDA 编写的），
# 那么在调用 forward 方法时，就会自动使用那个快速的核函数，而不是下面我们定义的这个 Python 版本。
# 如果找不到优化核，它会优雅地回退，执行下面的 Python forward 代码。
# 这是一个可选的性能增强功能。
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        文档字符串：LlamaRMSNorm 与 T5LayerNorm 是等价的。
        """
        # 调用父类 nn.Module 的构造函数，这是 PyTorch 模块的标准写法。
        super().__init__()

        # 定义一个可学习的参数 `weight`。
        # nn.Parameter() 会将一个张量注册为模型的参数，这意味着在训练期间，
        # PyTorch 的自动求导机制会计算它的梯度，并且优化器会更新它的值。
        # torch.ones(hidden_size) 将其初始化为全1的向量，形状与输入的隐藏层维度相同。
        # 初始化为1意味着在训练开始时，这个层不会改变输入向量的尺度。
        self.weight = nn.Parameter(torch.ones(hidden_size))

        # 定义一个非常小的常数 epsilon，用于防止除以零。
        # 在计算方差的平方根时，如果方差恰好为0，加上这个小的 epsilon 可以保证分母不为0，
        # 从而避免计算出 NaN (Not a Number) 或 Inf (Infinity)，保证数值稳定性。
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 1. 准备工作：保存原始数据类型，并将输入转换为 float32 进行计算

        # 保存输入张量 hidden_states 的原始数据类型（如 float16 或 bfloat16）。
        input_dtype = hidden_states.dtype

        # 将 hidden_states 转换为 float32 类型。
        # 这是为了提高计算过程中的数值精度，尤其是在进行 pow(2) 和 mean 操作时，
        # 使用 float32 可以防止数值下溢或上溢，得到更准确的方差。
        hidden_states = hidden_states.to(torch.float32)

        # 2. 计算 RMS (Root Mean Square)

        # hidden_states.pow(2): 对 hidden_states 中的每个元素进行平方（Square）。
        # .mean(-1, keepdim=True): 沿着最后一个维度（即特征维度）计算均值（Mean）。
        # `keepdim=True` 是一个关键参数，它让输出的维度得以保留。例如，如果输入形状是 [batch, seq_len, hidden_size]，
        # 那么计算均值后的形状将是 [batch, seq_len, 1] 而不是 [batch, seq_len]。
        # 这对于下一步的广播（broadcasting）除法至关重要。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # 3. 归一化输入

        # torch.rsqrt(x) 是一个高效的函数，用于计算 1 / sqrt(x) (即平方根的倒数, "rsqrt")。
        # `variance + self.variance_epsilon` 就是加上之前定义的 epsilon 来防止除零。
        # `hidden_states * torch.rsqrt(...)` 在数学上等价于 `hidden_states / torch.sqrt(...)`。
        # 由于 variance 的形状是 [..., 1]，而 hidden_states 是 [..., hidden_size]，
        # PyTorch 会利用广播机制，将 variance 的值应用到 hidden_states 的每一个特征上。
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 4. 应用可学习的增益（gain）并恢复原始数据类型

        # 将归一化后的 hidden_states 乘以可学习的 `weight` 参数。
        # 这允许模型在训练中学习如何动态地调整每个特征维度的尺度（"伸缩"），
        # 恢复或增强那些在归一化过程中可能被抑制的重要信息。
        # 最后，使用 .to(input_dtype) 将计算结果转换回原始的数据类型，以节省内存并与模型的其他部分保持一致。
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        # 这是一个辅助函数，用于自定义模块的打印输出信息。
        # 当你打印这个模块实例时（例如 `print(my_rms_norm)`），
        # 它会返回一个字符串，附加到默认的类名后面。
        # 这里的实现会显示 weight 的形状和 epsilon 的值，
        # 例如，输出会像这样：`LlamaRMSNorm((768,), eps=1e-06)`，非常便于调试。
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

# 该模块接收一个输入张量 x（主要用于获取设备和数据类型）和对应的 position_ids（表示每个token在序列中的位置），
# 然后计算并返回这两个位置上对应的旋转角度的余弦和正弦值。这些 cos 和 sin 值随后会在注意力模块中与Query和Key向量相乘，以注入位置信息。
# 这个类的一个关键特性是它支持上下文窗口扩展技术（如 YaRN、Linear Scaling 等），允许模型处理比原始训练时更长的序列。
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()

        # 1. 确定RoPE的扩展类型（Rope Scaling Type）
        # BC: "rope_type" was originally "type"  (BC代表向后兼容 "Backward Compatibility")
        # 检查配置文件中是否存在 'rope_scaling' 字典。这个字典包含了长上下文扩展的配置。
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            # 从字典中获取扩展类型。优先使用 "rope_type" 键，如果不存在，则使用旧的 "type" 键，以保持向后兼容。
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            # 如果没有配置 'rope_scaling'，则使用默认的RoPE实现。
            self.rope_type = "default"

        # 2. 存储原始和当前的最大序列长度
        # 'max_position_embeddings' 是模型在训练时支持的最大上下文长度。
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # 3. 选择并执行RoPE初始化函数
        # ROPE_INIT_FUNCTIONS 是一个字典，将 "default", "yarn", "linear" 等字符串映射到具体的初始化函数上。
        # 这里根据之前确定的 self.rope_type，选择一个合适的函数。
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # 4. 调用初始化函数，计算逆频率 (inv_freq) 和注意力缩放因子 (attention_scaling)
        # 这个初始化函数会根据选择的扩展方法（如YaRN）来计算RoPE的核心参数。
        # inv_freq: 逆频率，即 RoPE 公式中的 1 / (theta^(2i/d))。不同的扩展方法会修改 theta 或直接调整这些频率。
        # attention_scaling: 注意力缩放因子。对于YaRN等方法，除了调整频率，还需要对最终的cos/sin值进行缩放，以稳定注意力分数。
        #                    对于默认的RoPE，这个值通常是1.0。
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        # 5. 将 inv_freq 注册为模型的缓冲区 (buffer)
        # register_buffer: 告诉PyTorch将 inv_freq 作为模型状态的一部分（例如，它会随模型一起移动到GPU），
        #                 但它不是一个可学习的参数（即不会计算梯度）。
        # persistent=False: 表示在保存模型状态字典 (state_dict) 时，不保存这个缓冲区。
        #                 这可以节省磁盘空间，因为它可以根据config在加载时重新计算出来。
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq  # 保存一份原始的inv_freq，可能用于某些动态调整的场景。

    @torch.no_grad()  # 装饰器：表示该函数内的所有操作都不需要计算梯度。因为RoPE是固定的数学变换，这能提升效率。
    @dynamic_rope_update  # 装饰器：一个为高级用户设计的功能，用于处理动态RoPE类型。
    # 它可能会检查当前请求的上下文长度是否超过了缓存的长度，如果超过，则重新运行初始化函数以更新inv_freq。
    def forward(self, x, position_ids):
        # 1. 扩展 inv_freq 和 position_ids 的维度以进行矩阵乘法

        # self.inv_freq 的原始形状是 [head_dim / 2]。
        # [None, :, None] 操作后变为 [1, head_dim / 2, 1]。
        # .expand(...) 操作将其扩展到批次大小，形状变为 [batch_size, head_dim / 2, 1]。
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        # position_ids 的原始形状是 [batch_size, seq_len]。
        # [:, None, :] 操作后变为 [batch_size, 1, seq_len]。
        position_ids_expanded = position_ids[:, None, :].float()

        # 2. 在高精度(float32)下计算旋转角度

        # 检查设备类型，以确保在所有设备上都能安全地禁用混合精度。
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # torch.autocast: 这是一个上下文管理器，用于自动混合精度训练（如使用float16）。
        # enabled=False: 在这个代码块中，我们强制禁用混合精度，所有计算都使用float32。
        # 为什么？因为旋转角度的计算对数值精度敏感，使用低精度可能导致误差累积和模型性能下降。
        with torch.autocast(device_type=device_type, enabled=False):
            # 3. 核心计算：矩阵乘法得到频率 * 位置 = 角度
            # (inv_freq @ position_ids) 计算出每个维度在每个位置上的旋转角度。
            # 结果 freqs 的形状是 [batch_size, head_dim / 2, seq_len]。
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # 4. 准备完整的角度张量
            # RoPE中，成对的维度使用相同的旋转角度。这里通过拼接(cat)将角度复制一份。
            # 结果 emb 的形状变为 [batch_size, seq_len, head_dim]。
            emb = torch.cat((freqs, freqs), dim=-1)

            # 5. 计算cos和sin，并应用缩放因子
            # `emb.cos()` 逐元素计算余弦值。
            # `* self.attention_scaling` 应用之前从初始化函数中获得的缩放因子（对YaRN等方法至关重要）。
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # 6. 将结果转换回原始数据类型并返回
        # `to(dtype=x.dtype)` 确保输出的cos/sin张量与模型其他部分的类型一致（如bfloat16）。
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 函数文档字符串：旋转输入的一半隐藏维度。

    # ----------------------------------------------------------------------------------
    # 第1行：切分出张量的前半部分
    # ----------------------------------------------------------------------------------
    # `x` 是输入的张量，它可以有任意数量的维度（例如 [batch, heads, seq, dim]）。
    # `...` (省略号) 是 PyTorch/NumPy 的一个特性，意思是“选中所有前面的维度”。
    # `x.shape[-1]` 获取最后一个维度（即特征/头维度）的大小。
    # `// 2` 执行整数除法，找到中点。
    # 因此，`x[..., : x.shape[-1] // 2]` 选中了从开头到最后一个维度中点的所有数据。
    # 如果 x 是 [10, 20, 30, 40]，那么 x1 就是 [10, 20]。
    # 这个变量 `x1` 代表了我们概念中2D向量的所有 "x" 分量。
    x1 = x[..., : x.shape[-1] // 2]

    # ----------------------------------------------------------------------------------
    # 第2行：切分出张量的后半部分
    # ----------------------------------------------------------------------------------
    # 这与上一行类似。
    # 切片 `x.shape[-1] // 2 :` 的意思是“从中点到结尾”。
    # 如果 x 是 [10, 20, 30, 40]，那么 x2 就是 [30, 40]。
    # 这个变量 `x2` 代表了我们概念中2D向量的所有 "y" 分量。
    x2 = x[..., x.shape[-1] // 2:]

    # ----------------------------------------------------------------------------------
    # 第3行：取反、交换和拼接
    # ----------------------------------------------------------------------------------
    # 这是实现 `[-y, x]` 变换的核心操作。
    #
    # 1. `-x2`: 对张量的后半部分 (`x2`) 进行逐元素取反。
    #    如果 x2 是 [30, 40]，`-x2` 就变成了 [-30, -40]。这就是 `"-y"` 部分。
    #
    # 2. `x1`: 这是原始、未改变的前半部分。
    #    如果 x1 是 [10, 20]，它依然是 [10, 20]。这就是 `"x"` 部分。
    #
    # 3. `torch.cat((...), dim=-1)`: 沿最后一个维度 (`dim=-1`) 拼接元组 `(-x2, x1)` 中的张量。
    #    这里的顺序至关重要：`-x2` 在前，`x1` 在后。
    #    所以，它会将 [-30, -40] 和 [10, 20] 组合起来，生成最终的张量 [-30, -40, 10, 20]。
    #
    # 最终结果是一个新的张量，其原始的后半部分被取反并放在了开头，而原始的前半部分被放在了结尾。
    # 这就完美地创建了RoPE所需的“伙伴”向量。
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    渴望执行的注意力前向传播函数。

    Args:
        module (nn.Module): 包含注意力机制的模块，例如 LlamaAttention。
        query (torch.Tensor): 查询张量，形状通常为 (batch_size, num_heads, seq_len, head_dim)。
        key (torch.Tensor): 键张量，形状通常为 (batch_size, num_key_value_heads, seq_len, head_dim)。
        value (torch.Tensor): 值张量，形状通常为 (batch_size, num_key_value_heads, seq_len, head_dim)。
        attention_mask (Optional[torch.Tensor]): 注意力掩码，用于屏蔽不应参与注意力计算的 token。
        scaling (float): 缩放因子，通常为 head_dim 的平方根的倒数，用于缩放注意力权重。
        dropout (float): Dropout 概率，用于防止过拟合。
        **kwargs: 其他关键字参数。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 一个包含注意力输出和注意力权重的元组。
            - 注意力输出 (attn_output): 形状通常为 (batch_size, seq_len, num_heads * head_dim)。
            - 注意力权重 (attn_weights): 形状通常为 (batch_size, num_heads, seq_len, seq_len)。
    """
    # 1. 扩展 Key 和 Value 以匹配 Query 的 Head 数量
    key_states = repeat_kv(key, module.num_key_value_groups)
    #   - `repeat_kv` 函数：将 key 和 value 张量的 head 维度进行重复，以适应 Multi-Query Attention 或者 Grouped-Query Attention。
    #   - `module.num_key_value_groups`：表示 key 和 value 共享的 head 组的数量。
    #   -  目的: 为了让 key 和 value 的 head 数量与 query 的 head 数量相匹配，以便进行后续的注意力计算。
    value_states = repeat_kv(value, module.num_key_value_groups)

    # 2. 计算注意力权重
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    #   - `torch.matmul(query, key_states.transpose(2, 3))`：计算 query 和 key 的点积，得到注意力 logits。
    #   - `key_states.transpose(2, 3)`：将 key 张量的最后两个维度转置，以便进行矩阵乘法。
    #   - `scaling`：缩放因子，用于缩小点积的值，防止 softmax 后的梯度消失。

    # 3. 应用注意力掩码 (如果存在) key_states shape 为 (batch_size, num_key_value_heads, seq_len, head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        #   - `attention_mask`：注意力掩码，用于屏蔽不应该参与注意力计算的 token，例如 padding token。
        #   - `causal_mask`：因果掩码，用于确保模型只能关注到当前位置之前的 token，用于自回归生成任务。
        #   - `attn_weights + causal_mask`：将注意力掩码添加到注意力权重上，实现屏蔽的效果。
        attn_weights = attn_weights + causal_mask

    # 4. 对注意力权重进行 Softmax 归一化
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    #   - `nn.functional.softmax(attn_weights, dim=-1)`：对注意力权重进行 softmax 归一化，使其成为概率分布。
    #   - `dim=-1`：沿着最后一个维度进行 softmax 归一化，即对每个 query 位置的 key 位置进行归一化。
    #   - `dtype=torch.float32`：将注意力权重转换为 float32 类型，保证数值稳定性。
    #   - `.to(query.dtype)`: 将数据类型转换为与query张量相同的类型，确保计算的一致性。

    # 5. 应用 Dropout (如果需要)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    #   - `nn.functional.dropout(attn_weights, p=dropout, training=module.training)`：对注意力权重应用 dropout，防止过拟合。
    #   - `p=dropout`：dropout 概率。
    #   - `training=module.training`：只有在训练模式下才应用 dropout。

    # 6. 计算注意力输出
    attn_output = torch.matmul(attn_weights, value_states)
    #   - `torch.matmul(attn_weights, value_states)`：将注意力权重与 value 张量相乘，得到注意力输出。
    #   -  每个query会根据权重，关注不同的value信息。

    # 7. 调整输出形状
    attn_output = attn_output.transpose(1, 2).contiguous()
    #   - `attn_output.transpose(1, 2)`：将注意力输出的维度转置，使其形状变为 (batch_size, seq_len, num_heads, head_dim)。
    #   - `contiguous()`:  确保张量在内存中是连续存储的，避免一些潜在的问题。
    #     - 对于很多tensor操作，比如transpose()，view()等等，可能会导致tensor在内存中变成不连续存储。
    #     - 而如果后续你要调用一些需要连续存储的tensor的操作，就需要先调用contiguous()。

    # 8. 返回注意力输出和注意力权重
    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        """
        构造函数，初始化 LlamaAttention 层。

        Args:
            config (LlamaConfig): Llama 模型的配置对象。
            layer_idx (int): 当前 Attention 层在整个模型中的索引，用于 Cache 的管理。
        """
        super().__init__()  # 调用父类的构造函数。
        self.config = config  # 存储模型配置。
        self.layer_idx = layer_idx  # 存储当前层的索引。
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        #   - `head_dim`：每个注意力头的维度。如果配置中指定了 `head_dim`，则使用它；否则，通过 `hidden_size // config.num_attention_heads` 计算得到。
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        #   - `num_key_value_groups`：在 Grouped-Query Attention (GQA) 或 Multi-Query Attention (MQA) 中，key 和 value 头共享的组数。
        #   - 目的: 降低计算量和模型大小。
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于缩放注意力权重。等于 head_dim 的平方根的倒数。
        self.attention_dropout = config.attention_dropout  # 注意力 dropout 概率，用于防止过拟合。
        self.is_causal = True  # 是否是因果注意力，用于自回归生成任务。

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        #   - `q_proj`：线性层，用于将 hidden_states 转换为 query。
        #   - 输入维度：`config.hidden_size`。
        #   - 输出维度：`config.num_attention_heads * self.head_dim`，表示所有注意力头的 query 维度。
        #   - `bias=config.attention_bias`：是否使用偏置项。
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        #   - `k_proj`：线性层，用于将 hidden_states 转换为 key。
        #   - 输入维度：`config.hidden_size`。
        #   - 输出维度：`config.num_key_value_heads * self.head_dim`，表示所有 key-value 头的 key 维度。
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        #   - `v_proj`：线性层，用于将 hidden_states 转换为 value。
        #   - 输入维度：`config.hidden_size`。
        #   - 输出维度：`config.num_key_value_heads * self.head_dim`，表示所有 key-value 头的 value 维度。
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        #   - `o_proj`：线性层，用于将注意力输出转换为隐藏状态。
        #   - 输入维度：`config.num_attention_heads * self.head_dim`，表示所有注意力头的维度之和。
        #   - 输出维度：`config.hidden_size`。

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播函数，计算 Attention 层的输出。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)。
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): 位置嵌入，包括 cos 和 sin 值。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，用于屏蔽不应参与注意力计算的 token。
            past_key_value (Optional[Cache]): 过去的 key 和 value 状态，用于加速推理。
            cache_position (Optional[torch.LongTensor]): 缓存位置信息，用于更新缓存。
            **kwargs: 其他关键字参数，用于传递给 FlashAttention 等注意力机制。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
                包含注意力输出、注意力权重（如果需要）和过去的 key/value 状态（如果使用缓存）的元组。
        """
        # 1. 获取输入形状
        input_shape = hidden_states.shape[:-1]  # 输入形状， (batch_size, sequence_length)。
        hidden_shape = (*input_shape, -1, self.head_dim)  # 隐藏层形状，(batch_size, sequence_length, num_heads, head_dim)。

        # 2. 线性变换得到 Query, Key, Value
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        #   - `self.q_proj(hidden_states)`: 将输入 `hidden_states` 通过线性层 `q_proj` 投影到 Query 空间。
        #   - `.view(hidden_shape)`:  将输出张量reshape成想要的形状 `hidden_shape`, 方便后续计算
        #   - `.transpose(1, 2)`: 交换维度 1 和 2，使得 head 维度放到 seq_len 前面。
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        #   - 类似 `query_states`，不过这里是转换到 Key 空间。
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        #   - 类似 `query_states`，不过这里是转换到 Value 空间。

        # 3. 应用 RoPE (旋转位置编码)
        cos, sin = position_embeddings  # 从 position_embeddings 中获取 cos 和 sin 值。
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        #   - `apply_rotary_pos_emb`：应用 RoPE 到 query 和 key 张量。

        # 4. 更新 Cache (如果使用)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            #   - `cache_kwargs`：存储缓存相关的参数，包括 RoPE 的 sin 和 cos 值以及缓存位置。
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            #   - `past_key_value.update(...)`：更新 Cache，并将更新后的 key 和 value 状态返回。
            #   - `self.layer_idx`：当前 Attention 层的索引，用于 Cache 的管理。

        # 5. 选择 Attention 实现方式
        attention_interface: Callable = eager_attention_forward
        #   - `attention_interface`：一个 Callable 对象，指向具体的 Attention 实现函数。默认使用 `eager_attention_forward`。
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        #    - 根据模型配置 `self.config._attn_implementation` 选择不同的 Attention 实现方式。
        #    - `ALL_ATTENTION_FUNCTIONS`：一个字典，存储了各种 Attention 实现方式。

        # 6. 计算 Attention 输出和权重
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        #   - 调用具体的 Attention 实现函数 `attention_interface` 计算 Attention 输出和权重。

        # 7. 调整 Attention 输出的形状
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        #   - 将 Attention 输出 reshape 成 `(batch_size, sequence_length, hidden_size)` 的形状。
        #   - `contiguous()`：确保张量在内存中是连续存储的。

        # 8. 通过线性层进行输出投影
        attn_output = self.o_proj(attn_output)
        #   - `self.o_proj(attn_output)`：将 Attention 输出通过线性层 `o_proj` 投影到最终的输出空间。

        # 9. 返回结果
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaDecoderLayer(GradientCheckpointingLayer):
    """
    Llama 解码器层，是 Llama 模型的核心 building block。
    它包含自注意力机制、多层感知机 (MLP) 和 RMSNorm 归一化层，并通过残差连接来提高训练效率。
    该层继承自 GradientCheckpointingLayer，支持梯度检查点技术，以减少内存占用。
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        """
        构造函数，初始化 LlamaDecoderLayer。

        Args:
            config (LlamaConfig): 模型的配置对象。
            layer_idx (int): 当前层在整个模型中的索引，用于 Cache 的管理以及 RoPE 的计算。
        """
        super().__init__()  # 调用父类的构造函数，确保 GradientCheckpointingLayer 的初始化逻辑被执行。
        # 1. 隐藏层大小
        self.hidden_size = config.hidden_size
        #   - `hidden_size`：指定隐藏层的维度。

        # 2. 自注意力层
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        #   -  `self_attn`：LlamaAttention 类的实例，用于计算输入序列的自注意力。

        # 3. MLP 层
        self.mlp = LlamaMLP(config)
        #   - `mlp`：LlamaMLP 类的实例，用于对自注意力层的输出进行非线性变换。

        # 4. 输入归一化层
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #   - `input_layernorm`：LlamaRMSNorm 类的实例，用于对输入进行归一化。
        #   - RMSNorm 是一种 Layer Normalization 的变体，能够稳定训练过程。

        # 5. 注意力输出归一化层
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #   - `post_attention_layernorm`：LlamaRMSNorm 类的实例，用于对自注意力层的输出进行归一化。

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        前向传播函数，计算 LlamaDecoderLayer 的输出。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (batch_size, sequence_length, hidden_size)。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，用于屏蔽不应参与注意力计算的 token。
            position_ids (Optional[torch.LongTensor]): 位置 IDs，形状为 (batch_size, sequence_length)。
            past_key_value (Optional[Cache]): 过去的 key 和 value，用于加速推理。
            output_attentions (Optional[bool]): 是否输出注意力权重。
            use_cache (Optional[bool]): 是否使用缓存。
            cache_position (Optional[torch.LongTensor]): 缓存位置，用于更新缓存。
            position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]]): 位置嵌入 (RoPE)，cos 和 sin 值。
            **kwargs: 其他关键字参数，用于传递给 FlashAttention 等注意力机制。

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
                包含解码器层的输出和注意力权重（如果需要）的元组。
        """
        # 1. 残差连接
        residual = hidden_states
        #   -  将输入 `hidden_states` 存储在 `residual` 变量中，用于后面的残差连接。

        # 2. 输入归一化
        hidden_states = self.input_layernorm(hidden_states)
        #   - 使用 `self.input_layernorm` 对输入进行归一化。

        # 3. 自注意力
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        #   -  将归一化后的输入 `hidden_states` 传递给自注意力层 `self.self_attn`，并获取注意力输出和注意力权重。

        # 4. 残差连接
        hidden_states = residual + hidden_states
        #   - 将自注意力层的输出与原始输入 `residual` 相加，实现残差连接。
        #   - 残差连接有助于缓解梯度消失问题，并加速模型训练。

        # 5. 全连接层
        residual = hidden_states
        #   -  再次将输入存储在 `residual` 变量中，用于后面的残差连接。

        # 6. 注意力输出归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        #   - 使用 `self.post_attention_layernorm` 对自注意力层的输出进行归一化。

        # 7. MLP
        hidden_states = self.mlp(hidden_states)
        #   -  将归一化后的输入传递给 MLP 层 `self.mlp`，进行非线性变换。

        # 8. 残差连接
        hidden_states = residual + hidden_states
        #   - 将 MLP 层的输出与残差 `residual` 相加，实现残差连接。

        # 9. 构建输出
        outputs = (hidden_states,)
        #   - 创建一个包含隐藏状态 `hidden_states` 的元组作为输出。
        #   -   `hidden_states`：是解码器层的最终输出，用于传递给下一层或进行后续处理。

        # 10. 如果需要，添加注意力权重到输出
        if output_attentions:
            outputs += (self_attn_weights,)
            #    - 如果 `output_attentions` 为 True，则将自注意力权重 `self_attn_weights` 添加到输出元组中。

        # 11. 返回输出
        return outputs
        #   - 返回包含解码器层输出的元组。


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    """
    Llama 预训练模型基类。
    所有 Llama 模型都应继承自此类。它提供了一些通用的方法和属性，用于处理模型配置、权重初始化、设备管理和与 Hugging Face Transformers 库的集成。
    """
    # 1. 配置类
    config_class = LlamaConfig
    #   - `config_class`：指定模型使用的配置类为 `LlamaConfig`，该类定义了模型的各种参数。

    # 2. 模型前缀
    base_model_prefix = "model"
    #   - `base_model_prefix`：指定模型中核心模块的前缀，用于区分不同的模型组件。例如，`model.embed_tokens`、`model.layers` 等。

    # 3. 支持梯度检查点
    supports_gradient_checkpointing = True
    #   - `supports_gradient_checkpointing`：指示该模型支持梯度检查点（Gradient Checkpointing）技术。
    #   - 梯度检查点是一种在不存储所有中间激活的情况下训练大型模型的技术，通过在反向传播过程中重新计算激活来减少内存占用。

    # 4. 不分割的模块
    _no_split_modules = ["LlamaDecoderLayer"]
    #   - `_no_split_modules`：指定在进行模型并行化时，不进行分割的模块。
    #   - `LlamaDecoderLayer` 作为一个整体，不应该被分割到不同的设备上，因为它内部包含了多个相互依赖的子模块。

    # 5. 跳过设备放置的键
    _skip_keys_device_placement = ["past_key_values"]
    #   - `_skip_keys_device_placement`：在加载模型到设备时，指定跳过设备放置的键。
    #   - `past_key_values`：用于缓存历史状态的键，通常在生成任务中使用。延迟加载可以优化初始化时间。

    # 6. 对各种加速技术的支持
    _supports_flash_attn_2 = True
    #   - `_supports_flash_attn_2`：指示该模型支持 Flash Attention 2 技术，用于加速注意力计算。
    _supports_sdpa = True
    #   - `_supports_sdpa`：指示该模型支持 SDPA（Scaled Dot Product Attention），PyTorch 2.0 提供的一种加速注意力计算的方法。
    _supports_flex_attn = True
    #   - `_supports_flex_attn`：指示该模型支持 FlexAttention，一种更灵活的注意力机制，可以处理不同类型的 attention mask。

    # 7. 对各种缓存机制的支持
    _supports_cache_class = True
    #   - `_supports_cache_class`：指示该模型支持自定义的缓存类，用于管理和存储历史状态。
    _supports_quantized_cache = True
    #   - `_supports_quantized_cache`：指示该模型支持量化缓存，用于减少缓存的内存占用。
    _supports_static_cache = True
    #   - `_supports_static_cache`：指示该模型支持静态缓存，一种特殊的缓存机制，可以进一步加速推理。

    # 8. 支持可定制的注意力后端
    _supports_attention_backend = True
    #   - `_supports_attention_backend`：指示该模型支持选择不同的注意力后端实现。

    def _init_weights(self, module):
        """
        权重初始化方法。
        使用正态分布初始化模型的权重，并将偏置初始化为零。

        Args:
            module (nn.Module): 要初始化的模型模块。
        """
        # 1. 获取初始化范围
        std = self.config.initializer_range
        #   - 从模型配置中获取初始化范围，用于控制权重初始化的尺度。

        # 2. 初始化线性层的权重和偏置
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            #   - 使用均值为 0，标准差为 `std` 的正态分布初始化线性层的权重。
            if module.bias is not None:
                module.bias.data.zero_()
            #   - 如果线性层有偏置，则将其初始化为零。

        # 3. 初始化嵌入层的权重，并处理 padding 索引
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            #   - 使用均值为 0，标准差为 `std` 的正态分布初始化嵌入层的权重。
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            #   - 如果嵌入层有 `padding_idx`，则将对应位置的权重设置为零，以确保 padding token 不影响模型的计算。

        # 4. 初始化 LlamaRMSNorm 层的权重
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
            #   - 将 LlamaRMSNorm 层的权重初始化为 1，保证初始状态下不影响输入。

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer 解码器，由 *config.num_hidden_layers* 层组成。每层是一个 [`LlamaDecoderLayer`]。

    Args:
        config: LlamaConfig，模型配置
    """

    def __init__(self, config: LlamaConfig):
        """
        构造函数，初始化 LlamaModel。

        Args:
            config (LlamaConfig): 模型的配置对象。
        """
        super().__init__(config)  # 调用父类的构造函数，确保 PreTrainedModel 的初始化逻辑被执行。
        self.padding_idx = config.pad_token_id  #  `padding_idx`：padding token 的索引，用于在嵌入层中忽略 padding token。
        self.vocab_size = config.vocab_size  # `vocab_size`：词汇表的大小，即 token 的总数。

        # 1. 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        #     shape: (vocab_size, hidden_size)
        #   - `nn.Embedding`：将输入 token IDs 转换为连续的向量表示。
        #   - `config.vocab_size`：词汇表大小。
        #   - `config.hidden_size`：嵌入向量的维度。
        #   - `self.padding_idx`：指定 padding token 的索引，在计算损失时会被忽略。

        # 2. 解码器层列表
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        #   - `nn.ModuleList`：用于存储多个 `nn.Module` 对象，并将其作为一个整体进行管理。
        #   - `LlamaDecoderLayer(config, layer_idx)`：创建单个解码器层，`config` 是模型配置，`layer_idx` 是当前层的索引。
        #   - `config.num_hidden_layers`：解码器层的总数，决定了模型深度。

        # 3. 归一化层
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #   - `LlamaRMSNorm`：用于对最后一个隐藏层的输出进行归一化，以稳定训练并提高性能。
        #   - `config.hidden_size`：隐藏层大小。
        #   - `config.rms_norm_eps`：RMSNorm 中的 epsilon 值，用于防止除以零。

        # 4. 旋转位置嵌入
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        #   - `LlamaRotaryEmbedding`：用于编码位置信息，提供给自注意力机制。

        # 5. 梯度检查点
        self.gradient_checkpointing = False
        #   -  `gradient_checkpointing`：一个布尔值，指示是否启用梯度检查点技术。
        #      - 梯度检查点是一种在不存储所有中间激活的情况下训练大型模型的技术，通过在反向传播过程中重新计算激活来减少内存占用。

        # 6. 初始化权重和应用最终处理
        self.post_init()
        #   - 调用 `post_init` 方法，初始化模型的权重，并将模型设置为评估模式。

    def get_input_embeddings(self):
        """
        获取输入嵌入。

        Returns:
            nn.Embedding: 输入嵌入层。
        """
        return self.embed_tokens  # 返回模型的嵌入层。

    def set_input_embeddings(self, value):
        """
        设置输入嵌入。

        Args:
            value (nn.Embedding): 新的嵌入层。
        """
        self.embed_tokens = value  # 将模型的嵌入层替换为新的嵌入层。

    @can_return_tuple  # 指示该函数可以返回元组。
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)  # 添加输入文档字符串。
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        """
        前向传播函数，计算 LlamaModel 的输出。

        Args:
            input_ids (Optional[torch.LongTensor]): 输入 token IDs，形状为 (batch_size, sequence_length)。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，形状为 (batch_size, sequence_length)。
            position_ids (Optional[torch.LongTensor]): 位置 IDs，形状为 (batch_size, sequence_length)。
            past_key_values (Optional[Cache]): 过去的 key 和 value，用于加速推理。
            inputs_embeds (Optional[torch.FloatTensor]): 输入嵌入，如果提供了 input_ids，则不需要提供 inputs_embeds。
            use_cache (Optional[bool]): 是否使用缓存。
            output_attentions (Optional[bool]): 是否输出注意力权重。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态。
            cache_position (Optional[torch.LongTensor]): 缓存位置，用于更新缓存。
            **flash_attn_kwargs:  Flash Attention 相关的关键字参数。

        Returns:
            BaseModelOutputWithPast: 包含 last_hidden_state, past_key_values, hidden_states, attentions 的对象。
        """
        # 1. 根据配置设置是否输出注意力和隐藏状态
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #   - 如果 `output_attentions` 参数为 None，则使用模型配置中的 `output_attentions` 值。
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        #   - 如果 `output_hidden_states` 参数为 None，则使用模型配置中的 `output_hidden_states` 值。

        # 2. 根据配置设置是否使用缓存
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        #   - 如果 `use_cache` 参数为 None，则使用模型配置中的 `use_cache` 值。

        # 3. 检查是否只提供了一个输入：input_ids 或 inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        #   -  `^` 是异或运算符，用于检查是否只提供了 `input_ids` 或 `inputs_embeds` 中的一个。

        # 4. 梯度检查点和缓存不兼容，如果同时使用则给出警告
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        #   - 梯度检查点与缓存机制不兼容，如果同时启用，则关闭缓存以避免错误。

        # 5. TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        # 检查 past_key_values 的类型是否正确
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
        #   - 确保 `past_key_values` 参数的类型为 `Cache` 对象或 None。

        # 6. 如果没有提供 inputs_embeds，则使用嵌入层将 input_ids 转换为 inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        #   - 如果没有直接提供嵌入向量 `inputs_embeds`，则使用嵌入层 `self.embed_tokens` 将 `input_ids` 转换为嵌入向量。

        # 7. 如果使用缓存且没有提供 past_key_values，则创建一个 DynamicCache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        #   - 如果启用了缓存 (`use_cache=True`) 且没有提供 `past_key_values`，则创建一个 `DynamicCache` 对象，用于存储和管理历史状态。

        # 8. 如果没有提供 cache_position，则创建一个 cache_position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        #   -  (sequence_length,)计算缓存的起始位置，并创建一个 `cache_position` 张量，用于指示当前输入在缓存中的位置。

        # 9. 如果没有提供 position_ids，则使用 cache_position 创建一个 position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        #  position_ids shape   (1, sequence_length) -   如果没有直接提供 `position_ids`，则使用 `cache_position` 张量创建一个 `position_ids` 张量，用于指定输入 token 的位置信息。

        # 10. 更新因果掩码
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        #   - 根据输入、缓存和模型配置更新因果掩码，确保模型只能关注到当前位置之前的 token。

        # 11. 隐藏状态，初始化为输入嵌入
        hidden_states = inputs_embeds
        #   - 将 `hidden_states` 初始化为输入嵌入 `inputs_embeds`，作为 Transformer 解码器的输入。

        # 12. create position embeddings to be shared across the decoder layers
        # 创建位置嵌入，用于编码位置信息
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        #   - 使用 `self.rotary_emb` 生成位置嵌入，编码输入的位置信息。

        # 13. decoder layers
        # 所有隐藏状态，用于输出隐藏状态
        all_hidden_states = () if output_hidden_states else None
        #   - 初始化一个空的元组，用于存储所有解码器层的隐藏状态。
        # 所有自注意力权重，用于输出注意力权重
        all_self_attns = () if output_attentions else None
        #   - 初始化一个空的元组，用于存储所有自注意力层的注意力权重。

        # 14. 循环遍历所有解码器层
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # 如果需要输出隐藏状态，则添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 计算解码器层的输出
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            #  - 将 `hidden_states` 传递给当前解码器层，并获取输出。

            # 更新隐藏状态
            hidden_states = layer_outputs[0]
            #   - 使用当前解码器层的输出更新 `hidden_states`，作为下一个解码器层的输入。

            # 如果需要输出注意力权重，则添加到 all_self_attns 中
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 15. 对输出进行 RMSNorm 归一化
        hidden_states = self.norm(hidden_states)
        #   - 使用 `self.norm` 对最后一个解码器层的输出进行归一化。

        # 16. add hidden states from the last decoder layer
        # 如果需要输出隐藏状态，则添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 17. 返回 BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
            self,
            attention_mask: Union[torch.Tensor, "BlockMask"],
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool = False,
    ):
        """
        更新因果注意力掩码。
        该函数根据模型配置、输入张量、缓存状态和注意力掩码，生成或更新用于控制注意力计算的因果掩码。

        Args:
            attention_mask (Union[torch.Tensor, "BlockMask"]):
                - 2D 注意力掩码，形状为 `(batch_size, key_value_length)`
                - 或 4D 注意力掩码，形状为 `(batch_size, 1, query_length, key_value_length)`
                - 或 `BlockMask` 对象 (用于 FlexAttention)。
            input_tensor (torch.Tensor): 输入张量，形状为 `(batch_size, sequence_length, hidden_size)`。
            cache_position (torch.Tensor): 指示输入序列 token 在序列中的位置的索引，形状为 `(sequence_length)`。
            past_key_values (Cache): 用于存储过去 key 和 value 的缓存对象。
            output_attentions (bool):  指示是否输出注意力权重。

        Returns:
            torch.Tensor: 更新后的因果注意力掩码，形状为 `(batch_size, 1, query_length, key_value_length)` 或 `BlockMask` (用于 FlexAttention)。如果不需要掩码，则返回 None。
        """
        # 1. 根据注意力实现方式选择处理逻辑
        # 如果使用 flash_attention_2，且存在值为 0 的 attention_mask，则直接返回该 attention_mask；否则，返回 None
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None  # flash_attention_2 会直接使用 attention_mask 或 is_causal 参数

        # 如果使用 flex_attention，将 attention_mask 转换为 BlockMask 对象
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask  # flex_attention 使用 BlockMask 对象

        # 2. 处理 SDPA 注意力机制
        # SDPA 在某些情况下可以使用 is_causal 参数代替 attn_mask，以提升性能
        # 获取已处理的 token 数量
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        # 检查是否使用了静态缓存
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # 当 output attentions 为 True 时，sdpa 实现的前向方法调用 eager 实现的前向方法
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None  # SDPA 且满足特定条件，则不使用 attn_mask

        # 3. 准备因果掩码
        dtype = input_tensor.dtype  # 获取输入张量的数据类型
        sequence_length = input_tensor.shape[1]  # 获取输入张量的序列长度
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        #   -  计算目标序列长度 `target_length`，这取决于是否使用了静态缓存。
        #       - 如果使用了静态缓存，则目标长度是缓存的最大长度。
        #       - 否则，目标长度是已处理的 token 数量加上当前序列长度。

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        # 如果提供的 `attention` 掩码是 2D，则在这里生成一个因果掩码（4D）
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        #   - 调用 `_prepare_4d_causal_attention_mask_with_cache_position` 方法，根据输入信息生成 4D 因果掩码。

        # 4. 特殊处理 SDPA 的因果掩码
        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type in ["cuda", "xpu", "npu"]
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        # - 为了兼容 F.scaled_dot_product_attention 的高效内存路径，需要对 causal_mask 中完全被掩码的行进行特殊处理，使它们可以 attend 到所有 token。

        # 5. 返回因果掩码
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """
    Llama 模型，用于因果语言建模 (Causal Language Modeling)。
    用于生成文本，其任务是预测序列中的下一个 token。
    该模型继承自 LlamaPreTrainedModel 和 GenerationMixin，从而获得了 Llama 模型的通用功能和文本生成能力。
    """
    # 1. Tied 权重键
    _tied_weights_keys = ["lm_head.weight"]
    #   - `_tied_weights_keys`：指定需要进行权重绑定的键。
    #   - 在因果语言模型中，通常将嵌入层（`embed_tokens`）和语言模型头（`lm_head`）的权重绑定在一起，以减少参数数量并提高性能。

    # 2. 张量并行 (Tensor Parallelism) 计划
    _tp_plan = {"lm_head": "colwise_rep"}
    #   - `_tp_plan`：指定在进行张量并行训练时，`lm_head` 模块的并行策略。
    #   - `"colwise_rep"`：表示对 `lm_head` 的权重进行按列复制（colwise replication）。

    # 3. Pipeline 并行 (Pipeline Parallelism) 计划
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    #   - `_pp_plan`：指定在进行流水线并行训练时，`lm_head` 模块的输入和输出。
    #   - `(["hidden_states"], ["logits"])`：表示 `lm_head` 的输入是 `hidden_states`，输出是 `logits`。

    def __init__(self, config):
        """
        构造函数，初始化 LlamaForCausalLM。

        Args:
            config (LlamaConfig): 模型的配置对象。
        """
        super().__init__(config)  # 调用父类的构造函数，确保 LlamaPreTrainedModel 的初始化逻辑被执行。
        # 1. Llama 模型
        self.model = LlamaModel(config)
        #   - 创建一个 LlamaModel 实例，作为因果语言模型的核心模块。

        # 2. 词汇表大小
        self.vocab_size = config.vocab_size
        #   - 从模型配置中获取词汇表大小。

        # 3. 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #   - `lm_head`：一个线性层，用于将最后一个隐藏层的输出映射到词汇表大小，以预测下一个 token。
        #   - 输入维度：`config.hidden_size`，即隐藏层的维度。
        #   - 输出维度：`config.vocab_size`，即词汇表大小。
        #   - `bias=False`：指示不使用偏置项。

        # 4. 初始化权重和应用最终处理
        self.post_init()
        #   - 调用 `post_init` 方法，初始化模型的权重，并将模型设置为评估模式。

    def get_input_embeddings(self):
        """
        获取输入嵌入。

        Returns:
            nn.Embedding: 输入嵌入层。
        """
        return self.model.embed_tokens  # 返回底层 LlamaModel 的嵌入层。

    def set_input_embeddings(self, value):
        """
        设置输入嵌入。

        Args:
            value (nn.Embedding): 新的嵌入层。
        """
        self.model.embed_tokens = value  # 将底层 LlamaModel 的嵌入层替换为新的嵌入层。

    def get_output_embeddings(self):
        """
        获取输出嵌入（即语言模型头）。

        Returns:
            nn.Linear: 语言模型头。
        """
        return self.lm_head  # 返回语言模型头。

    def set_output_embeddings(self, new_embeddings):
        """
        设置输出嵌入（即语言模型头）。

        Args:
            new_embeddings (nn.Linear): 新的语言模型头。
        """
        self.lm_head = new_embeddings  # 将语言模型头替换为新的语言模型头。

    def set_decoder(self, decoder):
        """
        设置解码器。

        Args:
            decoder: 新的解码器。
        """
        self.model = decoder  # 将底层的 LlamaModel 替换为新的解码器。

    def get_decoder(self):
        """
        获取解码器。

        Returns:
            LlamaModel: 解码器。
        """
        return self.model  # 返回底层的 LlamaModel。

    @can_return_tuple  # 指示该函数可以返回元组。
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)  # 添加输入文档字符串。
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)  # 替换返回文档字符串。
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        前向传播函数，计算 LlamaForCausalLM 的输出。

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                输入序列 token 的索引。Padding 将默认被忽略。

                Indices 可以使用 [`AutoTokenizer`] 获得。

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于避免在 padding token 索引上执行注意力计算的掩码。 掩码值 selected in `[0, 1]`:

                - 1：表示 token **没有被掩码**，
                - 0：表示 token **被掩码**。

                [What are attention masks?](../glossary#attention-mask)

                Indices 可以使用 [`AutoTokenizer`] 获得。

                如果使用了 `past_key_values`，则可以选择只输入最后的 `input_ids`。

            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                位置嵌入中每个输入序列 token 的位置索引。 选择范围在 `[0, config.n_positions - 1]` 内。

                [What are position IDs?](../glossary#position-ids)
            past_key_values (`Cache`, *optional*):
                预先计算好的隐藏状态（自注意力模块中的 key 和 values），可以用于加速序列解码。 这通常由模型在解码的先前阶段返回的 `past_key_values` 组成，当 `use_cache=True` 或 `config.use_cache=True` 时。

                It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

                如果使用了 `past_key_values`，用户可以选择只输入最后的 `input_ids`。
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                可以选择不传递 `input_ids`，而是直接传递嵌入表示。

            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算 masked language modeling 损失的标签。索引应该在 `[0, ..., config.vocab_size]` 或 -100 内。

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                如果是一个 `int`，则计算最后 `logits_to_keep` 个 token 的 logits。如果为 0，则计算所有 `input_ids` 的 logits（特殊情况）。 只需要最后一个 token 的 logits 用于生成，并且只为该 token 计算它们可以节省内存，这对于长序列或大词汇量来说非常重要。 如果是一个 `torch.Tensor`，则必须是 1D 张量，对应于序列长度维度中要保留的索引。 这在使用 packed tensor 格式（批次和序列长度的单个维度）时很有用。

        Returns:
            CausalLMOutputWithPast: 包含 loss, logits, past_key_values, hidden_states, attentions 的对象。

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # 1. 根据配置设置是否输出注意力和隐藏状态
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #   - 如果 `output_attentions` 参数为 None，则使用模型配置中的 `output_attentions` 值。
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        #   - 如果 `output_hidden_states` 参数为 None，则使用模型配置中的 `output_hidden_states` 值。

        # 2. decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # 计算模型输出
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        #   - 调用底层的 LlamaModel 计算输出，并将结果存储在 `outputs` 变量中。
        #   -  `outputs` 是一个 `BaseModelOutputWithPast` 对象，包含模型的各种输出，例如：
        #       - `last_hidden_state`：最后一个隐藏层的输出。
        #       - `past_key_values`：缓存的 key 和 value。
        #       - `hidden_states`：所有隐藏层的输出（如果需要）。
        #       - `attentions`：注意力权重（如果需要）。

        # 3. 获取最后一个隐藏层的输出
        hidden_states = outputs.last_hidden_state
        #   - 从 `outputs` 中提取最后一个隐藏层的输出。

        # 4. Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # 只计算需要的 logits，并且在不计算损失时不要将它们转换为 float 类型
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        #  -  根据 `logits_to_keep` 的值创建一个切片对象，用于选择需要计算 logits 的 token。
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        #   - 将 `hidden_states` 通过线性层 `lm_head` 映射到 logits 空间，得到每个 token 的预测概率。

        # 5. 初始化损失
        loss = None
        #   - 初始化损失变量为 None。

        # 6. 如果提供了 labels，则计算损失
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        #   - 如果提供了标签 `labels`，则使用 `self.loss_function` 计算因果语言建模损失。
        #   - `self.loss_function`：一个损失函数，用于衡量模型的预测结果与真实标签之间的差异。

        # 7. 返回 CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        #   - 创建一个 `CausalLMOutputWithPast` 对象，并将损失、logits、缓存的 key 和 value、隐藏状态和注意力权重存储在该对象中。
        #   -  `CausalLMOutputWithPast` 是一个数据结构，用于组织和返回因果语言模型的输出。
@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    """
    Llama 模型，用于序列分类任务。
    该模型在 LLaMa 模型的基础上添加了一个序列分类头（线性层），用于将输入序列分类到不同的类别。
    它继承自 LlamaPreTrainedModel，从而获得了 Llama 模型的通用功能，例如加载预训练权重、保存模型等。
    该类适用于文本情感分析、主题分类等任务。
    """
    def __init__(self, config):
        """
        构造函数，初始化 LlamaForSequenceClassification。

        Args:
            config (LlamaConfig): 模型的配置对象，包含了模型结构、训练参数等信息。
        """
        super().__init__(config)  # 调用父类的构造函数，确保 LlamaPreTrainedModel 的初始化逻辑被执行，例如加载配置信息。

        # 1. 标签数量
        self.num_labels = config.num_labels
        #   - `num_labels`：指定分类任务的标签数量，例如情感分析任务的标签数量可以是 2（正面、负面）或 3（正面、负面、中性）。
        #   - 该参数从 LlamaConfig 中获取，需要在配置对象中预先定义。

        # 2. Llama 模型
        self.model = LlamaModel(config)
        #   - `model`：创建一个 LlamaModel 实例，作为序列分类任务的基础模型。
        #   - LlamaModel 负责处理输入序列，并生成隐藏状态表示。

        # 3. 线性分类层 (也称为 "scoring" 层或 "classification head")
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        #   - `score`：一个线性层，用于将最后一个隐藏层的输出（表示整个序列的特征）映射到标签数量，以进行分类。
        #   - 输入维度：`config.hidden_size`，即 LlamaModel 最后一个隐藏层的维度，代表了整个序列的特征向量。
        #   - 输出维度：`self.num_labels`，即标签数量，每个维度对应一个类别的得分。
        #   - `bias=False`：指示不使用偏置项。在某些情况下，去除偏置项可以提高模型性能。
        #   - 线性层的输出通常会经过 Softmax 函数进行归一化，得到每个类别的概率。

        # 4. 初始化权重和应用最终处理
        self.post_init()
        #   - 调用 `post_init` 方法，初始化模型的权重，并将模型设置为评估模式。
        #   -  `post_init` 方法是 PreTrainedModel 类提供的一个钩子函数，用于执行模型初始化后的操作。

    def get_input_embeddings(self):
        """
        获取输入嵌入。

        Returns:
            nn.Embedding: 输入嵌入层。
        """
        return self.model.embed_tokens  # 返回底层 LlamaModel 的嵌入层，用于将 token 转换为向量表示。

    def set_input_embeddings(self, value):
        """
        设置输入嵌入。

        Args:
            value (nn.Embedding): 新的嵌入层。
        """
        self.model.embed_tokens = value  # 将底层 LlamaModel 的嵌入层替换为新的嵌入层，用于替换模型的词汇表或调整嵌入维度。

    @can_return_tuple  # 指示该函数可以返回元组，允许用户根据需要选择返回哪些信息。
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)  # 添加输入文档字符串，从 LLAMA_INPUTS_DOCSTRING 常量中获取。
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        前向传播函数，计算 LlamaForSequenceClassification 的输出，执行序列分类任务的核心逻辑。

        Args:
            input_ids (Optional[torch.LongTensor]): 输入序列 token 的索引，形状为 (batch_size, sequence_length)。
            attention_mask (Optional[torch.Tensor]): 注意力掩码，形状为 (batch_size, sequence_length)。
            position_ids (Optional[torch.LongTensor]): 位置 IDs，形状为 (batch_size, sequence_length)。
            past_key_values (Optional[Cache]): 过去的 key 和 value，用于加速推理 (仅在 use_cache=True 时有效)。
            inputs_embeds (Optional[torch.FloatTensor]): 输入嵌入，如果提供了 input_ids，则不需要提供 inputs_embeds。
            labels (Optional[torch.LongTensor]): 用于计算序列分类/回归损失的标签，形状为 (batch_size,)。
                - 对于分类任务，每个值代表一个类别的索引。
                - 对于回归任务，每个值代表一个连续的目标值。
            use_cache (Optional[bool]): 是否使用缓存，用于加速推理。
            output_attentions (Optional[bool]): 是否输出注意力权重，用于可视化或分析模型行为。
            output_hidden_states (Optional[bool]): 是否输出隐藏状态，用于获取中间层特征。

        Returns:
            SequenceClassifierOutputWithPast: 包含 loss, logits, past_key_values, hidden_states, attentions 的对象。
            - loss: 如果提供了 labels，则返回计算得到的损失值，否则返回 None。
            - logits: 模型输出的 logits，形状为 (batch_size, num_labels)，表示每个类别的得分。
            - past_key_values:  如果 use_cache=True，则返回缓存的 key 和 value，用于加速后续的生成过程。
            - hidden_states: 如果 output_hidden_states=True，则返回所有隐藏层的输出，用于获取中间层特征。
            - attentions: 如果 output_attentions=True，则返回所有注意力层的权重，用于可视化或分析模型行为。
        """
        # r""" 用于创建原始字符串，可以包含特殊字符而无需转义。
        #  -  这段注释使用了 reStructuredText 格式，用于生成文档。

        # 1. 计算 Llama 模型的输出，将输入传递给底层的 LlamaModel
        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        #   - 将输入传递给底层的 LlamaModel，进行编码，得到 Transformer 模型的输出。
        #   - `transformer_outputs`：包含 LlamaModel 的输出，是一个 BaseModelOutputWithPast 对象。

        # 2. 获取最后一个隐藏层的输出，作为序列的整体表示
        hidden_states = transformer_outputs.last_hidden_state
        #   - 从 Transformer 模型的输出中提取最后一个隐藏层的输出。
        #   -  `hidden_states`：形状为 (batch_size, sequence_length, hidden_size)，表示每个 token 的隐藏状态。

        # 3. 使用线性分类层进行分类，将序列表示映射到 logits
        logits = self.score(hidden_states)
        #   - 将最后一个隐藏层的输出通过线性层 `self.score` 映射到 logits 空间，得到每个类别的预测分数。
        #   - `logits`：形状为 (batch_size, sequence_length, num_labels)，表示每个 token 对不同类别的预测得分。

        # 4. 获取 batch_size，用于后续的计算
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        #   -  根据是否提供了 input_ids 或 inputs_embeds 来确定 batch size。

        # 5. 确定用于分类的 token，需要考虑 padding 的情况
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        #   - 如果没有定义 `pad_token_id` 且 `batch_size` 大于 1，则抛出 ValueError，因为无法确定使用哪个 token 进行分类。
        #   - 这通常发生在处理变长序列且没有指定 padding token 的情况下。
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        #   - 如果没有定义 `pad_token_id`，则使用最后一个 token 进行分类。
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            # 为了处理左侧和右侧的 padding，我们获取最右侧的不等于 pad_token_id 的 token
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        #   - 如果定义了 `pad_token_id`，则找到每个序列中最后一个非 padding token 的位置。
        #   -  `non_pad_mask`：形状为 (batch_size, sequence_length)，用于指示哪些 token 是非 padding token。
        #   -  `token_indices`：形状为 (sequence_length)，表示 token 的位置索引。
        #   -  `last_non_pad_token`：形状为 (batch_size,)，表示每个序列中最后一个非 padding token 的位置索引。
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
        #   - 如果没有提供 `input_ids`，只有 `inputs_embeds`, 则使用最后一个 token 进行分类，并给出警告，因为无法检测 padding token。

        # 6. 获取用于分类的 logits
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        #   - 从 `logits` 中提取用于分类的 token 的 logits。
        #   -  `pooled_logits`：形状为 (batch_size, num_labels)，表示每个序列的分类得分。

        # 7. 初始化损失
        loss = None
        #   - 初始化损失变量为 None。

        # 8. 如果提供了 labels，则计算损失
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)
        #   - 如果提供了标签 `labels`，则使用 `self.loss_function` 计算序列分类损失。
        #   -  `self.loss_function`：负责计算损失，根据任务类型（分类或回归）选择合适的损失函数，例如交叉熵损失或均方误差损失。

        # 9. 返回 SequenceClassifierOutputWithPast
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        #   - 创建一个 `SequenceClassifierOutputWithPast` 对象，并将损失、logits、缓存的 key 和 value、隐藏状态和注意力权重存储在该对象中。
        #   -  `SequenceClassifierOutputWithPast` 是一个数据结构，用于组织和返回序列分类任务的输出结果。


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

# 从Hugging Face transformers库中导入基础模块
from ...activations import ACT2FN  # 激活函数工厂，如'silu'
from ...cache_utils import Cache  # KV缓存工具
from ...modeling_flash_attention_utils import FlashAttentionKwargs # Flash Attention参数
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS # 支持的注意力实现 (eager, sdpa, flash_attention_2)
from ...processing_utils import Unpack
from ...utils import logging
# 导入Llama模型的核心组件，DeepseekV3在此基础上构建
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)
# 导入DeepseekV3的配置文件类
from .configuration_deepseek_v3 import DeepseekV3Config


logger = logging.get_logger(__name__)

#均方根层归一化（RMSNorm）。这里直接继承Llama的实现，没有任何修改。
class DeepseekV3RMSNorm(LlamaRMSNorm):
    pass

#旋转位置编码（RoPE）。同样直接继承。重要的上下文扩展逻辑（如YaRN）实际上是在父类LlamaRotaryEmbedding中根据config.rope_scaling参数实现的。
class DeepseekV3RotaryEmbedding(LlamaRotaryEmbedding):
    pass

##功能: 这个函数为Query (q) 和 Key (k) 向量应用旋转位置编码。
##interleave (交错): 这是RoPE的一种实现方式。标准的RoPE将一个向量分成两半，然后进行旋转。
# 而交错方式将向量视为成对的元素 (x1, x2), (x3, x4), ... 并对每一对进行旋转。这种数据布局有时在特定硬件上更高效。
## TODO 备注: TODO let's just use the original freqcis computation to not have the view transpose + reshape! This is not optimized! 这条备注指出当前的实现（通过view, transpose, reshape）效率不高，未来可以优化为更直接的计算方式。
def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor. 它的形状通常是 [batch_size, num_heads, seq_len, head_dim].
        k (`torch.Tensor`): The key tensor.  它的形状通常是 [batch_size, num_heads, seq_len, head_dim].
        cos (`torch.Tensor`): RoPE的余弦部分。形状为 [batch_size, seq_len, head_dim].
        sin (`torch.Tensor`): RoPE的正弦部分。形状为 [batch_size, seq_len, head_dim].
        position_ids (`torch.Tensor`): 位置ID，用于从预先计算好的cos/sin表中查找相应的值。此函数内不直接使用，但在上游调用中用于生成cos和sin。
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            用于扩展 cos 和 sin 维度的轴，以便它们可以与 q 和 k 进行广播。
            对于形状为 [batch, heads, seq, dim] 的 q/k, unsqueeze_dim=1 会将 cos/sin 变为 [batch, 1, seq, dim]，从而可以和 heads 维度广播。
    Returns:
        `tuple(torch.Tensor)`: 返回经过旋转位置编码处理后的 query 和 key 张量。
    """
    # ----------------------------------------------------------------------------------
    # 步骤 1: 准备 cos 和 sin 以便进行广播
    # ----------------------------------------------------------------------------------

    # 在指定维度(默认为1，即heads维度)上增加一个维度。
    # 这使得 cos 的形状从 [batch, seq_len, head_dim] 变为 [batch, 1, seq_len, head_dim]。
    # 这样，在后续的乘法操作中，它就可以自动广播以匹配 `q` 和 `k` 的 `num_heads` 维度。
    cos = cos.unsqueeze(unsqueeze_dim)

    # 对 sin 执行相同的操作。
    sin = sin.unsqueeze(unsqueeze_dim)

    # ----------------------------------------------------------------------------------
    # 步骤 2: 重新排列 Query 张量以适配 `rotate_half` 函数
    # ----------------------------------------------------------------------------------

    # 获取 Query 张量 q 的维度信息。
    # b: batch_size, h: num_heads, s: sequence_length, d: head_dim
    b, h, s, d = q.shape

    # 核心操作1: 将最后一个维度 d 视为 d/2 个 2D 向量。
    # 形状从 [b, h, s, d] 变为 [b, h, s, d // 2, 2]。
    # 例如, [f1, f2, f3, f4] 变为 [[f1, f2], [f3, f4]]。
    # 这正是“交错”模式的体现：相邻的两个特征被视为一个对。
    q = q.view(b, h, s, d // 2, 2)

    # 核心操作2: 交换最后两个维度。
    # 形状从 [b, h, s, d // 2, 2] 变为 [b, h, s, 2, d // 2]。
    # 这一步是为了将所有成对向量的“第一部分”(x)和“第二部分”(y)分离开。
    # 现在张量在概念上是 [[f1, f3, ...], [f2, f4, ...]]。
    # 这样做是为了让 `rotate_half` 函数（它期望数据是前后两半的结构）能够正确工作。
    q = q.transpose(4, 3)

    # 核心操作3: 将数据重新展平回原始的 head_dim 维度。
    # 形状从 [b, h, s, 2, d // 2] 变回 [b, h, s, d]。
    # 此时，张量内部的数据布局已经被永久改变，现在前半部分是所有对的x分量，后半部分是所有对的y分量。
    q = q.reshape(b, h, s, d)

    # ----------------------------------------------------------------------------------
    # 步骤 3: 重新排列 Key 张量 (与 Query 完全相同的操作)
    # ----------------------------------------------------------------------------------

    # 获取 Key 张量 k 的维度信息。
    b, h, s, d = k.shape

    # 对 k 执行与 q 完全相同的 view -> transpose -> reshape 操作，
    # 以便将其数据也排列成 `rotate_half` 所需的格式。
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # ----------------------------------------------------------------------------------
    # 步骤 4: 应用旋转位置编码
    # ----------------------------------------------------------------------------------

    # 这是RoPE的核心数学公式: x' = x * cos(θ) - y * sin(θ)
    # 这里的实现是: q_embed = q * cos + rotate_half(q) * sin
    # `rotate_half(q)` 的作用是将张量的后半部分取反，然后与前半部分交换，等效于从 (x, y) 得到 (-y, x)。
    # 所以 `rotate_half(q) * sin` 实际上计算了 `(-y) * sin(θ)` 部分。
    # `q * cos` 计算了 `x * cos(θ)` 部分。
    # 两者相加，就完成了对所有2D向量的旋转。
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # 对 Key 张量 k 执行完全相同的旋转操作。
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 返回经过旋转编码后的 q 和 k。
    return q_embed, k_embed


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

# 低秩分解 (Low-Rank Decomposition): 它没有直接使用一个大的线性层将 hidden_states 投影到 Q, K, V。而是像 LoRA (Low-Rank Adaptation) 那样，先将输入投影到一个更小的“潜在”空间（由 q_lora_rank 和 kv_lora_rank 定义），然后再从这个潜在空间投影到最终的 Q, K, V 维度。这大大减少了参数量和计算量。
# 部分旋转编码 (Partial RoPE): 它不是对整个 Query 和 Key 向量应用旋转位置编码 (RoPE)
# 而是将它们切分成两部分：一部分应用 RoPE（_rope_head_dim），另一部分不应用 RoPE（_nope_head_dim）。这使得模型能够同时利用绝对位置信息（通过 RoPE）和相对内容信息（不通过 RoPE）。
# 共享的KV投影: Key 和 Value 的潜在表示是通过一个共享的投影层 (kv_a_proj_with_mqa) 生成的，这进一步提高了参数效率。
# 支持YaRN上下文扩展: 集成了 YaRN 技术，使其能够处理超长上下文。
class DeepseekV3Attention(nn.Module):
    # ...
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        # --- 1. 初始化基本配置 ---
        self.config = config
        self.layer_idx = layer_idx
        # GQA (Grouped-Query Attention) 相关，多少个Q头共享一组K/V头
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta # RoPE 的基数 theta

        # --- 2. MLA 和 部分RoPE 的核心参数 ---
        self.q_lora_rank = config.q_lora_rank          # Query 的低秩维度
        self.qk_rope_head_dim = config.qk_rope_head_dim # Q/K 向量中需要应用RoPE的部分的维度
        self.kv_lora_rank = config.kv_lora_rank        # Key/Value 的低秩维度
        self.v_head_dim = config.v_head_dim            # Value 向量的头维度
        self.qk_nope_head_dim = config.qk_nope_head_dim # Q/K 向量中不应用RoPE的部分的维度
        self.qk_head_dim = config.qk_head_dim          # Q/K 向量的总头维度 (rope + nope)

        self.is_causal = True # 这是一个自回归模型，需要因果掩码

        # --- 3. 定义 Query (Q) 的投影路径 ---
        # 步骤 A: 从 hidden_size 投影到低秩空间 q_lora_rank
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        # 在低秩空间进行一次 RMSNorm
        self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
        # 步骤 B: 从低秩空间投影到最终的 Q 维度 (所有头的维度总和)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # --- 4. 定义 Key (K) 和 Value (V) 的共享投影路径 ---
        # 步骤 A: 从 hidden_size 投影到一个共享的潜在空间。
        # 这个空间的维度是 KV低秩维度(kv_lora_rank) 和 K的RoPE部分维度(qk_rope_head_dim) 的总和。
        # K的RoPE部分被特殊处理，直接从这个投影中分离出来，不经过后续的低秩路径。
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        # 只对低秩部分进行 RMSNorm
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        # 步骤 B: 从低秩空间投影到最终的 K的非RoPE部分 和 V 的维度
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # --- 5. 定义输出投影层 (O) ---
        # 将所有头的注意力输出拼接后，投影回模型的 hidden_size
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # --- 6. 初始化注意力缩放因子，并集成YaRN ---
        # 标准的注意力缩放因子
        self.scaling = self.qk_head_dim ** (-0.5)
        # 如果配置了YaRN长上下文扩展
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                # 计算YaRN的温度缩放因子 mscale
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                # 将其应用到注意力缩放因子上，以稳定长上下文下的注意力分数
                self.scaling = self.scaling * mscale * mscale


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # ...
        batch_size, seq_length = hidden_states.shape[:-1]
        # 定义重塑 Q 和 K/V 后的目标形状
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # --- 1. 计算 Query (Q) 并分离 RoPE/non-RoPE 部分 ---
        # 完整的Q投影路径：hidden_states -> q_a_proj -> q_a_layernorm -> q_b_proj
        # .view(query_shape) 将其重塑为 [batch, seq_len, num_heads, qk_head_dim]
        # .transpose(1, 2) 将其变为 [batch, num_heads, seq_len, qk_head_dim] 以便计算
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
        # 将每个头的Q向量切分为不应用RoPE(q_pass)和应用RoPE(q_rot)两部分
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- 2. 计算 Key (K) 和 Value (V) 并分离 ---
        # 步骤A：共享投影
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # 从中直接分离出K的低秩部分(k_pass)和RoPE部分(k_rot)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # 步骤B：对K的低秩部分进行后续投影
        # k_pass -> kv_a_layernorm -> kv_b_proj -> reshape & transpose
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        # 从投影结果中分离出最终的K的非RoPE部分(k_pass)和Value(value_states)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # --- 3. 应用旋转位置编码 (RoPE) ---
        # 准备K的RoPE部分
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings  # 获取预先计算好的cos/sin值
        # 根据配置，对 q_rot 和 k_rot 应用RoPE
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        # 对于GQA，需要将K的RoPE部分扩展以匹配Q的头数
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # --- 4. 重组 Q 和 K ---
        # 将非RoPE部分和RoPE部分拼接回来，形成最终的Q和K
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # --- 5. KV 缓存和注意力计算 ---
        # 如果正在进行自回归生成，更新并使用KV缓存
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Flash Attention 2 的一个兼容性处理
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # 选择合适的注意力计算后端（如eager, sdpa, flash_attention_2）
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get(self.config._attn_implementation,
                                                                    eager_attention_forward)

        # 执行注意力计算！
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,  # 传入我们可能被YaRN修改过的缩放因子
            **kwargs,
        )

        # --- 6. 后处理和输出 ---
        # 移除Flash Attention 2的填充
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        # 将输出重塑并进行最终的输出投影
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV3DecoderLayer(LlamaDecoderLayer, nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        nn.Module().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV3PreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, DeepseekV3RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, DeepseekV3TopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)


class DeepseekV3Model(LlamaModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]


class DeepseekV3ForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "DeepseekV3PreTrainedModel",
    "DeepseekV3Model",
    "DeepseekV3ForCausalLM",
]

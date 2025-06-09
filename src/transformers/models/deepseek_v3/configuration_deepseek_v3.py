# coding=utf-8
# Copyright 2025 bzantium and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeekV3 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V3)

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
"""DeepSeekV3 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DeepseekV3Config(PretrainedConfig):
    r"""
    这是用于存储 [`DeepseekV3Model`] 配置的类。它根据指定的参数来实例化一个 DeepSeek 模型，定义了模型的架构。
    使用默认值实例化一个配置，将产生一个与 DeepSeek-V3 相似的配置。
    例如 [bzantium/tiny-deepseek-v3](https://huggingface.co/bzantium/tiny-deepseek-v3)

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。
    更多信息请阅读 [`PretrainedConfig`] 的文档。


    参数:
        vocab_size (`int`, *可选*, 默认为 129280):
            DeepSeek 模型的词汇表大小。定义了在调用 [`DeepseekV3Model`] 时，`inputs_ids` 可以表示的不同token的数量。

        hidden_size (`int`, *可选*, 默认为 7168):
            隐藏表示的维度。即模型中各层特征向量的大小。

        intermediate_size (`int`, *可选*, 默认为 18432):
            （非MoE层中）标准MLP前馈网络的中间层维度。

        moe_intermediate_size (`int`, *可选*, 默认为 2048):
            （MoE层中）每个专家的MLP前馈网络的中间层维度。

        num_hidden_layers (`int`, *可选*, 默认为 61):
            Transformer解码器中的隐藏层数量。

        num_attention_heads (`int`, *可选*, 默认为 128):
            Transformer解码器中每个注意力层的注意力头数量。

        num_key_value_heads (`int`, *可选*, 默认为 128):
            用于实现分组查询注意力（GQA）的键/值头的数量。
            - 如果 `num_key_value_heads` == `num_attention_heads`，模型使用多头注意力（MHA）。
            - 如果 `num_key_value_heads` == 1，模型使用多查询注意力（MQA）。
            - 否则，使用GQA。

        n_shared_experts (`int`, *可选*, 默认为 1):
            共享专家的数量。这些专家会被所有token使用。

        n_routed_experts (`int`, *可选*, 默认为 256):
            被路由的专家的数量。token会被路由到这些专家中的一部分。

        routed_scaling_factor (`float`, *可选*, 默认为 2.5):
            被路由的专家的输出的缩放因子。

        kv_lora_rank (`int`, *可选*, 默认为 512):
            用于键（Key）和值（Value）投影的LoRA矩阵的秩（即潜在空间的维度）。

        q_lora_rank (`int`, *可选*, 默认为 1536):
            用于查询（Query）投影的LoRA矩阵的秩。

        qk_rope_head_dim (`int`, *可选*, 默认为 64):
            查询/键的注意力头中，**使用**旋转位置编码（RoPE）的部分的维度。

        v_head_dim (`int`, *可选*, 默认为 128):
            值（Value）的注意力头的维度。

        qk_nope_head_dim (`int`, *可选*, 默认为 128):
            查询/键的注意力头中，**不使用**旋转位置编码（RoPE）的部分的维度。

        n_group (`int`, *可选*, 默认为 8):
            用于路由专家的分组数量。

        topk_group (`int`, *可选*, 默认为 4):
            每个token选择的分组数量（确保每个token选择的专家仅限于这 `topk_group` 个组内）。

        num_experts_per_tok (`int`, *可选*, 默认为 8):
            每个token选择的专家数量。

        first_k_dense_replace (`int`, *可选*, 默认为 3):
            在模型的浅层（底层）使用的稠密（非MoE）层的数量。
            结构大致为：embed -> dense -> dense -> dense -> moe -> moe ... -> lm_head
                                  \---- k个稠密层 ----/

        norm_topk_prob (`bool`, *可选*, 默认为 `True`):
            是否对被路由的专家的权重进行归一化。

        hidden_act (`str` or `function`, *可选*, 默认为 `"silu"`):
            解码器中使用的非线性激活函数（例如 "gelu", "relu" 或 "silu"）。

        max_position_embeddings (`int`, *可选*, 默认为 4096):
            该模型可能使用的最大序列长度。也是RoPE计算的原始基础长度。

        initializer_range (`float`, *可选*, 默认为 0.02):
            用于初始化所有权重矩阵的截断正态分布初始化器的标准差。

        rms_norm_eps (`float`, *可选*, 默认为 1e-06):
            RMS归一化层中使用的epsilon值，以防止除以零。

        use_cache (`bool`, *可选*, 默认为 `True`):
            模型是否应返回最后的键/值注意力（KV缓存），用于加速自回归生成。

        pad_token_id (`int`, *可选*):
            填充token的ID。

        bos_token_id (`int`, *可选*, 默认为 0):
            序列开始（beginning of stream）token的ID。

        eos_token_id (`int`, *可选*, 默认为 1):
            序列结束（end of stream）token的ID。

        pretraining_tp (`int`, *可选*, 默认为 1):
            （实验性特性）预训练期间使用的张量并行等级。用于确保预训练结果的精确复现。

        tie_word_embeddings (`bool`, *可选*, 默认为 `False`):
            是否将输入词嵌入的权重与输出解码层的权重绑定。

        rope_theta (`float`, *可选*, 默认为 10000.0):
            RoPE嵌入的基周期（base period）。

        rope_scaling (`Dict`, *可选*):
            包含RoPE嵌入缩放配置的字典。用于上下文扩展（如YaRN）。
            期望格式为 `{"type": "策略名称", "factor": 缩放因子}`。

        rope_interleave (`bool`, *可选*, 默认为 `True`):
            是否交错实现旋转位置编码。

        attention_bias (`bool`, 默认为 `False`, *可选*, 默认为 `False`):
            在自注意力机制的Q, K, V和输出投影层中是否使用偏置（bias）。

        attention_dropout (`float`, *可选*, 默认为 0.0):
            注意力概率的dropout比率。

    ```python
    >>> from transformers import DeepseekV3Model, DeepseekV3Config

    >>> # 初始化一个 Deepseek-V3 风格的配置
    >>> configuration = DeepseekV3Config()

    >>> # 从一个已加载的模型中访问配置
    >>> model = DeepseekV3Model(configuration)
    >>> configuration = model.config
    ```"""
    # 模型类型标识符
    model_type = "deepseek_v3"
    # 在推理时可以从模型输出中忽略的键（因为KV缓存是模型内部状态）
    keys_to_ignore_at_inference = ["past_key_values"]

    # 定义了在使用张量并行(TP)时，如何切分模型的权重。
    # 这对于在多GPU上训练或推理大模型至关重要。
    # "local_colwise": 按列切分，每个GPU持有一部分列，输入在所有GPU上复制。
    # "local_rowwise": 按行切分，每个GPU持有一部分行，输入在所有GPU上切分。
    # "local": 表示该模块在每个GPU上都是一个独立的、完整的副本。
    # "gather": 表示需要从所有GPU收集结果。
    base_model_tp_plan = {  # TODO: 只有在层数 > first_k_dense_replace 时才复制注意力层
        "layers.*.mlp.experts.*.gate_proj": "local_colwise",
        "layers.*.mlp.experts.*.up_proj": "local_colwise",
        "layers.*.mlp.experts.*.down_proj": "local_rowwise",
        "layers.*.mlp.experts.*": "local",  # 每个专家被包裹在一个ModuleList中
        "layers.*.mlp.shared_experts.gate_proj": "local_colwise",
        "layers.*.mlp.shared_experts.up_proj": "local_colwise",
        "layers.*.mlp.shared_experts.down_proj": "local_rowwise",
        "layers.*.mlp.shared_experts": "local",
        "layers.*.mlp.gate_proj": "local_colwise",
        "layers.*.mlp.up_proj": "local_colwise",
        "layers.*.mlp.down_proj": "local_rowwise",
        "layers.*.mlp": "gather",  # 这是唯一需要收集结果的时刻
    }

    # 定义了在使用流水线并行(PP)时，如何在不同的GPU阶段之间传递数据。
    # (["input_ids"], ["inputs_embeds"]) 表示：接收 input_ids，输出 inputs_embeds。
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
            self,
            vocab_size=129280,
            hidden_size=7168,
            intermediate_size=18432,
            moe_intermediate_size=2048,
            num_hidden_layers=61,
            num_attention_heads=128,
            num_key_value_heads=128,
            n_shared_experts=1,
            n_routed_experts=256,
            routed_scaling_factor=2.5,
            kv_lora_rank=512,
            q_lora_rank=1536,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128,
            n_group=8,
            topk_group=4,
            num_experts_per_tok=8,
            first_k_dense_replace=3,
            norm_topk_prob=True,
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=0,
            eos_token_id=1,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            rope_interleave=True,
            attention_bias=False,
            attention_dropout=0.0,
            **kwargs,
    ):
        # 将所有传入的参数赋值给类的实例变量 self.*。
        # 这样，模型的各个部分就可以通过 self.config.* 来访问这些超参数。
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        # 计算一些衍生参数
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # 在一些实现中，head_dim 特指应用RoPE的部分
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave

        # 为了向后兼容，如果 num_key_value_heads 未指定，则默认为 num_attention_heads (即MHA)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # 验证和处理RoPE缩放配置
        # BC: (Backward Compatibility) 如果 rope_scaling 字典中存在旧的 'type' 字段，将其复制为新的 'rope_type' 字段
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # 确保YaRN等方法中的关键参数是浮点数类型
        if self.rope_scaling is not None:
            for key in ["beta_fast", "beta_slow", "factor"]:
                if key in self.rope_scaling:
                    self.rope_scaling[key] = float(self.rope_scaling[key])

        # 调用一个外部函数来验证RoPE相关的配置是否合法
        rope_config_validation(self)

        # 调用父类 PretrainedConfig 的构造函数，处理如 pad_token_id, bos_token_id, eos_token_id 等通用参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ["DeepseekV3Config"]

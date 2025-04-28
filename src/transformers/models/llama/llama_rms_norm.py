import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm # 假设代码在 transformers 库中

# 设定随机种子，使结果可复现
torch.manual_seed(42)

hidden_size = 4
batch_size = 2
sequence_length = 3

# 创建一些随机的隐藏状态
hidden_states = torch.randn((batch_size, sequence_length, hidden_size))
print("Original hidden_states:\n", hidden_states)

t1=hidden_states.pow(2)
print(f"\nMean of squares:\n", t1)
t2=t1.mean(dim=-1, keepdim=True)
print(f"\nMean of squares (keepdim=True):\n", t2)

# 创建 LlamaRMSNorm 实例
rms_norm = LlamaRMSNorm(hidden_size=hidden_size, eps=1e-6) # 设置 eps 保证数值稳定性

# 对 hidden_states 应用 RMSNorm
normalized_hidden_states = rms_norm(hidden_states)

print("\nNormalized hidden_states:\n", normalized_hidden_states)
print("\nLlamaRMSNorm Weight:\n", rms_norm.weight)
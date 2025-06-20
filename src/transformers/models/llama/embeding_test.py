import torch
import torch.nn as nn

# 1. 设置配置参数
vocab_size = 10000
hidden_size = 256
padding_idx = 0

# 2. 创建嵌入层 (模拟 LlamaModel 的一部分)
embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

# 3. 创建 DynamicCache
past_key_values = None  # 初始时，没有缓存

# 4. 模拟生成过程 (循环生成 5 个 token)
for i in range(5):
    # a. 模拟生成一个 token ID
    if i == 0:
        input_ids = torch.tensor([[100]]) # 初始的 token ID
    else:
        input_ids = torch.tensor([[predicted_id]])

    # b. 将 token ID 转换为嵌入向量
    inputs_embeds = embed_tokens(input_ids)

    # c. 如果没有提供 cache_position，则创建一个 cache_position
    cache_position = None  # 每次循环开始时，cache_position 设为 None，模拟没有显式提供
    if cache_position is None:
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # d. 模拟 LlamaModel.forward() 的其他部分 (这里只打印 cache_position)
    print(f"Iteration {i+1}:")
    print("  input_ids:", input_ids)
    print("  inputs_embeds shape:", inputs_embeds.shape)
    print("  past_key_values is None:", past_key_values is None)
    print("  past_seen_tokens:", past_seen_tokens)
    print("  cache_position:", cache_position)

    # e. 模拟更新 past_key_values (这里只是为了让 get_seq_length() 返回正确的值)
    class MockCache:  # 创建一个模拟的 Cache 类
        def __init__(self, length):
            self.length = length
        def get_seq_length(self):
            return self.length
    if past_key_values is None:
        past_key_values = MockCache(1) # 第一次调用
    else:
        past_key_values = MockCache(past_key_values.get_seq_length() + 1)  # 更新长度

    # f. 假设模型预测的下一个 token ID
    predicted_id = torch.randint(1, vocab_size, (1,)).item()
    print("  predicted_id:", predicted_id)
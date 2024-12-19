import torch

termination_token_id = 3
log_pterm = []
log_pf = []
temperature = 0.9

# (n_samples=3, seq_len=3) generated_text_len = 2
encoded_input = torch.tensor([[3,2,1,1], [0,2,0,2], [2,1,2,3]])
# prompt_length = 2
skip_first = 2

# (n_samples=3)
active_seqs = torch.tensor([True, False, True])

# (n_samples=3, seq_len = 3, vocab_size=4), finished "logits = logits[:, skip_first - 1 :]"
logits = torch.tensor([[[2,7,3,8], [1,5,2,3], [3,7,1,2]], 
                       [[8,6,2,7], [7,3,5,1], [1,5,4,8]], 
                       [[3,7,8,4], [5,5,1,8], [9,5,2,1]]], dtype=torch.double)

# (n_samples, seq_len, vocab_size)
logprob = logits.log_softmax(dim=-1)

# 提取采样后的语句中的生成部分的 token id 序列，并进行维度扩展
token_ids = encoded_input[:, skip_first:].unsqueeze(-1)


# 收集 token_ids 对应的概率的对数
logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
# 逐步累加每句话的采样的所有词汇的概率，即每步可以停止时，当前生成的句子的概率之和
logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)，即给定 prompt 下，生成的句子的概率之和


# 获取每句话所有词汇位置的终止标记的概率，并作为初始 reward
reward = logprob[
    :, :, termination_token_id
]  # logP(generated[i+1]=term | prompt + generated[:i+1])，在i+1处停止时的概率

reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)

print("logprob:\n", logprob)
print("token_ids:\n", token_ids)
print("logPF:\n", logPF)
print("logP:\n", logP)
print("logprob:\n", logprob)
print("reward:\n", reward)


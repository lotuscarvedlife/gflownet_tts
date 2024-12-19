import torch

termination_token_id = 2
log_pterm = []
log_pf = []
temperature = 0.9


# (n_samples=3)
active_seqs = torch.tensor([True, False, True])

# (n_samples=3, vocab_size=4)
logits = torch.tensor([[1,5,2,3], [8,6,2,7], [3,7,8,4]], dtype=torch.double)


# 进行温度处理，让分布更尖或者更平缓
prob = (logits / temperature).softmax(dim=-1)
# 根据概率采样，生成每一句的 token id
token_ids = torch.multinomial(prob, num_samples=1)



# (n_samples, vocab_size)
logprob = logits.log_softmax(dim=-1)

# probability of terminating
log_pterm.append(
    torch.where(
        active_seqs,
        logprob[:, termination_token_id],
        0.0,
    )
)

# probability of forward
log_pf.append(
    torch.where(
        active_seqs,
        logprob.gather(-1, token_ids).squeeze(-1),
        0.,
    )
)

print("token_ids:\n", token_ids)
print("logprob:\n", logprob)
print("log_pterm:\n", log_pterm)
print("log_pf:\n", log_pf)







# a = torch.tensor([[[1,2,4,4], [2,5,1,4], [6,4,4,4]], [[5,1,1,4], [1,5,7,4], [5,3,4,4]]], dtype=torch.float)
# a = torch.tensor([[1,2,4,4], [2,5,1,4], [6,4,4,4], [5,1,1,4], [1,5,7,4], [5,3,4,4]], dtype=torch.float)

# a_mask = (a != 4)[:,1:]
# print(a_mask)
# a_mask = torch.cat(
#     (
#         a_mask.new_ones(a_mask.shape[0], 1),
#         a_mask,
#     ),
#     dim=-1,
# )
# print(a_mask)

# a[~a_mask] = 0.0

# print(a)

# print(a.cumsum(dim=-1))
# a = torch.tensor([[1,2,4,4], [2,5,4,1], [6,7,4,2], [5,4,1,3], [1,5,7,1], [5,3,1,8]])
# b = torch.tensor([1,2,3,4,5])
# print(a.log_softmax(-1))
# print(a.shape)
# print(a[:, 1:].shape)
# print(a.shape)
# a = a.unsqueeze(-1)
# b = b.unsqueeze(-1)
# b = b.squeeze(-1)
# print(a)
# print(a.shape)
# print(b)
# print(b.shape)
# a = a[:,-1,:]
# print(a)
# print(a.shape)
# a = (a/0.1).softmax(dim=-1)
# print(a)
# a = a.sum(dim=-1)
# print(a)

# active_seqs = torch.tensor([True, True, True])
# token_ids = torch.tensor([[2], [5], [4]])
# termination_token_id = 5
# active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
# print(active_seqs)

# log_pf = []
# active_seqs = torch.tensor([True, True, False])
# token_ids = torch.tensor([[1], [0], [2]])
# logprob = [torch.tensor([0.1, 0.2, 0.7]), torch.tensor([0.8, 0.1, 0.1]), torch.tensor([0.3, 0.2, 0.5])]
# print(a.gather(-1, token_ids).squeeze(-1))
# print(torch.stack(logprob, dim=1))
# print(logprob.gather(-1, token_ids).squeeze(-1))
# log_pf.append(
#     torch.where(
#         active_seqs,
#         logprob.gather(-1, token_ids).squeeze(-1),
#         0,
#     )
# )
# print(log_pf)
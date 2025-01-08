import torch
import random

generated_audio = torch.tensor([1,2,3,4,5])
print(generated_audio[-3:])

# 创建一个示例的一维张量
# tensor = torch.tensor([1, 2, 3, 4, 5])

# # 使用 tolist() 方法将张量转换为列表，并使用 join() 函数连接
# string_representation = ' '.join(map(str, tensor.tolist()))

# print(string_representation)  # 输出: "1 2 3 4 5"
# a = [[False]*4 for _ in range(20)]
# a[1][2] = True
# print(a)
# print(sum(sum(i) for i in a))

# # a = torch.where(torch.tensor([True, False, False]), torch.tensor([1,2,3]), 0)
# b = [[[3,2,1,4,5],[2,1,3,4,2],[2,4,6,1,2],[5,3,1,3,2]],
#      [[2,1,4,5,6],[4,5,8,1,2],[1,4,1,3,8],[2,1,3,3,9]]]
# b = torch.tensor(b)
# b = [[[3,2,1,4],[2,1,3,4],[2,4,6,1]],
#      [[2,1,4,5],[4,5,8,1],[1,4,1,3]]]
# cur_generated = [[] for _ in range(2)]
# idx = 0
# for i in b:
#     for j in i:
#         try:
#             cur_generated[idx].append(torch.tensor(j))
#         except:
#             print(idx)
#     idx += 1

# # print(c[:][-1])
# all_token_ids = torch.stack([j[-1] for j in cur_generated]).unsqueeze(-1)
# active_seqs = torch.tensor([True, False])
# logprob = torch.rand(2,4,10)
# print(logprob[:,0,:].shape)
# result = torch.where(
#     active_seqs,
#     logprob.gather(-1, all_token_ids).squeeze(-1).sum(dim=-1),
#     0
# )
# print(logprob)
# print(logprob.gather(-1, all_token_ids).squeeze(-1))
# print(result)

# # 创建示例张量
# tensor = torch.rand(20, 4, 10)
# # print(torch.full((20, 1, 0), -1))
# new_token_ids = []
# for i in range(4):
#     # if i!=0:
#     #     temp = torch.cat([torch.full((20, i), -1), temp], dim=-1)
#     # if 4-i-1!=0:
#     #     temp = torch.cat([temp, torch.full((20, 4-1-i), -1)], dim=-1)
#     new_token_ids.append(torch.cat([torch.full((20, i), -1),
#                                     tensor[:, i, :], 
#                                     torch.full((20, 4-1-i), -1)], 
#                                     dim=-1))

# new_token_ids = torch.stack(new_token_ids, dim=1)
# print(new_token_ids)
# print(new_token_ids.shape)



# token_ids = [[[1],[2],[3],[4]],[[4],[3],[2],[1]]]
# token_ids = torch.tensor(token_ids)
# print(b.gather(-1, token_ids).squeeze(-1).sum(dim=0)*4)
# y_len = 188
# mask_len_min = 1
# mask_len_max = 600
# min_gap = 5

# mask_intervals = []
# non_mask_intervals = []


# param = float('poisson1'[len("poisson"):])
# poisson_sample = torch.poisson(torch.tensor([param]))
# n_spans = int(poisson_sample.clamp(1, 3).item())

# print(n_spans)

# starts = random.sample(range(1, y_len-1-mask_len_min), n_spans)
# starts = sorted(starts)

# for j in range(len(starts)-1, 0, -1):
#     if starts[j] - starts[j-1] < min_gap:
#         del starts[j] # If elements are too close, delete the later one
# assert len(starts) > 0

# temp_starts =  starts + [y_len]
# gaps = [temp_starts[j+1] - temp_starts[j] for j in range(len(temp_starts)-1)]

# print(temp_starts)
# print(gaps)

# ends = []

# for j, (start, gap) in enumerate(zip(starts, gaps)):
#     mask_len = random.randint(mask_len_min, mask_len_max)
#     # if mask_len > gap * self.args.max_mask_portion: # make sure the masks are not overlapping with each other
#     if mask_len > gap - 1: # make sure the masks are not overlapping with each other
#         # temp_mask_start = int(0.6*gap*self.args.max_mask_portion)
#         # temp_mask_end = int(gap*self.args.max_mask_portion)
#         temp_mask_start = 1
#         temp_mask_end = gap - 1
#         mask_len = random.randint(temp_mask_start, temp_mask_end)
#     ends.append(start + mask_len)

# mask_intervals.append([(s,e) for s,e in zip(starts, ends)])
# non_mask_intervals.append([(ns,ne) for ns, ne in zip([0]+ends, starts+[y_len])])

# print(mask_intervals)
# print(non_mask_intervals)
import torch
a = torch.tensor([[1,2,4,4], [2,5,1,4], [6,1,4,4], [5,1,1,4], [1,5,7,4], [5,3,4,1]], dtype=torch.float)
b = torch.tensor([[5,1,1,4], [1,5,7,4], [5,3,4,1], [1,2,4,4], [2,5,1,4], [6,1,4,4]], dtype=torch.float)

batch_idx = torch.arange(a.size(0))
gen_len = (
    (a[:, 1:] == 4).byte().argmax(dim=-1)
)
print(a * 10)
# print(a[0].ndim)
# print(a.cumsum(dim=-1)+b)
# print(gen_len)
# print(b[batch_idx, gen_len])

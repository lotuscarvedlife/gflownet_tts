import torch

def print_all(*args, **kwargs):
    if args:
        print("位置参数:")
        for i, arg in enumerate(args, start=1):
            print(f"  参数 {i}: {arg} (类型: {type(arg)})")
    if kwargs:
        print("关键字参数:")
        for key, value in kwargs.items():
            print(f"  {key}: {value} (类型: {type(value)})")

a = torch.tensor([[[ 229, 1066, 1861,  196],
         [1470, 1816, 1018,  198],
         [2006, 1392, 1438, 1245],
         [ 494,  929, 1257,  179],
         [ 758,  929,  198,  828],
         [1642, 1025, 1404, 1384]]])

a = [[a[0]]]

# for b in a[0]:
#     print(b.shape)
# print(a.shape)

a = [[0]]
a += [[] for _ in range(2)]
a.append([1])
print_all(a = a)


# from collections import namedtuple
# import typing as tp
# LayoutCoord = namedtuple('LayoutCoord', ['t', 'q'])  # (timestep, codebook index)
# PatternLayout = tp.List[tp.List[LayoutCoord]]
# n_q = 4
# delays = list(range(n_q))
# out: PatternLayout = [[]]
# max_delay = max(delays)
# flatten_first = 0
# empty_initial = 0
# timesteps = 10
# if empty_initial:      # 添加空白初始，[[],[],[]] -> [empty_initial+1, _]
#     out += [[] for _ in range(empty_initial)]
# if flatten_first:      # 在最开始的几个位置中添加单独的位置坐标
#     for t in range(min(timesteps, flatten_first)):
#         for q in range(n_q):
#             out.append([LayoutCoord(t, q)])  # [[], [(0,0)], [(0,1)], [(0,2)], [(0,3)], [(1,0)], ...]

# for t in range(flatten_first, timesteps + max_delay):
#     v = []
#     for q, delay in enumerate(delays):
#         t_for_q = t - delay
#         if t_for_q >= flatten_first:
#             v.append(LayoutCoord(t_for_q, q))
#     out.append(v)

# print(out)
# print(len(out))
import numpy as np
a = np.array([[1, 2, 3], 
                   [4, 5, 6]])
print( a.shape) # (2, 3)
print( a.T.shape) # (3, 2) ，使用 .T 即可

b = np.arange(24).reshape(2, 3, 4)
print(b.shape) # (2, 3, 4)
print(b)
# 使用 np.transpose 指定新顺序
# 原来的轴是 (0, 1, 2) -> 变成 (2, 1, 0)
# 这意味着原来的最后一维变成了第一维，原来的第一维变成了最后一维
transposed_b = np.transpose(b, axes=(2, 1, 0))
print(transposed_b.shape) # (4, 3, 2)
print(transposed_b)#想象成一个魔方翻过来
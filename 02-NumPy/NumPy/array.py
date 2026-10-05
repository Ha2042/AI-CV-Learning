import numpy as np
a=np.array([1,2,3,4,5])
print(a)
print(a.ndim)
b=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(b)
print(b.ndim)

#矩阵乘法
c=np.array([[2,3,6],[1,4,7]])
print(c@b)
print(b@c.reshape(3,2))
import numpy as np
a=np.array([1,2,3,4,5])#一维数组没有行列之分，不存在 shape 为 (, 5) 这样语法的数组
print(a)
print(a.shape)#维度
print(a.reshape(1,5))#默认化为二维
print(a.reshape(5,1))

b=np.array([[1,2],[4,5],[7,8]])
print(b)
print(b.shape)
print(b.reshape(2,3))

d=np.array([1,2,3,4,5,6,7,8,9,10])
# 让 NumPy 自己算有几行，反正必须是 5 列
c= d.reshape(-1, 5)
print(c)
print(c.shape) 

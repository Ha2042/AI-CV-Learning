import numpy as np
a=np.array([1,2,3,4,5])
print(a[2])
a[2]=9
print(a[2])
print(a[:3])#打印索引为0-2（包括2）的元素
print(a[:-1])#最后一个元素不取
print(a[::2])#包括自身，每隔两个取一个

b=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(b[0,:])#取一行
print(b[1:,:2])#“,”前是取2,3行，“,”后是取列1,2

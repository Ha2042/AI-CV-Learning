import numpy as np
a=np.arange(9).reshape(3,3)
print(a)
b=1
print(a+b)#每一个都加b"1"
c = np.arange(3)[::-1].reshape(3,1)
print(a+c)#每一列加列[2,1,0]
d = np.arange(3)[::-1]
print(a+d)#每一行加行[2,1,0]
print(c)
print(d)
print(c+d)
import torch
x = torch.rand(5, 3)
print(x)
print(x[1,:])
print(x[1,2].item())
print(x.shape)
print(x.dtype)
y=torch.rand(5,3)
y.add_(x)#原地操作加法
z=torch.sub(x,y)#减法

#reshape a tensor
r=x.view(15)#一维
print(r)
print(r.size())

p=x.view(-1,5)#列确定，自动分配行
print(p)
print(p.size())

#NumPy<->PyTorch
import numpy as np
a=torch.ones(5)
print(a)
b=a.numpy()
print(b)

c=np.ones(3)
print(c)
d=torch.from_numpy(c)
print(d)

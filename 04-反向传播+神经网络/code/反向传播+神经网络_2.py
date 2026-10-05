import torch
x=torch.tensor([[1.0,2.0],[3.0,4.0]])
y=torch.tensor([[1.0],[0.0]])
w1=torch.tensor([[0.5,0.1],[0.2,0.3]],requires_grad=True)#requires_grad=True->此张量要计算梯度
b1=torch.tensor([0.1,0.2],requires_grad=True)
w2=torch.tensor([[0.4],[-0.2]],requires_grad=True)
b2=torch.tensor([0.1],requires_grad=True)
z1=x@w1+b1
a1=torch.relu(z1)#ReLU
z2=a1@w2+b2
Loss=0.5*torch.mean((z2-y)**2)
print("Loss:",Loss.item())
Loss.backward()
print("w2:\n",w2.grad)
print("b2:\n",b2.grad)
print("w1:\n",w1.grad)
print("b1:\n",b1.grad)
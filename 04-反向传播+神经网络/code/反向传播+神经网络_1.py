import numpy as np
x=np.array([[1.0,2.0],[3.0,4.0]])
y=np.array([[1.0],[0.0]])
w1=np.array([[0.5,0.1],[0.2,0.3]])
b1=np.array([0.1,0.2])
w2=np.array([[0.4],[-0.2]])
b2=np.array([0.1])
lr=0.1
#forward pass
z1=x@w1+b1
a1=np.maximum(0,z1)#ReLU
z2=a1@w2+b2
Loss=0.5*np.mean((z2-y)**2)#mean求平均，Loss是所有损失值的平均
print("forward pass z2:\n",z2)
print("initial loss:\n",Loss)
#backward pass
grad_z2=(z2-y)/x.shape[0]#(z2-y)/N=损失对z2的梯度

grad_w2=a1.T@grad_z2
grad_b2=np.sum(grad_z2,axis=0)
grad_a1=grad_z2@w2.T

grad_z1=grad_a1*(z1>0)#反ReLU

grad_w1=x.T@grad_z1
grad_b1=np.sum(grad_z1,axis=0)
#更新参数
w2-=lr*grad_w2
b2-=lr*grad_b2
w1-=lr*grad_w1
b1-=lr*grad_b1

print("\n更新后的参数:")
print("w2:\n",w2)
print("b2:\n",b2)
print("w1:\n",w1)
print("b1:\n",b1)
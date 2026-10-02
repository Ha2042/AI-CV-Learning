# 04-反向传播+神经网络

# 2026-10-02

# Today I Learned:

- 反向传播算法（numpy实现，梯度，损失值逻辑）
- 函数激活
- 参数更新

# Problem

## 不明白ReLU以及反向ReLU

- 函数ReLU(x)=max(0,x)[x>0->导数=1，x<=0->导数=0]
- 反向ReLU:在前向时被ReLU灭掉的位置（即负数），梯度变为0，不再往前传，正数位置的梯度继续往前传

## 矩阵反向传播中，grad_z2=(z2-y)/x.shape[0]不懂为什么要这样计算

- Loss=0.5*np.mean((z2-y)**2)#loss是均方误差（MSE）,mean求平均，Loss是所有损失值的平均
- grad_z2=(z2-y)/x.shape[0]#(z2-y)/N=损失对z2的梯度（这里的x.shape[0]->batch_size(N)）

# Next

Build an MLP from scratch and train it on MNIST.


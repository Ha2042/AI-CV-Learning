import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class TinyDataset(Dataset):
    def __init__(self):
        # 假设有 20 个样本(房产信息)，每个样本 3 个特征(面积，房间数，距地铁距离)，标签是 1 个数
        self.x = torch.randn(20, 3)          # 特征
        self.y = torch.randn(20, 1)          # 标签

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TinyDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
# 定义模型（nn.Module + forward）
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()                  # 定义网络里有哪些层（这里只用一个线性层）
        self.linear = nn.Linear(3, 1)       # 输入 3 维 → 输出 1 维（根据3个特征预测房价）

    def forward(self, x):
        # 定义数据怎么从输入变成输出
        out = self.linear(x)
        return out

model = TinyModel()
# 定义损失函数 和 优化器
criterion = nn.MSELoss()                    # 均方误差损失（回归常用）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降
# 训练循环（把所有东西串起来）
for epoch in range(5):                      # 训练 5 轮
    for features, labels in dataloader:     # 每次取一个 batch
        # ---- 前向传播 ----
        predictions = model(features)       # 调用 forward
        
        # ---- 计算损失 ----
        loss = criterion(predictions, labels)
        
        # ---- 反向传播 + 更新参数 ----
        optimizer.zero_grad()               # 清空上一次的梯度
        loss.backward()                     # 计算梯度
        optimizer.step()                    # 根据梯度更新模型参数

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
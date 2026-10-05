import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
class pytorch_dataset_demo(Dataset):
    def __init__(self):
        xy=np.loadtxt('pytorch_dataset_demo.csv',delimiter=",",dtype=np.float32,skiprows=1)
        #取所有行（样本）
        self.x=torch.from_numpy(xy[:,1:5])#代表特征，用来做预测的输入数据
        #只取第一列
        # 取第5列（label列）作为标签，或者直接用 xy[:, -1] 取最后一列
        self.y = torch.from_numpy(xy[:, 5]) #代表标签/目标值，预测正确答案
        self.n_samples=xy.shape[0]# row->sample,column->feature/label.返回第一个维度，也是样本的数量
    def __getitem__(self, index):
        return self.x[index],self.y[index]#返回一个元组
    def __len__(self):
        return self.n_samples
    
dataset=pytorch_dataset_demo()
first_data=dataset[0]
features,labels=first_data
print(features,labels)
print("样本数量:", len(dataset))
print("特征维度:", dataset.x.shape)
print("标签维度:", dataset.y.shape)

dataloader=DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=0)#按指定规则打包数据
#可以用 for 循环优雅地替代 iter + next
for features, labels in dataloader:
        print("一个 batch 的特征形状:", features.shape)
        print("一个 batch 的标签形状:", labels.shape)
        break  # 只查看第一个 batch
datatiter=iter(dataloader)#将dataloader转化为一个迭代器
data = next(datatiter)#从迭代器中取出第一个Batch的数据
features,labels=data
print(features,labels)
#for循坏遍历DataLoader
for epoch in range(5):
     for features,labels in dataloader:
          print(features,labels)
#iter=epoch*(samples/batch_size)
#数据集（Dataset） = 一整块披萨（30块）。
#轮次（Epoch） = 吃完一整块披萨（把30块全吃完）。
#批大小（Batch Size） = 你每一口塞进嘴里的披萨数量。
#迭代（Iteration） = 你每嚼一次、咽下去的动作（完成一次参数更新）
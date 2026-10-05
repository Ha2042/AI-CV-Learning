import torch
from torch import  nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d,ReLU,MaxPool2d
train_dataset=torchvision.datasets.CIFAR10("../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
train_dataloader=DataLoader(train_dataset,batch_size=64)
test_dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_dataloader=DataLoader(test_dataset,batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=Conv2d(3,3,3,stride=1,padding=0)#()里三个数字分别代表“输入通道”，“输出通道”，“kernel”
        self.relu1=ReLU()#激活
        self.pool_1=MaxPool2d(2)#最大池化 2×2 池化窗口，池化不改变通道数和批量数。
        self.conv2=Conv2d(3,16,3,stride=1,padding=0)
        self.relu2=ReLU()
        self.pool_2=MaxPool2d(2)
        self.conv3=Conv2d(16,6,3,stride=1,padding=0)
        self.relu3=ReLU()
        self.pool_3=MaxPool2d(2)
        self.linear=nn.Linear(24,10)#相当于把x一维化，但batch_size不改变
    def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool_1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool_2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.pool_3(x)
            x = torch.flatten(x,1)#展平成一维
            x = self.linear(x)
            return x
     
model=CNN()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
#测试
def evaluate(model,test_dataloader):
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for images,labels in test_dataloader:
            output=model(images)
            prediction=torch.argmax(output,dim=1)
            correct+=(prediction==labels).sum().item()#True加1，False加0
            total+=labels.size(0)
        return correct/total
    
#训练
for epoch in range(10):
    model.train()
    epoch_loss=0
    for images,labels in train_dataloader:
        output=model(images)
        loss=criterion(output,labels)
        epoch_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()# 根据梯度更新参数
    avg_loss = epoch_loss / len(train_dataloader)   # len(dataloader) = 总样本数 / batch_size 
    acc = evaluate(model, test_dataloader)
    print(f"Epoch:{epoch+1},Loss:{avg_loss:.4f},Accuracy:{acc:.2%}")

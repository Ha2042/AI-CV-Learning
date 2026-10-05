import torch
from torch import  nn
import torchvision
from torch.utils.data import DataLoader
train_dataset=torchvision.datasets.MNIST("../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
train_dataloader=DataLoader(train_dataset,batch_size=64)
test_dataset=torchvision.datasets.MNIST("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_dataloader=DataLoader(test_dataset,batch_size=64)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,214)
        self.fc2=nn.Linear(214,10)
    def forward(self,x):
        x=x.view(-1,784)#展平
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=MLP()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
def evaluate(model,test_dataloader):
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for images,labels in test_dataloader:
            output=model(images)
            prediction=torch.argmax(output,dim=1)
            correct+=(prediction==labels).sum().item()
            total+=labels.size(0)
        return correct/total
    

for epoch in range(3):
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


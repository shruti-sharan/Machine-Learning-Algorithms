import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import matplotlib.pyplot as plt
import torchvision

#GPU training
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

batch_size = 64
test_batch_size = 64

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Define Model
class FC_Net(nn.Module):
    def __init__(self):
        super(FC_Net,self).__init__()
        self.layer1=nn.Linear(32*32*3,1024,bias=True)
        self.layer2=nn.Linear(1024,512,bias=True)
        self.layer3=nn.Linear(512,256,bias=True)
        self.layer4=nn.Linear(256,128,bias=True)
        self.layer5=nn.Linear(128,64,bias=True)
        self.layer6=nn.Linear(64,32,bias=True)
        self.layer7=nn.Linear(32,10,bias=True)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=F.relu(self.layer6(x))
        x=F.relu(self.layer7(x))
        return x

net=FC_Net()
net.to(device)
lr=1e-4

#Define loss function and Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=lr)

#Training the network
loss_list=[]
for epoch in range(30):
    running_loss=0.0
    A_loss=[]
    for i,data in enumerate(train_loader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)

        optimizer.zero_grad()

        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        #Statistics
        running_loss+=loss.item()
        A_loss.append(loss.item())
        if i%100 ==99:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print("average loss in %d epoch= %.3f" %(epoch +1, np.mean(A_loss)))
    loss_list.append(np.mean(A_loss))
print("overall average loss",np.mean(loss_list))

print('Finish Training')

#Testing the network

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images,labels=images.to(device),labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#Visualise the loss

epoch=list(range(30))
plt.figure()
plt.plot(epoch,losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("MLP with Activation")
plt.show()






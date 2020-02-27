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
import pickle

#GPU Activation
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
    print(len(train))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    print(len(test))
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
print(len(train_loader))
print(len(test_loader))


#define model

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(64,50,kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(50,40,kernel_size=5, stride=1, padding=2)
        self.conv4=nn.Conv2d(40,16,kernel_size=5, stride=2, padding=2)
        self.fc1=nn.Linear(4096,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,10)


    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

net=CNN()
net.to(device)
lr=3e-4

#Define loss function and Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=lr,weight_decay=5e-4)

#Training the network
correct = 0
total = 0
loss_list=[]

for epoch in range(100):
    running_loss=0.0
    A_loss=[]
    for i,data in enumerate(train_loader):
        inputs,labels=data
        inputs,labels=inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        #Statistics
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss+=loss.item()
        A_loss.append(loss.item())
        if i%100 ==99:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    print("average loss in %d epoch = %.3f" %(epoch + 1, np.mean(A_loss)))
    loss_list.append(np.mean(A_loss))
print("overall average loss",np.mean(loss_list))


print('Finish Training')

print('Training Accuracy of the network on the 60000 train images: %d %%' % (
    100 * correct / total))


#Testing the network

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels=images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

epochs=list(range(100))
plt.figure()
plt.plot(epochs,loss_list)
plt.xlabel("No of epochs")
plt.ylabel("Average loss")
plt.title("CNN with Activation")
plt.show()







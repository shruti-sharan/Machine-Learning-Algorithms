#Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data/patrick-data/pytorch/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data/patrick-data/pytorch/data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().view(-1)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().view(-1)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

#defining the model
input_size=28*28
num_classes=1
lr=0.01
model = nn.Linear(input_size, num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)                   #without momentum
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=True)     #with momentum
print(model)
print(model.parameters)


#defining the loss functions

def logistic_regression_loss(model,labels):
  one = torch.Tensor([1])
  temp=model.t()*labels
  temp2=torch.exp(-temp)
  temp3=torch.add(one,temp2)
  loss=torch.mean(torch.log(temp3))
  return loss

def hinge_loss(model,labels):
  zero=torch.Tensor([0])
  loss=torch.mean(torch.max(1 - model.t() * labels, zero))
  return loss

#visualization of loss functions
def visualize(losses,s):
  plt.plot(losses)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title(s)
  plt.show()
    

# Training the Logistic Regression Model
num_epochs=20
total_step=len(train_loader)
losses=[]
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        #Convert labels from 0,1 to -1,1
        labels = Variable(2*(labels.float()-0.5))
        # Forward pass
        optimizer.zero_grad()
        pred = (model(images))
        loss=logistic_regression_loss(pred,labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
    #Print your results every epoch
        if (i+1) % 198 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))
            losses.append(loss.item())
visualize(losses,"Logistic Regression Loss")



# Test the Linear Regression Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    
    pred=torch.sigmoid(model(images))
    
    pred[pred>0.5]=1
    pred[pred<0.5]=0
    
    correct += (pred.view(-1).long()== labels).sum()
    total += images.shape[0]
print('Accuracy of the Logistic Regression model on the test images: %f %%' % (100 * (correct.float() / total)))




# Training the SVM Model
num_epochs=20
total_step=len(train_loader)
losses=[]
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        #Convert labels from 0,1 to -1,1
        labels = Variable(2*(labels.float()-0.5))
        # Forward pass
        optimizer.zero_grad()
        pred = (model(images))
        loss=hinge_loss(pred,labels)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
    #Print your results every epoch
        if (i+1) % 198 == 0:
            print ('Epoch [{}/{}],  Loss: {:.4f}' 
                   .format(epoch+1, num_epochs,  loss.item()))
            losses.append(loss.item())
visualize(losses,"SVM Loss")


# Test the SVM Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    labels = Variable((2*labels)-1)
    pred=model(images)
    ## Put your prediction code here
    pred[pred>0]=1
    pred[pred<0]=-1
    
    correct += (pred.view(-1).long()== labels).sum()
    total += images.shape[0]
print('Accuracy of the SVM model on the test images: %f %%' % (100 * (correct.float() / total)))



    

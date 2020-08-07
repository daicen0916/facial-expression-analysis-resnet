# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:33:16 2020

@author: daicen
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset, DataLoader
import gc

# Training settings
parser = argparse.ArgumentParser(description='CNN on SFEW dataset')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

#%%
'''
shape after first conv and pooling  torch.Size([20, 18, 89, 89])
shape after 2nd conv and pooling  torch.Size([20, 15, 21, 21])
shape after 3rd conv and pooling  torch.Size([20, 12, 9, 9])
shape after 4th conv and pooling  torch.Size([20, 10, 3, 3])
shape after first fully connection  torch.Size([12, 20])
shape after first fully connection  torch.Size([12, 7])
output size  torch.Size([12, 7])
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(5, 8, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(8, 10, kernel_size=5,stride=2)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.bn1 = nn.BatchNorm2d(5)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 7)
        # self.fc3 = nn.Linear(35,7)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn3(self.conv3(x))
        x = F.relu(F.max_pool2d(x, 2))
        # print(x.shape)
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def train(model,train_loader,optimizer, criterion,epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx , (data, target)in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        _,pred = torch.max(torch.softmax(output,dim=1), 1)
        #output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 or batch_idx==len(train_loader)-1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (1+batch_idx) * len(data), len(train_loader.dataset),
                100. * (1+batch_idx) / len(train_loader), loss.item()))

    # train_loss /= len(train_loader)
    print('Train set: Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return train_loss


def test(model,test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx , (data, target)in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).item()
        _,pred = torch.max(torch.softmax(output,dim=1), 1)
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
#%%
if __name__=='__main__':
    
    transform = transforms.Compose([transforms.CenterCrop(360),
                                    # transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    dataset = datasets.ImageFolder('Subset For Assignment SFEW',transform)
    
    kf = KFold(n_splits=5,shuffle = True)
    kf.get_n_splits(dataset)   #split the dataset to 5 folders
    acc_list, loss_list = list(),list()
    x =1
    max_list = list()
    for train_index, valid_index in kf.split(dataset):
        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(dataset, valid_index)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, **kwargs)
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, 
                                  shuffle=True, **kwargs)
        model = Net()
        if args.cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            train_loss = train(model,train_loader,optimizer, criterion,epoch)
            test_acc = test(model,valid_loader,criterion)
            acc_list.append(test_acc)
            loss_list.append(train_loss)
        max_list.append(max(acc_list))
        path = 'model/experiment5_'+str(x)+'.pt'
        torch.save(model.state_dict(),path)
        del model



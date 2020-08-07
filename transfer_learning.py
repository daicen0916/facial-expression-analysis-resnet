# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:05:24 2020

@author: daicen
Running this script will cost 3 hours
"""
#%%
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import  Subset, DataLoader
from torchvision import datasets, models
import time
import os
from prettytable import PrettyTable
import numpy as np
#%%
def train(model,train_loader,epoch,optimizer,args):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx , (data, target)in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = args.criterion(output, target)
        train_loss += loss.item()
        pred = F.softmax(output,dim=1).max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        if (batch_idx+1)%5==0:
            print('Epoch %d, %d batches are trained'%(epoch,batch_idx+1))
            
    train_loss /= len(train_loader)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return train_loss


def test(model,test_loader,args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += args.criterion(output, target).item()
        pred = F.softmax(output,dim=1).max(1, keepdim=True)[1]# get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def build_model (num_hidden):
    resnet = models.resnet18(pretrained=True)
    layers = list(resnet.children())
    for i in range(len(layers) - 3):
        for param in layers[i].parameters():
            param.requires_grad = False

    num_ftrs = resnet.fc.in_features
    
    resnet.fc = nn.Sequential(
                      nn.Linear(num_ftrs, num_hidden), 
                      nn.ReLU(),
                      nn.Dropout(0.2),
                      nn.Linear(num_hidden,7))
    return resnet



def cross_validation(dataset,args,save=False,prefix='',**kwargs):
    args = args
    start = time.time()
    kf = KFold(n_splits=5,shuffle = True)
    kf.get_n_splits(dataset)#split the dataset to 5 folders
    loss_matrix = torch.zeros((5,args.epochs))
    acc_matrix = torch.zeros((5,args.epochs))
    fold_index = 0
    for train_index, valid_index in kf.split(dataset):
        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(dataset, valid_index)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, **kwargs)
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, 
                                  shuffle=True, **kwargs)
        
        model = build_model(args.num_hidden)
        if args.cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(1, args.epochs + 1):
            current = time.time()
            running_time = int(current-start)
            hours, minutes = running_time//3600,(running_time%3600)/60
            print('running %d hours, %.2f minutes'%(hours,minutes))
            print('%d th fold,lr = %.4f,num_hidden = %d'%(fold_index,args.lr,\
                                                          args.num_hidden))
            training_loss = train(model,train_loader,epoch,optimizer,args)
            test_acc = test(model,valid_loader,args)
            loss_matrix[fold_index,epoch-1] = training_loss
            acc_matrix[fold_index,epoch-1] = test_acc
            if save :
                if not os.path.exists(prefix+'/'+str(fold_index)):
                    os.mkdir(prefix +'/'+str(fold_index))
                path = prefix+'/%d/%d.pt'%(fold_index,epoch)
                torch.save(model.state_dict(),path)
        fold_index+=1
    return loss_matrix,acc_matrix
            
    
#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CNN on SFEW dataset')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=120, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--criterion',default = nn.CrossEntropyLoss())
    # parser.add_argument('--log-interval', type=int, default=20, metavar='N',
    #                     help='how many batches to wait before logging training status')
    parser.add_argument('--num_hidden', type=int, default=200, metavar='N',
                        help='how many hidden neurons in fully connected layer')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    dataset = datasets.DatasetFolder('face_images',torch.load,
                                     extensions =tuple('.pt'))
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                [train_size, test_size])
    criterion = nn.CrossEntropyLoss()
#%% select hyper parameter
    lr_list = [0.005,0.001,0.0005]
    hidden_list = [100,200,300]
    indices = ((l,h) for l in range(len(lr_list)) for h in range(len(hidden_list)))
    para_loss = torch.zeros(len(lr_list),3,5,args.epochs)
    para_acc = torch.zeros(len(lr_list),3,5,args.epochs)
    start = time.time()
    for l,h in indices:
        args.lr,args.num_hidden = lr_list[l],hidden_list[h]
        loss_matrix,acc_matrix = cross_validation(dataset,args,**kwargs)
        para_loss[l,h] = loss_matrix
        para_acc[l,h] = acc_matrix
        current = time.time()
        running_time = int(current-start)
        hours, minutes = running_time//3600,(running_time%3600)/60
        print('running %d hours, %.2f minutes'%(hours,minutes))
    
    if not os.path.exists('para_selection'):
        os.mkdir('para_selection')
    torch.save(para_loss,'para_selection/loss_1.pt')
    torch.save(para_acc,'para_selection/acc_1.pt')
    current = time.time()
    running_time = int(current-start)
    hours, minutes = running_time//3600,(running_time%3600)/60
    print('1st selection running %d hours, %.2f minutes'%(hours,minutes))
    
    para_acc = torch.load('para_selection/acc_1.pt')
    max_acc,_= torch.max(para_acc,dim=3)
    selection_res = torch.mean(max_acc,dim=2)
    table = PrettyTable()
    table.field_names = ['lr\\num_hidden']+hidden_list
    for i in range(3):
        table.add_row([lr_list[i]]+ list(np.round(selection_res[i].numpy(),4)))
    print(table)

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:13:20 2020

@author: daicen
"""

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
import math
import os
from transfer_learning import cross_validation

#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CNN on SFEW dataset')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=120, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
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
    parser.add_argument('--num_hidden', type=int, default=100, metavar='N',
                        help='how many hidden neurons in fully connected layer')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    dataset = datasets.DatasetFolder('grayscale_faces',torch.load,
                                     extensions =tuple('.pt'))
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                [train_size, test_size])    

    para_loss = torch.zeros(3,5,args.epochs)
    para_acc = torch.zeros(3,5,args.epochs)
    start = time.time()
    if not os.path.exists('grayscale_model'):
        os.mkdir('grayscale_model')
    for i in range(3):
        prefix = 'grayscale_model/%d'%(i)
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        loss_matrix,acc_matrix = cross_validation(train_dataset,args,save=True,\
                                                  prefix = prefix,grayscale=True,\
                                                  **kwargs)
        para_loss[i] = loss_matrix
        para_acc[i] = acc_matrix
    torch.save(para_loss,'grayscale_model/loss_1.pt')
    torch.save(para_acc,'grayscale_model/acc_1.pt')
    current = time.time()
    running_time = int(current-start)
    hours, minutes = running_time//3600,(running_time%3600)/60
    print('generate 15 models, using %d hours, %.2f minutes'%(hours,minutes))
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:32:31 2020

@author: daicen
running this script will cost 2 hours.
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import  Subset, DataLoader
from torch.nn.utils import prune
from torchvision import datasets, models
import time
import os
from transfer_learning import cross_validation,test,build_model,train,test
import glob
from shutil import copyfile
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
#%% methods for feature analysis

def magnitude_analysis(W_mat1,W_mat2):
    P_ij =torch.div(W_mat1.abs().transpose(0,1),W_mat1.abs().sum(dim=1))
    P_jk =torch.div(W_mat2.abs().transpose(0,1),W_mat2.abs().sum(dim=1))
    Q_ik = P_ij.mm(P_jk)
    
    return Q_ik.sum(dim=1).cpu().detach()

def distinctiveness_analysis(W_mat):
    input_list = np.arange(0,512)
    input_comb = list(combinations(input_list,2))
    
    angle_matrix = torch.zeros(512,512) # result of model W

    w_mat = W_mat.detach().abs()
    if args.cuda:
        w_mat.cuda()
        angle_matrix.cuda()
    norm_vet = torch.norm(w_mat, dim=0)
    #calculate the cosine similarity and the angle
    for inputs in input_comb:
        input_1,input_2 = inputs[0],inputs[1]
        upper = (w_mat[:,input_1].dot(w_mat[:,input_2])).sum()
        lower = norm_vet[input_1]*norm_vet[input_2]
        cos_sim = upper.div(lower)
        cos = torch.acos(cos_sim)*180/3.14 # in degree
        angle_matrix[input_1,input_2] += cos
        angle_matrix[input_2,input_1] += cos #it's a symmetric matrix
    angle_matrix.cpu()
    aggregated = angle_matrix.sum(dim=0) / 511 
    return angle_matrix,aggregated

def prune_model(model,index_list):
    mag_w_mask = torch.zeros(100,512)
    for index in index_list:
        mag_w_mask[:,index] = 1
    module = model.fc[0]    
    pruned_w= prune.custom_from_mask(module, name='weight', mask=mag_w_mask)
    model.fc[0] = pruned_w
    return model

def random_prune_model(model,ratio): 
    module = model.fc[0]    
    pruned_w= prune.random_unstructured(module, name='weight',amount=ratio)#prune.custom_from_mask(module, name='weight', mask=mag_w_mask)
    model.fc[0] = pruned_w
    return model

def retrain_model (model,datasets,optimizer,args):
    retrain_acc = torch.zeros(args.epochs)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, 
                                                [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, 
                                  shuffle=True, **kwargs)    
    for epoch in range(1, args.epochs + 1):       
        training_loss = train(model,train_loader,epoch,optimizer,args)
        test_acc = test(model,valid_loader,args)
        retrain_acc[epoch-1]=test_acc

    return retrain_acc
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
                        help='learning rate (default: 0.005)')
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

    dataset = datasets.DatasetFolder('face_images',torch.load,
                                     extensions =tuple('.pt'))
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                [train_size, test_size])    
    
    test_indices = np.array(test_dataset.indices)    
    np.save('test_indices.npy', test_indices) #backup the split for repeat experiment
    
    para_loss = torch.zeros(3,5,args.epochs)
    para_acc = torch.zeros(3,5,args.epochs)
    start = time.time()
    if not os.path.exists('RGB_model'):
        os.mkdir('RGB_model')
    
    for i in range(3):
        prefix = 'RGB_model/%d'%(i)
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        loss_matrix,acc_matrix= cross_validation(train_dataset,args,save=True,\
                                                  prefix = prefix,**kwargs)
        para_loss[i] = loss_matrix
        para_acc[i] = acc_matrix
    torch.save(para_loss,'RGB_model/loss_1.pt')
    torch.save(para_acc,'RGB_model/acc_1.pt')
    current = time.time()
    running_time = int(current-start)
    hours, minutes = running_time//3600,(running_time%3600)/60
    print('generate 15 models, using %d hours, %.2f minutes'%(hours,minutes))
    
    para_acc= torch.load('RGB_model/acc_1.pt')
#%% select model for further analysis
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, 
                                  shuffle=True, **kwargs)
    path_list = glob.glob('RGB_model/*/*/*.*')
    test_acc = torch.zeros(3,5,15)
    for path in path_list:
        num_run,fold = int(path.split('\\')[1]),int(path.split('\\')[2])
        ep = int((path.split('\\')[-1]).split('.')[0])
        if ep>35:
            model = build_model(args.num_hidden)
            model.cuda()
            model.load_state_dict(torch.load(path))
            acc=test(model,test_loader,args)
            test_acc[num_run,fold,ep-36] = acc

    max_acc,max_arg = torch.max(test_acc,dim=2)
    if not os.path.exists('selected_model'):
        os.mkdir('selected_model')
        
    indices = ((n,f) for n in range(3) for f in range(5))
    args.lr = args.lr/5
    args.epochs=50
    selected_acc = torch.zeros(15)
    index = 0
    for n,f in indices:
        ep = max_arg[n,f]+36
        path = 'RGB_model/%d/%d/%d.pt'%(n,f,ep)
        model = build_model(args.num_hidden)
        if args.cuda:
            model.cuda()
        model.load_state_dict(torch.load(path))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        retrain_model(model,train_dataset,optimizer,args)
        selected_acc[index]=test(model,test_loader,args)
        filename = '%d.pt'%ep
        newfilename = 'selected_model/%d_%d_%d.pt'%(n,f,ep)
        torch.save(model.state_dict(),newfilename)
        index+=1
#%% magnitude analysis
    path_list = glob.glob('selected_model/*.*')
    mag_matrix = torch.zeros(15,512)
    index = 0
    for path in path_list: 
        model = build_model(args.num_hidden)
        model.cuda()
        model.load_state_dict(torch.load(path))
        W_mat1 = model.state_dict().get('fc.0.weight')
        W_mat2 = model.state_dict().get('fc.3.weight')
        mag = magnitude_analysis(W_mat1, W_mat2)
        mag_matrix[index] = mag
        index+=1
        
    mag_mean = torch.mean(mag_matrix,dim=0) #avg mag of 15 models
            
    x= np.arange(0, 512)
    mean = torch.mean(mag_mean)
    std = torch.std(mag_mean)
    l = torch.zeros(512)+mean
    l1 = l+std
    l2 = l-std
    plt.figure()
    plt.plot(mag_mean,'.b',markersize=8)
    plt.plot(l,'k')
    plt.plot(l1,'--k')
    plt.plot(l2,'--k')
    plt.ylabel('avg magnitude')
    plt.suptitle('avg magnitude v.s. features')
    plt.show()
    
    
    index_list = []
    for i in range(512):
        if float(mag_mean[i])>float(mean-std) :
            index_list.append(i)
#%% distinctiveness analysis
    
    path_list = glob.glob('selected_model/*.*')
    angle_res = torch.zeros(15,512,512)
    aggregate_res = torch.zeros(15,512) #functional analysis result
    index = 0
    for path in path_list: 
        start = time.time()
        model = build_model(args.num_hidden)
        model.cuda()
        model.load_state_dict(torch.load(path))
        W_mat1 = model.state_dict().get('fc.0.weight')
        angle_matrix,aggregated = distinctiveness_analysis(W_mat1)
        angle_res[index] = angle_matrix
        aggregate_res[index] = aggregated
        index+=1
        print('for 1 model, use %.2f minutes'%((time.time()-start)/60))
    avg_angle_matrix= torch.mean(angle_res,dim=0)
    avg_angle_matrix[np.where(angle_matrix==0)] = np.nan
    plt.figure()
    plt.imshow(avg_angle_matrix,extent = [0.5,511.5,511.5,0.5])
    plt.xticks(np.arange(0,512))
    plt.yticks(np.arange(0,512))
    plt.colorbar()
    
    avg_model_u = torch.mean(aggregate_res,dim=0) #average 15 models
    angle_mean = torch.mean(avg_model_u)    
    angle_std = torch.std(avg_model_u)
    l_dis = torch.zeros(512)+angle_mean
    l1_dis = l_dis+angle_std
    l2_dis = l_dis-angle_std
    plt.figure()
    plt.plot(avg_model_u,'.b',markersize=8)
    plt.plot(l_dis,'k')
    plt.plot(l1_dis,'--k')
    plt.plot(l2_dis,'--k')
    plt.ylabel('avg angle')
    plt.suptitle('avg angle among 15 models v.s. features')
    
    index_list_dis = []
    for i in range(512):
        if float(avg_model_u[i])>float(angle_mean-angle_std) :
            index_list_dis.append(i)
#%% pruning the network based on magnitude analysis
    mag_pruned_acc = torch.zeros(15)
    
    index = 0
    for path in path_list: 
        model = build_model(args.num_hidden)
        model.load_state_dict(torch.load(path))
        model = prune_model(model,index_list)
        if args.cuda:
            model.cuda()
        test_acc=test(model, test_loader, args)
        mag_pruned_acc[index] = test_acc
        index+=1
    
    mag_pruned_mean = torch.mean(mag_pruned_acc)
    mag_pruned_std = torch.std(mag_pruned_acc)
#%% pruning based on distinctiveness analysis
    dis_pruned_acc = torch.zeros(15)
    
    index = 0
    for path in path_list: 
        model = build_model(args.num_hidden)
        model.load_state_dict(torch.load(path))
        model = prune_model(model,index_list_dis)
        if args.cuda:
            model.cuda()
        test_acc=test(model, test_loader, args)
        dis_pruned_acc[index] = test_acc
        index+=1
    
    dis_pruned_mean = torch.mean(dis_pruned_acc)
    dis_pruned_std = torch.std(dis_pruned_acc)
#%% random pruning the features
    rand_pruned_acc = torch.zeros(15)
    
    index = 0
    for path in path_list: 
        model = build_model(args.num_hidden)
        model.load_state_dict(torch.load(path))
        ratio = 1-(len(index_list)/512)
        model = random_prune_model(model,ratio)
        if args.cuda:
            model.cuda()
        test_acc=test(model, test_loader, args)
        rand_pruned_acc[index] = test_acc
        index+=1
        
    rand_pruned_mean = torch.mean(rand_pruned_acc)
    rand_pruned_std = torch.std(rand_pruned_acc)
    
    fig, ax = plt.subplots()
    ax.plot(selected_acc,'k',linewidth=0.8,label = 'no prune')
    ax.plot(mag_pruned_acc,'+r',label = 'mag prune')
    ax.plot(dis_pruned_acc,'^g', label = 'func prune')
    ax.plot(rand_pruned_acc,'xb',label = 'random prune')
    plt.ylim((80,100))
    plt.ylabel('test accuray(%)')
    plt.suptitle('test accuracy after pruning')
    legend = ax.legend(loc='lower right')
    plt.show()
    
#%% pruning 0-100% features
    index = 0
    ratio_acc = torch.zeros(15,21)
    for path in path_list: 
        for i in range(21):
            model = build_model(args.num_hidden)
            model.load_state_dict(torch.load(path))
            ratio = 0.05*i
            model = random_prune_model(model,ratio)
            if args.cuda:
                model.cuda()
            test_acc=test(model, test_loader, args)
            ratio_acc[index,i] = test_acc
            print('model %d, %d percent features are pruned'%(index, 5*i))
        index+=1
    avg_ratio_acc = torch.mean(ratio_acc,dim=0)
    std_ratio_acc = torch.std(ratio_acc,dim=0)
    x = np.linspace(0,1, num =21)
    plt.figure()
    plt.plot(x,avg_ratio_acc)
    plt.ylim((0,100))
    plt.ylabel('test accuray(%)')
    plt.suptitle('test accuracy v.s. pruning ratio')
 #%%   prune 85-100% features
    index = 0
    small_ratio_acc = torch.zeros(15,16)
    for path in path_list: 
        for i in range(16):
            model = build_model(args.num_hidden)
            model.load_state_dict(torch.load(path))
            #ratio = 0.05*i
            model = random_prune_model(model,(85+i)*0.01)
            if args.cuda:
                model.cuda()
            test_acc=test(model, test_loader, args)
            small_ratio_acc[index,i] = test_acc
            print('model %d, %d percent features are pruned'%(index, 85+i))
        index+=1
    avg_ratio_acc1 = torch.mean(small_ratio_acc,dim=0)
    std_ratio_acc1 = torch.std(small_ratio_acc,dim=0)
    x = np.linspace(0.85,1, num =16)
    plt.figure()
    # for i in range(15):
    #     plt.plot(x,ratio_acc[i])
    plt.plot(x,avg_ratio_acc1)
    plt.ylim((0,100))
    plt.ylabel('test accuray(%)')
    plt.suptitle('test accuracy v.s. pruning ratio')

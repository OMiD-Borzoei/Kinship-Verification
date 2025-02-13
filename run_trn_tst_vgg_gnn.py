
import numpy as np
import time
import pickle

import os
os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.set_default_tensor_type(torch.FloatTensor)
# torch.set_default_dtype(torch.FloatTensor) 
# torch.set_default_device('cuda:0')
# torch.set_default_device('cpu')

# import sys
# sys.path.append('..')
# directory = path.path(__file__).abspath()
# sys.path.append(directory.parent.parent)

from utils_residual_gnn import set_seed, seed_worker, set_device, reset_weights, myoptimizer, Criterion
from utils_residual_gnn import train_residual_vgg, evaluate_metric, detailed_evaluate_metric
from utils_residual_gnn import save_checkpoint, load_checkpoint
# from utils_residual_gnn import features_collate as collate
from utils_residual_gnn import features_collate4metric as collate

from data_loading import KinFeaturetDataset
# from all_transform_compositions import imagenet_test_transform, imagenet_train_transform
# from all_transform_compositions import kin_transform_test, kin_transform_train

# from GNN_mixture import GNN_Mixture
# from GNN_mixture_v2 import GNN_Mixture
# from GNN_mixture_v3 import GNN_Mixture
from GNN_Residual_VGG import GNN_Residual_VGG
from center_loss import CenterLoss

import csv

import settings as st

SEED = 20240804
set_seed(seed=SEED)
DEVICE = set_device()


"""change made by OMiD"""
def write_details(path, fold_num, epoch, train_acc, test_acc, my_loss, BCE, MSEP, MSEN, MSEC, triplet, cross):
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        to_write = []
        
        to_write.append(f'{fold_num}')
        to_write.append(f'{epoch: .0f}')
        
        to_write.append(f'{train_acc*100: .1f}')
        to_write.append(f'{test_acc*100: .0f}')
        
        to_write.append(f'{my_loss: .1f}')
        to_write.append(f'{BCE: .3f}')
        to_write.append(f'{MSEP: .3f}')
        to_write.append(f'{MSEN: .3f}')
        to_write.append(f'{MSEC: .3f}')
        to_write.append(f'{triplet: .3f}')
        to_write.append(f'{cross: .2f}')
        
        
        writer.writerow(to_write)
    file.close()
    
def mean_max_acc(path, printt=False):
    # top_acc_of_each_fold    
    max_test_accs = [0, 0, 0, 0, 0]
    # (first_occurance_of_top_acc, last_occurance_of_top_acc)
    max_test_accs_epochs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    
    with open(path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            fold_num, test_acc = int(row[0]), float(row[3]) 
            if max_test_accs[fold_num-1] < test_acc:
                max_test_accs[fold_num-1] = test_acc
                max_test_accs_epochs[fold_num-1][0] = int(row[1])
                max_test_accs_epochs[fold_num-1][1] = int(row[1])
            
            elif max_test_accs[fold_num-1] == test_acc:
                max_test_accs_epochs[fold_num-1][1] = int(row[1])

    file.close()
    
    if printt:
        print(max_test_accs, '\t', max_test_accs_epochs, '\t\t', sum(max_test_accs)/5)
        
    return (sum(max_test_accs)/5)

def mean_dataset(ls):
    accs = sum([mean_max_acc('detail_'+i+'.csv') for i in ls])
    print('Dataset Accuracy =', accs/len(ls))       


if __name__ == '__main__':
    
    ls = ['father_dau', 'father_son', 'mother_dau', 'mother_son']
    # mean_dataset(ls)
    for csv_metadata_file_path in ls:
        
        csv_metadata_file = csv_metadata_file_path + '.csv'
        detail_path = 'detail_' + csv_metadata_file
        # mean_max_acc(detail_path, printt=True)
        # continue
        # detail_path = 'detail.csv'
        file =  open(detail_path, mode='w', newline='')
        writer = csv.writer(file)
        data = ['fold_num', ' epoch', ' train_acc', ' test_acc', ' my_loss', ' bce', ' msep', ' msen', ' msec', ' triplet', ' cross']
        writer.writerow(data)
        file.close()


        data_path = os.path.join(r'D:\Uni\Term 9\Project\kinship\data')
        

        # csv_metadata_file = 'mother_dau.csv'
        csv_metadata_file_fullpath = os.path.join(data_path, csv_metadata_file)
        kin_type = csv_metadata_file[:-4]

        # features_file = 'resnet-features-2000.npz'
        # features_file = 'residual-features-avgpool-2000.npz'
        # features_file = 'resnet-risidual-640.npz'
        features_file = 'vggface2_resnet_residual_2688.npz'
        # features_file = 'resnet-risidual-640-five-landmarks.npz'
        features_file_fullpath = os.path.join(data_path, features_file)

        n_folds = 5
        n_epochs = st.EPOCHS#200
        batch_size = st.BATCH_SIZE#64#16#2#5#25# 10
        num_landmark = 9
        # num_landmark = 5

        for fold_num in range (5, 0, -1):#(5, 0, -1):#range(1,n_folds+1):#range(1,2):#range(1,n_folds+1):#range(4,6):#range(3,4):#range(1,n_folds+1):#range(1,2):#range(3,4):#
            
            """Change made by omid"""
            bce_weight = st.BCE_STARTING_WEIGHT        # 1
            ccl_weight = st.CCL_STARTING_WEIGHT     # 0.01
            
            bce_decay_rate = st.BCE_WEIGHT_DECAY    # 1  
            ccl_decay_rate = st.CCL_DECAY_RATE  #1.1 gave 0.95

            max_acc = 0.5

            # datasets
            trn_set = KinFeaturetDataset(csv_metadata_file_fullpath, features_file_fullpath,
                                    isTrain=True, fold_num=fold_num)
        
            # print(trn_set[0])
            trn_dataloader = DataLoader(trn_set, batch_size=batch_size, shuffle=True, collate_fn=collate, worker_init_fn=seed_worker)
            # one_datapoint = next(iter(trn_dataloader))
            # print(one_datapoint)
            
            tst_set = KinFeaturetDataset(csv_metadata_file_fullpath, features_file_fullpath,
                                        isTrain=False, fold_num=fold_num)
            
            tst_dataloader = DataLoader(tst_set, batch_size=batch_size, shuffle=False, collate_fn=collate, worker_init_fn=seed_worker)
        
            # model = GNN_Mixture() 
            # model = GNN_Metric() 
            input_dim = 2688
            output_dim= 1
            model = GNN_Residual_VGG(input_dim, output_dim)

        #     # ## setting model's parameters
        #     # model.apply(reset_weights)
            model.to(DEVICE)
        #     model.to(DEVICE)
        #     #.to(torch.float64)
            loss = nn.CrossEntropyLoss()
        #     # loss = nn.BCELoss()

            criterion = Criterion()
        #     # opt_parameters = {'lr':1e-2}
            # opt_parameters = {'lr':1e-3}
            opt_parameters = {'lr':1e-4}
        #     # opt_parameters = {'lr':1e-5}
            opt_parameters['weight_decay']=0
            
            optimizer = myoptimizer(model, opt_parameters)
            
            lr_milestones = [ 20, 40, 60,90, 120, 150, 200]
            lr_decay = 0.5 ## learning rate decay
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay)

            for epoch in range(n_epochs):#for epoch in range(5):

                start = time.time()
                model.isTrain = True
                # train_loss, train_acc = train_mixture(model, trn_dataloader, loss, optimizer, DEVICE)
                """Change made by OMID"""
                
                
                train_loss, train_acc, my_loss, bce, msep, msen, msec, triplet, cross = train_residual_vgg(model, trn_dataloader, criterion, optimizer, DEVICE, ccl_weight, bce_weight, epoch)
                model.isTrain = False
                
                """Change made by OMID"""
                ccl_weight *= ccl_decay_rate
                bce_weight *= bce_decay_rate
                
                            
                test_loss, test_acc = evaluate_metric(model, tst_dataloader, criterion, DEVICE)
                scheduler.step()
                
                """change made by OMiD"""
                write_details(detail_path, fold_num, epoch+1, train_acc, test_acc, my_loss, bce, msep, msen, msec, triplet, cross)
                
                
                # if (test_acc > max_acc):
                #     max_acc = test_acc
                #     # torch.save(model, SAVE_PATH)
                #     FILENAME = 'model_residual_vgg_'+str(kin_type)+'_fold_'+str(fold_num)+'_batch_size'+str(batch_size)+'_epochs'+str(n_epochs)+f'_{test_acc*100:.0f}.pt'
                #     SAVE_PATH = os.path.join('CheckPoints', FILENAME)
                #     save_checkpoint(model, optimizer, SAVE_PATH, epoch, kin_type, batch_size, fold_num)
                #print(f'{time.time()-start:.2f} seconds')
        
        mean_max_acc(detail_path, printt=True)
    
    mean_dataset(ls)
                 


        
# import os
from dgl.data.utils import Subset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl

import numpy as np
# from mlp_soft import MLP_Parent_Child_High_Dim as MLP_layer
from mlp_soft import MLP_Parent_XenterLoss as MLP_layer
# from mlp_soft import MLP_Parent_Child as MLP_layer

# from mlp_hard import MLP_Rich as MLP_layer
# from mlp_hard import MLP_Sigmo as MLP_layer

# from self_attention import SelfAttention

class ConvolutionGGN_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

        # self.dropout1d_h = nn.Dropout()
        # self.dropout1d_e = nn.Dropout(0.8)

        self.act_fn = nn.ReLU()

    def message_func(self, edges):
        Bx_j = edges.src['BX']
        # e_j = Ce_j + Dxj + Ex
        e_j = edges.data['CE'] + edges.src['DX'] + edges.dst['EX']
        edges.data['E'] = e_j
        return {'Bx_j' : Bx_j, 'e_j' : e_j}

    def reduce_func(self, nodes):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        e_j = nodes.mailbox['e_j']
        # sigma_j = σ(e_j)
        # σ_j = torch.sigmoid(e_j)

        # σ_j = torch.tanh(e_j)
        # h = Ax + Σ_j η_j * Bxj
        # h = Ax + torch.sum(σ_j * Bx_j, dim=1) / (torch.sum(σ_j, dim=1)+np.finfo(np.float32).eps)  #epsilon = sys.float_info.epsilon
        h = Ax + torch.sum(e_j * Bx_j, dim=1) #/ (torch.sum(σ_j, dim=1)+np.finfo(np.float32).eps)  #epsilon = sys.float_info.epsilon
        
        h = self.act_fn(h)

        return {'H' : h}
    
    def forward(self, g, X, E_X):

        g.ndata['H']  = X
        g.ndata['AX'] = self.A(X) 
        #t=g.ndata['AX']
        g.ndata['BX'] = self.B(X) 
        g.ndata['DX'] = self.D(X)
        g.ndata['EX'] = self.E(X) 
        g.edata['E']  = E_X
        g.edata['CE'] = self.C(E_X)
        
        #print(f',  forward_func AX_shape: {t.size()}, AX_value: {t}')
        
        g.update_all(self.message_func, self.reduce_func)
        
        H = g.ndata['H'] # result of graph convolution
        E = g.edata['E'] # result of graph convolution

        # H *= snorm_n # normalize activation w.r.t. graph node size
        # E *= snorm_e # normalize activation w.r.t. graph edge size
        
        H = self.bn_node_h(H) # batch normalization  
        E = self.bn_node_e(E) # batch normalization  

        # H = self.dropout1d_h(H) # drop out
        # E = self.dropout1d_e(E) # drop out for edges

        # H = torch.relu(H) # non-linear activation
        # E = torch.relu(E) # non-linear activation

        H = X + H # residual connection
        E = E_X + E # residual connection

        H = torch.relu(H) # non-linear activation
        E = torch.relu(E) # non-linear activation

        return H, E

class ConvolutionGGN(nn.Module):
    

    # def __init__(self, input_dim, hidden_dim, output_dim, L, readout, double_residual):
    def __init__(self, input_dim, output_dim):        
        super().__init__()
        # super(GatedGCN,self).__init__()

        self.input_feature_dim = input_dim
        self.output_dim = output_dim

        h_dim_list = []
        layer_size = input_dim
        while(layer_size>=32):
            h_dim_list.append(layer_size)
            layer_size //=4
        
        e_dim_list = h_dim_list.copy()
        h_dim_list.insert(0,self.input_feature_dim)
        e_dim_list.insert(0,1)

        self.h_dim_list = h_dim_list
        self.e_dim_list = e_dim_list
                
        self.embedding_h_layers = nn.ModuleList([
                nn.Linear( self.h_dim_list[idx], self.h_dim_list[idx+1]) for idx in range(len(self.h_dim_list)-1)
        ])
        self.embedding_e_layers = nn.ModuleList([
                nn.Linear( self.e_dim_list[idx], self.e_dim_list[idx+1]) for idx in range(len(self.e_dim_list)-1)
        ])
        self.ConvolutionGGN_layers = nn.ModuleList([
            ConvolutionGGN_layer(self.h_dim_list[idx+1], self.h_dim_list[idx+1]) for idx in range(len(self.h_dim_list)-1)
        ])
       
        # self.GatedGCN_layers = nn.ModuleList([
        #     ConvolutionGGN_layer(hidden_dim, hidden_dim) for _ in range(L)
        # ])


        self.mlp_input_dim_parent = sum(self.h_dim_list[1:])
        self.mlp_input_dim_child = sum(self.h_dim_list[1:])

        self.MLP_layer = MLP_layer(self.mlp_input_dim_parent, self.mlp_input_dim_child, self.output_dim)
        # self.dropout1d = nn.Dropout()
        
    def forward(self, g, X, E):
        
        H = X 
        # H = Xnew
        mean_lst = []     
        entire_lst = []  

        parent_gnn_feature_lst = []
        child_gnn_feature_lst = []

        for i in range(len(self.h_dim_list)-1):

            H = self.embedding_h_layers[i](H)
            E = self.embedding_e_layers[i](E)
            H, E = self.ConvolutionGGN_layers[i](g, H, E)

            num_features = H.shape[1]
            
            parent = H.reshape(g.batch_size,-1, num_features)[:,0:-1:2,:]
            parent = torch.mean(parent, dim=1)

            child = H.reshape(g.batch_size,-1, num_features)[:,1:-1:2,:]
            child = torch.mean(child, dim=1)

            parent_gnn_feature_lst.append(parent)
            child_gnn_feature_lst.append(child)

        # parent_mean = torch.cat((parent_gnn_feature_lst[0], parent_gnn_feature_lst[1],parent_gnn_feature_lst[2],parent_gnn_feature_lst[3],parent_gnn_feature_lst[4]), dim=1)
        # child_mean = torch.cat((child_gnn_feature_lst[0], child_gnn_feature_lst[1], child_gnn_feature_lst[2], child_gnn_feature_lst[3], child_gnn_feature_lst[4]), dim=1)
        parent_mean = torch.cat((parent_gnn_feature_lst[0], parent_gnn_feature_lst[1],parent_gnn_feature_lst[2],parent_gnn_feature_lst[3]), dim=1)
        child_mean = torch.cat((child_gnn_feature_lst[0], child_gnn_feature_lst[1], child_gnn_feature_lst[2], child_gnn_feature_lst[3]), dim=1)
        # parent_mean = torch.cat((parent_gnn_feature_lst[0], parent_gnn_feature_lst[1]), dim=1)
        # child_mean = torch.cat((child_gnn_feature_lst[0], child_gnn_feature_lst[1]), dim=1)


        y, center_feature = self.MLP_layer(parent_mean, child_mean)
        
        # if np.isnan(y.detach().numpy().any()):
        #     print(f'after mlp yyyyyy: {y}')

        # # print(y.shape)
        
        return y, parent_mean, child_mean, center_feature




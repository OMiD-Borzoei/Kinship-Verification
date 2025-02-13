from random import choice
import numpy as np
import networkx as nx
# import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from torchvision import models
# from torchvision.models import ResNet18_Weights

# os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
import dgl
from dgl.data.utils import Subset
import settings as st

# from patch_generate import PatchGenerate
# from ConvolutionGN_att_self import ConvolutionGGN
# from ConvolutionGN import ConvolutionGGN
# from ConvolutionGN_graph2 import ConvolutionGGN
from ConvolutionGN import ConvolutionGGN

from mlp_soft import MLP_layer as MLP_layer_id

class GNN_Residual_VGG(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:#__init__(self, *args, **kwargs) -> None: #__init__(self, batch_size, num_landmarks, *args, **kwargs) -> None: #
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.feature = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        # self.feature = nn.Sequential(*list(self.feature.children())[:-1])

        # self.ml = nn.Linear(5,10)
        # self.gnn_model = ConvolutionGGN(input_dim=128, output_dim=1) 
        self.gnn_model = ConvolutionGGN(self.input_dim, self.output_dim) 
        # self.gnn_model = ConvolutionGGN(input_dim=512, hidden_dim=64, output_dim=1, L=2, readout='mean', double_residual=False) 
        # self.gnn_model = ConvolutionGGN(input_dim=16*512, hidden_dim=64, output_dim=1, L=2, readout='mean', double_residual=False) 
        # self.num_samples = batch_size
        # self.num_landmarks = num_landmarks
        # self.n_nodes = 2 * self.num_landmarks
        # self.device = device

        self.list_graph_creation = [func for func in dir(GNN_Residual_VGG) if callable(getattr(GNN_Residual_VGG, func)) and func.startswith("create_graph_")]
        # self.num_graphs = len(self.list_graph_creation)
        # self.isTrain = True

        self.part = 4#2#4#3
        member_features_list = []
        layer_size = self.input_dim
        while(layer_size>=32):
            member_features_list.append(layer_size)
            layer_size //=4
        member_features_dim = sum(member_features_list)

        MLP_input_dim = member_features_dim // self.part
        self.id_layers = nn.ModuleList([
            MLP_layer_id(MLP_input_dim, st.config[st.I]["OUTPUTDIM"]) for _ in range(self.part) #MLP_layer_id(160, 400) for _ in range(self.part)
        ])

    def create_graph_type_1(self, n_nodes): #pair_graph_conseq
        
        Adj = np.zeros( (n_nodes, n_nodes) )

        for node in np.arange(0, n_nodes, 2):
            Adj[node,node+1] = 1
            Adj[node+1, node]= 1

        Graph = dgl.from_networkx(nx.from_numpy_array(Adj))

        return Graph 
    
    #two_entire_nodes_connected
    def create_graph_type_2(self, n_nodes):

        Adj = np.zeros( (n_nodes, n_nodes) )

        Adj[0,1] = 1
        Adj[1,0] = 1

        for node in np.arange(2, n_nodes, 2):
            Adj[node,0] = 1
            Adj[0,node] = 1
            Adj[node+1, 1] = 1
            Adj[1, node+1] =1

        Graph = dgl.from_networkx(nx.from_numpy_array(Adj))

        return Graph 
    
    #two_complete_graphs_connected
    def create_graph_type_3(self, n_nodes):
        Adj = np.zeros( (n_nodes, n_nodes) )
        Adj[0,1] = 1
        Adj[1,0] = 1

        Adj[0:n_nodes:2, 0:n_nodes:2] = 1
        Adj[1:n_nodes:2, 1:n_nodes:2] = 1

        for node in np.arange(n_nodes):
            Adj[node,node] = 0

        Graph = dgl.from_networkx(nx.from_numpy_array(Adj))

        return Graph
    
    # one_complete
    def create_graph_type_4(self, n_nodes):
        Adj = np.ones( (n_nodes, n_nodes) )

        for node in np.arange(n_nodes):
            Adj[node,node] = 0

        Graph = dgl.from_networkx(nx.from_numpy_array(Adj))

        return Graph

    # two_complete_graphs_connected_pairnodes_connected
    def create_graph_type_5(self, n_nodes):
        Adj = np.zeros( (n_nodes, n_nodes) )

        Adj[0:n_nodes:2, 0:n_nodes:2] = 1
        Adj[1:n_nodes:2, 1:n_nodes:2] = 1



        for node in np.arange(0,n_nodes,2):
            Adj[node,  node+1] = 1
            Adj[node+1, node]  = 1

        for node in np.arange(n_nodes):
            Adj[node, node] = 0

        Graph = dgl.from_networkx(nx.from_numpy_array(Adj))

        return Graph

    
    def score_random_graph(self, x1_batch, x2_batch):
        [num_samples, num_landmarks, num_features] = x1_batch.shape
        num_nodes = 2 * num_landmarks 
        device = x1_batch.device   

        list_graph = []        
        for i in range(num_samples):
            create_graph_function_name = choice(self.list_graph_creation)
            create_graph = getattr(self, create_graph_function_name)
            graph = create_graph(num_nodes).to(device)
            # graph5 = self.creat_graph_two_complete_graphs_connected_pairnodes_connected(num_nodes).to(device)
            parent_features = x1_batch[i,:,:] #self.give_features_of_parent(i, x1_lst)
            child_features  = x2_batch[i,:,:] #self.give_features_of_parent(i, x2_lst)
            # node_features = torch.cat( (torch.stack(parent_features),torch.stack(child_features)),  dim=0)
            # node_features = torch.cat( (torch.stack(parent_features),torch.stack(child_features)),  dim=1)
            node_features = torch.cat( ( parent_features, child_features),  dim=0)
            graph.ndata['feat'] = node_features#.reshape([-1,num_features])
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1).to(device)
            list_graph.append(graph)
        
        batch_graphs = dgl.batch(list_graph)
        batch_X2 = batch_graphs.ndata['feat']
        batch_E2 = batch_graphs.edata['feat']

        batch_scores = self.gnn_model(batch_graphs, batch_X2, batch_E2)
        
        return batch_scores
    
    def prepare_data_graph(self, x1_batch, x2_batch, create_graph_func):

        [num_samples, num_landmarks, num_features] = x1_batch.shape
        num_nodes = 2 * num_landmarks 
        device = x1_batch.device

        # construct 10 graphs for pair(x1,x2)
        list_graphs = []        
        for i in range(num_samples):
            
            create_graph = getattr(self, create_graph_func)
            graph = create_graph(num_nodes).to(device)
            # graph = self.create_garph_type_1(num_nodes).to(device)

            parent_features = x1_batch[i,:,:] #self.give_features_of_parent(i, x1_lst)
            child_features  = x2_batch[i,:,:] #self.give_features_of_parent(i, x2_lst)
            # node_features = torch.cat( (torch.stack(parent_features),torch.stack(child_features)),  dim=0)
            # node_features = torch.cat( (torch.stack(parent_features),torch.stack(child_features)),  dim=1)
            node_features = torch.cat( ( parent_features, child_features),  dim=1)
            graph.ndata['feat'] = node_features.reshape([-1,num_features])
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1).to(device)
            list_graphs.append(graph)
        
        batch_graphs = dgl.batch(list_graphs)
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        return batch_graphs, batch_X, batch_E

    def score_graph(self, x1_batch, x2_batch, create_graph_func):

        [num_samples, num_landmarks, num_features] = x1_batch.shape
        num_nodes = 2 * num_landmarks 
        device = x1_batch.device

        # construct 10 graphs for pair(x1,x2)
        list_graphs = []        
        for i in range(num_samples):
            
            create_graph = getattr(self, create_graph_func)
            graph = create_graph(num_nodes).to(device)
            # graph = self.create_garph_type_1(num_nodes).to(device)

            parent_features = x1_batch[i,:,:] #self.give_features_of_parent(i, x1_lst)
            child_features  = x2_batch[i,:,:] #self.give_features_of_parent(i, x2_lst)
            # node_features = torch.cat( (torch.stack(parent_features),torch.stack(child_features)),  dim=0)
            # node_features = torch.cat( (torch.stack(parent_features),torch.stack(child_features)),  dim=1)
            node_features = torch.cat( ( parent_features, child_features),  dim=1)
            graph.ndata['feat'] = node_features.reshape([-1,num_features])
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1).to(device)
            list_graphs.append(graph)
        
        batch_graphs = dgl.batch(list_graphs)
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        batch_scores, f_parent, f_child, center_feature = self.gnn_model(batch_graphs, batch_X, batch_E)

        return batch_scores, f_parent, f_child, center_feature

    def forward(self, x1_batch, x2_batch):
      
        selection_create_graph_func = 'create_graph_type_1'
        # selection_create_graph_func = 'create_graph_type_2'
        # selection_create_graph_func = 'create_graph_type_3'
        # selection_create_graph_func = 'create_graph_type_4'
        # selection_create_graph_func = 'create_graph_type_5'
        y, f_parent, f_child, center_feature = self.score_graph(x1_batch, x2_batch, selection_create_graph_func)
        # batch_graphs, batch_X, batch_E = self.prepare_data_graph(x1_batch, x2_batch, selection_create_graph_func)
        # y, f_parent, f_child = self.gnn_model(batch_graphs, batch_X, batch_E)

        f = torch.cat((f_parent, f_child), dim=0)
        # f = self.part_avgpool(f)
        f = f.view(f.size(0), -1, self.part)
        # f = f.view(f.size(0), -1, 2, 1)
        # part = {}
        # predict = {}
        # # get six part feature batchsize*2048*6
        # for i in range(self.part):
        #     part[i] = torch.squeeze(f[:, :, i])
        #     predict[i] = self.id_layers[i](part[i])
        # family_id = []
        # for i in range(self.part):
        #     family_id.append(predict[i])

        part = {}
        predict_family_id_dict = {}
        # predict_family_id_dict[0] = self.id_layers[0](f)
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            # part[i] = torch.squeeze(f[:, :, i])
            part[i] = torch.squeeze(f[:, :, i])
            predict_family_id_dict[i] = self.id_layers[i](part[i])


        # if(self.isTrain):
        #     batch_scores = self.score_random_graph(x1_batch, x2_batch)
        # else:
        #     batch_scores = self.score_graph(x1_batch, x2_batch, selection_create_graph_func)
        return y, f_parent, f_child, predict_family_id_dict, center_feature
        # return batch_score
        # return y

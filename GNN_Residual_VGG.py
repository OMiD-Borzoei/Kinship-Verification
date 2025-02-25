from random import choice
import numpy as np
import networkx as nx
import torch
from torch import Tensor
import torch.nn as nn

import dgl
from dgl.heterograph import DGLGraph
import settings as st

from ConvolutionGN import ConvolutionGGN

from mlp_soft import MLP_layer as MLP_layer_id


class GNN_Residual_VGG(nn.Module):

    def __init__(self, input_dim, output_dim, model_type:tuple) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_model = ConvolutionGGN(self.input_dim, self.output_dim)
        self.model_type = model_type

        self.list_graph_creation = [func for func in dir(GNN_Residual_VGG) if callable(
            getattr(GNN_Residual_VGG, func)) and func.startswith("create_graph_")]

        self.part = 4
        member_features_list = []
        layer_size = self.input_dim
        while (layer_size >= 32):
            member_features_list.append(layer_size)
            layer_size //= 4
        member_features_dim = sum(member_features_list)

        MLP_input_dim = member_features_dim // self.part
        self.id_layers = nn.ModuleList([
            MLP_layer_id(MLP_input_dim, st.config[st.I]["OUTPUTDIM"]) for _ in range(self.part)
        ])

    def create_graph_type_1(self, n_nodes: int):  # pair_graph_conseq

        Adj = np.zeros((n_nodes, n_nodes))

        for node in np.arange(0, n_nodes, 2):
            Adj[node, node+1] = 1
            Adj[node+1, node] = 1

        Graph = dgl.from_networkx(nx.from_numpy_array(Adj))

        return Graph

    def score_random_graph(self, x1_batch: Tensor, x2_batch: Tensor):
        num_samples, num_landmarks, _ = x1_batch.shape
        num_nodes = 2 * num_landmarks
        device = x1_batch.device

        list_graph = []
        for i in range(num_samples):
            create_graph_function_name = choice(self.list_graph_creation)
            create_graph = getattr(self, create_graph_function_name)
            graph: DGLGraph = create_graph(num_nodes).to(device)
            parent_features = x1_batch[i, :, :]
            child_features = x2_batch[i, :, :]
            node_features = torch.cat(
                (parent_features, child_features),  dim=0)
            graph.ndata['feat'] = node_features  # .reshape([-1,num_features])
            graph.edata['feat'] = torch.ones(
                graph.number_of_edges(), 1).to(device)
            list_graph.append(graph)

        batch_graphs = dgl.batch(list_graph)
        batch_X2 = batch_graphs.ndata['feat']
        batch_E2 = batch_graphs.edata['feat']

        batch_scores = self.gnn_model(batch_graphs, batch_X2, batch_E2)

        return batch_scores

    def prepare_data_graph(self, x1_batch: Tensor, x2_batch: Tensor, create_graph_func: str):
        [num_samples, num_landmarks, num_features] = x1_batch.shape
        num_nodes = 2 * num_landmarks
        device = x1_batch.device
        list_graphs = []
        for i in range(num_samples):

            create_graph = getattr(self, create_graph_func)
            graph: DGLGraph = create_graph(num_nodes).to(device)
            parent_features = x1_batch[i, :, :]
            child_features = x2_batch[i, :, :]
            node_features = torch.cat(
                (parent_features, child_features),  dim=1)
            graph.ndata['feat'] = node_features.reshape([-1, num_features])
            graph.edata['feat'] = torch.ones(
                graph.number_of_edges(), 1).to(device)
            list_graphs.append(graph)

        batch_graphs = dgl.batch(list_graphs)
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        return batch_graphs, batch_X, batch_E

    def score_graph(self, x1_batch: Tensor, x2_batch: Tensor, create_graph_func: str):
        num_landmarks = 1
        [num_samples, _, num_features] = x1_batch.shape
    
        num_nodes = 2 * num_landmarks
        device = x1_batch.device

        list_graphs = []
        for i in range(num_samples):

            create_graph = getattr(self, create_graph_func)
            graph: DGLGraph = create_graph(num_nodes).to(device)

            parent_features = x1_batch[i, self.model_type[1], :]
            child_features = x2_batch[i, self.model_type[1], :]
            
            parent_features = parent_features.unsqueeze(0) 
            child_features = child_features.unsqueeze(0)
            
            node_features = torch.cat(
                (parent_features, child_features),  dim=1)
            graph.ndata['feat'] = node_features.reshape([-1, num_features])
            graph.edata['feat'] = torch.ones(
                graph.number_of_edges(), 1).to(device)
            list_graphs.append(graph)

        batch_graphs = dgl.batch(list_graphs)
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        batch_scores, f_parent, f_child, center_feature = self.gnn_model(
            batch_graphs, batch_X, batch_E)

        return batch_scores, f_parent, f_child, center_feature

    def forward(self, x1_batch: Tensor, x2_batch: Tensor):

        selection_create_graph_func = 'create_graph_type_1'
        y, f_parent, f_child, center_feature = self.score_graph(
            x1_batch, x2_batch, selection_create_graph_func)

        f = torch.cat((f_parent, f_child), dim=0)
        f = f.view(f.size(0), -1, self.part)
        part = {}
        predict_family_id_dict = {}
        for i in range(self.part):
            part[i] = torch.squeeze(f[:, :, i])
            predict_family_id_dict[i] = self.id_layers[i](part[i])

        return y, f_parent, f_child, predict_family_id_dict, center_feature

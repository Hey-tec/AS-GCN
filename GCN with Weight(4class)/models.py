import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa


class ResGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, edge_weight):
        super(ResGCN, self).__init__()
        skip = True
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.blocks = []

        self.block1 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        # self.block2 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        # self.block3 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        # self.block4 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        # self.block5 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        # self.block6 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        # self.block7 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)

        self.final = GCNConv(in_channels, out_channels, normalize=True)

        self.MLPPredictor = MLPPredictor(4, 4)

    def forward(self, data):
        x = data.x.float()

        x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        x = self.final(x, self.edge_index, self.edge_weight)
        # x = F.log_softmax(x, dim=1)
        return self.MLPPredictor(x)

class MLPPredictor(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = torch.nn.Linear(in_features * 2, out_classes)
        AS_List = []
        AS_file = open("data/file_2017_04_01/AS_Features.txt")
        for line in AS_file:
            num_list = line.strip().split()
            AS_List.append(num_list[0])
        self.AS_List = AS_List
        edge_list = []
        with open("data/file_2017_04_01/line_graph_edge.txt") as f:
            for edge in f:
                edges = edge.strip().split()
                edge_list.append([self.AS_List.index(edges[0]), self.AS_List.index(edges[1])])
        self.edge_list = edge_list

    # graph 是整个网络图      h 是 GNN模型的输出
    def forward(self, h):
        output = []
        for edge in self.edge_list:
            # dot = h[edge[0]] * h[edge[1]] * torch.Tensor([-1, -1, -1])
            dot = self.W(torch.cat([h[edge[0]], h[edge[1]]])).to("cuda:0")
            dot = F.log_softmax(dot, dim=0)
            output.append(dot)
        return (torch.cat(output, 0)).reshape(len(self.edge_list), -1)

class ResGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, edge_weight, use_identity=True, block_size=1, use_gcd=False):
        super(ResGCNBlock, self).__init__()

        self.use_identity = use_identity
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True).cuda()
        self.conv2 = GCNConv(in_channels, in_channels, normalize=True).cuda()
        self.conv3 = GCNConv(in_channels, in_channels, normalize=True).cuda()

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, self.edge_index, self.edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, self.edge_index, self.edge_weight))
        x = F.normalize(x)
        if self.use_identity:
            x = (x + identity)/2
        return x




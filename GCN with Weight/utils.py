# Author : Pey
# Time : 2021/4/9 10:45
# File_name : utils.py

# --------- Import Model ---------#
import random
import torch
import numpy as np
from torch_geometric.data import Data
# --------- Sub Function ---------#
def normalize(data, std_norm=False):
    '''
    Normalizes data between 0 and 1
    '''

    if std_norm:
        data_norm = (data - np.mean(data))/np.std(data)
    else:
        max_dist = np.max(data)
        min_dist = np.min(data)

        data_norm = (data - min_dist)/(max_dist - min_dist)

    return data_norm

def get_node_features(data_parents_path):
    features = []
    with open(data_parents_path + "AS_Features.txt", "rb") as file:
        for line in file:
            ASes = line.strip().split()
            AS_features = []
            for i in range(len(ASes)-1):
                AS_features.append(float(ASes[i+1]))
            features.append(AS_features)

    features = np.array(features)
    # Normalize each column individually
    columns = features.shape[1]
    for i in range(columns):
        features[:, i] = normalize(features[:, i], std_norm=True)

    return np.array(features)

def get_data(data_file):
    with open(data_file + 'line_graph_edge_label.txt', 'rb') as file:
        edges = []
        y = []
        for line in file:
            src, dst, rel = line.strip().split()
            edges.append((int(src.decode('utf-8')), int(dst.decode('utf-8'))))
            y.append(int(rel.decode('utf-8')))

    with open(data_file + "line_weight.txt", "rb") as file:
        edge_ratio = []
        for line in file:
            src, dst, ratio = line.strip().split()
            edge_ratio.append(float(ratio.decode('utf-8')))
    # edge_weights = deal_edge_weight(edge_ratio)
    edge_weights = edge_ratio

    # Create train test valid masks
    np.random.seed(0)
    num_nodes = len(y)
    random_order = np.array(range(num_nodes))
    train_percent = 0.8
    valid_percent = 0.1
    train_nodes = int(train_percent * num_nodes)
    valid_nodes = int(valid_percent * num_nodes)
    train_mask, valid_mask, test_mask = np.zeros(num_nodes, dtype=bool), np.zeros(num_nodes, dtype=bool), np.zeros(num_nodes, dtype=bool)
    train_mask[random_order[:train_nodes]] = True
    valid_mask[random_order[train_nodes:train_nodes+valid_nodes]] = True
    test_mask[random_order[train_nodes+valid_nodes:]] = True

    edge_index = torch.tensor(edges, dtype=torch.long)
    # y = np.array(y).reshape(-1, 1)
    y = np.array(y)
    node_values = get_node_features(data_file)
    x = torch.tensor(node_values, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    data = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_weights, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    print(data.x.shape[1])
    print(len(data.y[data.train_mask]))
    print(len(data.y[data.test_mask]))
    print(len(data.y[data.valid_mask]))

    return data

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# --------- Main Function --------#
if __name__ == "__main__":
    print("Start coding...")
    get_data("./data/file_2017_04_01/")

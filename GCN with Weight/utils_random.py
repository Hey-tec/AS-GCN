# Author : Pey
# Time : 2021/4/9 10:45
# File_name : utils_random.py

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

def get_data(data_parents_path):
    with open(data_parents_path + 'line_graph_edge_label.txt', 'rb') as file:
        edges = []
        y = []
        for line in file:
            src, dst, rel = line.strip().split()
            edges.append((int(src.decode('utf-8')), int(dst.decode('utf-8'))))
            y.append(int(rel.decode('utf-8')))

    with open(data_parents_path + "line_weight.txt", "rb") as file:
        edge_ratio = []
        for line in file:
            src, dst, ratio = line.strip().split()
            edge_ratio.append(float(ratio.decode('utf-8')))
    # edge_weights = deal_edge_weight(edge_ratio)
    edge_weights = edge_ratio

    # Create train test valid masks
    num_nodes = len(y)
    random_order = np.array(range(num_nodes))
    train_percent = 0.6
    valid_percent = 0.2
    train_nodes = int(train_percent * num_nodes)
    valid_nodes = int(valid_percent * num_nodes)

    idx_edge = list(range(num_nodes))
    idx_train = random.sample(idx_edge, train_nodes)
    for elem in idx_train:
        idx_edge.remove(elem)
    idx_val = random.sample(idx_edge, valid_nodes)
    for elem in idx_val:
        idx_edge.remove(elem)
    idx_test = idx_edge

    # 保存每一次随机选取的路径
    dataset = {}
    dataset['train'] = idx_train
    dataset['valid'] = idx_val
    dataset['test'] = idx_test
    np.save("./Experiment Random Dataset/test.npy", dataset)

    train_mask, valid_mask, test_mask = np.zeros(num_nodes, dtype=bool), np.zeros(num_nodes, dtype=bool), np.zeros(num_nodes, dtype=bool)
    train_mask[idx_train] = True
    valid_mask[idx_val] = True
    test_mask[idx_test] = True

    edge_index = torch.tensor(edges, dtype=torch.long)
    # y = np.array(y).reshape(-1, 1)
    y = np.array(y)
    node_values = get_node_features(data_parents_path)
    x = torch.tensor(node_values, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    data = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_weights, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    print(data.x.shape[1])
    return data

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# --------- Main Function --------#
if __name__ == "__main__":
    print("Start coding...")
    get_data("./Data/AS_2020_12_01/")

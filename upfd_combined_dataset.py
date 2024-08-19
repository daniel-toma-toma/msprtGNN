import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, random_split
from torch_geometric.datasets import UPFD
import numpy as np
import random
from collections import Counter
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.transforms import ToUndirected


def extract_k_hop_subgraph(data, start_node, num_hops):
    subset, edge_index, _, _ = k_hop_subgraph(start_node, num_hops, data.edge_index, relabel_nodes=True, directed=False, flow="target_to_source")
    subgraph_data = Data(x=data.x[subset], edge_index=edge_index, y=data.y)
    return subgraph_data

def relabel(data, label_mapping):
    data.y = torch.tensor([label_mapping[label.item()] for label in data.y], dtype=torch.long)
    return data

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        noise = torch.randn(data.x.size()) * self.std + self.mean
        data.x = data.x + noise
        return data

def load_and_relabel(dataset_name, label_mapping, feature):
    transform = None
    train_data = UPFD(root='data', feature=feature, name=dataset_name, split='train', transform=transform)
    test_data = UPFD(root='data', feature=feature, name=dataset_name, split='val', transform=transform)
    val_data = UPFD(root='data', feature=feature, name=dataset_name, split='test', transform=transform)
    num_features = train_data.num_features
    train_data = [relabel(data, label_mapping) for data in train_data]
    #train_data = [extract_k_hop_subgraph(data, 0, 4) for data in train_data]
    val_data = [relabel(data, label_mapping) for data in val_data]
    test_data = [relabel(data, label_mapping) for data in test_data]
    #transform = AddGaussianNoise(mean=0.0, std=1.0)
    #test_data = [transform(data) for data in test_data]
    #train_data = [transform(data) for data in train_data]
    return train_data, val_data, test_data, num_features


def undersample(data):
    labels = [d.y.item() for d in data]
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    undersampled_data = []
    label_data_map = {label: [] for label in label_counts.keys()}

    for d in data:
        label_data_map[d.y.item()].append(d)

    for label in label_data_map.keys():
        undersampled_data.extend(random.Random(42).sample(label_data_map[label], min_count))

    random.Random(42).shuffle(undersampled_data)
    return undersampled_data

def create_subgraphs(data, num_subgraphs=10, min_nodes=3, max_nodes=40):
    subgraphs = []
    num_nodes = data.num_nodes
    for _ in range(num_subgraphs):
        N = random.randint(min_nodes, min(max_nodes, num_nodes))
        subgraph_nodes = torch.arange(N)

        subgraph_x = data.x[subgraph_nodes]
        subgraph_edge_index = data.edge_index[:, (data.edge_index[0, :] < N) & (data.edge_index[1, :] < N)]

        subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, y=data.y)
        subgraphs.append(subgraph_data)
    return subgraphs

def enhance(dataset):
    enhanced_data_list = []
    for data in dataset:
        subgraphs = create_subgraphs(data)
        enhanced_data_list.extend(subgraphs)
    return enhanced_data_list

def get_combined_upfd_dataset(num_classes, features):
    if num_classes == 3:
        gossipcop_label_mapping = {0: 0, 1: 1}
        politifact_label_mapping = {0: 0, 1: 2}
    elif num_classes == 4:
        gossipcop_label_mapping = {0: 0, 1: 2}
        politifact_label_mapping = {0: 1, 1: 3}
    elif num_classes == 2:
        gossipcop_label_mapping = {0: 0, 1: 1}
        politifact_label_mapping = {0: 0, 1: 1}
    gossipcop_train, gossipcop_val, gossipcop_test, num_features = load_and_relabel('gossipcop', gossipcop_label_mapping, features)
    politifact_train, politifact_val, politifact_test, num_features = load_and_relabel('politifact', politifact_label_mapping, features)
    full_gossipcop_dataset = gossipcop_train + gossipcop_val + gossipcop_test
    full_politifact_dataset = politifact_train + politifact_val + politifact_test
    full_dataset = full_gossipcop_dataset + full_politifact_dataset
    undersampled_dataset = undersample(full_dataset)
    train_size = int(len(undersampled_dataset) * 0.4)
    val_size = int(len(undersampled_dataset) * 0.4)
    test_size = len(undersampled_dataset) - val_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, val_test_data = random_split(undersampled_dataset, [train_size, val_size + test_size], generator=generator)
    val_data, test_data = random_split(val_test_data, [val_size, test_size], generator=generator)
    #train_data += enhance(train_data)
    return train_data, val_data, test_data, num_features

def get_combined_upfd_dataloader(train_data, val_data, test_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    return train_loader, val_loader, test_loader

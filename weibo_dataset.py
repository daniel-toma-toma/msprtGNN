import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, random_split
from torch_geometric.data import InMemoryDataset
from upfd_combined_dataset import undersample, extract_k_hop_subgraph
from torch_geometric.transforms import BaseTransform
import random

class WeiboDataset(InMemoryDataset):
    def __init__(self, root, features=None, num_classes=3, transform=None, pre_transform=None, pre_filter=None):
        self.feature = features
        self.num_class = num_classes
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['pytorch_geom_weibo_dataset.pt']

    @property
    def processed_file_names(self):
        if self.num_class == 2:
            return ['data.pt']
        else:
            return ['data3.pt']

    def process(self):
        raw_data_path = self.raw_paths[0]
        raw_data = torch.load(raw_data_path)

        data_list = []
        for item in raw_data:
            x = item['x']
            '''
            if self.feature == "content":
                user_desc = item['user_desc']
                text = item['text']
                text = (text+user_desc) / 2
                x = torch.hstack((x, text))
            '''
            y = item['y']
            if self.num_class == 2:
                y = 0 if y==0 else 1
            data = Data(x=x, edge_index=item['edge_index'], y=y)
            data_list.append(data)
        print(len(data_list))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

class StandardizeFeatures(BaseTransform):
    def forward(self, data):
        x = data.x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        data.x = (x - mean) / std
        return data

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

def get_weibo_dataset(num_classes, features):
    if num_classes > 3:
        return NotImplementedError
    transform = StandardizeFeatures()
    weibo_dataset = WeiboDataset(root='weibo_dataset', num_classes=num_classes, features=features, pre_transform=transform)
    undersampled_dataset = undersample(weibo_dataset)
    train_size = int(len(undersampled_dataset) * 0.7)
    val_size = int(len(undersampled_dataset) * 0.1)
    test_size = len(undersampled_dataset) - val_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, val_test_data = random_split(undersampled_dataset, [train_size, val_size + test_size], generator=generator)
    #train_data = [extract_k_hop_subgraph(data, 0, 5) for data in train_data]
    val_data, test_data = random_split(val_test_data, [val_size, test_size], generator=generator)
    val_data = [data for data in val_data]
    test_data = [data for data in test_data]
    enhanced_train_data = enhance(train_data)
    return train_data, val_data, test_data, weibo_dataset.num_features, enhanced_train_data

def get_weibo_dataloader(train_data, val_data, test_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    return train_loader, val_loader, test_loader

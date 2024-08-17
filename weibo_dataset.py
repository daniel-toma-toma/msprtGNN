import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, random_split
from torch_geometric.data import InMemoryDataset
from upfd_combined_dataset import undersample, extract_k_hop_subgraph
from torch_geometric.transforms import BaseTransform

class WeiboDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['pytorch_geom_weibo_dataset.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        raw_data_path = self.raw_paths[0]
        raw_data = torch.load(raw_data_path)

        data_list = []
        for item in raw_data:
            x = item['x']
            user_desc = item['user_desc']
            text = item['text']
            text = (text+user_desc) / 2
            x = torch.hstack((x, text))
            data = Data(x=x, edge_index=item['edge_index'], y=item['y'])
            data_list.append(data)
        print(len(data_list))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

class StandardizeFeatures(BaseTransform):
    def __call__(self, data):
        x = data.x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        data.x = (x - mean) / std
        return data

def get_weibo_dataset():
    transform = StandardizeFeatures()
    weibo_dataset = WeiboDataset(root='weibo_dataset', pre_transform=transform)
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
    return train_data, val_data, test_data, weibo_dataset.num_features

def get_weibo_dataloader(train_data, val_data, test_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    return train_loader, val_loader, test_loader

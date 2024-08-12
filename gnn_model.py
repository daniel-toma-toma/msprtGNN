import torch
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool, GraphConv, GraphSAGE, GATConv, SAGEConv, SAGPooling, EdgeConv, GINEConv, GINConv
from torch.nn import Linear, Sequential, ReLU
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

class GIN_markov(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(GIN_markov, self).__init__()
        self.num_classes = output_dim
        self.mlp = Sequential(
            Linear(input_dim, int(hidden_dim)),
            ReLU(),
            )
        nn = Sequential(
            Linear(int(hidden_dim), int(output_dim))
        )
        self.conv1 = GINConv(nn, train_eps=True, eps=2.0)
        self.pool = global_mean_pool
        self.T = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.mlp(x)

        z = self.conv1(x, edge_index)
        z = torch.cat([F.log_softmax(z, dim=1)], dim=0)
        z = self.pool(z, batch)  # log(Pr(Zi | Fi-1))
        z = z / self.T

        z = F.log_softmax(z, dim=1)
        return z

    def reset(self):
        pass
import torch
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool, GraphConv, GraphSAGE, GATConv, SAGEConv, SAGPooling, EdgeConv, GINEConv, GINConv
from torch.nn import Linear, Sequential, ReLU
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
class UPFDNet(torch.nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels,
                 concat=False):
        super().__init__()
        self.concat = concat

        if model == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif model == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        elif model == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)

        if self.concat:
            self.lin0 = Linear(in_channels, hidden_channels)
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

    def reset(self):
        pass

    def forward(self, data):
        x, edge_index, batch = [data.x, data.edge_index, data.batch]
        h = self.conv1(x, edge_index).relu()
        h = global_max_pool(h, batch)

        if self.concat:
            # Get the root node (tweet) features of each graph:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h.log_softmax(dim=-1)

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=128, output_dim=3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = [data.x, data.edge_index, data.batch]
        h = self.conv1(x, edge_index).relu()
        h = global_mean_pool(h, batch)
        h = self.lin2(h)
        return h.log_softmax(dim=-1)

    def reset(self):
        pass
class GCN_markov(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(GCN_markov, self).__init__()
        self.num_classes = output_dim
        num_heads = 2
        #hidden_dim = int(hidden_dim/num_heads) * num_heads
        self.conv1 = GATConv(input_dim, int(hidden_dim/num_heads), heads=num_heads)
        #self.conv1 = GCNConv(input_dim, hidden_dim, improved=False)
        self.local_fc = Linear(hidden_dim, output_dim)
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        x, edge_index, batch = [data.x, data.edge_index, data.batch]
        z = F.relu(self.conv1(x, edge_index)) # node embedding
        z = self.local_fc(z)
        z = torch.cat([F.log_softmax(z, dim=1)], dim=0)  # log Pr(Zi | Zi-1). Softmax is applied per node
        z = global_add_pool(z, batch) # log(Pr(Zi | Fi-1))
        z = self.w * z
        z = F.log_softmax(z, dim=1)
        return z

    def reset(self):
        pass

class GCN_markov2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(GCN_markov2, self).__init__()
        self.num_classes = output_dim
        num_heads = 2
        #hidden_dim = int(hidden_dim/num_heads) * num_heads
        self.conv1 = GATConv(input_dim, int(hidden_dim/num_heads), heads=num_heads)
        #self.conv1 = GCNConv(input_dim, hidden_dim, improved=False)
        self.local_fc = Linear(hidden_dim, output_dim)
        self.w = nn.Parameter(torch.tensor(1.0))
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(-1.0))

    def get_subgraph_without_root_and_leaves(self, x, edge_index, batch):
        num_nodes = edge_index.max().item() + 1

        # Find all root nodes (nodes with no incoming edges)
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        in_degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
        root_nodes = (in_degree == 0).nonzero(as_tuple=True)[0]

        # Find all leaf nodes (nodes with no outgoing edges)
        out_degree = torch.zeros(num_nodes, dtype=torch.long)
        out_degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
        leaf_nodes = (out_degree == 0).nonzero(as_tuple=True)[0]

        # Create mask to exclude root and leaf nodes
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[root_nodes] = False
        mask[leaf_nodes] = False

        # Get the node indices to keep
        nodes_to_keep = mask.nonzero(as_tuple=True)[0]
        if nodes_to_keep.numel() == 0:
            return None, None, None

        # Create the subgraph
        subgraph_edge_index, _ = subgraph(nodes_to_keep, edge_index, relabel_nodes=True)
        subgraph_x = x[nodes_to_keep]
        if batch is not None:
            subgraph_batch = batch[nodes_to_keep]
        else:
            subgraph_batch = batch

        return subgraph_x, subgraph_edge_index, subgraph_batch

    def forward(self, data):
        x, edge_index, batch = [data.x, data.edge_index, data.batch]
        z = F.relu(self.conv1(x, edge_index)) # node embedding
        z = self.local_fc(z)
        z = torch.cat([F.log_softmax(z, dim=1)], dim=0)  # log Pr(Zi | Zi-1). Softmax is applied per node
        z = global_add_pool(z, batch) # log(Pr(Zi | Fi-1))

        y, y_edge_index, batch = self.get_subgraph_without_root_and_leaves(x, edge_index, batch)
        #if y is None or y.numel() == 0:
        if True:
            z = self.a * z
        else:
            y_edge_index = torch.empty((2, 0), dtype=torch.long)
            y = F.relu(self.conv1(y, y_edge_index))
            y = self.local_fc(y)
            y = torch.cat([F.log_softmax(y, dim=1)], dim=0)
            y = global_add_pool(y, batch)
            z = self.a * z + self.b * y

        z = F.log_softmax(z, dim=1)
        return z

    def reset(self):
        pass


class GIN_markov(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(GIN_markov, self).__init__()
        self.num_classes = output_dim
        self.mlp = Sequential(
            Linear(input_dim, int(hidden_dim)),
            ReLU(),
            #Linear(int(hidden_dim), int(hidden_dim)),
            #ReLU()
            )
        nn = Sequential(
            #Linear(int(input_dim),int(hidden_dim)),
            #ReLU(),
            Linear(int(hidden_dim), int(output_dim))
        )
        self.conv1 = GINConv(nn, train_eps=True, eps=2.0)
        #self.conv1 = GCNConv(int(hidden_dim), int(output_dim), improved=True)
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
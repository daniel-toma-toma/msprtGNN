import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
"""
The GCN, GAT, and GraphSAGE implementation
"""
class upfdGNN(torch.nn.Module):
	def __init__(self, num_features, nhid, num_classes, dropout_ratio=0.0):
		super(upfdGNN, self).__init__()
		self.num_features = num_features
		self.nhid = nhid
		self.num_classes = num_classes
		self.dropout_ratio = dropout_ratio
		self.model = 'sage'

		if self.model == 'gcn':
			self.conv1 = GCNConv(self.num_features, self.nhid)
		elif self.model == 'sage':
			self.conv1 = SAGEConv(self.num_features, self.nhid)
		elif self.model == 'gat':
			self.conv1 = GATConv(self.num_features, self.nhid)
		self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

	def reset(self):
		pass

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		edge_attr = None
		x = F.relu(self.conv1(x, edge_index, edge_attr))
		x = gmp(x, batch)
		x = F.log_softmax(self.lin2(x), dim=-1)
		return x

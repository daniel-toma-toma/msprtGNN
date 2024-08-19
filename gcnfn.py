import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv
import torch

"""

GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder; 
the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional 
comment word2vec (spaCy) embeddings plus 10-dimensional profile features 

Paper: Fake News Detection on Social Media using Geometric Deep Learning
Link: https://arxiv.org/pdf/1902.06673.pdf


Model Configurations:

Vanilla GCNFN: args.concat = False, args.feature = content
UPFD-GCNFN: args.concat = True, args.feature = spacy

"""

class GCNFN(torch.nn.Module):
	def __init__(self, num_features, nhid, num_classes):
		super(GCNFN, self).__init__()
		self.num_features = num_features
		self.num_classes = num_classes
		self.nhid = nhid
		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)
		self.fc1 = Linear(self.nhid * 2, self.nhid)
		self.fc2 = Linear(self.nhid, self.num_classes)

	def reset(self):
		pass

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.log_softmax(self.fc2(x), dim=-1)
		return x
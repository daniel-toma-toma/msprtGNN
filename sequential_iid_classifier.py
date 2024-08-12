import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sequential_markov_classifier import get_edge_subgraphs, calc_iid_eta

def classify_edges(classifier, data):
    edge_subgraphs = get_edge_subgraphs(data)
    batch = torch.zeros(2, dtype=torch.long)
    z = []
    for edge in edge_subgraphs:
        out = classifier(edge)
        edge_z = out.argmax(dim=1)
        z += [edge_z]
    data.z = z
    return data

def get_iid_eta(dataset, num_classes, num_z, load=False):
    if load:
        iid_eta = torch.load("iid_eta.pt", weights_only=True)
        return iid_eta, None
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        iid_eta = torch.zeros((num_classes, num_z))
        iid_label_counter = torch.zeros((num_classes, num_z))
        for label in range(num_classes):
            iid_eta[label], iid_label_counter[label] = calc_iid_eta(loader, num_z, label)
        torch.save(iid_eta, "iid_eta.pt")
        return iid_eta, iid_label_counter


class sequential_iid(torch.nn.Module):
    def __init__(self, num_classes, eta_z):
        super(sequential_iid, self).__init__()
        self.num_classes = num_classes
        self.pi0 = 1/num_classes
        self.p = torch.ones(self.num_classes) / self.num_classes
        self.eta_z = eta_z
        self.num_classes = num_classes
        self.labels = list(range(num_classes))

    def reset(self):
        self.p = torch.ones(self.num_classes) / self.num_classes

    def forward(self, subgraph):
        z = subgraph.z[-1]
        pr_new_z_given_h = self.eta_z[:, z].squeeze()
        pr_z_and_h = torch.mul(self.p, pr_new_z_given_h)
        pr_z = torch.sum(pr_z_and_h)
        self.p = pr_z_and_h / pr_z
        return torch.log(self.p)
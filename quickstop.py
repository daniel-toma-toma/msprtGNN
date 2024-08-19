import torch
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

tmp_dir = "./tmp/"

def count_transitions(graph: Data, num_z):
    transition_counts = defaultdict(int)
    for i in range(len(graph.z)):
        transition_counts[graph.z[i-1], graph.z[i]] += 1
    transition_counts_array = torch.zeros((num_z, num_z), dtype=torch.float32)
    for (i, j), count in transition_counts.items():
        transition_counts_array[i, j] = count
    return transition_counts_array

def is_row_stochastic(matrix):
    if torch.any(matrix < 0):
        return False
    row_sums = torch.sum(matrix, axis=1)
    ones = torch.ones_like(row_sums)
    return torch.allclose(row_sums, ones)

def init_transition_counter(num_z):
    transition_counts = torch.zeros((num_z, num_z), dtype=torch.float32)
    return transition_counts

def calc_alpha(loader, device, num_z, label):
    transition_counts = init_transition_counter(num_z)
    for data in tqdm(loader, desc=f'calculating alpha for class {label}'):
        if data.y != label:
            continue
        data = data.to(device)
        transition_counts += count_transitions(data, num_z)
    row_sums = transition_counts.sum(dim=1, keepdim=True)
    transition_matrix = transition_counts / row_sums
    for i in range(transition_matrix.shape[0]):
        if row_sums[i] == 0:
            transition_matrix[i,:] = torch.ones(num_z) / num_z
    assert is_row_stochastic(transition_matrix)
    return transition_matrix

def calc_eta(loader, device, num_z, label):
    label_counter = torch.zeros(num_z)
    edges_counter = 0
    for data in tqdm(loader, desc=f'calculating eta for class {label}'):
        if data.y != label:
            continue
        data = data.to(device)
        label_counter[data.z[0]] += 1
        edges_counter +=1
    eta_z = label_counter / edges_counter
    assert torch.allclose(torch.sum(eta_z), torch.ones(1))
    return eta_z, label_counter

def get_quickstop_alpha_and_eta(dataset_name, dataset, device, num_classes, num_z, load=False):
    if load:
        alpha = torch.load(tmp_dir+dataset_name+"_quickstop_alpha.pt", weights_only=True)
        eta = torch.load(tmp_dir+dataset_name+"_quickstop_eta.pt", weights_only=True)
        return alpha, eta
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        alpha = torch.zeros((num_classes, num_z, num_z))
        eta = torch.zeros((num_classes, num_z))
        for label in range(num_classes):
            eta[label], label_counter = calc_eta(loader, device, num_z, label)
            alpha[label] = calc_alpha(loader, device, num_z, label)
        torch.save(alpha, tmp_dir+dataset_name+"_quickstop_alpha.pt")
        torch.save(eta, tmp_dir+dataset_name+"_quickstop_eta.pt")
        return alpha, eta


class quickstop(torch.nn.Module):
    def __init__(self, num_classes, alpha, eta_z):
        super(quickstop, self).__init__()
        self.num_classes = num_classes
        self.pi0 = 1/num_classes
        self.p = torch.ones(self.num_classes) / self.num_classes
        self.pr_z_given_h = torch.ones(self.num_classes)
        self.alpha = alpha
        self.eta_z = eta_z
        self.num_classes = num_classes
        self.labels = list(range(num_classes))
        self.previous_z = None

    def reset(self):
        self.p = torch.ones(self.num_classes) / self.num_classes
        self.pr_z_given_h = torch.ones(self.num_classes)
        self.previous_z = None

    def calc_pr_z_and_h_if_not_connected_to_source(self, curr_z):
        for j in self.labels:
            self.pr_z_given_h[j] = self.pr_z_given_h[j] * self.alpha[j][self.previous_z, curr_z]

    def calc_pr_z_and_h_if_connected_to_source(self, z):
        for j in self.labels:
            self.pr_z_given_h[j] = self.pr_z_given_h[j] * self.eta_z[j][int(z)]

    def update_posterior_probabilites(self):
        pr_z_and_h = torch.zeros(self.num_classes)
        for j in range(len(self.labels)):
            pr_z_and_h[j] = self.pi0 * self.pr_z_given_h[j]
        pr_z = torch.sum(pr_z_and_h)
        for j in self.labels:
            if pr_z >= pr_z_and_h[j]:
                assert True
            self.p[j] = pr_z_and_h[j] / pr_z

    def forward(self, subgraph):
        new_node_id = subgraph.x.shape[0] - 1
        z = subgraph.z[-1]
        if new_node_id == 1:
            self.calc_pr_z_and_h_if_connected_to_source(z)
        else:
            self.calc_pr_z_and_h_if_not_connected_to_source(z)
        self.update_posterior_probabilites()
        self.previous_z = z
        return torch.log(self.p)
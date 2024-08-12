import torch
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

def get_incoming_edge_index(subgraph, node):
    edge_index = subgraph.edge_index
    incoming_edge_index = (edge_index[1] == node).nonzero()
    return incoming_edge_index


def get_parent_node(subgraph, child_node):
    edge_index = subgraph.edge_index
    incoming_edge_index = get_incoming_edge_index(subgraph, child_node)
    if incoming_edge_index.numel() > 1:
        incoming_edge_index = incoming_edge_index[-1]
    elif incoming_edge_index.numel() == 0:
        incoming_edge_index = incoming_edge_index[-1]
    parent_node = edge_index[0, incoming_edge_index].item()
    return parent_node

def get_curr_z(subgraph, node):
    edges = get_incoming_edge_index(subgraph, node)
    z = subgraph.z[edges[0][0].int()]
    return z

def get_previous_z(subgraph, node):
    parent_node = get_parent_node(subgraph, node)
    prev_z = get_curr_z(subgraph, parent_node)
    return prev_z

def get_edge_subgraphs(data):
    edge_subgraphs = []
    for i in range(data.edge_index.shape[1]):
        edge = data.edge_index[:, i].view(2, 1)
        nodes = edge.flatten().unique()
        subgraph_x = data.x[nodes]
        node_mapping = {node.item(): idx for idx, node in enumerate(nodes)}
        subgraph_edge_index = edge.clone().apply_(lambda x: node_mapping[x])
        edge_subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index)
        edge_subgraphs.append(edge_subgraph)
    return edge_subgraphs


def relabel_logits(probs, threshold=0.0):
    primary_label = torch.argmax(probs)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    sorted_probs = sorted_probs.squeeze()
    sorted_indices = sorted_indices.squeeze()
    secondary_label = None
    if sorted_probs[1] >= threshold:
        secondary_label = sorted_indices[1]
    assert secondary_label is not None
    num_labels = probs.shape[0]
    d = secondary_label if secondary_label < primary_label else secondary_label - 1
    new_label = primary_label * (num_labels - 1) + d
    return [new_label]


def relabel_logits(probs, threshold=0.0):
    primary_labels = torch.argmax(probs, dim=1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
    secondary_labels = sorted_indices[:, 1]
    valid_secondary = sorted_probs[:, 1] >= threshold
    assert valid_secondary.all()
    num_labels = probs.shape[1]
    d = torch.where(secondary_labels < primary_labels,
                    secondary_labels,
                    secondary_labels - 1)
    new_labels = primary_labels * (num_labels - 1) + d
    return new_labels

def classify_edges(classifier, data, num_classes, num_z):
    edge_subgraphs = get_edge_subgraphs(data)
    edge_subgraphs = Batch.from_data_list(edge_subgraphs)
    out = torch.exp(classifier(edge_subgraphs))
    if num_z == num_classes:
        preds = out.argmax(dim=1)
        data.z = preds
    else:
        data.z = relabel_logits(out, threshold=0.0)
    return data

def z_classify_dataset(dataset, classifier, num_classes, num_z, name="z_class_dataset.pt", load=False):
    if load:
        z_classified_dataset = torch.load(name)
    else:
        z_classified_dataset = []
        for i in tqdm(range(len(dataset)), desc=f'z-classifying dataset'):
            z_classified_dataset += [classify_edges(classifier, dataset[i], num_z, num_classes)]
        torch.save(z_classified_dataset, name)
    return z_classified_dataset

max_depth = 20

def depth_first_search(graph, node, parent, parent_z, edge_index, labels, transition_counts, depth, visited=None):
    if depth == max_depth:
        return
    if visited is None:
        visited = set()
    if node not in visited:
        visited.add(node)
        children = edge_index[1][edge_index[0] == node]
        for child in children:
            if child == parent:
                continue
            curr_z = get_curr_z(graph, child)
            if parent_z != -1:
                transition_counts[parent_z, curr_z] += 1
            depth_first_search(graph, child.item(), node, curr_z, edge_index, labels, transition_counts, depth+1, visited)

def count_transitions(graph: Data, num_z):
    labels = graph.y
    edge_index = graph.edge_index
    transition_counts = defaultdict(int)
    root = 0
    depth_first_search(graph, root, -1, -1, edge_index, labels, transition_counts, 0)
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

def init_transition_counter(eta_iid, num_z):
    transition_counts = torch.zeros((num_z, num_z), dtype=torch.float32)
    s = 10
    transition_counts[:] = eta_iid * s
    return transition_counts

def calc_alpha(loader, device, num_z, label, eta_iid, num_classes):
    transition_counts = init_transition_counter(eta_iid, num_z)
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
        root = 0
        edge_index = data.edge_index
        edge_indices_from_root = torch.where(edge_index[0] == root)[0]
        for edge_index in edge_indices_from_root:
            label_counter[data.z[edge_index]] += 1
            edges_counter +=1
    eta_z = label_counter / edges_counter
    assert torch.allclose(torch.sum(eta_z), torch.ones(1))
    return eta_z, label_counter

def get_alpha_and_eta(dataset, device, num_classes, num_z, eta_iid, load=False):
    if load:
        alpha = torch.load("alpha.pt")
        eta = torch.load("eta.pt")
        return alpha, eta
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        alpha = torch.zeros((num_classes, num_z, num_z))
        eta = torch.zeros((num_classes, num_z))
        for label in range(num_classes):
            eta[label], label_counter = calc_eta(loader, device, num_z, label)
            alpha[label] = calc_alpha(loader, device, num_z, label, eta_iid[label], num_classes)
        torch.save(alpha, "alpha.pt")
        torch.save(eta, "eta.pt")
        return alpha, eta


class sequential_markov(torch.nn.Module):
    def __init__(self, num_classes, alpha, eta_z):
        super(sequential_markov, self).__init__()
        self.num_classes = num_classes
        self.pi0 = 1/num_classes
        self.p = torch.ones(self.num_classes) / self.num_classes
        self.pr_z_given_h = torch.ones(self.num_classes)
        self.alpha = alpha
        self.eta_z = eta_z
        self.num_classes = num_classes
        self.labels = list(range(num_classes))

    def reset(self):
        self.p = torch.ones(self.num_classes) / self.num_classes
        self.pr_z_given_h = torch.ones(self.num_classes)

    def calc_pr_z_and_h_if_not_connected_to_source(self, subgraph, new_node_id):
        curr_z = subgraph.z[-1]
        previous_z = get_previous_z(subgraph, new_node_id)
        for j in self.labels:
            A_j = self.alpha[j][previous_z, curr_z]
            self.pr_z_given_h[j] = self.pr_z_given_h[j] * A_j

    def calc_pr_z_and_h_if_connected_to_source(self, z):
        for j in self.labels:
            A_j= self.eta_z[j][int(z)]
            self.pr_z_given_h[j] = self.pr_z_given_h[j] * A_j

    def is_connected_to_root(self, edge_index, new_node):
        root_node = 0
        is_connected = torch.eq(edge_index, torch.tensor([[root_node], [new_node]])).all(dim=0).any()
        return is_connected

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
        if self.is_connected_to_root(subgraph.edge_index, new_node_id):
            self.calc_pr_z_and_h_if_connected_to_source(z)
        else:
            self.calc_pr_z_and_h_if_not_connected_to_source(subgraph, new_node_id)
        self.update_posterior_probabilites()
        return torch.log(self.p)
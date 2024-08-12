import torch
from upfd_combined_dataset import get_combined_upfd_dataloader, get_combined_upfd_dataset
from gnn_model import GIN_markov
import numpy as np
from sklearn.metrics import classification_report, recall_score
import torch.nn as nn
from sequential_test import sequential_test
import matplotlib.pyplot as plt
from sequential_markov_classifier import get_alpha_and_eta, sequential_markov, z_classify_dataset
from quickstop import get_quickstop_alpha_and_eta, quickstop
from torch_geometric.loader import DataLoader
from sequential_iid_classifier import get_iid_eta, sequential_iid
from weibo_dataset import get_weibo_dataset, get_weibo_dataloader
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.nn import global_add_pool

def train(model, optimizer, train_loader, device, l1_lambda=0.0):
    model.train()
    criterion = nn.NLLLoss()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        if l1_lambda > 0.0:
            for param in model.mlp.parameters():
                loss += l1_lambda * torch.sum(torch.abs(param))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    all_predicted = np.array([])
    all_labels = np.array([])
    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs
        all_predicted = np.append(all_predicted, pred)
        all_labels = np.append(all_labels, data.y)
    print(classification_report(all_labels, all_predicted))
    min_recall = np.min(recall_score(all_labels, all_predicted, average=None))
    acc = total_correct / total_examples
    return acc

def train_loop(model, train_loader, val_loader, test_loader, device, lr=0.005, weight_decay=0.01, dataset='', l1_lambda=0.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.8, 0.995))
    best_test_acc = 0.0
    for epoch in range(1, 61):
        loss = train(model, optimizer, train_loader, device, l1_lambda=l1_lambda)
        train_acc = test(model, train_loader, device)
        #val_acc = test(model, val_loader, device)
        test_acc = test(model, test_loader, device)
        general_err = train_acc - test_acc
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}, generalization err: {general_err:.2%}')
        if test_acc >= best_test_acc:
            torch.save(model.state_dict(), dataset + '_model.pt')
            best_test_acc = test_acc


def tune_hyperparams(train_loader, val_loader, test_loader, device, num_features, num_classes):
    lrs = [0.001, 0.002, 0.004]
    #weight_decays = [round(i * 0.001, 3) for i in range(1, 10, 1)]
    weight_decays = [0.01]
    print(lrs)
    print(weight_decays)
    x = []
    y = []
    z = []
    for lr in lrs:
        for weight_decay in weight_decays:
            print(f'lr {lr:.4f}, decay: {weight_decay:.4f}')
            model = GIN_markov(num_features, output_dim=num_classes).to(device)
            train_loop(model, train_loader, val_loader, test_loader, device, lr, weight_decay)
            val_acc = test(model, val_loader, device)
            x += [lr]
            y += [weight_decay]
            z += [val_acc]
            print(f'lr {lr:.4f}, decay: {weight_decay:.4f}, acc: {val_acc:.4f}')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('lr')
    ax.set_ylabel('weight decay')
    ax.set_zlabel('acc')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    w = np.vstack((x, y, z)).T
    w = w[w[:, 2].argsort()]
    print(x)
    print(y)
    print(z)
    print(w)
    plt.show()


def maximal_distance_from_first_node(data, start_node=0):
    G = to_networkx(data, to_undirected=True)
    lengths = nx.single_source_shortest_path_length(G, start_node)
    max_distance = max(lengths.values())
    return max_distance

def plot_eta_iid(eta_iid, num_classes):
    if num_classes == 4:
        x_axis = ["0,1", "0,2", "0,3", "1,0", "1,2", "1,3",
                  "2,0", "2,1", "2,3", "3,0", "3,1", "3,2"]
    elif num_classes == 3:
        x_axis = ["0,1", "0,2", "1,0", "1,2", "2,0", "2,1"]
    else:
        raise NotImplementedError("num_classes not implemented")

    plt.figure(figsize=(10, 6))
    bar_width = 0.1
    x_indices = np.arange(len(x_axis))
    if False:
        for i in range(eta_iid.shape[0]):
            plt.bar(x_indices + i * bar_width, eta_iid[i], width=bar_width, label=f'Hypothesis {i}')

        plt.xlabel('Z values (label1.label2)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Z values')
        plt.xticks(x_indices + bar_width * (eta_iid.shape[0] - 1) / 2, x_axis)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    for i in range(len(eta_iid)):
        print((i, eta_iid[i]))

def print_dataset_stats(dataset):
    total_graphs = len(dataset)
    total_edges = 0
    total_vertices = 0
    vertices_mul = 1
    label_counts = {}
    depths = []
    for data in dataset:
        total_edges += data.num_edges
        total_vertices += data.num_nodes
        vertices_mul *= (data.num_nodes ** (1/total_graphs))
        label = data.y.item()
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1
        depth = maximal_distance_from_first_node(data)
        depths += [depth]
    avg_edges_per_graph = total_edges / total_graphs
    avg_vertices_per_graph = total_vertices / total_graphs
    geom_mean_nodes_per_graph = vertices_mul
    print(f'Average number of edges per graph: {avg_edges_per_graph}')
    print(f'Average number of vertices per graph: {avg_vertices_per_graph}')
    print(f'geom Average number of vertices per graph: {geom_mean_nodes_per_graph}')
    print(f'Number of graphs per label: {label_counts}')
    print(f'maximal depth: {np.max(depths)}')
    print(f'average depth: {np.mean(depths)}')
    return geom_mean_nodes_per_graph

def get_stationary_transition(P):
    n = P.shape[0]
    A = np.transpose(P) - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n)
    b = np.append(b, 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

def offline_msprt_algo(train_data, test_data, gnn_classifier, num_classes, num_z, device):
    z_classified_train_dataset = z_classify_dataset(train_data, gnn_classifier, num_classes, num_z,
                                                    name="z_class_train_dataset.pt", load=load_z_dataset)
    alpha, eta, eta_iid = get_alpha_and_eta(z_classified_train_dataset, device, num_classes, num_z, load=load_alpha)
    naive_iid_model = sequential_iid(num_classes, eta_iid)
    markov_model = sequential_markov(num_classes, alpha, eta)
    #print(eta_iid)
    z_classified_test_dataset = z_classify_dataset(test_data, gnn_classifier, num_classes, num_z,
                                                   name="z_class_test_dataset.pt", load=load_z_dataset)
    test_loader = DataLoader(z_classified_test_dataset, batch_size=1, shuffle=True)
    eta_iid_test, iid_label_counter_test = get_iid_eta(z_classified_test_dataset, num_classes, num_z, load=load_alpha)
    #plot_eta_iid(eta_iid_test, num_classes)
    #print(eta)
    #print(alpha)
    for i in range(num_classes):
        stationary_p = get_stationary_transition(alpha[i])
        #print(stationary_p)
    return markov_model, naive_iid_model, test_loader

def offline_quickstop_algo(train_data, test_data, gnn_classifier, num_classes, num_z, device):
    z_classified_train_dataset = z_classify_dataset(train_data, gnn_classifier, num_classes, num_z,
                                                    name="z_class_train_dataset.pt", load=True)
    quick_alpha, quick_eta = get_quickstop_alpha_and_eta(z_classified_train_dataset, device, num_classes, num_z, load=load)
    quickstop_model = quickstop(num_classes, quick_alpha, quick_eta)
    #print("quickstop:")
    #print(quick_eta)
    #print(quick_alpha)
    for i in range(num_classes):
        stationary_p = get_stationary_transition(quick_eta[i])
        #print(stationary_p)
    return quickstop_model

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

load = True
load_gnn = load
load_z_dataset = load
load_alpha = load
dataset = "weibo"

def main():
    if dataset == "upfd":
        num_classes = 4
        num_z = num_classes * (num_classes-1)
        train_data, val_data, test_data, num_features = get_combined_upfd_dataset(num_classes)
        train_loader, val_loader, test_loader = get_combined_upfd_dataloader(train_data, val_data, test_data, batch_size=32)
        lr, weight_decay, hidden_dim, l1_lambda = 0.001, 0.001, 512, 0.000001
        gnn_threshold = 0.99
        eps = 1e-10
        msprt_threshold = 0.999999
    elif dataset == "weibo":
        num_classes = 3
        num_z = num_classes * (num_classes-1)
        train_data, val_data, test_data, num_features = get_weibo_dataset()
        train_data += val_data
        train_loader, val_loader, test_loader = get_weibo_dataloader(train_data, val_data, test_data, batch_size=32)
        lr, weight_decay, hidden_dim, l1_lambda = 0.001, 0.001, 64, 0.000001
        gnn_threshold = 0.5
        msprt_threshold = 0.9999
        quickstop_threshold = 0.8
    print(f'Train dataset size: {len(train_data)}, Validation dataset size: {len(val_data)}, Test dataset size: {len(test_data)}')
    T = print_dataset_stats(train_data + val_data + test_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_classifier = GIN_markov(num_features, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
    if not load_gnn:
        train_loop(gnn_classifier, train_loader, val_loader, test_loader, device, lr, weight_decay, dataset, l1_lambda)
    gnn_classifier.load_state_dict(torch.load(dataset + '_model.pt'))
    gnn_classifier.T = T
    gnn_classifier.pool = global_add_pool
    test_acc = test(gnn_classifier, test_loader, device)
    print(f'Loaded model. test acc: {test_acc:.4f}')
    print(f'GINConv eps: {gnn_classifier.conv1.eps.item():.4f}')
    if True:
        markov_model, naive_iid_model, test_loader = offline_msprt_algo(train_data, test_data, gnn_classifier, num_classes, num_z, device)
        quickstop_model = offline_quickstop_algo(train_data, test_data, gnn_classifier, num_classes, num_z, device)
    print("quickstop model:")
    sequential_test(quickstop_model, device, test_loader, pvalue=quickstop_threshold)
    print("naive iid model:")
    sequential_test(naive_iid_model, device, test_loader, pvalue=msprt_threshold)
    print("markov model:")
    sequential_test(markov_model, device, test_loader, pvalue=msprt_threshold)
    print("GNN model:")
    sequential_test(gnn_classifier, device, test_loader, pvalue=gnn_threshold)


if __name__ == '__main__':
    main()

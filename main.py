import torch
from upfd_combined_dataset import get_combined_upfd_dataloader, get_combined_upfd_dataset
from msprtgnn import GIN_markov
import numpy as np
from sklearn.metrics import classification_report, recall_score
import torch.nn as nn
from sequential_test import sequential_test
import matplotlib.pyplot as plt
from markov_msprt import get_alpha_and_eta, sequential_markov, z_classify_dataset
from quickstop import get_quickstop_alpha_and_eta, quickstop
from torch_geometric.loader import DataLoader
from naive_msprt import get_iid_eta, sequential_iid
from weibo_dataset import get_weibo_dataset, get_weibo_dataloader
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.nn import global_add_pool
import utils
from upfd_gnn import upfdGNN
from gcnfn import GCNFN

sci_mode=False
np.set_printoptions(suppress=not sci_mode)
torch.set_printoptions(sci_mode=sci_mode)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print(classification_report(all_labels, all_predicted, zero_division=0.0))
    min_recall = np.min(recall_score(all_labels, all_predicted, average=None))
    acc = total_correct / total_examples
    return acc

def train_loop(model, train_loader, test_loader, device, lr=0.005, weight_decay=0.01, dataset='', l1_lambda=0.0, filename=''):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.8, 0.995))
    best_test_acc = 0.0
    for epoch in range(0, num_epochs):
        loss = train(model, optimizer, train_loader, device, l1_lambda=l1_lambda)
        train_acc = test(model, train_loader, device)
        #val_acc = test(model, val_loader, device)
        test_acc = test(model, test_loader, device)
        general_err = train_acc - test_acc
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}, generalization err: {general_err:.2%}')
        if test_acc >= best_test_acc:
            torch.save(model.state_dict(), filename)
            best_test_acc = test_acc


def tune_hyperparams(train_loader, val_loader, test_loader, device, num_features, num_classes):
    lrs = [0.001, 0.002, 0.004]
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
            train_loop(model, train_loader, test_loader, device, lr, weight_decay)
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
    total_harmonic = 0
    vertices_mul = 1
    label_counts = {}
    depths = []
    for data in dataset:
        total_edges += data.num_edges
        total_vertices += data.num_nodes
        vertices_mul *= (data.num_nodes ** (1/total_graphs))
        total_harmonic += 1 / data.num_nodes
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
    harmonic_mean = total_graphs / total_harmonic
    print(f'Average number of edges per graph: {avg_edges_per_graph}')
    print(f'Average number of vertices per graph: {avg_vertices_per_graph}')
    print(f'geom Average number of vertices per graph: {geom_mean_nodes_per_graph}')
    print(f'harmonic_mean of vertices per graph: {harmonic_mean}')
    print(f'Number of graphs per label: {label_counts}')
    print(f'maximal depth: {np.max(depths)}')
    print(f'average depth: {np.mean(depths)}')
    return harmonic_mean

def get_stationary_transition(P):
    n = P.shape[0]
    A = np.transpose(P) - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n)
    b = np.append(b, 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

def train_naive(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z):
    filename = dataset + "_z_class_train_dataset.pt"
    z_classified_train_dataset = z_classify_dataset(train_data, edge_classifier, num_classes, num_z, name=filename, load=load_z_dataset)
    iid_eta, iid_label_counter = get_iid_eta(dataset, z_classified_train_dataset, num_classes, num_z, load=load_alpha)
    naive_iid_model = sequential_iid(num_classes, iid_eta)
    return naive_iid_model

def train_markovmsprt(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z):
    filename = dataset + "_z_class_train_dataset.pt"
    z_classified_train_dataset = z_classify_dataset(train_data, edge_classifier, num_classes, num_z, name=filename, load=True)
    alpha, eta = get_alpha_and_eta(dataset, z_classified_train_dataset, device, num_classes, num_z, load=load_alpha)
    markov_model = sequential_markov(num_classes, alpha, eta)
    return markov_model

def train_quickstop(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z):
    filename = dataset + "_z_class_train_dataset.pt"
    z_classified_train_dataset = z_classify_dataset(train_data, edge_classifier, num_classes, num_z, name=filename, load=True)
    quick_alpha, quick_eta = get_quickstop_alpha_and_eta(dataset, z_classified_train_dataset, device, num_classes, num_z, load=load)
    quickstop_model = quickstop(num_classes, quick_alpha, quick_eta)
    return quickstop_model

def train_msprtgnn(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z):
    if dataset == "upfd3" or dataset == "upfd4":
        lr, weight_decay, hidden_dim, l1_lambda = 0.001, 0.001, 512, 0.000001
    elif dataset == "weibo":
        lr, weight_decay, hidden_dim, l1_lambda = 0.001, 0.001, 64, 0.000001
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    gnn_classifier = GIN_markov(num_features, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
    filename = dataset+"_msprtgnn.pt"
    if not load_gnn:
        train_loop(gnn_classifier, train_loader, test_loader, device, lr, weight_decay, dataset, l1_lambda, filename)
    gnn_classifier.load_state_dict(torch.load(filename, weights_only=True))
    gnn_classifier.T = T * 2
    gnn_classifier.pool = global_add_pool
    test_acc = test(gnn_classifier, test_loader, device)
    return gnn_classifier

def train_gcnfn(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z):
    lr, weight_decay, hidden_dim, l1_lambda = 0.001, 0.001, 128, 0.0
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    gnn_classifier = GCNFN(num_features, nhid=hidden_dim, num_classes=num_classes).to(device)
    filename = dataset+"_gcnfn.pt"
    if not load_gnn:
        train_loop(gnn_classifier, train_loader, test_loader, device, lr, weight_decay, dataset, l1_lambda, filename=filename)
    gnn_classifier.load_state_dict(torch.load(filename, weights_only=True))
    test_acc = test(gnn_classifier, test_loader, device)
    return gnn_classifier

def train_gcnfn(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z):
    lr, weight_decay, hidden_dim, l1_lambda = 0.001, 0.001, 128, 0.0
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    gnn_classifier = GCNFN(num_features, nhid=hidden_dim, num_classes=num_classes).to(device)
    filename = dataset+"_gcnfn.pt"
    if not load_gnn:
        train_loop(gnn_classifier, train_loader, test_loader, device, lr, weight_decay, dataset, l1_lambda, filename=filename)
    gnn_classifier.load_state_dict(torch.load(filename, weights_only=True))
    test_acc = test(gnn_classifier, test_loader, device)
    return gnn_classifier

train_list = [
    #"msprtGNN",
    #"upfd-sage",
    "gcnfn" ,
    #"naive",
    #"markovMSPRT",
    #"quickstop",
    #"HGFND",
    ]

test_list = [
    #"msprtGNN",
    #"upfd-sage",
    "gcnfn" ,
    #"naive",
    #"markovMSPRT",
    #"quickstop",
    #"HGFND",
    ]

models_train_fn_dict = {
    "msprtGNN" : train_msprtgnn,
    #"upfd-sage": train_upfd,
    "gcnfn" : train_gcnfn,
    "naive" : train_naive,
    "markovMSPRT" : train_markovmsprt,
    "quickstop" : train_quickstop,
    #"HGFND" : train_hgfnd,
}

upfd3_threshold_dict = {
    "msprtGNN" : 0.62,
    #"upfd-sage": 0.9,
    "gcnfn" : 0.9,
    "naive" : 0.998,
    "markovMSPRT" : 0.99998,
    "quickstop" : 0.93,
    #"HGFND"" : train_hgfnd,
}
upfd4_threshold_dict = {
    "msprtGNN" : 0.62,
    #"upfd-sage": 0.9,
    "gcnfn" : 0.9,
    "naive" : 0.998,
    "markovMSPRT" : 0.99998,
    "quickstop" : 0.93,
    #"HGFND"" : train_hgfnd,
}
weibo_threshold_dict = {
    "msprtGNN" : 0.36, #0.42,
    #"upfd-sage": 0.9,
    "gcnfn" : 0.55,
    "naive" : 0.99,
    "markovMSPRT" : 0.99,
    "quickstop" : 0.7,
    #"HGFND" : train_hgfnd,
}
threshold_dict = {
    "upfd3" : upfd3_threshold_dict,
    "upfd4" : upfd4_threshold_dict,
    "weibo": weibo_threshold_dict,
}

load = False
load_gnn = load
load_z_dataset = load
load_alpha = load
dataset = "weibo"
batch_size = 32
num_epochs = 60
hyper_tune_th = True

def main():
    if dataset == "upfd3":
        num_classes = 3
        train_data, val_data, test_data, num_features = get_combined_upfd_dataset(num_classes)
    if dataset == "upfd4":
        num_classes = 4
        train_data, val_data, test_data, num_features = get_combined_upfd_dataset(num_classes)
    elif dataset == "weibo":
        num_classes = 3
        train_data, val_data, test_data, num_features = get_weibo_dataset(num_classes)
    num_z = num_classes * (num_classes - 1)
    train_data += val_data
    print(f'Train dataset size: {len(train_data)}, Test dataset size: {len(test_data)}')
    T = print_dataset_stats(train_data + val_data + test_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dict = {}
    edge_classifier_name = "msprtGNN"
    edge_classifier = None
    for model_name in train_list:
        train_fn = models_train_fn_dict[model_name]
        model = train_fn(train_data, num_classes, num_features, T, test_data, edge_classifier, num_z)
        models_dict[model_name] = model
        if model_name == edge_classifier_name:
            edge_classifier = model
    if edge_classifier is not None:
        filename = dataset + "_z_class_test_dataset.pt"
        z_classified_test_dataset = z_classify_dataset(test_data, edge_classifier, num_classes, num_z,name=filename, load=load_z_dataset)
        test_loader = DataLoader(z_classified_test_dataset, batch_size=1, shuffle=True)
    else:
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    optimal_thrs = {}
    results = []
    print("Model & Accuracy & Average Deadline & Bayesian Risk")
    for model_name in test_list:
        print(f"Testing model {model_name}")
        model = models_dict[model_name]
        bayesian_risks = {}
        if hyper_tune_th:
            #base = 1-0.71
            #thresholds = [1-base*4, 1-base*2, 1-base*np.sqrt(2), 1-base, 1-base/np.sqrt(2), 1-base/2, 1-base/4]
            thresholds = [0.55]
        else:
            thresholds = [threshold_dict[dataset][model_name]]
        for th in thresholds:
            print(f"threshold:{th}")
            result = sequential_test(model, device, test_loader, pvalue=th, model_name=model_name)
            bayesian_risks[th] = result.risk
            results += [result]
            result.print_results()
        if hyper_tune_th:
            min_th, min_bayesian_risk = utils.find_min_value_key(bayesian_risks)
            optimal_thrs[model_name] = min_th
            utils.plot_dict(bayesian_risks)
    for result in results:
        result.print_results()

    print(optimal_thrs)


if __name__ == '__main__':
    main()

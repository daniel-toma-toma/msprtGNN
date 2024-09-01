import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from torch_geometric.utils import to_networkx, from_networkx
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_network(G):
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='gray')
    plt.title("Graph Visualization")
    plt.show()

def add_nodes_by_posting_time(data):
    G = to_networkx(data, to_undirected=False)
    nodes = list(range(1, data.x.shape[0]))
    subgraph_nodes = set([0])
    for node in nodes:
        subgraph_nodes.add(node)
        subgraph = G.subgraph(subgraph_nodes)
        subgraph_data = from_networkx(subgraph)
        node_features = data.x[list(subgraph_nodes)]
        subgraph_data.x = node_features
        #plot_network(subgraph)
        if hasattr(data, 'z'):
            # Create a mapping from edges to their indices in the original edge index
            edge_indices = {tuple(edge): idx for idx, edge in enumerate(data.edge_index.t().tolist())}
            subgraph_edges = list(subgraph.edges)
            subgraph_edge_indices = [edge_indices[edge] for edge in subgraph_edges]
            subgraph_edge_attr = np.array(data.z)[subgraph_edge_indices]
            subgraph_data.z = subgraph_edge_attr
        yield subgraph_data


class Results:
    def __init__(self, model_name, accuracy, deadline, risk):
        self.model_name = model_name
        self.accuracy = accuracy
        self.deadline = deadline
        self.risk = risk

    def print_results(self):
        print(f"\\texttt{{{self.model_name}}} & {self.accuracy:.2f} & {self.deadline:.2f} & {self.risk:.2f}\\\\")


# Example usage:
# result = Results("Model1", 0.95, 0.93, "2024-08-18", "Low")
# result.print_results()

#def calc_risk(pred_list, label_list, T_list, c_prop=1.0, c_err=100000.0):
def calc_risk(pred_list, label_list, T_list, c_prop=1.0, c_err=1000.0):
    propagation_risk = c_prop * np.mean(T_list)
    count_wrong_pred = sum(1 for pred, label in zip(pred_list, label_list) if pred != label)
    wrong_pred_error = count_wrong_pred / len(pred_list)
    error_risk = c_err * wrong_pred_error
    return error_risk + propagation_risk

def sequential_test(model, device, loader, model_name, is_seq=True, pvalue=0.7, eps = 0.2, max_t=1000):
    correct = 0
    num_nonconverge = 0
    n = 0
    T_list = []
    decision_list = []
    label_list = []
    T_sum = 0
    full_trace_T_sum = 0
    all_predicted = np.array([])
    all_labels = np.array([])
    t_correct_all = np.zeros((max_t, 2))
    model.eval()
    for data in tqdm(loader, desc=f'testing'):
        model.reset()
        data = data.to(device)
        full_trace_T_sum += data.num_nodes
        posterior_pr_array = []
        pred_array = []
        with torch.no_grad():
            pred = None
            did_converge = False
            T = 0
            for subgraph in add_nodes_by_posting_time(data):
                out = torch.exp(model(subgraph)).squeeze()
                posterior_pr = torch.max(out)
                pred = out.argmax(dim=0)
                T += 1
                if (is_seq and subgraph.num_nodes > 2 and posterior_pr > pvalue and pred_array[-1] == pred and posterior_pr_array[-1] > pvalue - eps):
                    did_converge = True
                    break
                else:
                    posterior_pr_array += [posterior_pr]
                    pred_array += [pred]
                #if T == 2: # for ablation study
                #    break
                if not is_seq and T < max_t:
                    if T == max_t:
                        break
                    t_correct_all[T][0] += pred.eq(data.y).sum().item() #correct prediction
                    t_correct_all[T][1] += 1 #all predictions

            if not did_converge:
                num_nonconverge += 1
        n += 1
        T_sum += T
        T_list += [T]
        decision_list += [pred]
        label_list += [data.y]
        correct += pred.eq(data.y).sum().item()
        all_predicted = np.append(all_predicted, pred)
        all_labels = np.append(all_labels, data.y)
    bayesian_risk = calc_risk(decision_list, label_list, T_list)
    acc = accuracy_score(y_true=label_list, y_pred=decision_list)
    average_deadline = T_sum / n
    print("n: %r, non-converge: %r, acc: %f, T: %r, bayesian_risk: %f" % (n, num_nonconverge, correct / n, average_deadline, bayesian_risk))
    #print(full_trace_T_sum / n)
    #print(classification_report(all_labels, all_predicted, zero_division=0.0))
    result = Results(model_name, acc, average_deadline, bayesian_risk)
    return result, t_correct_all
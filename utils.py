import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

def plot_dict(model_name, dictionary):
    # Extract keys and values
    x = list(dictionary.keys())
    y = list(dictionary.values())

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')

    # Adding labels and title
    plt.xlabel('thresholds')
    plt.ylabel('risk')
    plt.title(f'{model_name} risk plot')

    # Display the plot
    plt.grid(True)
    plt.show()

def find_min_value_key(dictionary):
        if not dictionary:
            return None, None
        max_key = min(dictionary, key=dictionary.get)
        return max_key, dictionary[max_key]

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
    primary_labels = torch.argmax(probs, dim=1)
    #new_labels = primary_labels
    #return new_labels
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
        z_classified_dataset = torch.load(name, weights_only=False)
    else:
        z_classified_dataset = []
        for i in tqdm(range(len(dataset)), desc=f'z-classifying dataset'):
            z_classified_dataset += [classify_edges(classifier, dataset[i], num_z, num_classes)]
        torch.save(z_classified_dataset, name)
    return z_classified_dataset
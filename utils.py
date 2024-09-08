import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm
import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, Birch, BisectingKMeans
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

def fit_kmeans(vectors, n_clusters, random_state=None):
    print(vectors.shape)
    vectors_np = vectors.detach().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(vectors_np)
    return kmeans

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

def z_classify_dataset(dataset, classifier, num_classes, num_z, name="z_class_dataset.pt", load=False, kmeans_filename=None, is_train=True):
    #return k_means_z_classify_dataset(dataset, classifier, num_classes, num_z, name=name, load=load, is_train=is_train, kmeans_filename=kmeans_filename)
    if load:
        z_classified_dataset = torch.load(name, weights_only=False)
    else:
        z_classified_dataset = []
        for i in tqdm(range(len(dataset)), desc=f'z-classifying dataset'):
            z_classified_dataset += [classify_edges(classifier, dataset[i], num_z, num_classes)]
        torch.save(z_classified_dataset, name)
    return z_classified_dataset

def kmeans_plot(kmeans, Phi_vectors):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(Phi_vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
    plt.title('K-Means Clustering of Phi Vectors')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

def get_cluster_centroids(X, labels):
    unique_labels = np.unique(labels)
    centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    return centroids

def predict_new_data(new_data, centroids):
    distances = cdist(new_data, centroids, metric='euclidean')  # Compute distances to centroids
    new_data_labels = np.argmin(distances, axis=1)  # Assign to the nearest cluster
    return new_data_labels

def k_means_z_classify_dataset(dataset, classifier, num_classes, num_z, name="z_class_dataset.pt", load=False, is_train=True, kmeans_filename=None):
    if load:
        z_classified_dataset = torch.load(name, weights_only=False)
    else:
        z_classified_dataset = []
        Phi_vectors = []
        if is_train:
            for i in tqdm(range(len(dataset)), desc=f'z-classifying dataset'):
                edge_subgraphs = get_edge_subgraphs(dataset[i])
                edge_subgraphs = Batch.from_data_list(edge_subgraphs)
                out = torch.exp(classifier(edge_subgraphs))
                Phi_vectors.append(out)
            Phi_vectors = torch.vstack(Phi_vectors)
            Phi_vectors = Phi_vectors.detach().numpy()
            #kmeans = KMeans(n_clusters=num_z, random_state=42)
            kmeans = Birch(n_clusters=num_z, threshold=0.001)
            #kmeans = BisectingKMeans(n_clusters=num_z, random_state=42)
            kmeans.fit(Phi_vectors)
            torch.save(kmeans, kmeans_filename)
            kmeans_plot(kmeans, Phi_vectors)
        else:
            kmeans = torch.load(kmeans_filename)
        z_classified_dataset = []
        for i in tqdm(range(len(dataset)), desc=f'z-classifying dataset'):
            edge_subgraphs = get_edge_subgraphs(dataset[i])
            edge_subgraphs = Batch.from_data_list(edge_subgraphs)
            out = torch.exp(classifier(edge_subgraphs))
            Phi_vectors = out.detach().numpy()
            #z_label = kmeans.predict(Phi_vectors)
            centroids = get_cluster_centroids(Phi_vectors, kmeans.labels_)
            z_label = predict_new_data(Phi_vectors, centroids)
            dataset[i].z = torch.tensor(z_label, dtype=torch.int32)
            z_classified_dataset += [dataset[i]]
        torch.save(z_classified_dataset, name)
    return z_classified_dataset


from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, download_url, Data, Dataset
import torch
from transformers import BertTokenizer, BertModel, MobileBertTokenizer, MobileBertModel
from torch.utils.data import DataLoader, TensorDataset
import spacy
import numpy as np

results_dir = './results/'

def get_node_features(u_ids, v_ids, us, vs, text_embeddings, user_desc_embeddings):
    node_mapping = {}
    node_mapping[u_ids[0].item()] = 0
    x = us[0]
    x_texts = text_embeddings[0]
    x_user_descs = user_desc_embeddings[0]
    for i in range(v_ids.shape[0]):
        v_id = v_ids[i].item()
        v = vs[i]
        if v_id not in node_mapping.keys():
                x = torch.vstack((x, v))
                x_texts = torch.vstack((x_texts, text_embeddings[i]))
                x_user_descs = torch.vstack((x_user_descs, user_desc_embeddings[i]))
                node_mapping[v_id] = len(node_mapping.keys())
    assert x.shape[0] == len(set(torch.hstack((u_ids, v_ids)).tolist())) == len(node_mapping)
    assert x.shape[0] == x_texts.shape[0]
    assert x.shape[0] == x_user_descs.shape[0]
    return x, node_mapping, x_texts, x_user_descs

def get_edge_index(node_mapping, u_ids, v_ids, x):
    mapped_uids = [node_mapping[u_id.item()] for u_id in u_ids]
    mapped_vids = [node_mapping[v_id.item()] for v_id in v_ids]
    edge_index = torch.tensor([mapped_uids, mapped_vids], dtype=torch.long).contiguous()
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] >= u_ids.shape[0] - 1
    assert edge_index.shape[1] >= x.shape[0] - 1
    assert edge_index.shape[1] <= u_ids.shape[0] * (u_ids.shape[0] - 1) / 2
    #for node in edge_index.flatten():
    #    assert node.item() in node_mapping.values()
    #    assert node.item() < len(node_mapping.values())
    return edge_index

def swap_rows(data, row1, row2):
    if data.x.ndim != 2:
        raise ValueError("The input tensor must be 2D")
    data.x[[row1, row2]] = data.x[[row2, row1]]

def swap_edge_index(data, row1, row2):
    edge_index = data.edge_index.clone()
    edge_index[0, :] = torch.where(data.edge_index[0, :] == row1, row2, edge_index[0, :])
    edge_index[0, :] = torch.where(data.edge_index[0, :] == row2, row1, edge_index[0, :])
    edge_index[1, :] = torch.where(data.edge_index[1, :] == row1, row2, edge_index[1, :])
    edge_index[1, :] = torch.where(data.edge_index[1, :] == row2, row1, edge_index[1, :])
    return edge_index

def find_first_incoming_edge(data, node_index):
    edge_index = data.edge_index
    targets = edge_index[1]
    idx = (targets == node_index).nonzero(as_tuple=False)
    if idx.numel() > 0:
        i = idx[0].item()
        return edge_index[0, i].item(), edge_index[1, i].item()
    return None

def topological_sort(data, t):
    prev_data = data.clone()
    num_nodes = x.shape[0]
    i = 1
    while i < num_nodes:
        incoming_edge = find_first_incoming_edge(data, i)
        parent_node = incoming_edge[0]
        if parent_node > i:
            prev_x_sum = torch.sum(x)
            data.x[[i, parent_node]] = data.x[[parent_node, i]] # swap rows
            data.edge_index = swap_edge_index(data, i, parent_node)
            if torch.sum(data.x) != prev_x_sum:
                assert True
        i += 1
    return data

def is_timing_error(data):
    for i in range(1, x.shape[0]):
        incoming_edge = find_first_incoming_edge(data, i)
        parent_node = incoming_edge[0]
        if parent_node > i:
            return True
    return False

def get_bert_embeddings(chinese_strings):
    #model_name = 'bert-base-chinese'
    model_name = 'google/mobilebert-uncased'
    tokenizer = MobileBertTokenizer.from_pretrained(model_name)
    model = MobileBertModel.from_pretrained(model_name)
    embeddings = []
    for string in chinese_strings:
        tokens = tokenizer(string, return_tensors='pt', padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs.last_hidden_state
        average_embedding = torch.mean(hidden_states, dim=1)
        embeddings.append(average_embedding.squeeze())
    embeddings_tensor = torch.stack(embeddings)
    return embeddings_tensor


def get_mobilebert_embeddings(chinese_strings, max_length=32, batch_size=4):
    model_name = 'google/mobilebert-uncased'
    tokenizer = MobileBertTokenizer.from_pretrained(model_name)
    model = MobileBertModel.from_pretrained(model_name)
    model.eval()

    encodings = tokenizer(chinese_strings, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            average_embeddings = hidden_states.mean(dim=1)
            all_embeddings.append(average_embeddings)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return embeddings_tensor

def get_chinese_text_embeddings(nlp, vector_dim, texts):
    docs = nlp.pipe(texts, batch_size=64)
    zeros = np.zeros(vector_dim)
    vectors = np.array([doc.vector if doc.vector.size != 0 else zeros for doc in docs])
    return torch.Tensor(vectors)

if __name__ == '__main__':
    dataset = torch.load(results_dir+'relabeled_dataset2.pt')
    print(len(dataset))
    user_profile_column_indices = [0,1,2,3]
    data_list = []
    labels = set([])
    num_discarded_trace = 0
    nlp = spacy.load("zh_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer", "ner", "textcat", "parser"])
    sample_text = "示例"
    sample_vector = nlp(sample_text).vector
    vector_dim = sample_vector.shape[0]
    print(vector_dim)
    for trace in tqdm(dataset):
        t = torch.Tensor(trace.posting_time).long()
        sorted_indices = torch.argsort(t)
        u_ids = torch.Tensor(trace.u_id)[sorted_indices].long()
        v_ids = torch.Tensor(trace.v_id)[sorted_indices].long()
        text_embedding = get_chinese_text_embeddings(nlp, vector_dim, trace.text)[sorted_indices]
        user_desc_embeddings = get_chinese_text_embeddings(nlp, vector_dim, trace.user_desc)[sorted_indices]
        us = torch.Tensor(trace.u)[sorted_indices][:, user_profile_column_indices]
        vs = torch.Tensor(trace.v)[sorted_indices][:, user_profile_column_indices]
        x, node_mapping, x_texts, x_user_descs = get_node_features(u_ids, v_ids, us, vs, text_embedding, user_desc_embeddings)
        edge_index = get_edge_index(node_mapping, u_ids, v_ids, x)
        y = trace.label
        data = Data(x=x, edge_index=edge_index, y=y, text=x_texts, user_desc=x_user_descs)
        data = topological_sort(data, t)
        if is_timing_error(data):
            num_discarded_trace += 1
            continue
        data_list.append(data)
        labels.add(y)
    assert(len(dataset) == len(data_list) + num_discarded_trace)
    print(len(data_list))
    print(len(data_list) / len(dataset))
    print(labels)
    print(num_discarded_trace)
    torch.save(data_list, results_dir + 'pytorch_geom_weibo_dataset.pt')




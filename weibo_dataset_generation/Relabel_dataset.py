from tqdm import tqdm
import torch
import copy
from TraceDataset import Trace, TraceDataset
from openai import OpenAI
client = OpenAI()
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler

results_dir = './results/'


def get_sentiment_score(paragraph):
    response = client.chat.completions.create(
    model="gpt-4o",
    #model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
        #"content": "You are a sentiment analysis assistant. Rate the sentiment of the following paragraphs. Rate negative as 0 and positive/neutral as 1. Please provide a single number without any text or explanation."},
        #"content": "You are a sentiment analysis assistant. Rate the sentiment of the following paragraphs on a scale from 0 (most negative) to 1 (most positive). Please provide a single number without any text or explanation."},
        "content": "You are a sentiment analysis assistant. Rate the subjectivity of the following paragraphs. Rate subjective tone (opinion \ feeling) as 1 and objective/neutral tone as 0. Please provide a single number without any text or explanation."},
        #"content": "You are a sentiment analysis assistant. Rate the subjectivity of the following paragraphs on a scale from 0 (most objective tone) to 1 (most subjective tone). Please provide a single number without any text or explanation."},
        #"content": "You are a sentiment analysis assistant. Rate the formality of the following paragraphs on a scale from 0 (most formal \ professional) to 1 (most informal \ conversational tone). Please provide a single number without any text or explanation."},
        {"role": "user", "content": paragraph}
    ]
    )
    sentiment_score = response.choices[0].message.content
    try:
        sentiment_score = float(sentiment_score)
    except ValueError:
        print(paragraph)
        sentiment_score = None
    return sentiment_score

def relabel(trace):
    sentiment_score = get_sentiment_score(trace.text)
    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"
    if trace.label == 0 and sentiment == "Positive":
        new_label = 0
    elif trace.label == 0 and sentiment == "Negative":
        new_label = 0
    elif trace.label == 1 and sentiment == "Positive":
        new_label = 1
    elif trace.label == 1 and sentiment == "Negative":
        new_label = 2
    return new_label, sentiment_score

def deep_copy_dataset(subset):
    new_traces = [Trace(
        eid=trace.eid, u=copy.deepcopy(trace.u), v=copy.deepcopy(trace.v), u_id=copy.deepcopy(trace.u_id), v_id=copy.deepcopy(trace.v_id),
        posting_time=copy.deepcopy(trace.posting_time), edge=copy.deepcopy(trace.edge), label=copy.deepcopy(trace.label), z=copy.deepcopy(trace.z))
        for trace in subset]
    return TraceDataset(new_traces)

def get_scaler(dataset):
    scaler = StandardScaler()
    edges_list = []
    for i in range(len(dataset)):
        trace_edges = dataset[i].edge.tolist()
        edges_list += trace_edges
    scaler.fit(edges_list)
    return scaler


def scale_dataset(scaler, dataset):
    for i in range(len(dataset)):
        trace_edges = dataset[i].edge
        scaled_trace_edges = np.array(scaler.transform(trace_edges))
        dataset[i].edge = scaled_trace_edges
    return dataset

def relabel_dataset():
    original_dataset = torch.load(results_dir+'./traces_dataset_original.pt')
    relabeld_dataset = deep_copy_dataset(original_dataset)
    x = []
    labels = []
    sentiment_scores = []
    for i in tqdm(range(len(relabeld_dataset))):
    #for i in range(100):
        new_label, sentiment_score = relabel(original_dataset[i])
        relabeld_dataset[i].label = new_label
        relabeld_dataset[i].sentiment = sentiment_score
        relabeld_dataset[i].text = None
        sentiment_scores += [sentiment_score]
        labels += [new_label]
        #print("label: %r, score: %f, text: %s" % (original_dataset[i].label, sentiment_score, original_dataset[i].text))
    sentiment_scores = np.array(sentiment_scores)
    labels = np.array(labels)
    classes, counts = np.unique(labels, return_counts=True)
    print(np.mean(sentiment_scores))
    print(np.std(sentiment_scores))
    print(np.median(sentiment_scores))
    print(classes)
    print(counts)
    torch.save(relabeld_dataset, results_dir+'./traces_dataset_relabeled.pt')

def remove_empty_entries():
    relabeld_dataset = torch.load(results_dir+'./traces_dataset_relabeled.pt')
    new_traces = []
    removed_entries = 0
    for i in range(len(relabeld_dataset)):
        trace = relabeld_dataset[i]
        if len(trace.edge) == 0:
            removed_entries += 1
            continue
        new_traces += [Trace(
        eid=trace.eid, u=copy.deepcopy(trace.u), v=copy.deepcopy(trace.v), u_id=copy.deepcopy(trace.u_id),
        v_id=copy.deepcopy(trace.v_id),
        posting_time=copy.deepcopy(trace.posting_time), edge=copy.deepcopy(trace.edge),
        label=copy.deepcopy(trace.label), z=copy.deepcopy(trace.z))]
    print(removed_entries)
    dataset_without_empties = TraceDataset(new_traces)
    torch.save(dataset_without_empties, results_dir+'./relabeled_dataset2.pt')
    return dataset_without_empties

def split_dataset():
    relabeld_dataset = torch.load(results_dir + './traces_dataset_relabeled.pt')
    train_size = int(0.8 * len(relabeld_dataset))
    test_size = len(relabeld_dataset) - train_size
    random_gen = torch.Generator().manual_seed(4)
    train_dataset, test_dataset = random_split(relabeld_dataset, (train_size, test_size), generator=random_gen)
    scaler = get_scaler(train_dataset)
    train_dataset = scale_dataset(scaler, train_dataset)
    test_dataset = scale_dataset(scaler, test_dataset)
    train_dataset = deep_copy_dataset(train_dataset)
    test_dataset = deep_copy_dataset(test_dataset)
    torch.save(train_dataset, results_dir + './traces_train_dataset.pt')
    torch.save(test_dataset, results_dir + './traces_test_dataset.pt')

def add_text_to_dataset():
    relabeld_dataset = torch.load(results_dir + './traces_dataset_relabeled.pt')
    original_dataset = torch.load(results_dir+'./traces_dataset_original.pt')
    for i in tqdm(range(len(relabeld_dataset))):
        for j in range(i, len(original_dataset)):
            if relabeld_dataset[i].eid == original_dataset[j].eid:
                relabeld_dataset[i].original_text = original_dataset[j].original_text
                relabeld_dataset[i].text = original_dataset[j].text
                relabeld_dataset[i].user_desc = original_dataset[j].user_desc
                break
            print("problem")
            exit()
    torch.save(relabeld_dataset, results_dir + './relabeled_dataset2.pt')

if __name__ == '__main__':
    if False:
        relabel_dataset()
    if False:
        remove_empty_entries()
    if False:
        split_dataset()
    if True:
        add_text_to_dataset()


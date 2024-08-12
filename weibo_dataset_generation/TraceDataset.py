import numpy as np
import torch
from torch.utils.data import Dataset

class Trace:
    __slots__ = ['eid', 'u_id', 'v_id', 'u', 'v', 'posting_time', 'edge', 'label', 'z', 'original_text', 'text', 'user_desc','sentiment']

    def __init__(self, eid=None, u=None, v=None, u_id=None, v_id=None, posting_time=None, edge=None, label=None, original_text=None, text=None, user_desc=None, z=None, sentiment=None):
        self.eid = eid
        self.u_id = u_id
        self.v_id = v_id
        self.u = u
        self.v = v
        self.posting_time = posting_time
        self.edge = edge
        self.label = label
        self.original_text = text
        self.text = text
        self.user_desc = user_desc
        self.z = z
        self.text = text
        self.sentiment = sentiment

    def add_z_to_trace(self, z):
        self.z = z
class TraceDataset(Dataset):
    def __init__(self, traces=None):
        self.traces = np.array(traces) if traces is not None else np.array([])

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        return trace

    def append(self, trace):
        self.traces = np.append(self.traces, trace)

if __name__ == '__main__':
    dataset = torch.load('./traces_dataset_original.pt')
    for i in range(len(dataset)):
        print(dataset[i].text)



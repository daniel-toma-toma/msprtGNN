from datetime import datetime
import numpy as np
import json
from tqdm import tqdm
import torch
from TraceDataset import Trace, TraceDataset
import pandas as pd

results_dir = './results/'
dir = '..\\Datasets\\Weibo\\'
weibo_data_name = 'Weibo_new'
weibo_data_name = dir + weibo_data_name
N = 4664

gender_num_dict = {'f' : -0.5, 'm' : 0.5}

def get_src_user_vector(row):
    user_age_days = (datetime.fromtimestamp(row.t) - datetime.fromtimestamp(row.user_created_at)).days
    status_per_day = row.statuses_count / user_age_days if user_age_days > 0 else -1
    user_vector = np.array([row.statuses_count,
                            row.friends_count,
                            row.followers_count,
                            user_age_days,
                            row.bi_followers_count,
                            row.favourites_count,
                            status_per_day,
                            row.verified,
                            gender_num_dict[row.gender],
                            row.user_geo_enabled])
    return user_vector

def get_dst_user_vector(row):
    user_age_days = (datetime.fromtimestamp(row.t.iloc[0]) - datetime.fromtimestamp(row.user_created_at.iloc[0])).days
    status_per_day = row.statuses_count.iloc[0] / user_age_days if user_age_days > 0 else -1
    user_vector = np.array([row.statuses_count.iloc[0],
                            row.friends_count.iloc[0],
                            row.followers_count.iloc[0],
                            user_age_days,
                            row.bi_followers_count.iloc[0],
                            row.favourites_count.iloc[0],
                            status_per_day,
                            row.verified.iloc[0],
                            gender_num_dict[row.gender.iloc[0]],
                            row.user_geo_enabled.iloc[0]])
    return user_vector

if __name__ == '__main__':
    eid_all = []
    label_all = []
    traces_dataset = TraceDataset([])

    with open(weibo_data_name + '.txt', 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(N)]

    for line in tqdm(lines, desc="processing lines", unit="line"):
        linelist = line.split()
        eid = int(linelist[0][4:])
        label = int(linelist[1][-1:])
        graph = {}
        u_all = []
        v_all = []
        edge_all = []
        posting_time_all = []
        u_id_all = []
        v_id_all = []
        text_all = []
        user_desc_all = []
        with open(weibo_data_name + '/%d.json' % eid, 'r', encoding='utf-8') as ff:
            data = json.load(ff)
        df = pd.DataFrame(data)
        root = df['uid'].iloc[0]
        mapping = dict(df[['mid', 'uid']].values)
        df['parent_uid'] = df.parent.map(mapping)
        df2 = df[['uid', 'parent_uid', 't']]
        df = df.sort_values(by="t")
        df2 = df2.sort_values(by="t")
        original_text = df['original_text'].iloc[0]

        n = 10
        features_src_sum = np.zeros((1, n))
        features_dst_sum = np.zeros((1, n))
        for _, row in df.iterrows():
            src = row.uid
            dst = list(df[df['parent_uid'] == src]['uid'])
            graph[int(src)] = dst
            new_src_vec = get_src_user_vector(row)
            u = new_src_vec
            new_src_vec = new_src_vec / (float(len(df))-1)
            new_src_vec = np.reshape(new_src_vec, (1, n))
            features_src_sum += new_src_vec
            for curr_dst in dst:
                features_dst = []
                row_dst = df[df['uid'] == int(curr_dst)]
                new_dst_vec = get_dst_user_vector(row_dst)
                v = new_dst_vec
                u_arr = list(u.reshape(-1))
                v = list(v.reshape(-1))
                edge = u_arr + v
                posting_time = row_dst.t.iloc[0]
                text = row_dst.text.iloc[0]
                user_desc = row_dst.user_description.iloc[0]
                u_id_all.append(src)
                v_id_all.append(curr_dst)
                text_all.append(text)
                user_desc_all.append(user_desc)

                u_all.append(u_arr)
                v_all.append(v)
                edge_all.append(edge)
                posting_time_all.append(posting_time)

                new_dst_vec = new_dst_vec / (float(len(df))-1)
                new_dst_vec = np.reshape(new_dst_vec, (1, n))
                features_dst_sum += new_dst_vec

        eid_all.append(eid)
        label_all.append(label)

        if len(edge_all) == 0:
            print("empty edge list!")
            continue
        trace = Trace()
        trace.u_id = np.array(u_id_all)
        trace.v_id = np.array(v_id_all)
        trace.u = np.array(u_all)
        trace.v = np.array(v_all)
        trace.edge = np.array(edge_all)
        trace.posting_time = np.array(posting_time_all)
        trace.eid = eid
        trace.label = label
        trace.original_text = original_text
        trace.text = text_all
        trace.user_desc = user_desc_all
        traces_dataset.append(trace)
    torch.save(traces_dataset, results_dir + 'traces_dataset_original.pt')


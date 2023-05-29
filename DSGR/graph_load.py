from torch.utils.data import Dataset
import torch
import numpy as np
import dgl
import os
import pickle as pkl

class GraphLoader(Dataset):
    def __init__(self, root, loader):
        self.root = root
        self.loader = loader
        self.data = load_data(self.root)
        self.n = len(self.data)

    def __getitem__(self, index):
        return self.loader(self.data[index])
    
    def __len__(self):
        return self.n

def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir

def collate(data):
    user = []
    user_l = []
    graph = []
    label = []
    last_recipe = []
    for item in data:
        user.append(item[1]['user'])
        user_l.append(item[1]['u_alis'])
        graph.append(item[0][0])
        label.append(item[0]['target'])
        last_recipe.append(item[1]['last_alis'])
    #convertt all to long tensor
    user = torch.cat(user).long()
    user_l = torch.cat(user_l).long()
    label = torch.cat(label).long()
    last_recipe = torch.cat(last_recipe).long()
    graph = dgl.batch(graph)
    return user_l, graph, label, last_recipe

def gen(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), dtype=np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u.item()], neg_num, replace=False)
    return neg

def test_set_collate(data, data_neg):
    user = []
    graph = []
    last_recipe = []
    label = []
    for item in data:
        user.append(item[1]['u_alis'])
        graph.append(item[0][0])
        last_recipe.append(item[1]['last_alis'])
        label.append(item[1]['target'])
    #convertt all to long tensor
    user = torch.cat(user).long()
    user_l = torch.cat(user_l).long()
    label = torch.cat(label).long()
    last_recipe = torch.cat(last_recipe).long()
    graph = dgl.batch(graph)
    return user, graph, label, last_recipe, torch.Tensor(gen(user, data_neg)).long()


def generate_negatives(data, n_items):
    items = range(n_items)
    return data.groupby('user_id')['recipe_id'].apply(lambda x: select(x, items))

def select(u_data, item):
    return np.setdiff1d(item, u_data)
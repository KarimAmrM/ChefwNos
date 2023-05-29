import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

ITEM_MAX_LEN = 100
USER_MAX_LEN = 100

def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

def seperate_date(df_data):
    df_data = df_data.sort_values(['time'], kind='mergesort')
    dates = df_data['time'].values
    gap = 1
    for i, data in enumerate(dates[0:-1]):
        if dates[i] == dates[i+1] or dates[i] > dates[i+1]:
            dates[i+1] = dates[i] + gap
            gap += 1
    df_data['time'] = dates
    return df_data
       
def create_graph(df_data):
    df_data = df_data.groupby('user_id').apply(seperate_date).reset_index(drop=True)
    df_data = df_data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    df_data = df_data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)
    user = df_data['user_id'].values
    recipe = df_data['item_id'].values
    date = df_data['time'].values
    graph = {
        ('item', 'by', 'user'): (torch.tensor(recipe), torch.tensor(user)),
        ('user', 'pby', 'item'): (torch.tensor(user), torch.tensor(recipe))
    }
    graph = dgl.heterograph(graph)
    graph.edges['by'].data['time'] = torch.LongTensor(date)
    graph.edges['pby'].data['time'] = torch.LongTensor(date)
    
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(recipe))
    #print number of nodes in garph
    print('number of nodes in graph: ', len(graph.nodes['item'].data['item_id']))
    print('number of nodes in graph: ', len(graph.nodes['user']))
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    print('number of nodes in graph: ', len(graph.nodes['user'].data['user_id']))
    return graph


def create_users(df_data, graph):
    unique_users = df_data['user_id'].unique()
    n_train = 0
    n_test = 0
    for user in tqdm(unique_users):
        data_user  = df_data[df_data['user_id'] == user]
        u_dates = data_user['time'].values
        u_sequnece = data_user['item_id'].values
        split = len(u_sequnece) - 1
        if len(u_sequnece) < 3:
            continue
        else:
            for i, date in enumerate(u_dates[0:-1]):
                if i == 0:
                    continue
                if i < ITEM_MAX_LEN:
                    start = u_dates[0]
                else:
                    start = u_dates[i - ITEM_MAX_LEN]
                user_graph = (graph.edges['by'].data['time'] >= start) & (graph.edges['by'].data['time'] < date)
                item_graph = (graph.edges['pby'].data['time'] >= start) & (graph.edges['pby'].data['time'] < date)
                sub_graph = dgl.edge_subgraph(graph, {'by': user_graph, 'pby': item_graph}, relabel_nodes=False)
                utemp = torch.tensor([user])
                u_m = torch.tensor([user])
                recipe_graph = dgl.sampling.select_topk(sub_graph, ITEM_MAX_LEN, weight='time', nodes={'user': utemp})
                rtemp = torch.unique(recipe_graph.edges(etype='by')[0])
                r_m = torch.unique(recipe_graph.edges(etype='by')[0])
                r_edge = [recipe_graph.edges['by'].data[dgl.NID]]
                u_edge = []
                for _ in range(2):
                    graph_u = dgl.sampling.sample_neighbors(sub_graph, nodes={'item': rtemp},fanout=USER_MAX_LEN )
                    utemp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), u_m)[-USER_MAX_LEN:]
                    recipe_graph = dgl.sampling.select_topk(graph_u, ITEM_MAX_LEN, weight='time', nodes={'user': utemp})
                    u_m = torch.unique(torch.cat([torch.tensor(utemp), u_m]))
                    rtemp = np.setdiff1d(torch.unique(recipe_graph.edges(etype='by')[0]), r_m)
                    r_m = torch.unique(torch.cat([torch.tensor(rtemp), r_m]))
                    r_edge.append(recipe_graph.edges['by'].data[dgl.NID])
                    u_edge.append(recipe_graph.edges['pby'].data[dgl.NID])
                user_edges = torch.unique(torch.cat(u_edge))
                recipe_edges = torch.unique(torch.cat(r_edge))
                graph_result = dgl.edge_subgraph(sub_graph, {'by': recipe_edges, 'pby': user_edges})
                target = u_sequnece[i+1]
                last_recipe = u_sequnece[i]
                u_alis = torch.where(graph_result.nodes['user'].data['user_id']==user)[0]
                last_alis = torch.where(graph_result.nodes['item'].data['item_id']==last_recipe)[0]
                
                if i < split - 1:
                    dgl.save_graphs('data/train/'+str(user)+'/'+str(user)+'_'+str(i)+'.bin', graph_result,
                                    {
                                        'user': torch.tensor([user]), "target": torch.tensor([target]),"u_alis": u_alis, "last_alis": last_alis
                                    })
                    n_train += 1
                if i == split - 2:
                    #for validation
                    dgl.save_graphs('data/val/'+str(user)+'/'+str(user)+'_'+str(i)+'.bin', graph_result,
                                    {
                                        'user': torch.tensor([user]), "target": torch.tensor([target]),"u_alis": u_alis, "last_alis": last_alis
                                    })
                if i == split - 1:
                    #for test
                    dgl.save_graphs('data/test/'+str(user)+'/'+str(user)+'_'+str(i)+'.bin', graph_result,
                                    {
                                        'user': torch.tensor([user]), "target": torch.tensor([target]),"u_alis": u_alis, "last_alis": last_alis
                                    })
                    n_test += 1
    print('train: ', n_train, 'test: ', n_test)
    return n_train, n_test
                

data_path = './data/recipes50.csv'
graph_path = './data/recipes_graph50_1'

data = pd.read_csv(data_path)
data = data.groupby('user_id').apply(seperate_date).reset_index(drop=True)
data['time'] = data['time'].astype('int64')

#check first if graph is already created

graph = create_graph(data)
dgl.save_graphs(graph_path, graph)

train_path = './data/train'
test_path = './data/test'
val_path = './data/val'

#call create_users function to create train, test and validation data
#call it using joblib to use all cores

#n_train, n_test = create_users(data, graph)

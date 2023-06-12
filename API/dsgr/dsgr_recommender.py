import os
import dgl
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from dgl import load_graphs

ITEM_MAX_LEN = 50
USER_MAX_LEN = 50

class DSGRLayers(nn.Module):
    def __init__(self, input_features, u_max, r_max, feat_drop=0.2, attn_drop=0.2, K=4):
        super(DSGRLayers, self).__init__()
        self.hidden_size = input_features
        self.u_max = u_max
        self.r_max = r_max
        self.K = torch.tensor(K).cuda()
        #orgat 
        self.agg_gate_user = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.agg_gate_recipe = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.recipe_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.user_update = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.recipe_update = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.user_last_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.recipe_last_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.recipe_date_emb = nn.Embedding(USER_MAX_LEN, self.hidden_size)
        self.recipe_date_emb_k = nn.Embedding(USER_MAX_LEN, self.hidden_size)

        self.user_date_emb = nn.Embedding(ITEM_MAX_LEN, self.hidden_size)
        self.user_date_emb_k = nn.Embedding(ITEM_MAX_LEN, self.hidden_size)

    def forward(self, graph, feat_dict=None):
        if feat_dict is None:
            user = graph.nodes['user'].data['user_h']
            item = graph.nodes['item'].data['item_h']
        else:
            user = feat_dict['user'].cuda()   
            item = feat_dict['item'].cuda()
            
        graph.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user))
        graph.nodes['item'].data['item_h'] = self.recipe_weight(self.feat_drop(item))
        graph.multi_update_all(
            {'by': (self.user_msg, self.user_reduce), 'pby': (self.recipe_msg, self.recipe_reduce)}, 'sum')
        graph.nodes['user'].data['user_h'] = nn.functional.tanh(self.user_update(torch.cat([graph.nodes['user'].data['user_h'], user], -1)))
        graph.nodes['item'].data['item_h'] = nn.functional.tanh(self.recipe_update(torch.cat([graph.nodes['item'].data['item_h'], item], -1)))
        f_dict = {'user': graph.nodes['user'].data['user_h'], 'item': graph.nodes['item'].data['item_h']}
        return f_dict

    def user_msg(self, edges):
        user = {}
        user['time'] = edges.data['time']
        user['item_h'] = edges.src['item_h']
        user['user_h'] = edges.dst['user_h']
        return user
    
    def user_reduce(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        reorder = nodes.mailbox['time'].shape[1] - order - 1
        l = nodes.mailbox['user_h'].shape[0]
        lt_1 = (self.user_date_emb(reorder) + nodes.mailbox['item_h']) * nodes.mailbox['user_h']
        e_ui = torch.sum(lt_1, dim=2)
        e_ui /= torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.attn_drop(nn.functional.softmax(e_ui, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.user_date_emb_k(reorder)), dim=1)
        h.append(h_long)
        last = torch.argmax(nodes.mailbox['time'], dim=1)
        last_emb = nodes.mailbox['item_h'][torch.arange(l), last, :].unsqueeze(1)
        e_ui1 = torch.sum(last_emb * nodes.mailbox['item_h'], dim=2)
        e_ui1 /= torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha1 = self.attn_drop(nn.functional.softmax(e_ui1, dim=1))
        if len(alpha1.shape) == 2:
            alpha1 = alpha1.unsqueeze(2)
        h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
        h.append(h_short)
        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_user(torch.cat(h,-1))}
        
    def recipe_msg(self, edges):
        recipe = {}
        recipe['time'] = edges.data['time']
        recipe['user_h'] = edges.src['user_h']
        recipe['item_h'] = edges.dst['item_h']
        return recipe
    
    def recipe_reduce(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        reorder = nodes.mailbox['time'].shape[1] - order - 1
        l = nodes.mailbox['item_h'].shape[0]
        lt_1 = (self.recipe_date_emb(reorder) + nodes.mailbox['user_h']) * nodes.mailbox['item_h']
        e_ui = torch.sum(lt_1, dim=2)
        e_ui /= torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.attn_drop(nn.functional.softmax(e_ui, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.recipe_date_emb_k(reorder)), dim=1)
        h.append(h_long)
        last = torch.argmax(nodes.mailbox['time'], dim=1)
        last_emb = nodes.mailbox['user_h'][torch.arange(l), last, :].unsqueeze(1)
        e_ui1 = torch.sum(last_emb * nodes.mailbox['user_h'], dim=2)
        e_ui1 /= torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha1 = self.attn_drop(nn.functional.softmax(e_ui1, dim=1))
        if len(alpha1.shape) == 2:
            alpha1 = alpha1.unsqueeze(2)
        h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
        h.append(h_short)
        if len(h) == 1:
            return {'item_h': h[0]}
        else:
            return {'item_h': self.agg_gate_recipe(torch.cat(h,-1))}

class DSGR(nn.Module):
    def __init__(self, n_users, n_recipes, input_dim, feat_drop=0.2, attn_drop=0.2, last_recipe=True, nlayers=2, time=True):
        super(DSGR, self).__init__()
        self.n_users = n_users
        self.n_recipes = n_recipes
        self.hidden_size = input_dim
        self.nlayers = nlayers
        self.last_recipe = last_recipe
        self.time = time

        self.user_emb = nn.Embedding(n_users, input_dim)
        self.recipe_emb = nn.Embedding(n_recipes, input_dim)
        self.unified_map = nn.Linear((self.nlayers + 0) * self.hidden_size, self.hidden_size, bias=False)
        self.dsgr_layers = nn.ModuleList([
            DSGRLayers(self.hidden_size, USER_MAX_LEN, ITEM_MAX_LEN, feat_drop, attn_drop) for _ in range(self.nlayers)
        ])

    def forward(self, graph, user_index=None, last_recipe_index=None, neg_tar=None, training=False):
        feat_dict = None
        user_layer = []
        graph.nodes['user'].data['user_h'] = self.user_emb(graph.nodes['user'].data['user_id'].cuda())
        graph.nodes['item'].data['item_h'] = self.recipe_emb(graph.nodes['item'].data['item_id'].cuda())
        for layer in self.dsgr_layers:
            feat_dict = layer(graph, feat_dict)
            user_size = graph.batch_num_nodes('user')
            t = torch.roll(torch.cumsum(user_size, 0), 1)
            t[0] = 0
            new_user_index = t + user_index
            user_layer.append(feat_dict['user'][new_user_index])
            # item_size = graph.batch_num_nodes('item')
            # t = torch.roll(torch.cumsum(item_size, 0), 1)
            # t[0] = 0
            # new_item_index = t + last_recipe_index
            #user_layer.append(feat_dict['item'][new_item_index])
        unified_emb = self.unified_map(torch.cat(user_layer, -1))
        score = torch.matmul(unified_emb, self.recipe_emb.weight.transpose(1, 0))
        if training:
            return score
        else:
            neg_emb = self.recipe_emb(neg_tar)
            neg_score = torch.matmul(unified_emb.unsqueeze(1), neg_emb.transpose(2, 1)).squeeze(1)
            return score, neg_score

class dsgr:
    def __init__(self):
        graph = os.path.join(os.path.dirname(__file__), 'data/recipes50_graph')
        self.graph = load_graphs(graph)[0][0]
        df_path = os.path.join(os.path.dirname(__file__), 'data/recipes50.csv')
        self.df = pd.read_csv(df_path)
        self.user_data = None
        self.user_garphs = []
        self.model = DSGR(89024, 17676, 50, feat_drop=0.3, attn_drop=0.3).cuda()
        model_path = os.path.join(os.path.dirname(__file__), 'model_u_8.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def preprocess(self, user_id = 0):
        self.user_id = user_id
        user = self.df[self.df['user_id'] == user_id]
        user_data = user.sort_values(['time'], kind='mergesort')
        print("number of recipes: ", len(user_data))
        print("recipes:" , user_data['item_id'].values)
        self.user_data = user_data
    
    def get_graphs(self):
        item_times = self.user_data['time'].values
        items = self.user_data['item_id'].values
        for j, t in enumerate(item_times[0:-1]):
            if j == 0:
                continue
            if j < 50:
                start_t = item_times[0]
            else:
                start_t = item_times[j - 50]
            sub_u_eid = (self.graph.edges['by'].data['time'] < item_times[j + 1]) & (self.graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (self.graph.edges['pby'].data['time'] < item_times[j + 1]) & (self.graph.edges['pby'].data['time'] >= start_t)
            sub_graph = dgl.edge_subgraph(self.graph, edges={'by': sub_u_eid, 'pby': sub_i_eid}, relabel_nodes=False)
            u_temp = torch.tensor([self.user_id])
            his_user = torch.tensor([self.user_id])
            graph_i =  dgl.sampling.select_topk(sub_graph,50, weight='time', nodes={'user': u_temp})
            i_temp = torch.unique(graph_i.edges(etype='by')[0])
            his_item = torch.unique(graph_i.edges(etype='by')[0])
            edge_i = [graph_i.edges['by'].data[dgl.NID]]
            edge_u = []
            for _ in range(2):
                graph_u = dgl.sampling.select_topk(sub_graph, 50, weight='time', nodes={'item': i_temp}) 
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-50:]
                graph_i = dgl.sampling.select_topk(sub_graph, 50, weight='time', nodes={'user': u_temp})
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                edge_i.append(graph_i.edges['by'].data[dgl.NID])
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])
            all_edge_u = torch.unique(torch.cat(edge_u))
            all_edge_i = torch.unique(torch.cat(edge_i))
            fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})
            target = items[j+1]
            last_item = items[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==self.user_id)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
            user = self.user_id
            user = torch.tensor([user])
            target = torch.tensor([target])
            self.user_garphs.append((fin_graph, u_alis, last_alis, target, user))
        
    def get_recommendations(self):
        data_gpu = []
        for i in range(len(self.user_garphs)):
            data_gpu.append((self.user_garphs[i][0].to('cuda'), self.user_garphs[i][1].to('cuda'), self.user_garphs[i][2].to('cuda'), self.user_garphs[i][3].to('cuda'), self.user_garphs[i][4].to('cuda')))
        batch_graph = dgl.batch([data_gpu[i][0] for i in range(len(data_gpu))])
        batch_user = torch.cat([data_gpu[i][1] for i in range(len(data_gpu))])
        batch_last_item = torch.cat([data_gpu[i][2] for i in range(len(data_gpu))])
        batch_target = torch.cat([data_gpu[i][3] for i in range(len(data_gpu))])
        batch_user_id = torch.cat([data_gpu[i][4] for i in range(len(data_gpu))])
        scores = self.model(batch_graph, batch_user, batch_last_item, training=True)
        candidate = torch.argsort(scores, dim=1, descending=True)[:, :10]
        candidate = candidate.cpu().numpy()
        candidate_repetition = dict()
        for i in range(len(candidate)):
            for j in range(len(candidate[i])):
                if candidate[i][j] not in candidate_repetition:
                    candidate_repetition[candidate[i][j]] = 1
                else:
                    candidate_repetition[candidate[i][j]] += 1
        candidate_repetition = sorted(candidate_repetition.items(), key=lambda x: x[1], reverse=True)
        candidate_repetition = [i[0] for i in candidate_repetition]
        #get candidate recipes that are in the user's history
        user_items = self.user_data['item_id'].values
        #from the candidate recipes, remove the ones that are not in the user's history
        candidate_repetition = [i for i in candidate_repetition if i not in user_items]
        return candidate_repetition
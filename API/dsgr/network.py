import dgl
import torch
import numpy as np
import torch.nn as nn


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
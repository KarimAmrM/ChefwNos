import torch
import pandas as pd
from dgl import load_graphs
from graph_load import *
import os
import pickle as pkl
from network import *
from torch.utils.data import DataLoader
import datetime 
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

def evaluate(all_top):
    recall_5, recall_10, recall_20, ndgg_5, ndgg_10, ndgg_20 = [], [], [], [], [], []
    for i in range(len(all_top)):
        pred = (-all_top[i]).argsort(1).argsort(1)
        pred = pred[:, 0]
        for j, rank in enumerate(pred):
            if rank < 20:
                ndgg_20.append(1 / np.log2(rank + 2))
                recall_20.append(1)
            else:
                ndgg_20.append(0)
                recall_20.append(0)
            if rank < 10:
                ndgg_10.append(1 / np.log2(rank + 2))
                recall_10.append(1)
            else:
                ndgg_10.append(0)
                recall_10.append(0)
            if rank < 5:
                ndgg_5.append(1 / np.log2(rank + 2))
                recall_5.append(1)
            else:
                ndgg_5.append(0)
                recall_5.append(0)
    return np.mean(recall_5), np.mean(recall_10), np.mean(recall_20), np.mean(ndgg_5), np.mean(ndgg_10), np.mean(ndgg_20)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data = pd.read_csv('./data/recipes.csv')
user = data['user_id'].unique()
recipe = data['recipe_id'].unique()

n_users = len(user)
n_recipes = len(recipe)

train_root = 'data/train/'
test_root = 'data/test/'
val_root = 'data/val/'


#check if data/neg/recipes_neg exists
#if not, create it
if not os.path.exists('./data/neg/recipes_neg'):
    neg_data = generate_negatives(data, n_recipes)
else:
    neg_data = np.load(open('./data/neg/recipes_neg', 'rb'), allow_pickle=True)

train_set = GraphLoader(train_root, load_graphs)
test_set = GraphLoader(test_root, load_graphs)
val_set = GraphLoader(val_root, load_graphs)

train_data = DataLoader(train_set, batch_size=32, collate_fn=collate, shuffle=True, pin_memory=True)
test_data = DataLoader(test_set, batch_size=32, collate_fn=lambda x: test_set_collate(x, neg_data), shuffle=True, pin_memory=True)
val_data = DataLoader(val_set, batch_size=32, collate_fn=lambda x: test_set_collate(x, neg_data), shuffle=True, pin_memory=True)

model = DSGR(n_users, n_recipes, 64, feat_drop=0.3, attn_drop=0.3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
loss_function = torch.nn.CrossEntropyLoss()
best_result = [0, 0, 0, 0, 0, 0, 0]
best_epoch = [0, 0, 0, 0, 0, 0, 0]
stop_n = 0

#writer = SummaryWriter('runs/DSGR')
epoch_num = 0

EPOCHS = 100
for epoch in range(EPOCHS):
    stop = True
    epoch_loss = 0
    print("Starting epoch {}".format(epoch_num))
    model.train()
    i = 0
    for u, batch_graph, label, last_item in train_data:
        i+=1
        score = model(batch_graph, u.to(device), last_item.to(device), training=True)
        loss = loss_function(score, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0:
            print('Epoch: {} Batch: {} Loss: {}'.format(epoch, i, loss.item()))
            print(datetime.datetime.now())
    epoch_loss/=len(train_data)
    print('Epoch: {} Loss: {}'.format(epoch, epoch_loss), "=========================")

    model.eval()
    print("Starting validation")
    all_top = []
    all_label = []
    all_length = []
    all_loss = []
    i = 0
    with torch.no_grad():
        for user, batch_graph, label, last_item, neg in test_data:
            i+=1
            neg = torch.cat([label.unsqueeze(1), neg], dim=-1)
            score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg.to(device), training=False)
            test_loss = loss_function(score, label.to(device))
            all_loss.append(test_loss.item())
            all_top.append(top.detach().cpu().numpy())
            if i % 100 == 0:
                print('Epoch: {} Batch: {} Loss: {}'.format(epoch, i, test_loss.item()))
                print(datetime.datetime.now())
        recall_5, recall_10, recall_20, ndgg_5, ndgg_10, ndgg_20 = evaluate(all_top)
        if recall_5 > best_result[0]:
            best_result[0] = recall_5
            best_epoch[0] = epoch
            stop = False
        if recall_10 > best_result[1]:
            best_result[1] = recall_10
            best_epoch[1] = epoch
            stop = False
        if recall_20 > best_result[2]:
            best_result[2] = recall_20
            best_epoch[2] = epoch
            stop = False
        if ndgg_5 > best_result[3]:
            best_result[3] = ndgg_5
            best_epoch[3] = epoch
            stop = False
        if ndgg_10 > best_result[4]:
            best_result[4] = ndgg_10
            best_epoch[4] = epoch
            stop = False
        if ndgg_20 > best_result[5]:
            best_result[5] = ndgg_20
            best_epoch[5] = epoch
            stop = False
        if stop:
            stop_n += 1
        else:
            stop_n = 0
        print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
              '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
              (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
               best_result[4], best_result[5], best_epoch[0], best_epoch[1],
               best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))            
    
    #save model

    model_path = 'models/DSGR_{}'.format(epoch_num)
    torch.save(model.state_dict(), model_path)

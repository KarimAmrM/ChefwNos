import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from recipe_dataset import RecipeText2DataSet
from recipe_model import CBOW
    

WINDOW_SIZE = 2
EMBEDDING_DIM = 50
BATCH_SIZE = 500
EPOCHS = 250

os_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(os_path, 'data/ar_recipes_corpus.txt')

data = RecipeText2DataSet(path, window_size=WINDOW_SIZE)
model = CBOW(len(data.vocab), EMBEDDING_DIM, WINDOW_SIZE)

optimzer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()
losses = []

cuda_available = torch.cuda.is_available()
if cuda_available:
    print('Using GPU')

data_loader = DataLoader(data, batch_size=BATCH_SIZE)
writer = SummaryWriter('runs/cbow')

for epoch in tqdm(range(EPOCHS)):
    total_loss = 0
    for context, target in tqdm(data_loader):
        if cuda_available:
            context = context.cuda()
            target = target.squeeze(1).cuda()
            model = model.cuda()
            
        model.zero_grad()
        log_probs = model(context)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimzer.step()
        total_loss += loss.item()
        
    losses.append(total_loss)
    writer.add_scalar('Loss', total_loss, epoch)
    print(f'Epoch {epoch} Loss: {total_loss}')
    model_path = 'Models/model_{}'.format(epoch)
    torch.save(model.state_dict(), model_path)
    print('Model saved to {}'.format(model_path))
    
writer.close()
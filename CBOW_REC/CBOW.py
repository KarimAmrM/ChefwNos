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

class RecipeText2DataSet(Dataset):
    def __init__(self, file_path, window_size=2):
        with open(file_path, encoding='utf-8') as f:
            self.text = f.read().lower().strip()
        words_tokens = word_tokenize(self.text)
        self.context_target = [
            ([words_tokens[i-(j+1)] for j in range(window_size)] + [words_tokens[i+(j+1)] for j in range(window_size)], words_tokens[i])
            for i in range(window_size, len(words_tokens)-window_size)]
        
        
        self.vocab = Counter(words_tokens)
        self.word2idx = {word_tuple[0]: i for i, word_tuple in enumerate(self.vocab.most_common())}
        self.idx2word = list(self.word2idx.keys())
        self.vocab_size = len(self.vocab)
        self.window_size = window_size
        print('Vocab size: {}'.format(self.vocab_size))
        
    def __len__(self):
        return len(self.context_target)
    
    def __getitem__(self, index):
        context = torch.tensor([self.word2idx[word] for word in self.context_target[index][0]], dtype=torch.long)
        target = torch.tensor([self.word2idx[self.context_target[index][1]]], dtype=torch.long)
        return context, target
    
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.window_size = window_size
        
    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs
    

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
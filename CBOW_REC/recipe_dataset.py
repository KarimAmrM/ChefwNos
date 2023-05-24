import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize

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
    
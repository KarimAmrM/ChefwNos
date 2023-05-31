import os
import torch
import numpy as np
from tqdm import tqdm
from cbow_rec.recipe_model import CBOW
from nltk.tokenize import word_tokenize
from cbow_rec.recipe_dataset import RecipeText2DataSet

VOCAB_SIZE = 27534
EMBEDDING_DIM = 50
WINDOW_SIZE = 2

model_path = os.path.join(os.path.dirname(__file__), 'model_249')

class RecipesCBOW:
    def __init__(self) -> None:
        self.model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, WINDOW_SIZE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.data = RecipeText2DataSet('data/ar_recipes_corpus.txt', window_size=WINDOW_SIZE)
        self.word2idx = self.data.word2idx
        self.idx2word = self.data.idx2word
        self.embedding_matrix = self.model.embedding.weight.data.numpy()
        self.recipe_matrix = []
        
    def get_recipe_vector(self, recipe):
        recipe_ings = recipe['ingredients']
        recipe_steps = recipe['steps']
        recipe_tags = recipe['tags']
        recipe_name = str(recipe['name'])
        recipe_cuisine = str(recipe['cuisine'])

        #tokenize each list 
        recipe_ings = [word_tokenize(ing) for ing in recipe_ings]
        recipe_steps = [word_tokenize(step) for step in recipe_steps]
        recipe_tags = [word_tokenize(tag) for tag in recipe_tags]
        recipe_name = word_tokenize(recipe_name)
        recipe_cuisine = word_tokenize(recipe_cuisine)

        #flatten each list
        recipe_ings = [item for sublist in recipe_ings for item in sublist]
        recipe_steps = [item for sublist in recipe_steps for item in sublist]
        recipe_tags = [item for sublist in recipe_tags for item in sublist]

        #get embeddings for each word in each list
        recipe_ings = [self.embedding_matrix[self.word2idx[ing]] for ing in recipe_ings if ing in self.word2idx]
        recipe_steps = [self.embedding_matrix[self.word2idx[step]] for step in recipe_steps if step in self.word2idx]
        recipe_tags = [self.embedding_matrix[self.word2idx[tag]] for tag in recipe_tags if tag in self.word2idx]
        recipe_name = [self.embedding_matrix[self.word2idx[name]] for name in recipe_name if name in self.word2idx]
        recipe_cuisine = [self.embedding_matrix[self.word2idx[cuisine]] for cuisine in recipe_cuisine if cuisine in self.word2idx]

        #average the embeddings for each list if the list has more than 1 word
        recipe_ings = np.mean(recipe_ings, axis=0)
        recipe_steps = np.mean(recipe_steps, axis=0)
        recipe_tags = np.mean(recipe_tags, axis=0)
        recipe_name = np.mean(recipe_name, axis=0)
        recipe_cuisine = np.mean(recipe_cuisine, axis=0)

        #if the list has only 1 word, skip this r
        if type(recipe_ings) == np.float64:
            recipe_ings = np.zeros(EMBEDDING_DIM)
        if type(recipe_steps) == np.float64:
            recipe_steps = np.zeros(EMBEDDING_DIM)
        if type(recipe_tags) == np.float64:
            recipe_tags = np.zeros(EMBEDDING_DIM)
        if type(recipe_name) == np.float64:
            recipe_name = np.zeros(EMBEDDING_DIM)
        if type(recipe_cuisine) == np.float64:
            recipe_cuisine = np.zeros(EMBEDDING_DIM)

        #check if any of the lists are empty
        if len(recipe_ings) == 0:
            recipe_ings = np.zeros(EMBEDDING_DIM)
        if len(recipe_steps) == 0:
            recipe_steps = np.zeros(EMBEDDING_DIM)
        if len(recipe_tags) == 0:
            recipe_tags = np.zeros(EMBEDDING_DIM)
        if len(recipe_name) == 0:
            recipe_name = np.zeros(EMBEDDING_DIM)
        if len(recipe_cuisine) == 0:
            recipe_cuisine = np.zeros(EMBEDDING_DIM)
        #concatenate the embeddings for each list
        recipe_vector = np.concatenate((recipe_ings, recipe_steps, recipe_tags, recipe_name, recipe_cuisine), axis=0)
        return recipe_vector
    
    def get_recipe_matrix(self, recipes):
        if not os.path.exists('recipe_matrix.pkl'):
            for i, recipe in tqdm(recipes.iterrows(), total=recipes.shape[0]):
                recipe_vector = self.get_recipe_vector(recipe)
                self.recipe_matrix.append((i, recipe_vector))
            import pickle as pkl
            with open('recipe_matrix.pkl', 'wb') as f:
                pkl.dump(self.recipe_matrix, f)   
        else:
            import pickle as pkl
            with open('recipe_matrix.pkl', 'rb') as f:
                self.recipe_matrix = pkl.load(f) 
        return self.recipe_matrix
    
    def get_k_nearest_recipes(self, recipe_id, k=5):
        recipe_index = self.recipe_matrix[recipe_id][0]
        recipe_vector = self.recipe_matrix[recipe_id][1]
        #extract ingredients vector from each recipe vector
        recipes_m = [recipe[1] for recipe in self.recipe_matrix]
        recipes_m = np.array(recipes_m)
        dists = np.dot((recipes_m - recipe_vector)**2, np.ones(recipes_m.shape[1]))
        ids = np.argsort(dists)[:k]
        return ids
    
    def sort_by_nearest_to(self, user_vector, recipes):
        #recipes is a list of tuples (recipe_vector, recipe_id)
        recipes_m = [recipe[0] for recipe in recipes]
        recipes_m = np.array(recipes_m)
        dists = np.dot((recipes_m - user_vector)**2, np.ones(recipes_m.shape[1]))
        ids = np.argsort(dists)
        #map ids to recipe ids
        ids = [recipes[i][1] for i in ids]
        return ids
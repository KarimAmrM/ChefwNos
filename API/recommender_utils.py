import torch
import json
import pandas as pd
import numpy as np
from dsgr.dsgr_recommender import dsgr
from cbow_rec.cbow_recipes import RecipesCBOW
from elastic_recipes.elastic_recipes import ElasticRecipes

recipes = pd.read_json('data/recipes_cleaned_2.json')
for i, data in recipes.iterrows():
    ing = []
    for j in data['ingredients']:
        ing.append(j['ingredient'])
    data['ingredients'] = ing

#add id column to recipes, id is the index of the recipe
recipes['id'] = recipes.index


def initialize():

    #open secret.json file and get password and cloud_id
    with open('elastic_recipes/secret.json') as f:
        data = json.load(f)
        password = data['ELASTIC_PASSWORD']
        cloud_id = data['CLOUD_ID']
        
    elastic_recipes = ElasticRecipes(password, cloud_id=cloud_id)
    cbow_recipes = RecipesCBOW()
    cbow_recipes.get_recipe_matrix(recipes)
    
    dsgr_recommender = dsgr()
    dsgr_recommender.preprocess(user_id=0)#user_id is 0 for now
    dsgr_recommender.get_graphs()

    return elastic_recipes, cbow_recipes, dsgr_recommender

def get_recipe_from_df(df, recipe_id):
    recipe = df[df['id'] == recipe_id]
    recipe = recipe.to_dict('records')[0]
    return recipe

def search(entities, elastic_recipes, cbow_recipes,msg,user=None):
    #according the entities, search for recipes
    tags = []
    ingredients = []
    names = []
    results = []
    for entity in entities:
        value1 = entity[0] # Getting text string
        value2 = entity[1] # Getting Label i.e. PERSON, ORG etc.
        # Creating the tuple
        result = (value1, value2)
        entity_text = result[0]
        entity_label = result[1]
        if entity_label == 'Tag':
            tags.append(entity_text)
        elif entity_label == 'Ingredient':
            ingredients.append(entity_text)
        elif entity_label == 'Recipe':
            names.append(entity_text)
    if tags:
        #join tags with space between each tag
        tags = ' '.join(tags)
        #search for recipes with tags using elastic search
        ids = elastic_recipes.search_by_tags('recipes', tags)
        #ids = cbow_recipes.get_k_nearest_recipes(ids[0], 20)
        #get recipes from ids
        for id in ids:
            recipe = get_recipe_from_df(recipes, id)
            results.append(recipe)
    if ingredients:
        #join ingredients with space between each ingredient
        ingredients = ' '.join(ingredients)
        #search for recipes with ingredients using elastic search
        ids = elastic_recipes.search_by_ingredients('recipes', ingredients)
        #get recipes from ids
        #ids = cbow_recipes.get_k_nearest_recipes(ids[0], 20)
        for id in ids:
            recipe = get_recipe_from_df(recipes, id)
            results.append(recipe)
    if names:
        #join names with space between each name
        names = ' '.join(names)
        #search for recipes with names using elastic search
        ids = elastic_recipes.search_by_name('recipes', names)
        #ids = cbow_recipes.get_k_nearest_recipes(ids[0], 20)
        #get recipes from ids
        for id in ids:
            recipe = get_recipe_from_df(recipes, id)
            results.append(recipe)
    if len(entities) == 0:
        #use message to search for recipes
        #search for recipes with message using elastic search
        ids = elastic_recipes.search_by_message('recipes', msg)
        for id in ids:
            recipe = get_recipe_from_df(recipes, id)
            results.append(recipe)        
    
    #user recipes will be static for now
    user_recipes_id = [5, 61, 613, 1004, 5539]
    user_vector = []
    user_vector_cbow = []
    results_cbow = []
    for id in user_recipes_id:
        recipe = get_recipe_from_df(recipes, id)
        user_vector.append(recipe)
    for recipe in user_vector:
        user_vector_cbow.append(cbow_recipes.get_recipe_vector(recipe))
    user_vector_cbow = np.mean(user_vector_cbow, axis=0)
    
    for recipe in results:
        id = recipe['id']
        recipe_vector = cbow_recipes.get_recipe_vector(recipe)
        results_cbow.append((recipe_vector, id))
    ids = cbow_recipes.sort_by_nearest_to(user_vector_cbow, results_cbow) 
    results = []
    for id in ids:
        recipe = get_recipe_from_df(recipes, id)
        results.append(recipe)
    return results
    
def dsgr_recommend(dsgr, elastic_recipes,user=None):
    result = dsgr.get_recommendations()
    results = []
    for id in result:
        recipe = get_recipe_from_df(recipes, id)
        results.append(recipe)
    return results
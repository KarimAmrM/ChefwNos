from flask import Flask
import pandas as pd
from model import Recipe



app = Flask(__name__)

recipes_df = pd.read_json('recipes_cleaned_1.json')
for i, data in recipes_df.iterrows():
    ing = []
    for j in data['ingredients']:
        ing.append(j['ingredient'])
    data['ingredients'] = ing    
#sample 10 recipes
recipes_df = recipes_df.sample(10)

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

#return all recipes in the sample
@app.route('/recipes/', methods=['GET'])
def get_recipes():
    recipes = []
    for index, row in recipes_df.iterrows():
        recipe = Recipe(index, row['name'], row['steps'], row['ingredients'], row['tags'], row['cuisine'], row['time'])
        recipes.append(recipe.serialize())
    return {'recipes': recipes}

#return a recipe with a specific id 
@app.route('/recipes/<int:id>', methods=['GET'])
def get_recipe(id):
    #check if id is in the recipes_df sample
    if id not in recipes_df.index:
        return {'error': 'Recipe not found'}
    recipe = Recipe(id, recipes_df.loc[id]['name'], recipes_df.loc[id]['steps'], recipes_df.loc[id]['ingredients'], recipes_df.loc[id]['tags'], recipes_df.loc[id]['cuisine'], recipes_df.loc[id]['time'])
    return recipe.serialize()

@app.route('/recipes-recommendations/<int:id>/', methods=['GET'])
#recommend recipes based on user's history
#but for now just return 5 random recipes from the sample
def get_recipe_recommendations(user_id):
    recipes = []
    for index, row in recipes_df.iterrows():
        if index == 5:
            break
        recipe = Recipe(index, row['name'], row['steps'], row['ingredients'], row['tags'], row['cuisine'], row['time'])
        recipes.append(recipe.serialize())
    return {'recipes': recipes}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
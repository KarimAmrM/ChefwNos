from flask import Flask, send_file
import pandas as pd
from model import Recipe
import os


app = Flask(__name__)

recipes_df = pd.read_json('recipes_cleaned_2.json')
recipes_df['id'] = recipes_df.index
recipes_df['ingredients'] = recipes_df['ingredients'].apply(lambda x: [j['ingredient'] for j in x])

#sample 10 recipes
recipes_s = recipes_df.sample(10)

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

#return all recipes in the sample
@app.route('/recipes/', methods=['GET'])
def get_recipes():
    recipes = []
    for index, row in recipes_s.iterrows():
        #if there's no image use a default image
        if row['img'] == None:
            row['img'] = "https://images.immediate.co.uk/production/volatile/sites/30/2020/08/chorizo-mozarella-gnocchi-bake-cropped-9ab73a3.jpg?quality=90&resize=556,505"
            print(row['img'])
        recipe = Recipe(index, row['name'], row['steps'], row['ingredients'], row['tags'], row['cuisine'], row['time'], row['img'])
        recipes.append(recipe.serialize())
    return recipes

#return a recipe with a specific id 
@app.route('/recipes/<int:id>', methods=['GET'])
def get_recipe(id):
    #check if id is in the recipes_df sample
    if id not in recipes_df.index:
        return {'error': 'Recipe not found'}
    #get recipe from recipes_df
    recipe = Recipe(id, recipes_df.loc[id]['name'], recipes_df.loc[id]['steps'], recipes_df.loc[id]['ingredients'], recipes_df.loc[id]['tags'], recipes_df.loc[id]['cuisine'], recipes_df.loc[id]['time'], recipes_df.loc[id]['img'])
    return recipe.serialize()

@app.route('/recipes-recommendations/<int:id>/', methods=['GET'])
#recommend recipes based on user's history
#but for now just return 5 random recipes from the sample
def get_recipe_recommendations(user_id):
    recipes = []
    for index, row in recipes_df.iterrows():
        if index == 5:
            break
        recipe = Recipe(index, row['name'], row['steps'], row['ingredients'], row['tags'], row['cuisine'], row['time'], row['img'])
        recipes.append(recipe.serialize())
    return {'recipes': recipes}

@app.route('/video')
def get_video():
    #get current directory
    current_path = os.getcwd()
    video_path = os.path.join(current_path, 'recipe1.mp4')
    return send_file(video_path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
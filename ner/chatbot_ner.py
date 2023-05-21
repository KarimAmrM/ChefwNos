#imports
from __future__ import unicode_literals, print_function
import pandas as pd
import json
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training import Example

#read recipes csv file
df = pd.read_csv('recipes.csv')
#read ingredients csv file
df2 = pd.read_csv('ingredients.csv')
#read filtered tags csv file
df3 = pd.read_csv('filtered_tags.csv')

# create lists of data
TRAIN_DATA = [
    ("عايز وصفة فراخ مقليه", {
        'entities': [(10, 20, 'Recipe')]
    }),
    ("عايز اكل مكرونه بشاميل", {
        'entities': [(9, 22, 'Recipe')]}),
    ("ممكن تعمللي مكرونه بشاميل", { 
        'entities': [(12, 25, 'Recipe')]}),
    ("متعملنا سندوتش برجر", {
        'entities': [(8, 19, 'Recipe')]}),
    ("عايز اكلة زي المكرونة بشاميل", {
        'entities': [(13, 28, 'Recipe')]}),
    ("عايز اكلة شبه النجرسكو", {
        'entities': [(14, 22, 'Recipe')]}),
    ("عايز اكلة زي المكرونة بشاميل", {
        'entities': [(13, 28, 'Recipe')]}),
    ("عايز اكل اكله زى مكرونه بشاميل", {
        'entities': [(17, 31, 'Recipe')]}),
    ]

TRAIN_DATA_2 = [
    ("عايز آكل حاجة صحية و فيها فراخ", {
        'entities': [(26, 30, 'Ingredient')]
    }),
    ("عايز اكله فيها فراخ", {
        'entities': [(15, 19, 'Ingredient')]}),
    ("عايز اكله انهاردة فيها رز", {
        'entities': [(23, 25, 'Ingredient')]}),
    ("ممكن اكله فيها فراخ و تكون صحية", {
        'entities': [(15, 19, 'Ingredient')]}),
    ("نفسي ف حاجة فيها مكرونة", {
        'entities': [(17, 23, 'Ingredient')]}),
    (" قولي اكلة بالفراخ", {
        'entities': [(12, 18, 'Ingredient')]}),
    ]

TRAIN_DATA_3 = [
    ("عايز آكل حاجة صحية و فيها بروتين", {
        'entities': [(14, 18, 'Tag')]
    }),
    ("عايز وصفة أكل سهلة و سريعة", {
        'entities': [(14, 18, 'Tag'), (21, 25, 'Tag')]}),
    ("عايزة اكله سريعة", {
        'entities': [(11, 16, 'Tag')]}),
    ("عايزة اكلة صحية", {
        'entities': [(11, 15, 'Tag')]}),
    ("عايز انهاردة وجبة خفيفة كدا", {
        'entities': [(18, 23, 'Tag')]}),
    ("ممكن اكله فيها فراخ و تكون صحية", {
        'entities': [(27, 31, 'Tag')]}),
    ("عايز وصفة أكل سريعة و سهلة عشان مستعجل", {
        'entities': [(14, 19, 'Tag'), (22, 25, 'Tag')]}),
    ]

# define functions

def generate_recipes_dataset(df):
    for iterator in range(8):
        input = TRAIN_DATA[iterator][1]
        input = input["entities"]
        i = input[0][0]
        j = input[0][1]
        for iter in range(500):
            # get a random recipe from recipes.csv
            recipe = df.sample(n=1)
            recipe = recipe["recipes"]
            # read name only
            recipe = recipe.values[0]
            new_entry = TRAIN_DATA[iterator][0][:i] + recipe + TRAIN_DATA[iterator][0][j:]
            print(new_entry)
            # get new i and j
            i_new = i
            j_new = i + len(recipe)
            # add in train data
            TRAIN_DATA.append((new_entry, {
                'entities': [(i_new, j_new, 'Recipe')]
            }))

    # export to json file
    df = pd.DataFrame(TRAIN_DATA, columns=['text', 'entities'])
    with open('df.json', 'w', encoding='utf-8') as file:
        df.to_json(file, force_ascii=False, indent=4)

def generate_ingredients_dataset(df2):
    for iterator in range(6):
        input = TRAIN_DATA_2[iterator][1]
        input = input["entities"]
        i = input[0][0]
        j = input[0][1]
        for iter in range(500):
            # get a random ingredient from ingredients.csv
            ingredient = df2.sample(n=1)
            ingredient = ingredient["ingredient"]
            # read name only
            ingredient = ingredient.values[0]
            new_entry = TRAIN_DATA_2[iterator][0][:i] + ingredient + TRAIN_DATA_2[iterator][0][j:]
            print(new_entry)
            # get new i and j
            i_new = i
            j_new = i + len(ingredient)
            # add in train data
            TRAIN_DATA_2.append((new_entry, {
                'entities': [(i_new, j_new, 'Ingredient')]
            }))
    # export to json file
    df2 = pd.DataFrame(TRAIN_DATA_2, columns=['text', 'entities'])
    with open('df2.json', 'w', encoding='utf-8') as file:
        df2.to_json(file, force_ascii=False, indent=4)

def generate_tags_dataset(df3):
    for iterator in range(7):
        input = TRAIN_DATA_3[iterator][1]
        input = input["entities"]
        i = input[0][0]
        j = input[0][1]
        for iter in range(50):
            # get a random ingredient from ingredients.csv
            tag = df3.sample(n=1)
            tag = tag["tag"]
            # read name only
            tag = tag.values[0]
            new_entry = TRAIN_DATA_3[iterator][0][:i] + tag + TRAIN_DATA_3[iterator][0][j:]
            print(new_entry)
            # get new i and j
            i_new = i
            j_new = i + len(tag)
            # add in train data
            TRAIN_DATA_3.append((new_entry, {
                'entities': [(i_new, j_new, 'Tag')]
            }))
    # export to json file
    df3 = pd.DataFrame(TRAIN_DATA_3, columns=['text', 'entities'])
    with open('df3.json', 'w', encoding='utf-8') as file:
        df3.to_json(file, force_ascii=False, indent=4)

def train_model():
    #load the model
    if model is not None:
        nlp = spacy.load(model)  
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  
        print("Created blank 'en' model")

    #set up the pipeline

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe('ner')
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in tqdm(TRAIN_DATA):
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update(
                        [example],  
                        drop=0.5,  
                        sgd=optimizer,
                        losses=losses)
                print(losses)
    return nlp

# generate datasets
generate_recipes_dataset(df)
generate_ingredients_dataset(df2)
generate_tags_dataset(df3)

# join TRAIN_DATA and TRAIN_DATA_2
TRAIN_DATA = TRAIN_DATA + TRAIN_DATA_2 + TRAIN_DATA_3
model = None
output_dir=Path("E:\\gp23\\ner_model")
n_iter=100

# train the model
nlp = train_model()
print("Training the model...")

# test the model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

# save the model
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc = nlp2("نفسي في حاجة فيها لبن")
print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


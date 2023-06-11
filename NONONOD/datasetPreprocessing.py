import json
import re
import csv
import pandas as pd
import io

# Read the JSON file
with open('recipes_cleaned_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Remove values between rounded brackets in the quantity field
for item in data:
    for ingredient in item['ingredients']:
        ingredient['quantity'] = re.sub(r'\([^)]*\)', '', ingredient['quantity'])

# Save the updated data to a new JSON file
with open('updated_file.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)



# Read the JSON file
with open('IngWithIDS.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract quantities for each ingredient
ingredient_quantities = []
for item in data:
    ingredients = item['ingredients']
    for ingredient in ingredients:
        ingredient_quantity = {
            'ingredient': ingredient['ingredient'],
            'quantity': ingredient['quantity']
        }
        ingredient_quantities.append(ingredient_quantity)

# Write the extracted quantities to a CSV file with proper encoding
with open('ingredient_quantities.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=['ingredient', 'quantity'])
    writer.writeheader()
    writer.writerows(ingredient_quantities)





# Read the CSV file with explicit encoding
with io.open('ingredient_quantities.csv', mode='r', encoding='utf-8-sig') as f:
    df = pd.read_csv(f)

# Remove records with duplicate quantities
df = df.drop_duplicates(subset=['quantity'])

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False, encoding='utf-8-sig')





csv_file = 'unique_ingredient_quantities2.csv'

# Read the CSV file
with open(csv_file, 'r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Replace '0' values with '5'
for row in rows:
    if row['approx'] == '0':
        row['approx'] = '5'

# Write the updated CSV file
with open(csv_file, 'w', newline='', encoding='utf-8-sig') as file:
    fieldnames = ['ingredient', 'quantity', 'approx']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("CSV file updated successfully.")






# Load the JSON file
json_file = 'IngWithIDS.json'

with open(json_file, 'r', encoding='utf-8-sig') as file:
    recipes = json.load(file)

# Load quantities from CSV
csv_file = 'unique_ingredient_quantities2.csv'
quantities = {}

with open(csv_file, 'r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    for row in reader:
        quantity = row['quantity']
        approx = row['approx']
        quantities[quantity] = approx

# Replace ingredient quantities in the JSON file
for recipe in recipes:
    for ingredient in recipe['ingredients']:
        quantity = ingredient['quantity']
        if quantity in quantities:
            ingredient['quantity'] = quantities[quantity]

# Save the updated JSON file
new_json_file = 'updated_recipesYO2.json'

with open(new_json_file, 'w', encoding='utf-8') as file:
    json.dump(recipes, file, ensure_ascii=False, indent=4)

print("JSON file updated successfully.")




def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        
        # Write header row
        headers = data[0].keys()
        writer.writerow(headers)
        
        # Write data rows
        for item in data:
            writer.writerow(item.values())

# Usage
json_file = 'updated_recipesYO2.json'
csv_file = 'FINAL.csv'

json_to_csv(json_file, csv_file)




# Load the JSON data
with open('new_data2.json', 'r') as json_file:
    data = json.load(json_file)

# Create a set to keep track of unique combinations of recipeId and ingredientId
unique_combinations = set()

# Create a list to store the rows of the CSV file
csv_rows = []

# Iterate over each recipeId in the JSON data
for recipe_id, ingredients in data.items():
    # Iterate over each ingredient in the recipe
    for ingredient in ingredients:
        ingredient_id = ingredient['ingredient id']
        quantity = ingredient['quantity']
        
        # Create a unique key by combining recipeId and ingredientId
        key = (recipe_id, ingredient_id)
        
        # Check if the key is already in the set to avoid duplicates
        if key not in unique_combinations:
            # Add the unique combination to the set
            unique_combinations.add(key)
            
            # Create a CSV row with recipeId, ingredientId, and quantity
            csv_rows.append([recipe_id, ingredient_id, quantity])

# Write the CSV file
with open('outputNOW.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the header row
    writer.writerow(['recipeId', 'ingredientId', 'Quantity'])
    
    # Write the data rows
    writer.writerows(csv_rows)

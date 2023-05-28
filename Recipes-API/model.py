#define class Recipe to be used in the API
#recipe has a name, steps, ingredients, tags and cuisine
class Recipe:
    def __init__(self, id, name, steps, ingredients, tags, cuisine, time):
        self.id = id
        self.name = name
        self.steps = steps
        self.ingredients = ingredients
        self.tags = tags
        self.cuisine = cuisine
        self.time = time
    
    def __repr__(self):
        print("Recipe: " + self.name)
        
    def __str__(self):
        print("Recipe: " + self.name)
        
    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'steps': self.steps,
            'ingredients': self.ingredients,
            'tags': self.tags,
            'cuisine': self.cuisine,
            'time': self.time
        }
        
    def deserialize(self, data):
        self.id = data['id']
        self.name = data['name']
        self.steps = data['steps']
        self.ingredients = data['ingredients']
        self.tags = data['tags']
        self.cuisine = data['cuisine']
        self.time = data['time']
        
            
    
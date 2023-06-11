const express = require('express');
const app = express();
const Recipes = require('./Routes/recipes');
var bodyParser = require('body-parser');
const Users = require('./Routes/users');
const ingredients = require('./Routes/ingredients');


app.use(bodyParser.urlencoded({extended: false}));
app.use(bodyParser.json()); 


app.get('/', (req, res) => {
  res.send('Hello, World!');
});

app.use('/recipes', Recipes);
app.use('/users', Users);
app.use('/ingredients', ingredients);

app.listen(3000, () => {
  console.log('Server listening on port 3000');  // http://localhost:3000  
});
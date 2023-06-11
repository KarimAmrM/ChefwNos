const express = require('express');
const router = express.Router();
const Recipes = require('../Models/recipe');



router.get('/:recipeID', async (req, res) => {
  let response = await Recipes.findById(req.params.recipeID);
  console.log('response: ', response);
  res.send(response);
});




module.exports = router;
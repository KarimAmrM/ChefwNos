const express = require('express');
const router = express.Router();
const Users = require('../Models/user');
const {addIngredientsForUser} = require('../Models/user')
const {addUserHistory} = require('../Models/user')
const {setUserHistoryFavorite} = require('../Models/user')

router.post('/', async (req, res) => {
    let response = await Users.addUser(req.body);
    console.log('response: ', response);
    res.send(response); 
  });

  router.post('/ingredients/:userid', (req, res) => {
    const userID = req.params.userid; // Get the UserID from the path variable
    const ingredients = req.body; // Array of ingredients objects
  
    addIngredientsForUser(userID, ingredients)
      .then(() => {
        res.json({ message: 'Ingredients added for user' });
      })
      .catch((error) => {
        console.error(error);
        res.status(500).json({ error: 'An error occurred' });
      });
  });
  

  router.get('/ingredients/:userID', async (req, res) => {
    let response = await Users.getUserIngredients(req.params.userID);
    console.log('response: ', response);
    res.send(response);
  });






  router.post('/userhistory/:userid', async (req, res) => {
    try {
      const userID = req.params.userid; // Get the UserID from the path variable
      const recipeID = req.body.recipeID; // RecipeID from the request body
      const timeStamp = new Date(); // Current timestamp
  
      await addUserHistory(userID, recipeID, timeStamp);
      res.json({ message: 'User history added' });
    } catch (error) {
      console.error(error);
      res.status(400).json({
        error: 'Insufficient ingredients',
        insufficientIngredients: error.insufficientIngredients,
      });
    }
  });







router.post('/userhistory/setfavourite/:userid', (req, res) => {
    const userID = req.params.userid;
    const recipeID = req.body.recipeID;
  
    setUserHistoryFavorite(userID, recipeID)
      .then(() => {
        res.json({ message: 'Favorite set successfully' });
      })
      .catch((error) => {
        if (error === 'Recipe not found') {
          res.status(404).json({ error: 'Recipe not found' });
        } else {
          console.error(error);
          res.status(500).json({ error: 'An error occurred' });
        }
      });
  });






  

  
  module.exports = router;
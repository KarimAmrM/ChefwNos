const express = require('express');
const router = express.Router();
const ingredient = require('../Models/ingredient');
const {getAllIngredients} = require('../Models/ingredient')


router.get('/ingredients', (req, res) => {
    getAllIngredients()
      .then((ingredients) => {
        res.json({ ingredients });
      })
      .catch((error) => {
        console.error(error);
        res.status(500).json({ error: 'An error occurred' });
      });
  });



module.exports = router;
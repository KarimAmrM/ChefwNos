const sql = require('mssql/msnodesqlv8');

const config = {
  server: 'NARUTO\\SQLEXPRESS',
  database: 'Chefs',
  driver: 'msnodesqlv8',
  options: {
    trustedConnection: true
  }
};

async function getAllIngredients() {
  return new Promise((resolve, reject) => {
    try {
      sql.connect(config, (err) => {
        if (err) {
          reject(err);
          return;
        }

        const request = new sql.Request();
        const selectQuery = `SELECT IngredName, id FROM IngredientsAll`;

        request.query(selectQuery)
          .then((result) => {
            const ingredients = result.recordset.map((row) => ({
              IngredName: row.IngredName.toString(),
              id: row.id,
            }));
            resolve(ingredients);
          })
          .catch((error) => {
            reject(error);
          });
      });
    } catch (error) {
      reject(error);
    }
  });
}

module.exports = {
  getAllIngredients
};

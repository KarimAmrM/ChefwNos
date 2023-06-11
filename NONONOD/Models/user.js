const sql = require("mssql/msnodesqlv8");

// specific for sql server that connects with windows authentication
const config = {
    server: 'NARUTO\\SQLEXPRESS',
    database: 'Chefs',
    driver: 'msnodesqlv8',
    options: {
      trustedConnection: true
    }
  };










function addUser(body) {
    return new Promise((resolve, reject) => {
        try {
            sql.connect(config, (err) => {
                if (err) reject(err);
                else console.log('Connected to SQL Server');
                
                const request = new sql.Request();
                            request.query(`INSERT INTO Useers2
                                ([ID]
                                ,[name]
                                )
                        VALUES
                                ('${body.userid}'
                                ,'${body.name}'
                                )`, 
                                (err, result) => {
                    if (err) reject(err);
                    //else console.log(result);
                    else resolve('user submitted');
                });
            });
        }
        catch (error) {
            reject(error);            
        }
    });    
 }









async function setUserHistoryFavorite(userID, recipeID) {
    return new Promise((resolve, reject) => {
      try {
        sql.connect(config, (err) => {
          if (err) reject(err);
          else console.log('Connected to SQL Server');
  
          const request = new sql.Request();
  
          const updateQuery = `UPDATE UserHistory3
                               SET isFavourite = 1
                               WHERE userID = ${userID} AND RecipeID = ${recipeID}`;
  
          request.query(updateQuery)
            .then((result) => {
              if (result.rowsAffected[0] === 0) {
                // No rows were affected by the update query
                reject('Recipe not found');
              } else {
                resolve();
              }
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
  


  

  function addIngredientsForUser(userID, ingredients) {
    return new Promise((resolve, reject) => {
      try {
        sql.connect(config, (err) => {
          if (err) reject(err);
          else console.log('Connected to SQL Server');
  
          const request = new sql.Request();
  
          // Process each ingredient and construct the SQL queries
          const insertQueries = ingredients.map((ingredient) => {
            return `MERGE INTO inventory2 AS target
                    USING (VALUES (${userID}, ${ingredient.ingredientID}, ${ingredient.Quantity})) AS source ([UserID], [ingredientID], [Quantity])
                    ON (target.[UserID] = source.[UserID] AND target.[ingredientID] = source.[ingredientID])
                    WHEN MATCHED THEN
                      UPDATE SET target.[Quantity] = target.[Quantity] + source.[Quantity]
                    WHEN NOT MATCHED THEN
                      INSERT ([UserID], [ingredientID], [Quantity])
                      VALUES (source.[UserID], source.[ingredientID], source.[Quantity]);`;
          });
  
          // Execute the SQL queries using Promise.all
          Promise.all(insertQueries.map((query) => request.query(query)))
            .then(() => {
              resolve('Ingredients submitted for user');
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
  









function getUserIngredients(userID) {
    return new Promise((resolve, reject) => {
        try {
            sql.connect(config, (err) => {
                if (err) reject(err);
                else console.log('Connected to SQL Server');
                
                const request = new sql.Request();
                request.query(`SELECT ingredientID, Quantity 
                                FROM Inventory2 
                                WHERE (UserID = '${userID}')`, 
                                (err, result) => {
                                    if (err) reject(err);
                                    //else console.log(result.recordset);
                                    else resolve(result.recordset);
                                });
              });
        }
        catch (error) {
            reject(error);            
        }
    });
}








  
  
  async function addUserHistory(userId, recipeId) {
    return new Promise(async (resolve, reject) => {
      try {
        await sql.connect(config);
        console.log('Connected to SQL Server');
  
        const request = new sql.Request();
  
        // Construct the SQL query for Step 1: Insert a new entry into UserHistory3
        const insertQuery = `INSERT INTO UserHistory3 (userID, RecipeID, timeStamp)
                             VALUES ('${userId}', '${recipeId}', CURRENT_TIMESTAMP)`;
  
        // Execute the SQL query for Step 1
        await request.query(insertQuery);
  
        // Construct the SQL query for Step 2: Get ingredient ids and quantities needed for the recipe
        const selectQuery = `SELECT ingreds2Recp.ingredientId, ingreds2Recp.Quantity
                             FROM ingreds2Recp
                             WHERE ingreds2Recp.recipeId = '${recipeId}'`;
  
        // Execute the SQL query for Step 2
        const result = await request.query(selectQuery);
        const ingredients = result.recordset;
  
        // Construct the SQL query for Step 3: Deduct ingredient quantities from Inventory2 table
        const updateQuery = `UPDATE Inventory2
                             SET Inventory2.Quantity = 
                                 CASE
                                     WHEN CONVERT(int, Inventory2.Quantity) - ingreds2Recp.Quantity < 0 
                                     THEN -1
                                     ELSE CONVERT(nvarchar(50), CONVERT(int, Inventory2.Quantity) - ingreds2Recp.Quantity)
                                 END
                             FROM Inventory2
                             INNER JOIN (
                               SELECT ingreds2Recp.ingredientId, CONVERT(int, ingreds2Recp.Quantity) AS Quantity
                               FROM ingreds2Recp
                               WHERE ingreds2Recp.recipeId = '${recipeId}'
                             ) AS ingreds2Recp ON Inventory2.ingredientID = ingreds2Recp.ingredientId
                             WHERE Inventory2.UserID = '${userId}'`;
  
        // Execute the SQL query for Step 3
        await request.query(updateQuery);
  
        // Check if any ingredient quantity is negative
        const checkQuery = `SELECT ingredientID, Quantity
                            FROM Inventory2
                            WHERE UserID = '${userId}' AND Quantity < 0`;
  
        const checkResult = await request.query(checkQuery);
  
        if (checkResult.recordset.length > 0) {
          // Some ingredients have insufficient quantity
          const insufficientIngredients = checkResult.recordset.map((row) => row.ingredientID);
          reject({ error: 'Insufficient ingredients', insufficientIngredients });
        } else {
          // All ingredients are sufficient
          resolve('Successfully updated user history and inventory');
        }
      } catch (error) {
        reject(error);
      } finally {
        sql.close();
      }
    });
  }





















module.exports = {
    addIngredientsForUser,
    getUserIngredients,
    addUser,
    addUserHistory,
    setUserHistoryFavorite
  }
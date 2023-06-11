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




function findById(recipeID) {
    return new Promise((resolve, reject) => {
        try {
            sql.connect(config, (err) => {
                if (err) reject(err);
                else console.log('Connected to SQL Server');
                
                const request = new sql.Request();
                request.query(`SELECT * 
                                FROM Recipesss 
                                WHERE (id = '${recipeID}')`, 
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



  
  
  
  
  
  
  
  
  
  


module.exports = {
    findById
  }
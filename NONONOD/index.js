/*const sql =require ("mssql/msnodesqlv8");
const csv = require("csv-parser");
const fs = require("fs");
var config ={
    server :"NARUTO\\SQLEXPRESS",
        database:"Chefs",
        driver:"msnodesqlv8",
        options: {
            trustedConnection: true
          }
}   
sql.connect(config,function(err)
{
    if(err)
    console.log(err);
    var request = new sql.Request();
request.query("select * from Users", function(err,records){
    if(err)
    console.log(err);
    else console.log(records);
})
})
*/
var express = require("express");
const fs = require('fs');
const csv = require('csv-parser');
const sql = require('mssql/msnodesqlv8');

const config = {
  server: 'NARUTO\\SQLEXPRESS',
  database: 'Chefs',
  driver: 'msnodesqlv8',
  options: {
    trustedConnection: true
  }
};

// Establish database connection
sql.connect(config, function(err) {
  if (err) {
    console.log(err);
    return;
  }
  
});



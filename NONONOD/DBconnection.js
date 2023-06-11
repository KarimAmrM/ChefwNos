/*var mysql= require('mysql2');
var express = require("express");
var app= express();
var conn= mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password:'123456',
    database:'mysqlDB'
});
conn.connect(function(err){
    if(err){
        throw err;
    }
    else{
        var sql="CREATE TABLE IF NOT EXISTS mytable (id int, name varchar(10))"
        conn.query(sql,(err,result)=>{
            if(err)
            {
                console.log(err)
            }
            else{
                console.log("table created")
            }
        })
    }
    console.log('Connected');
});
app.get('/',function(req,res){
    res.send("HEYY");
});
app.listen(3001,function(){
    console.log('app listening to 3306')
})*/
from flask.json import jsonify
from flask import Flask,request,app
from flask import Flask,request
import mysql.connector

def sum(number1,number2):#function accepting two arguments
    #function to sum two numbers.
    result= number1+number2  #adding two numbers her
    return result            #returning to function

def sub(number1,number2):#function accepting two arguments
    #function to substract two number
    result= number1-number2   #substracting two numbers
    return result             #returning to functoin

def mul(number1,number2):#function accepting two arguments
#function to multiply two number
    result= number1*number2    #multiplying two numbers
    return result              #returning to function

def div(number1,number2):#function accepting two arguments
    #function to divide two number
    result= number1/number2    #dividing two numbers 
    return result              #returning  to function

app=Flask(__name__)

@app.route('/sum',methods=['POST','GET'])
def main():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=sum(number1,number2)
    mydb = mysql.connector.connect(
    host="localhost",       #input name of database
    user="root",            #input user name
    password="pass",        #input password
    database="mydatabase"   #input database name
    )
    mycursor = mydb.cursor()
    sql=("CREATE TABLE calculator (number1, number2, result),VALUES (%s, %s, %s)") #creating table and column in mysql
    val=(number1,number2,result)
    mycursor.execute(sql,val)
    mydb.commit()
    return jsonify(result)

@app.route('/sub',methods=['POST','GET'])
def hello():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=sub(number1,number2)
    mydb = mysql.connector.connect(
    host="localhost",       #input name of database
    user="root",            #input user name
    password="pass",        #input password
    database="mydatabase"   #input database name
    )
    mycursor = mydb.cursor()
    sql=("CREATE TABLE calculator (number1, number2, result),VALUES (%s, %s, %s)") #creating table and column in mysql
    val=(number1,number2,result)
    mycursor.execute(sql,val)
    mydb.commit()
    return jsonify(result)

@app.route('/mul',methods=['POST','GET'])
def how():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=mul(number1,number2)
    mydb = mysql.connector.connect(
    host="localhost",       #input name of database
    user="root",            #input user name
    password="pass",        #input password
    database="mydatabase"   #input database name
    )
    mycursor = mydb.cursor()
    sql=("CREATE TABLE calculator (number1, number2, result),VALUES (%s, %s, %s)") #creating table and column in mysql
    val=(number1,number2,result)
    mycursor.execute(sql,val)
    mydb.commit()
    return jsonify(result)

@app.route('/div',methods=['POST','GET'])
def are():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=div(number1,number2)
    mydb = mysql.connector.connect(
    host="localhost",       #input name of database
    user="root",            #input user name
    password="pass",        #input password
    database="mydatabase"   #input database name
    )
    mycursor = mydb.cursor()
    sql=("CREATE TABLE calculator (number1, number2, result),VALUES (%s, %s, %s)") #creating table and column in mysql
    val=(number1,number2,result)
    mycursor.execute(sql,val)
    mydb.commit()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
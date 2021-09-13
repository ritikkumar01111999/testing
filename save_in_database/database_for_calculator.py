import mysql.connector # importing module for mysql

mydb = mysql.connector.connect(
  host="localhost",       #input name of database
  user="root",            #input user name
  password="pass",        #input password
  database="mydatabase"   #input database name
)

mycursor = mydb.cursor()

mycursor.execute("CREATE TABLE calculator (number1, number2, result)") #creating table and columnin mysql
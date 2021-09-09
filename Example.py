import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="pass",
  database="mydatabase"
)

mycursor = mydb.cursor()

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("Ritik", "saket, new delhi")
mycursor.execute(sql, val)

mydb.commit()

print(mycursor.rowcount, "record inserted.")
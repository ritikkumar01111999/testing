import mysql.connector

db = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = '',
    database = 'database_tag'
)

cursor = db.cursor()
cursor.execute('CREATE TABLE Tag_prdiction (discription CHAR(60), tag CHAR(60))')
class tagDb :
    def tag_db(self,testing,testing_result) :
        cursor.execute('INSERT INTO Tag_prdiction(testing,testing_result) VALUES(%s, %s)', (testing,testing_result))
        db.commit() 
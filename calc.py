from flask.json import jsonify
from flask import Flask,request,app
from flask import Flask,request

def sum(number1,number2):#function accepting two arguments
    #function to sum two numbers.
    result= number1+number2
    return result

def sub(number1,number2):#function accepting two arguments
    #function to substract two number
    result= number1-number2
    return result

def mul(number1,number2):#function accepting two arguments
#function to multiply two number
    result= number1*number2
    return result

def div(number1,number2):#function accepting two arguments
    #function to divide two number
    result= number1/number2
    return result

app=Flask(__name__)

@app.route('/sum',methods=['POST','GET'])
def main():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=sum(number1,number2)
    return jsonify(result)

@app.route('/sub',methods=['POST','GET'])
def hello():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=sub(number1,number2)
    return jsonify(result)

@app.route('/mul',methods=['POST','GET'])
def how():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=mul(number1,number2)
    return jsonify(result)

@app.route('/div',methods=['POST','GET'])
def are():
    number1=int(request.form.get('number1'))
    number2=int(request.form.get('number2'))
    result=div(number1,number2)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

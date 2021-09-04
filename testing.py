from attr import resolve_types
import pytest
import  unittest
from flask import json
from calc import sum,mul,sub,div

class TestSum(unittest.TestCase):
    def test_1(self):
        number1=10
        number2=20
        result=sum(number1,number2)
        self.assertEqual(result,True)
        print('test_1 succes')

    def test_2(self):
        number1=10
        number2=20
        result=sub(number1,number2)
        self.assertEqual(result,True)
        print('test_2 succes')

    def test_3(self):
        number1=10
        number2=20
        result=mul(number1,number2)
        self.assertEqual(result,True)
        print('test_3 succes')

    def test_4(self):
        number1=10
        number2=20
        result=div(number1,number2)
        self.assertEqual(result,True)
        print('test_4 succes')

    def test_5(self):
        number1=12
        number2=20
        result=sum(number1,number2)
        self.assertEqual(result,True)
        print('test_5 succes')

    def test_6(self):
        number1=100
        number2=20
        result=sub(number1,number2)
        self.assertEqual(result,True)
        print('test_6 succes')

    def test_7(self):
        number1=10
        number2=222
        result=mul(number1,number2)
        self.assertEqual(result,True)
        print('test_7 succes')

if __name__=="__main__":
   unittest.main()

import unittest
import calc

class TestCalc(unittest.TestCase):

    def test_sum1(self): # function to call self
#testing is being done for addition
        result=calc.sum(10,15) #calculation module is being called
        self.assertEqual(result,25)
        print('test 1 of sum successful')

    def test_sum2(self):# function to call self
        #testing is being done for addition
        result=calc.sum(12,7)#calculation module is being called
        self.assertEqual(result,19)
        print('test 2 of sum successful')

    def test_sum3(self):       # function to call self
        #testing is being done for addition
        result=calc.sum(15,5)#calculation module is being called
        self.assertEqual(result,20)
        print('test 3 of sum successful')

    def test_sum4(self):# function to call self
        #testing is being done for addition
        result=calc.sum(12,8)#calculation module is being called
        self.assertEqual(result,20)
        print('test 4 of sum successful')

    def test_sum5(self):# function to call self
        #testing is being done for addition
        result=calc.sum(10,15)#calculation module is being called
        self.assertEqual(result,25)
        print('test 5 of sum successful')

    def test_sum6(self):# function to call self
        #testing is being done for addition
        result=calc.sum(10,15)#calculation module is being called
        self.assertEqual(result,25)
        print('test 6 of sum successful')

    def test_sum7(self):# function to call self
        #testing is being done for addition
        result=calc.sum(10,15)#calculation module is being called
        self.assertEqual(result,25)
        print('test 7 of sum successful')

    def test_sum8(self):# function to call self
        #testing is being done for addition
        result=calc.sum(12,7)#calculation module is being called
        self.assertEqual(result,19)
        print('test 8 of sum successful')

    def test_sum9(self):# function to call self
        #testing is being done for addition
        result=calc.sum(10,15)#calculation module is being called
        self.assertEqual(result,25)
        print('test 9  of sum successful')

    def test_sum10(self):# function to call self
        #testing is being done for addition
        result=calc.sum(12,7)#calculation module is being called
        self.assertEqual(result,19)
        print('test 10 of sum successful')

    def test_sub1(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(10,5)#calculation module is being called
        self.assertEqual(result,5)
        print('test 1 of sub successful')

    def test_sub2(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(12,7)#calculation module is being called
        self.assertEqual(result,5)
        print('test 2 of sub successful')

    def test_sub3(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(15,5)#calculation module is being called
        self.assertEqual(result,10)
        print('test 3 of sub successful')

    def test_sub4(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(12,8)#calculation module is being called
        self.assertEqual(result,4)
        print('test 4 of sub successful')

    def test_sub5(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(10,15)#calculation module is being called
        self.assertEqual(result,-5)
        print('test 5 of sub successful')

    def test_sub6(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(100,25)#calculation module is being called
        self.assertEqual(result,75)
        print('test 6 of sub successful')

    def test_sub7(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(15,5)#calculation module is being called
        self.assertEqual(result,10)
        print('test 7 of sub successful')

    def test_sub8(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(12,8)#calculation module is being called
        self.assertEqual(result,4)
        print('test 8 of sub successful')

    def test_sub9(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(10,15)#calculation module is being called
        self.assertEqual(result,-5)
        print('test 9 of sub successful')

    def test_sub10(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(100,25)#calculation module is being called
        self.assertEqual(result,75)
        print('test 10 of sub successful')

    def test_mul1(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(2,4)#calculation module is being called
        self.assertEqual(result,8)
        print('test 1 of multiplication successful')

    def test_mul2(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(6,2)#calculation module is being called
        self.assertEqual(result,12)
        print('test 2 of multiplication successful')

    def test_mul3(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(15,5)#calculation module is being called
        self.assertEqual(result,75)
        print('test 3 of multiplication successful')

    def test_mul4(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(12,8)#calculation module is being called
        self.assertEqual(result,96)
        print('test 4 of multiplication successful')

    def test_mul5(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(10,15)#calculation module is being called
        self.assertEqual(result,150)
        print('test 5 of multiplication successful')

    def test_mul6(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(5,5)#calculation module is being called
        self.assertEqual(result,25)
        print('test 6 of multiplication successful')

    def test_mul7(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(15,5)#calculation module is being called
        self.assertEqual(result,75)
        print('test 7 of multiplication successful')

    def test_mul8(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(12,8)#calculation module is being called
        self.assertEqual(result,96)
        print('test 8 of multiplication successful')

    def test_mul9(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(10,15)#calculation module is being called
        self.assertEqual(result,150)
        print('test 9 of multiplication successful')

    def test_mul10(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(5,5)#calculation module is being called
        self.assertEqual(result,25)
        print('test 10 of multiplication successful')

    def test_div1(self):# function to call self
        #testing is being done for division
        result=calc.div(10,5)#calculation module is being called
        self.assertEqual(result,2)
        print('test 1 of division successful')

    def test_div2(self):# function to call self
        #testing is being done for division
        result=calc.div(12,6)#calculation module is being called
        self.assertEqual(result,2)
        print('test 2 of division successful')

    def test_div3(self):# function to call self
        #testing is being done for division
        result=calc.div(15,5)#calculation module is being called
        self.assertEqual(result,3)
        print('test 3 of division successful')

    def test_div4(self):# function to call self
        #testing is being done for division
        result=calc.div(12,3)#calculation module is being called
        self.assertEqual(result,4)
        print('test 4 of division successful')

    def test_div5(self):# function to call self
        #testing is being done for division
        result=calc.div(14,7)#calculation module is being called
        self.assertEqual(result,2)
        print('test 5 of division successful')

    def test_div6(self):# function to call self
        #testing is being done for division
        result=calc.div(100,25)#calculation module is being called
        self.assertEqual(result,4)
        print('test 6 of division successful')

    def test_div7(self):# function to call self
        #testing is being done for division
        result=calc.div(15,5)#calculation module is being called
        self.assertEqual(result,3)
        print('test 7 of division successful')

    def test_div8(self):# function to call self
        #testing is being done for division
        result=calc.div(12,3)#calculation module is being called
        self.assertEqual(result,4)
        print('test 8 of division successful')

    def test_div9(self):# function to call self
        #testing is being done for division
        result=calc.div(14,7)#calculation module is being called
        self.assertEqual(result,2)
        print('test 9 of division successful')

    def test_div10(self):# function to call self
        #testing is being done for division
        result=calc.div(100,25)#calculation module is being called
        self.assertEqual(result,4)
        print('test 10 of division successful')

if __name__=='__main__':
    unittest.main()

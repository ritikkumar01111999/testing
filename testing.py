
import unittest
import calc

class TestCalc(unittest.TestCase):

    def test_sum1(self): # function to call self
#testing is being done for addition
        result=calc.sum(10,15) #calculation module is being called
        self.assertEqual(result,25)
        print('test 1 of sum successful')

   

    def test_sub10(self):# function to call self
        #testing is being done for substraction
        result=calc.sub(100,25)#calculation module is being called
        self.assertEqual(result,75)
        print('test 10 of sub successful')

    

    def test_mul10(self):# function to call self
        #testing is being done for multiplication
        result=calc.mul(5,5)#calculation module is being called
        self.assertEqual(result,25)
        print('test 10 of multiplication successful')

    
    def test_div10(self):# function to call self
        #testing is being done for division
        result=calc.div(100,25)#calculation module is being called
        self.assertEqual(result,4)
        print('test 10 of division successful')

if __name__=='__main__':
    unittest.main()

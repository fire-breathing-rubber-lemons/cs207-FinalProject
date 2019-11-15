import unittest
import numpy as np
import pyad


class TestTensor(unittest.TestCase):
    '''
    Unit testing class for testing the Tensor Class and associated methods
    '''
    def test_get_value_and_deriv_return_tensor(self):
        '''
        Get the value and derivative of a tensor
        '''
        test_tensor = pyad.Tensor(5)
        output_tuple = test_tensor.get_value_and_deriv(test_tensor)

        self.assertIsInstance(output_tuple, tuple) 
        self.assertEqual(output_tuple[0], np.array(5))
        self.assertIsInstance(output_tuple[1], pyad.MultivariateDerivative)
        self.assertEqual(output_tuple[1].variables, {})

    def test_get_value_and_deriv_return_constant(self):
        '''
        Get the value and derivative of a constant
        '''
        test_tensor = pyad.Tensor(5)
        output_tuple = test_tensor.get_value_and_deriv(1)

        self.assertIsInstance(output_tuple, tuple) 
        self.assertEqual(output_tuple[0], 1)
        self.assertIsInstance(output_tuple[1], pyad.MultivariateDerivative)
        self.assertEqual(output_tuple[1].variables, {})

    def test_get_value_and_deriv_return_Variable(self):
        '''
        Get the value and derivative of a variable
        '''
        test_tensor = pyad.Tensor(5)
        test_variable = pyad.Variable('x', 2)
        output_tuple = test_tensor.get_value_and_deriv(test_variable)

        self.assertIsInstance(output_tuple, tuple) 
        self.assertEqual(output_tuple[0], np.array(2))
        self.assertIsInstance(output_tuple[1], pyad.MultivariateDerivative)
        self.assertEqual(output_tuple[1].variables, {'x':1})



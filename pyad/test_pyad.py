
import numpy as np
import pyad

def test_get_value_and_deriv_return_tensor():
    '''
    Get the value and derivative of a tensor
    '''
    test_tensor = pyad.Tensor(5)
    output_tuple = test_tensor.get_value_and_deriv(test_tensor)

    assert type(output_tuple) == tuple
    assert output_tuple[0] == np.array(5)
    assert type(output_tuple[1]) == pyad.MultivariateDerivative
    assert output_tuple[1].variables == {}

def test_get_value_and_deriv_return_constant():
    '''
    Get the value and derivative of a constant
    '''
    test_tensor = pyad.Tensor(5)
    output_tuple = test_tensor.get_value_and_deriv(1)

    assert type(output_tuple) == tuple 
    assert output_tuple[0] == 1
    assert type(output_tuple[1]) == pyad.MultivariateDerivative
    assert output_tuple[1].variables == {}

def test_get_value_and_deriv_return_Variable():
    '''
    Get the value and derivative of a variable
    '''
    test_tensor = pyad.Tensor(5)
    test_variable = pyad.Variable('x', 2)
    output_tuple = test_tensor.get_value_and_deriv(test_variable)

    assert type(output_tuple) == tuple
    assert output_tuple[0] == np.array(2)
    assert type(output_tuple[1]) == pyad.MultivariateDerivative
    assert output_tuple[1].variables == {'x':1}



import numpy as np
import math
import pyad

################################
#       End to End Tests       #
################################
def test_case_base_trigonometric():
    '''
    Try some end to end testing using a complex trigonometric function of sin, cos and tan
    '''
    x = pyad.var('x', 1)
    y = pyad.var('y', 0.5)
    z = pyad.var('z', 2)
    derivative_result = pyad.cos(x) + 3*(pyad.sin(y)**2) * pyad.cos(z) + pyad.tan(x)
    
    # Calculated from Wolfram Alpha
    true_value = 1.8107574187515
    true_x_deriv = 2.5840478360068
    true_y_deriv = -1.0505264651220
    true_z_deriv = -0.6270028955876

    assert math.isclose(float(derivative_result.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(derivative_result.d.variables['x']), true_x_deriv, rel_tol=1e-12)
    assert math.isclose(float(derivative_result.d.variables['y']), true_y_deriv, rel_tol=1e-12)
    assert math.isclose(float(derivative_result.d.variables['z']), true_z_deriv, rel_tol=1e-12)


################################
#       Unit Tests       #
################################
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



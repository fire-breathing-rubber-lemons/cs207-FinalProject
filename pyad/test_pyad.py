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


def test_case_base_inversetrig():
    '''
    Testing inverse trig functions
    '''
    x = pyad.var('x', 0.25)
    y = pyad.var('y', 0.5)
    z = pyad.var('z', 3)
    function = pyad.arcsin(x)**3 + pyad.arccos(y)**2 - z*pyad.arctan(z)

    # Calculated using numpy
    true_value = -2.634381651043423
    true_x_deriv = 0.1978236588118186
    true_y_deriv = -2.4183991523122903
    true_z_deriv = -1.5490457723982545

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['x']), true_x_deriv, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['y']), true_y_deriv, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['z']), true_z_deriv, rel_tol=1e-12)



def test_case_base_exp():
    '''
    Testing exponential function
    '''
    x = pyad.var('x', 2)
    function = pyad.exp(x) + pyad.exp(-x)

    # Calculated using numpy
    true_value = 7.524391382167263
    true_x_deriv = 7.253720815694038

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['x']), true_x_deriv, rel_tol=1e-12)


def test_case_base_log():
    '''
    Testing logarithm function
    '''
    x = pyad.var('x', 3)
    function = pyad.log(x)**2

    # Calculated using numpy
    true_value = 1.206948960812582
    true_x_deriv = 0.7324081924454066

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['x']), true_x_deriv, rel_tol=1e-12)

def test_case_base_root():
    '''
    Testing square root and cubic root functions
    '''
    x = pyad.var('x', 4)
    y = pyad.var('y', 8)
    function = pyad.sqrt(x) + pyad.cbrt(y)

    # Calculated using numpy
    true_value = 4
    true_x_deriv = 0.25
    true_y_deriv = 0.0833333333333333
    
    print(function.value)
    print(function.d.variables['x'])
    print(function.d.variables['y'])

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['x']), true_x_deriv, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['y']), true_y_deriv, rel_tol=1e-12)


def test_case_base_abs():
    '''
    Testing absolute value functin
    '''
    x = pyad.var('x', -2)
    function = pyad.abs(x)

    true_value = 2
    true_x_deriv = -1

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d.variables['x']), true_x_deriv, rel_tol=1e-12)



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


test_case_base_root()
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
    assert math.isclose(float(derivative_result.d['x']), true_x_deriv, rel_tol=1e-12)
    assert math.isclose(float(derivative_result.d['y']), true_y_deriv, rel_tol=1e-12)
    assert math.isclose(float(derivative_result.d['z']), true_z_deriv, rel_tol=1e-12)


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
    assert math.isclose(float(function.d['x']), true_x_deriv, rel_tol=1e-12)
    assert math.isclose(float(function.d['y']), true_y_deriv, rel_tol=1e-12)
    assert math.isclose(float(function.d['z']), true_z_deriv, rel_tol=1e-12)


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
    assert math.isclose(float(function.d['x']), true_x_deriv, rel_tol=1e-12)


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
    assert math.isclose(float(function.d['x']), true_x_deriv, rel_tol=1e-12)


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
    print(function.d['x'])
    print(function.d['y'])

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d['x']), true_x_deriv, rel_tol=1e-12)
    assert math.isclose(float(function.d['y']), true_y_deriv, rel_tol=1e-12)


def test_case_base_abs():
    '''
    Testing absolute value functin
    '''
    x = pyad.var('x', -2)
    function = pyad.abs(x)

    true_value = 2
    true_x_deriv = -1

    assert math.isclose(float(function.value), true_value, rel_tol=1e-12)
    assert math.isclose(float(function.d['x']), true_x_deriv, rel_tol=1e-12)


##########################
#       Unit Tests       #
##########################


#### Tests on the Tensor Class Object ####

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


def test_norm_tensor_single_value():
    '''
    Ensure the norm function correctly calculates euclidian norm
    '''
    test_tensor = pyad.Tensor(3)
    norm_output = test_tensor.norm()

    assert norm_output == 3.0


def test_norm_tensor_single_vector():
    '''
    Ensure the norm function correctly calculates euclidian norm
    '''
    test_tensor = pyad.Tensor([2,2,1])
    norm_output = test_tensor.norm()

    assert norm_output == 3.0


def test_repr():
    '''
    Ensure repr correctly prints the Tensor class
    '''
    test_tensor = pyad.Tensor(3)
    test_tensor2 = pyad.Tensor(3, d=pyad.MultivariateDerivative({'x':5}))

    assert test_tensor.__repr__() == 'Tensor(3, D())'
    assert test_tensor2.__repr__() == 'Tensor(3, D(x=5))'


def test_neg():
    '''
    Ensure negating works on both value and derivatives
    '''
    test_tensor = pyad.Tensor(3, d=pyad.MultivariateDerivative({'x':5}))
    neg_tensor = test_tensor.__neg__()

    assert neg_tensor.value == -3
    assert neg_tensor.d.variables == {'x':-5}


def test_add():
    '''
    Add two tensors together to get a new tensor
    '''
    test_tensor1 = pyad.Tensor(3, d=pyad.MultivariateDerivative({'x':5, 'y':10}))
    test_tensor2 = pyad.Tensor(5, d=pyad.MultivariateDerivative({'x':3, 'y':30}))

    new_tensor = test_tensor1 + test_tensor2

    assert type(new_tensor) == pyad.Tensor
    assert new_tensor.value == 8
    assert new_tensor.d.variables == {'x':8, 'y':40}


#### Tests on the MultivariateDerivative Class ####

def test_repr_mvd():
    '''
    Ensure multivariatederivates print correctly
    '''
    mvd = pyad.MultivariateDerivative({'x':10})

    assert mvd.__repr__() == 'D(x=10)'


def test_copy_mvd():
    '''
    Ensure copy is a new MVD object
    '''
    mvd = pyad.MultivariateDerivative({'x':10})
    mvd2 = mvd.copy()

    assert mvd2.variables == {'x':10}
    assert mvd2 != mvd


def test_getitem_mvd():
    '''
    Ensure get item returns correctly
    '''
    mvd = pyad.MultivariateDerivative({'x':10, 'y':3, 'z':1})

    assert mvd.__getitem__('x') == 10
    assert mvd.__getitem__('y') == 3
    assert mvd.__getitem__('z') == 1


def test_add_mvd():
    '''
    Test the add method for the multivariatederivative object
    '''
    mvd1 = pyad.MultivariateDerivative({'x':10, 'y':3, 'z':1})
    mvd2 = pyad.MultivariateDerivative({'x':10, 'y':3, 'a':1})

    mvd3 = mvd1 + mvd2

    assert mvd3.variables == {'x':20, 'y':6, 'z':1, 'a':1}


def test_mul_mvd():
    '''
    Test the multiply method for the multivariatederivative object
    '''
    mvd1 = pyad.MultivariateDerivative({'x':10, 'y':3, 'z':1})

    mvd2 = mvd1.mul(10)

    assert mvd2.variables == {'x':100, 'y':30, 'z':10}


#### Tests on non-class functions ####

def test_tensor_function():
    '''
    Check that the tensor wrapper around Tensor correctly returns a Tensor object
    '''
    test_tensor1 = pyad.tensor(5)

    assert type(test_tensor1) == pyad.Tensor
    assert test_tensor1.value == np.array(5)
    assert type(test_tensor1.d) == pyad.MultivariateDerivative
    assert test_tensor1.d.variables == {}


def test_var_function():
    '''
    Check that the var wrapper around Variable correctly returns a Variable
    '''
    test_var1 = pyad.var('x', 5)

    assert type(test_var1) == pyad.Variable
    assert test_var1.value == np.array(5)
    assert type(test_var1.d) == pyad.MultivariateDerivative
    assert test_var1.d.variables == {'x':1}

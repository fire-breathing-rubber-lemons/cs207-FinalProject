import numpy as np
import math
import pyad.forward_diff as fwd
import pytest

################################
#       End to End Tests       #
################################

def test_case_base_trigonometric():
    """
    Try some end to end testing using a complex trigonometric function of sin, cos and tan
    """
    x = fwd.var('x', 1)
    y = fwd.var('y', 0.5)
    z = fwd.var('z', 2)
    function = fwd.cos(x) + 3*(fwd.sin(y)**2) * fwd.cos(z) + fwd.tan(x)

    # Calculated from Wolfram Alpha
    true_value = 1.8107574187515
    true_x_deriv = 2.5840478360068
    true_y_deriv = -1.0505264651220
    true_z_deriv = -0.6270028955876

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['y'], true_y_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['z'], true_z_deriv, rel_tol=1e-12)


def test_case_base_inversetrig():
    """
    Testing inverse trig functions
    """
    x = fwd.var('x', 0.25)
    y = fwd.var('y', 0.5)
    z = fwd.var('z', 3)
    function = fwd.arcsin(x)**3 + fwd.arccos(y)**2 - z*fwd.arctan(z)

    # Calculated using numpy
    true_value = -2.634381651043423
    true_x_deriv = 0.1978236588118186
    true_y_deriv = -2.4183991523122903
    true_z_deriv = -1.5490457723982545

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['y'], true_y_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['z'], true_z_deriv, rel_tol=1e-12)


def test_case_base_hyperbolic():
    """
    Testing hyperbolic functions
    """
    x = fwd.var('x', 0.5)
    y = fwd.var('y', 1)
    z = fwd.var('z', 3)
    function = 2*fwd.sinh(x)*x + 3*fwd.cosh(y) + fwd.tanh(z)

    # Calculated using numpy
    true_value = 6.145391963626209
    true_x_deriv = 2.1698165761938757
    true_y_deriv = 3.525603580931404
    true_z_deriv = 0.009866037165440192

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['y'], true_y_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['z'], true_z_deriv, rel_tol=1e-12)


def test_case_base_exp():
    """
    Testing exponential function
    """
    x = fwd.var('x', 2)
    function = fwd.exp(x) + fwd.exp(-x)

    # Calculated using numpy
    true_value = 7.524391382167263
    true_x_deriv = 7.253720815694038

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)


def test_case_base_log():
    """
    Testing logarithm function
    """
    x = fwd.var('x', 3)
    function = fwd.log(x)**2

    # Calculated using numpy
    true_value = 1.206948960812582
    true_x_deriv = 0.7324081924454066

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)


def test_log2():
    """
    Testing logarithm base 2 function
    """
    x = fwd.var('x', 3)
    function = fwd.log2(x) ** 2

    # Calculated using numpy
    true_value = np.log2(3) ** 2
    true_x_deriv = 2 * np.log2(3) / 3 / np.log(2)

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)


def test_log10():
    """
    Testing logarithm base 10 function
    """
    x = fwd.var('x', 3)
    function = fwd.log10(x) ** 2

    # Calculated using numpy
    true_value = np.log10(3) ** 2
    true_x_deriv = 2 * np.log10(3) / 3 / np.log(10)

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)


def test_case_base_root():
    """
    Testing square root and cubic root functions
    """
    x = fwd.var('x', 4)
    y = fwd.var('y', 8)
    function = fwd.sqrt(x) + fwd.cbrt(y)

    # Calculated using numpy
    true_value = 4
    true_x_deriv = 0.25
    true_y_deriv = 0.0833333333333333

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)
    assert math.isclose(function.d['y'], true_y_deriv, rel_tol=1e-12)


def test_case_base_abs():
    """
    Testing absolute value functin
    """
    x = fwd.var('x', -2)
    function = fwd.abs(x)

    true_value = 2
    true_x_deriv = -1

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(function.d['x'], true_x_deriv, rel_tol=1e-12)


##########################
#       Unit Tests       #
##########################


#### Tests on the Tensor Class Object ####

def test_tensor_from_another_tensor():
    t1 = fwd.Tensor(5, d=fwd.MultivariateDerivative({'a': 2, 'b': 3}))
    t2 = fwd.Tensor(t1)
    assert t1.value == t2.value
    assert t1.d.variables == t2.d.variables


def test_get_value_and_deriv_return_tensor():
    """
    Get the value and derivative of a tensor
    """
    test_tensor = fwd.Tensor(5)
    output_tuple = test_tensor.get_value_and_deriv(test_tensor)

    assert type(output_tuple) == tuple
    assert output_tuple[0] == np.array(5)
    assert type(output_tuple[1]) == fwd.MultivariateDerivative
    assert output_tuple[1].variables == {}


def test_get_value_and_deriv_return_constant():
    """
    Get the value and derivative of a constant
    """
    test_tensor = fwd.Tensor(5)
    output_tuple = test_tensor.get_value_and_deriv(1)

    assert type(output_tuple) == tuple
    assert output_tuple[0] == 1
    assert type(output_tuple[1]) == fwd.MultivariateDerivative
    assert output_tuple[1].variables == {}


def test_get_value_and_deriv_return_Variable():
    """
    Get the value and derivative of a variable
    """
    test_tensor = fwd.Tensor(5)
    test_variable = fwd.Variable('x', 2)
    output_tuple = test_tensor.get_value_and_deriv(test_variable)

    assert type(output_tuple) == tuple
    assert output_tuple[0] == np.array(2)
    assert type(output_tuple[1]) == fwd.MultivariateDerivative
    assert output_tuple[1].variables == {'x':1}


def test_norm_tensor_single_value():
    """
    Ensure the norm function correctly calculates euclidian norm
    """
    test_tensor = fwd.Tensor(3)
    norm_output = test_tensor.norm()

    assert norm_output == 3.0


def test_norm_tensor_single_vector():
    """
    Ensure the norm function correctly calculates euclidian norm
    """
    test_tensor = fwd.Tensor([2,2,1])
    norm_output = test_tensor.norm()

    assert norm_output == 3.0


def test_repr():
    """
    Ensure repr correctly prints the Tensor class
    """
    test_tensor = fwd.Tensor(3)
    test_tensor2 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x':5}))

    assert test_tensor.__repr__() == 'Tensor(3, D())'
    assert test_tensor2.__repr__() == 'Tensor(3, D(x=5))'


def test_neg():
    """
    Ensure negating works on both value and derivatives
    """
    test_tensor = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x':5}))
    neg_tensor = test_tensor.__neg__()

    assert neg_tensor.value == -3
    assert neg_tensor.d.variables == {'x':-5}


def test_add():
    """
    Add two tensors together to get a new tensor
    """
    test_tensor1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x':5, 'y':10}))
    test_tensor2 = fwd.Tensor(5, d=fwd.MultivariateDerivative({'x':3, 'y':30}))

    new_tensor = test_tensor1 + test_tensor2

    assert type(new_tensor) == fwd.Tensor
    assert new_tensor.value == 8
    assert new_tensor.d.variables == {'x':8, 'y':40}


def test_radd():
    """
    Add a non-tensor and a tensor
    """
    t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
    res = 5 + t1

    assert isinstance(res, fwd.Tensor)
    assert res.value == 8
    assert res.d.variables == {'x': 5, 'y': 10}


def test_rsub():
    """
    Subtract a non-tensor and a tensor
    """
    t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
    res = 5 - t1

    assert isinstance(res, fwd.Tensor)
    assert res.value == 2
    assert res.d.variables == {'x': -5, 'y': -10}


def test_rtruedriv():
    """
    Divide a non-tensor and a tensor
    """
    t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
    res = -5 / t1

    assert isinstance(res, fwd.Tensor)
    assert res.value == -5/3
    assert res.d.variables == {'x': 25/9, 'y': 50/9}


def test_rpow():
    """
    Take a non-tensor to the power of a tensor
    """
    t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
    res = 4 ** t1

    assert isinstance(res, fwd.Tensor)
    assert res.value == 64
    assert res.d.variables == {'x': 320 * math.log(4), 'y': 640 * math.log(4)}


#### Tests on the MultivariateDerivative Class ####

def test_repr_mvd():
    """
    Ensure multivariatederivates print correctly
    """
    mvd = fwd.MultivariateDerivative({'x':10})

    assert mvd.__repr__() == 'D(x=10)'


def test_copy_mvd():
    """
    Ensure copy is a new MVD object
    """
    mvd = fwd.MultivariateDerivative({'x':10})
    mvd2 = mvd.copy()

    assert mvd2.variables == {'x':10}
    assert mvd2 != mvd


def test_getitem_mvd():
    """
    Ensure get item returns correctly
    """
    mvd = fwd.MultivariateDerivative({'x':10, 'y':3, 'z':1})

    assert mvd.__getitem__('x') == 10
    assert mvd.__getitem__('y') == 3
    assert mvd.__getitem__('z') == 1


def test_add_mvd():
    """
    Test the add method for the multivariatederivative object
    """
    mvd1 = fwd.MultivariateDerivative({'x':10, 'y':3, 'z':1})
    mvd2 = fwd.MultivariateDerivative({'x':10, 'y':3, 'a':1})

    mvd3 = mvd1 + mvd2

    assert mvd3.variables == {'x':20, 'y':6, 'z':1, 'a':1}


def test_mul_mvd():
    """
    Test the multiply method for the multivariatederivative object
    """
    mvd1 = fwd.MultivariateDerivative({'x':10, 'y':3, 'z':1})

    mvd2 = mvd1.mul(10)

    assert mvd2.variables == {'x':100, 'y':30, 'z':10}


#### Tests on non-class functions ####

def test_tensor_function():
    """
    Check that the tensor wrapper around Tensor correctly returns a Tensor object
    """
    test_tensor1 = fwd.tensor(5)

    assert type(test_tensor1) == fwd.Tensor
    assert test_tensor1.value == np.array(5)
    assert type(test_tensor1.d) == fwd.MultivariateDerivative
    assert test_tensor1.d.variables == {}


def test_var_function():
    """
    Check that the var wrapper around Variable correctly returns a Variable
    """
    test_var1 = fwd.var('x', 5)

    assert type(test_var1) == fwd.Variable
    assert test_var1.value == np.array(5)
    assert type(test_var1.d) == fwd.MultivariateDerivative
    assert test_var1.d.variables == {'x':1}


#### Tests on the Tensor Class Object ####

def test_comparisons():
    """
    Check that comparison operators work for Tensors
    """
    t1 = fwd.Tensor([1, 12, 3, 4, 5])
    t2 = fwd.Tensor([4, 2, 3, 1, 10])

    # test against another Tensor
    assert ((t1 < t2) == np.array([1, 0, 0, 0, 1])).all()
    assert ((t1 > t2) == np.array([0, 1, 0, 1, 0])).all()
    assert ((t1 <= t2) == np.array([1, 0, 1, 0, 1])).all()
    assert ((t1 >= t2) == np.array([0, 1, 1, 1, 0])).all()
    assert ((t1 == t2) == np.array([0, 0, 1, 0, 0])).all()
    assert ((t1 != t2) == np.array([1, 1, 0, 1, 1])).all()

    # test against a non-Tensor
    assert ((t1 < 3) == np.array([1, 0, 0, 0, 0])).all()
    assert ((t1 > 3) == np.array([0, 1, 0, 1, 1])).all()
    assert ((t1 <= 3) == np.array([1, 0, 1, 0, 0])).all()
    assert ((t1 >= 3) == np.array([0, 1, 1, 1, 1])).all()
    assert ((t1 == 3) == np.array([0, 0, 1, 0, 0])).all()
    assert ((t1 != 3) == np.array([1, 1, 0, 1, 1])).all()


def test_any_all():
    t1 = fwd.Tensor([0, 0, 0])
    assert t1.any() is False
    assert t1.all() is False

    t2 = fwd.Tensor([1, 2, 3])
    assert t2.any() is True
    assert t2.all() is True

    t3 = fwd.Tensor([0, 1, 2])
    assert t3.any() is True
    assert t3.all() is False

    t4 = fwd.Tensor(1)
    assert t4.any() is True
    assert t4.all() is True

    t5 = fwd.Tensor([])
    assert t5.any() is False
    assert t5.all() is True


def test_bool():
    """
    Check that you should only be able to call bool() on a singleton Tensor
    """
    t0 = fwd.Tensor(0)
    t1 = fwd.Tensor([1])
    t2 = fwd.Tensor([1, 2, 3, 4])

    assert bool(t0) is False
    assert bool(t1) is True

    with pytest.raises(ValueError):
        bool(t2)

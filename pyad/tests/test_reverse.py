import numpy as np
import math
import pyad.reverse_mode as rev
import pytest

################################
#       End to End Tests       #
################################
def test_reverse_by_example():
	x = rev.Tensor(0.5)
	y = rev.Tensor(4.2)
	z = rev.Tensor(3)
	f = x * y**3 + rev.sin(x) - rev.logistic(z)

	#set df seed
	f.backward()

	assert abs(f.value - (0.5*4.2**3+np.sin(0.5) - 1/(1+np.exp(-3)))) <= 1e-15
	assert abs(x.grad - (4.2**3 + np.cos(0.5))) <= 1e-15
	assert abs(y.grad - (3*0.5*4.2**2)) <= 1e-15
	assert abs(z.grad - (-np.exp(-3)/(1+np.exp(-3))**2)) <= 1e-15


def test_case_base_trigonometric():
    """
    Try some end to end testing using a complex trigonometric function of sin, cos and tan
    """
    x = rev.Tensor(1)
    y = rev.Tensor(0.5)
    z = rev.Tensor(2)
    function = rev.cos(x) + 3*(rev.sin(y)**2) * rev.cos(z) + rev.tan(x)

    function.backward()

    # Calculated from Wolfram Alpha
    true_value = 1.8107574187515
    true_x_deriv = 2.5840478360068
    true_y_deriv = -1.0505264651220
    true_z_deriv = -0.6270028955876

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)
    assert math.isclose(y.grad, true_y_deriv, rel_tol=1e-12)
    assert math.isclose(z.grad, true_z_deriv, rel_tol=1e-12)

def test_case_base_inversetrig():
    """
    Testing inverse trig functions
    """
    x = rev.Tensor(0.25)
    y = rev.Tensor(0.5)
    z = rev.Tensor(3)
    function = rev.arcsin(x)**3 + rev.arccos(y)**2 - z*rev.arctan(z)

    function.backward()

    # Calculated using numpy
    true_value = -2.634381651043423
    true_x_deriv = 0.1978236588118186
    true_y_deriv = -2.4183991523122903
    true_z_deriv = -1.5490457723982545

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)
    assert math.isclose(y.grad, true_y_deriv, rel_tol=1e-12)
    assert math.isclose(z.grad, true_z_deriv, rel_tol=1e-12)


def test_case_base_hyperbolic():
    """
    Testing hyperbolic functions
    """
    x = rev.Tensor(0.5)
    y = rev.Tensor(1)
    z = rev.Tensor(3)
    function = 2*rev.sinh(x)*x + 3*rev.cosh(y) + rev.tanh(z)
    function.backward()

    # Calculated using numpy
    true_value = 6.145391963626209
    true_x_deriv = 2.1698165761938757
    true_y_deriv = 3.525603580931404
    true_z_deriv = 0.009866037165440192

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)
    assert math.isclose(y.grad, true_y_deriv, rel_tol=1e-12)
    assert math.isclose(z.grad, true_z_deriv, rel_tol=1e-12)


def test_case_base_exp():
    """
    Testing exponential function
    """
    x = rev.Tensor(2)
    function = rev.exp(x) + rev.exp(-x)
    function.backward()

    # Calculated using numpy
    true_value = 7.524391382167263
    true_x_deriv = 7.253720815694038

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)


def test_case_base_log():
    """
    Testing logarithm function
    """
    x = rev.Tensor(3)
    function = rev.log(x)**2
    function.backward()

    # Calculated using numpy
    true_value = 1.206948960812582
    true_x_deriv = 0.7324081924454066

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)


def test_log2():
    """
    Testing logarithm base 2 function
    """
    x = rev.Tensor(3)
    function = rev.log2(x) ** 2
    function.backward()

    # Calculated using numpy
    true_value = np.log2(3) ** 2
    true_x_deriv = 2 * np.log2(3) / 3 / np.log(2)

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)


def test_log10():
    """
    Testing logarithm base 10 function
    """
    x = rev.Tensor(3)
    function = rev.log10(x) ** 2
    function.backward()

    # Calculated using numpy
    true_value = np.log10(3) ** 2
    true_x_deriv = 2 * np.log10(3) / 3 / np.log(10)

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)


def test_case_base_root():
    """
    Testing square root and cubic root functions
    """
    x = rev.Tensor(4)
    y = rev.Tensor(8)
    function = rev.sqrt(x) + rev.cbrt(y)
    function.backward()

    # Calculated using numpy
    true_value = 4
    true_x_deriv = 0.25
    true_y_deriv = 0.0833333333333333

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)
    assert math.isclose(y.grad, true_y_deriv, rel_tol=1e-12)


def test_case_base_abs():
    """
    Testing absolute value functin
    """
    x = rev.Tensor(-2)
    function = rev.abs(x)
    function.backward()

    true_value = 2
    true_x_deriv = -1

    assert math.isclose(function.value, true_value, rel_tol=1e-12)
    assert math.isclose(x.grad, true_x_deriv, rel_tol=1e-12)


# ##########################
# #       Unit Tests       #
# ##########################


# #### Tests on the Tensor Class Object ####

def test_children_Tensor():
	x = rev.Tensor(1)
	y = rev.Tensor(2)
	z = rev.Tensor(3)

	function = (x + y) * z
	function.backward()

	# Real children
	child1 = x.children
	child2 = y.children
	child3 = z.children
	child4 = child1[0][1].children
	child5 = child2[0][1].children

	# Leaf nodes
	child6 = child3[0][1].children
	child7 = child4[0][1].children
	child8 = child5[0][1].children

	assert(child6 == [])
	assert(child7 == [])
	assert(child8 == [])

	assert(child1 == child2)
	assert(child3 == child4 == child5)


def test_backward_Tensor():
	x = rev.Tensor(3)

	assert(x.grad_value == None)

	x.backward()

	assert(x.grad_value == 1)


def test_repr():
    """
    Ensure repr correctly prints the Tensor class
    """
    test_tensor = rev.Tensor(3)
    test_tensor2 = rev.Tensor(2)
    test_tensor2.backward()

    assert test_tensor.__repr__() == 'Tensor(3, D(None))'
    assert test_tensor2.__repr__() == 'Tensor(2, D(1.0))'


def test_neg():
    """
    Ensure negating works on both value and derivatives
    """
    test_tensor = rev.Tensor(3)
    neg_tensor = test_tensor.__neg__()

    assert neg_tensor.value == -3


def test_add():
    """
    Add two tensors together to get a new tensor
    """
    test_tensor1 = rev.Tensor(3)
    test_tensor2 = rev.Tensor(5)

    new_tensor = test_tensor1 + test_tensor2

    assert type(new_tensor) == rev.Tensor
    assert new_tensor.value == 8
    assert test_tensor1.children[0] == (1.0, new_tensor)
    assert test_tensor2.children[0] == (1.0, new_tensor)


def test_radd():
    """
    Add a non-tensor and a tensor
    """
    t1 = rev.Tensor(3)
    res = 5 + t1

    assert isinstance(res, rev.Tensor)
    assert res.value == 8
    assert t1.children[0] == (1.0, res)


def test_sub():
    """
    Subtract a tensor and a tensor
    """
    t1 = rev.Tensor(3)
    t2 = rev.Tensor(1)
    res = t1 - t2

    assert isinstance(res, rev.Tensor)
    assert res.value == 2
    assert t1.children[0] == (1.0, res)
    assert t2.children[0][0] == -t2.value
    assert t2.children[0][1].value == t2.__neg__().value


def test_rsub():
    """
    Subtract a non-tensor and a tensor
    """
    t1 = rev.Tensor(3)
    res = 5 - t1

    assert isinstance(res, rev.Tensor)
    assert res.value == 2
    assert t1.children[0][0] == -1
    assert t1.children[0][1].value == t1.__neg__().value


def test_truedriv():
    """
    Divide a tensor and a tensor
    """
    t1 = rev.Tensor(3)
    t2 = rev.Tensor(2)
    res = t1 / t2
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 3/2    
    assert t1.children[0][0] == 1/2
    assert t2.children[0][0] == -3/4


def test_rtruedriv():
    """
    Divide a non-tensor and a tensor
    """
    t1 = rev.Tensor(3)
    res = 2 / t1

    assert isinstance(res, rev.Tensor)
    assert res.value == 2/3
    assert t1.children[0][0] == -2/9
    assert t1.children[0][1].value == 2/3


def test_pow():
    """
    Take a tensor to the power of a constant
    """
    t1 = rev.Tensor(3)
    res = t1 ** 4

    assert isinstance(res, rev.Tensor)
    assert res.value == 81
    t1.children[0][0] == 108


def test_rpow():
    """
    Take a non-tensor to the power of a tensor
    """
    t1 = rev.Tensor(3)
    res = 4 ** t1

    assert isinstance(res, rev.Tensor)
    assert res.value == 64
    assert t1.children[0][0] == (64*np.log(4))


def test_logistic():
    """
    Tests the logistic function
    """
    t1 = rev.Tensor(3)
    res = rev.logistic(t1)

    expected_value = 1 / (1 + np.exp(-3))
    expected_deriv = np.exp(3) / (np.exp(3) + 1) ** 2

    assert math.isclose(res.value, expected_value, rel_tol=1e-12)
    assert math.isclose(t1.children[0][0], expected_deriv, rel_tol=1e-12)
    

#### Tests on the Tensor Class Object ####

def test_comparisons():
    """
    Check that comparison operators work for Tensors
    """
    t1 = rev.Tensor(2)
    t2 = rev.Tensor(2)
    t3 = rev.Tensor(1)
    t4 = rev.Tensor(3)

    # test against another Tensor
    assert  t1 == t2
    assert  t1 <= t2
    assert  t1 >= t2
    assert  t3 < t1
    assert  t3 <= t1
    assert  t4 > t1
    assert  t4 >= t1

    # test against a non-Tensor
    assert t1 == 2
    assert t1 <= 2
    assert t1 >= 2
    assert t1 < 3
    assert t1 > 1


def test_bool():
    """
    Check that you should only be able to call bool() on a singleton Tensor
    """
    t0 = rev.Tensor(0)
    t1 = rev.Tensor(1)
    t2 = rev.Tensor(2)

    assert bool(t0) is False
    assert bool(t1) is True
    assert bool(t2) is True
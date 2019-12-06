import numpy as np
import math
import pyad.reverse_mode as rev
import pytest

################################
#       End to End Tests       #
################################
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
    Subtract a non-tensor and a tensor
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


# def test_rtruedriv():
#     """
#     Divide a non-tensor and a tensor
#     """
#     t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
#     res = -5 / t1

#     assert isinstance(res, fwd.Tensor)
#     assert res.value == -5/3
#     assert res.d.variables == {'x': 25/9, 'y': 50/9}


# def test_rpow():
#     """
#     Take a non-tensor to the power of a tensor
#     """
#     t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
#     res = 4 ** t1

#     assert isinstance(res, fwd.Tensor)
#     assert res.value == 64
#     assert res.d.variables == {'x': 320 * math.log(4), 'y': 640 * math.log(4)}


# def test_logistic():
#     """
#     Tests the logistic function
#     """
#     t1 = fwd.Tensor(3, d=fwd.MultivariateDerivative({'x': 5, 'y': 10}))
#     res = fwd.logistic(t1)

#     expected_value = 1 / (1 + np.exp(-3))
#     expected_deriv = np.exp(3) / (np.exp(3) + 1) ** 2
#     assert math.isclose(res.value, expected_value, rel_tol=1e-12)
#     assert math.isclose(res.d['x'], 5 * expected_deriv, rel_tol=1e-12)
#     assert math.isclose(res.d['y'], 10 * expected_deriv, rel_tol=1e-12)


# #### Tests on the MultivariateDerivative Class ####

# def test_repr_mvd():
#     """
#     Ensure multivariatederivates print correctly
#     """
#     mvd = fwd.MultivariateDerivative({'x':10})

#     assert mvd.__repr__() == 'D(x=10)'


# def test_copy_mvd():
#     """
#     Ensure copy is a new MVD object
#     """
#     mvd = fwd.MultivariateDerivative({'x':10})
#     mvd2 = mvd.copy()

#     assert mvd2.variables == {'x':10}
#     assert mvd2 != mvd


# def test_getitem_mvd():
#     """
#     Ensure get item returns correctly
#     """
#     mvd = fwd.MultivariateDerivative({'x':10, 'y':3, 'z':1})

#     assert mvd.__getitem__('x') == 10
#     assert mvd.__getitem__('y') == 3
#     assert mvd.__getitem__('z') == 1


# def test_add_mvd():
#     """
#     Test the add method for the multivariatederivative object
#     """
#     mvd1 = fwd.MultivariateDerivative({'x':10, 'y':3, 'z':1})
#     mvd2 = fwd.MultivariateDerivative({'x':10, 'y':3, 'a':1})

#     mvd3 = mvd1 + mvd2

#     assert mvd3.variables == {'x':20, 'y':6, 'z':1, 'a':1}


# def test_mul_mvd():
#     """
#     Test the multiply method for the multivariatederivative object
#     """
#     mvd1 = fwd.MultivariateDerivative({'x':10, 'y':3, 'z':1})

#     mvd2 = mvd1.mul(10)

#     assert mvd2.variables == {'x':100, 'y':30, 'z':10}


# #### Tests on non-class functions ####

# def test_tensor_function():
#     """
#     Check that the tensor wrapper around Tensor correctly returns a Tensor object
#     """
#     test_tensor1 = fwd.tensor(5)

#     assert type(test_tensor1) == fwd.Tensor
#     assert test_tensor1.value == np.array(5)
#     assert type(test_tensor1.d) == fwd.MultivariateDerivative
#     assert test_tensor1.d.variables == {}


# def test_var_function():
#     """
#     Check that the var wrapper around Variable correctly returns a Variable
#     """
#     test_var1 = fwd.var('x', 5)

#     assert type(test_var1) == fwd.Variable
#     assert test_var1.value == np.array(5)
#     assert type(test_var1.d) == fwd.MultivariateDerivative
#     assert test_var1.d.variables == {'x':1}


# #### Tests on the Tensor Class Object ####

# def test_comparisons():
#     """
#     Check that comparison operators work for Tensors
#     """
#     t1 = fwd.Tensor([1, 12, 3, 4, 5])
#     t2 = fwd.Tensor([4, 2, 3, 1, 10])

#     # test against another Tensor
#     assert np.array_equal(t1 < t2, np.array([1, 0, 0, 0, 1]))
#     assert np.array_equal(t1 > t2, np.array([0, 1, 0, 1, 0]))
#     assert np.array_equal(t1 <= t2, np.array([1, 0, 1, 0, 1]))
#     assert np.array_equal(t1 >= t2, np.array([0, 1, 1, 1, 0]))
#     assert np.array_equal(t1 == t2, np.array([0, 0, 1, 0, 0]))
#     assert np.array_equal(t1 != t2, np.array([1, 1, 0, 1, 1]))

#     # test against a non-Tensor
#     assert np.array_equal(t1 < 3, np.array([1, 0, 0, 0, 0]))
#     assert np.array_equal(t1 > 3, np.array([0, 1, 0, 1, 1]))
#     assert np.array_equal(t1 <= 3, np.array([1, 0, 1, 0, 0]))
#     assert np.array_equal(t1 >= 3, np.array([0, 1, 1, 1, 1]))
#     assert np.array_equal(t1 == 3, np.array([0, 0, 1, 0, 0]))
#     assert np.array_equal(t1 != 3, np.array([1, 1, 0, 1, 1]))


# def test_any_all():
#     """
#     Tests the any() and all() methods
#     """
#     t1 = fwd.Tensor([0, 0, 0])
#     assert t1.any() is False
#     assert t1.all() is False

#     t2 = fwd.Tensor([1, 2, 3])
#     assert t2.any() is True
#     assert t2.all() is True

#     t3 = fwd.Tensor([0, 1, 2])
#     assert t3.any() is True
#     assert t3.all() is False

#     t4 = fwd.Tensor(1)
#     assert t4.any() is True
#     assert t4.all() is True

#     t5 = fwd.Tensor([])
#     assert t5.any() is False
#     assert t5.all() is True


# def test_bool():
#     """
#     Check that you should only be able to call bool() on a singleton Tensor
#     """
#     t0 = fwd.Tensor(0)
#     t1 = fwd.Tensor([1])
#     t2 = fwd.Tensor([1, 2, 3, 4])

#     assert bool(t0) is False
#     assert bool(t1) is True

#     with pytest.raises(ValueError):
#         bool(t2)


# ### Tests for vector operations ###
# def test_simple_vector_arithmetic():
#     """
#     Tests simple arithmetic (add/sub/mul/div) for vectors
#     """
#     v1 = fwd.var('x', 5)
#     v2 = fwd.var('y', 10)

#     t1 = fwd.Tensor([1, 2, 3])
#     t2 = fwd.Tensor([5, 6, 7])
#     res = t1*v1 - t2*v2

#     assert np.array_equal(res.value, np.array([-45, -50, -55]))
#     assert np.array_equal(res.d['x'], np.array([1, 2, 3]))
#     assert np.array_equal(res.d['y'], np.array([-5, -6, -7]))

#     res = 5 + (res * res * res) / res
#     assert np.array_equal(res.value, np.array([2030, 2505, 3030]))
#     assert np.array_equal(res.d['x'], np.array([-90, -200, -330]))
#     assert np.array_equal(res.d['y'], np.array([450, 600, 770]))


# def test_scalar_var_getitem():
#     """
#     Tests indexing into a Tensor with scalar variables
#     """

#     v1 = fwd.var('x', 5)
#     v2 = fwd.var('y', 10)

#     t1 = fwd.Tensor([1, 2, 3])
#     t2 = fwd.Tensor([5, 6, 7])

#     res = t1*v1 - t2*v2
#     res = 5 + (res * res * res) / res

#     assert isinstance(res[0], fwd.Tensor)
#     assert res[0].value == 2030
#     assert res[0].d['x'] == -90
#     assert res[0].d['y'] == 450

#     assert isinstance(res[1], fwd.Tensor)
#     assert res[1].value == 2505
#     assert res[1].d['x'] == -200
#     assert res[1].d['y'] == 600

#     assert isinstance(res[2], fwd.Tensor)
#     assert res[2].value == 3030
#     assert res[2].d['x'] == -330
#     assert res[2].d['y'] == 770

#     assert isinstance(res[0:2], fwd.Tensor)
#     assert np.array_equal(res[0:2].value, np.array([2030, 2505]))
#     assert np.array_equal(res[0:2].d['x'], np.array([-90, -200]))
#     assert np.array_equal(res[0:2].d['y'], np.array([450, 600]))

#     assert isinstance(res[[0, 2]], fwd.Tensor)
#     assert np.array_equal(res[[0, 2]].value, np.array([2030, 3030]))
#     assert np.array_equal(res[[0, 2]].d['x'], np.array([-90, -330]))
#     assert np.array_equal(res[[0, 2]].d['y'], np.array([450, 770]))


# def test_scalar_var_setitem():
#     """
#     Tests setting indexes in a Tensor with scalar variables
#     """
#     v1 = fwd.var('x', 5)
#     v2 = fwd.var('y', 10)
#     v3 = fwd.var('z', 5)

#     t = fwd.Tensor([1, 2, 3, 4]) * v1 + v2 - 2 * v3

#     assert np.array_equal(t.value, np.array([5, 10, 15, 20]))
#     assert np.array_equal(t.d['x'], np.array([1, 2, 3, 4]))
#     assert np.array_equal(t.d['y'], np.array([1, 1, 1, 1]))
#     assert np.array_equal(t.d['z'], np.array([-2, -2, -2, -2]))

#     t[0] = t[1]
#     assert np.array_equal(t.value, np.array([10, 10, 15, 20]))
#     assert np.array_equal(t.d['x'], np.array([2, 2, 3, 4]))
#     assert np.array_equal(t.d['y'], np.array([1, 1, 1, 1]))
#     assert np.array_equal(t.d['z'], np.array([-2, -2, -2, -2]))

#     t[1:3] = t[3]
#     assert np.array_equal(t.value, np.array([10, 20, 20, 20]))
#     assert np.array_equal(t.d['x'], np.array([2, 4, 4, 4]))
#     assert np.array_equal(t.d['y'], np.array([1, 1, 1, 1]))
#     assert np.array_equal(t.d['z'], np.array([-2, -2, -2, -2]))

#     # adding a variable that previously wasn't there
#     t[0] = fwd.var('q', 100)
#     assert np.array_equal(t.value, np.array([100, 20, 20, 20]))
#     assert np.array_equal(t.d['x'], np.array([0, 4, 4, 4]))
#     assert np.array_equal(t.d['y'], np.array([0, 1, 1, 1]))
#     assert np.array_equal(t.d['z'], np.array([0, -2, -2, -2]))
#     assert np.array_equal(t.d['q'], np.array([1, 0, 0, 0]))


# def test_vector_var():
#     """
#     Basic arithmetic tests for vector variables
#     """
#     v1 = fwd.var('x', [1, 2, 3])
#     v2 = fwd.var('y', [2, 3, 4])

#     assert np.array_equal(v1.d['x'], np.eye(3))
#     assert np.array_equal(v2.d['y'], np.eye(3))

#     res = (v1 + 1) * (v2 + 1)
#     assert np.array_equal(res.d['x'], np.diag([3, 4, 5]))
#     assert np.array_equal(res.d['y'], np.diag([2, 3, 4]))

#     # try out invalid variable declarations
#     with pytest.raises(ValueError):
#         fwd.var('x', [])

#     with pytest.raises(ValueError):
#         fwd.var('x', np.ones((2, 2)))


# def test_vector_var_getitem():
#     """
#     Tests getting indexes in a Tensor with vector variables
#     """
#     v1 = fwd.var('x', [1, 2, 3])
#     v2 = fwd.var('y', [2, 3, 4])
#     res = 6 * (v1 + 1) / (v2 - 1)

#     assert np.array_equal(res.value, np.array([12, 9, 8]))
#     assert np.array_equal(res.d['x'], np.diag([6, 3, 2]))
#     assert np.array_equal(res.d['y'], np.diag([-12, -9/2, -8/3]))

#     assert np.array_equal(res[0].value, 12)
#     assert np.array_equal(res[0].d['x'], np.array([6, 0, 0]))
#     assert np.array_equal(res[0].d['y'], np.array([-12, 0, 0]))

#     assert np.array_equal(res[1].value, 9)
#     assert np.array_equal(res[1].d['x'], np.array([0, 3, 0]))
#     assert np.array_equal(res[1].d['y'], np.array([0, -9/2, 0]))

#     assert np.array_equal(res[2].value, 8)
#     assert np.array_equal(res[2].d['x'], np.array([0, 0, 2]))
#     assert np.array_equal(res[2].d['y'], np.array([0, 0, -8/3]))


# def test_vector_var_setitem():
#     """
#     Tests setting indexes in a Tensor with vector variables
#     """
#     v1 = fwd.var('x', [1, 2, 3])
#     v2 = fwd.var('y', [2, 3, 4])
#     res = 2 * v1 * v2

#     res[0] += res[1] + res[2] / 2
#     assert np.array_equal(res.value, [28, 12, 24])
#     assert np.array_equal(res.d['x'], [[4, 6, 4], [0, 6, 0], [0, 0, 8]])
#     assert np.array_equal(res.d['y'], [[2, 4, 3], [0, 4, 0], [0, 0, 6]])


# def test_shape_and_len():
#     """
#     Tests len() and .shape of a Tensor
#     """
#     v1 = fwd.Tensor(0)
#     assert v1.shape == ()
#     with pytest.raises(TypeError):
#         len(v1)

#     v2 = fwd.Tensor([0])
#     assert v2.shape == (1,)
#     assert len(v2) == 1

#     v3 = fwd.Tensor([0, 1, 2, 3])
#     assert v3.shape == (4,)
#     assert len(v3) == 4

#     v4 = fwd.Tensor([[1, 2, 3], [4, 5, 6]])
#     assert v4.shape == (2, 3)
#     assert len(v4) == 2


# def test_iter():
#     """
#     Tests iterating over a Tensor
#     """
#     v1 = fwd.Tensor(0)
#     with pytest.raises(TypeError):
#         iter(v1)

#     v2 = fwd.Tensor([0, 1], d=fwd.MultivariateDerivative({'x': [1, 2], 'y': [3, 4]}))
#     res = []
#     for x in v2:
#         res.append(x)

#     assert res[0].value == 0
#     assert res[0].d['x'] == 1
#     assert res[0].d['y'] == 3

#     assert res[1].value == 1
#     assert res[1].d['x'] == 2
#     assert res[1].d['y'] == 4


# def test_stack():
#     """
#     Tests for stacking Tensors
#     """

#     # stack with scalar variables
#     t1 = fwd.Tensor([1, 2, 3]) * fwd.var('x', 2)
#     t2 = fwd.Tensor([4, 5, 6]) * fwd.var('y', 3)

#     res = fwd.stack([t1, t2])
#     assert np.array_equal(res.value, np.array([[2, 4, 6], [12, 15, 18]]))
#     assert np.array_equal(res.d['x'], np.array([[1, 2, 3], [0, 0, 0]]))
#     assert np.array_equal(res.d['y'], np.array([[0, 0, 0], [4, 5, 6]]))

#     # stack with vector variables
#     t1 = fwd.Tensor([2, 3, 4]) * fwd.var('x', [1, 2, 3])
#     t2 = fwd.Tensor([5, 6, 7]) * fwd.var('y', [4, 5, 6])

#     res = fwd.stack([t1.sum(), t2.sum()])
#     assert np.array_equal(res.value, np.array([20, 92]))
#     assert np.array_equal(res.d['x'], [[2, 3, 4], [0, 0, 0]])
#     assert np.array_equal(res.d['y'], [[0, 0, 0], [5, 6, 7]])

#     res = res + fwd.var('a', 1)
#     res = res * fwd.var('b', [3, 4])
#     assert np.array_equal(res.d['a'], [3, 4])
#     assert np.array_equal(res.d['b'], [[21, 0], [0, 93]])

#     # test stack of an empty list
#     with pytest.raises(ValueError):
#         fwd.stack([])

#     # test stack with inconsistent shapes
#     with pytest.raises(ValueError):
#         fwd.stack([fwd.var('x', 1), fwd.var('x', [1, 2, 3])])

# # Simple test suite explaining how Tensor works
# #Initialization
# def test_illu():
# 	x = rev.Tensor(0.5)
# 	y = rev.Tensor(4.2)
# 	z = rev.Tensor(3)
# 	f = x * y**3 + rev.sin(x) - rev.logistic(z)

# 	#set df seed
# 	f.backward()

# 	# f_val = f.value
# 	# x_val = x.value
# 	# x_grad = x.grad
# 	# y_grad = y.grad
# 	# z_grad = z.grad

# 	assert abs(f.value - (0.5*4.2**3+np.sin(0.5) - 1/(1+np.exp(-3)))) <= 1e-15
# 	assert abs(x.grad - (4.2**3 + np.cos(0.5))) <= 1e-15
# 	assert abs(y.grad - (3*0.5*4.2**2)) <= 1e-15
# 	assert abs(z.grad - (-np.exp(-3)/(1+np.exp(-3))**2)) <= 1e-15
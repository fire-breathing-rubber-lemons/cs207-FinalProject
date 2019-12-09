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

    # set df seed
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
    """
    Makes sure that the .backward() seed is set correct
    """
    x = rev.Tensor(3)
    assert x.grad_value is None

    x.backward()
    assert x.grad_value == 1

    # make sure you can't call .backward() on a non-scalar
    y = x * [1, 2, 3]
    with pytest.raises(ValueError):
        y.backward()


def test_repr():
    """
    Ensure repr correctly prints the Tensor class
    """
    test_tensor = rev.Tensor(3)
    test_tensor2 = rev.Tensor(2)
    test_tensor2.backward()

    assert test_tensor.__repr__() == 'Tensor(3.0, D(None))'
    assert test_tensor2.__repr__() == 'Tensor(2.0, D(1.0))'


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
    new_tensor.backward()

    assert type(new_tensor) == rev.Tensor
    assert new_tensor.value == 8
    assert test_tensor1.grad == 1.0
    assert test_tensor2.grad == 1.0
    assert test_tensor1.children[0][1] == new_tensor
    assert test_tensor2.children[0][1] == new_tensor


def test_radd():
    """
    Add a non-tensor and a tensor
    """
    t1 = rev.Tensor(3)
    res = 5 + t1
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 8
    assert t1.grad == 1.0
    assert t1.children[0][1] == res


def test_sub():
    """
    Subtract a tensor and a tensor
    """
    t1 = rev.Tensor(3)
    t2 = rev.Tensor(1)
    res = t1 - t2
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 2
    assert t2.grad == -t2.value
    assert t2.children[0][1].value == t2.__neg__().value


def test_rsub():
    """
    Subtract a non-tensor and a tensor
    """
    t1 = rev.Tensor(3)
    res = 5 - t1
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 2
    assert t1.grad == -1
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
    assert t1.grad == 1/2
    assert t2.grad == -3/4
    assert t1.children[0][1].value == res.value
    assert t2.children[0][1].value == res.value


def test_rtruedriv():
    """
    Divide a non-tensor and a tensor
    """
    t1 = rev.Tensor(3)
    res = 2 / t1
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 2/3
    assert t1.grad == -2/9
    assert t1.children[0][1].value == res.value


def test_pow():
    """
    Take a tensor to the power of a constant
    """
    t1 = rev.Tensor(3)
    res = t1 ** 4
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 81
    assert t1.grad == 108
    assert t1.children[0][1].value == res.value


def test_rpow():
    """
    Take a non-tensor to the power of a tensor
    """
    t1 = rev.Tensor(3)
    res = 4 ** t1
    res.backward()

    assert isinstance(res, rev.Tensor)
    assert res.value == 64
    assert t1.grad == (64*np.log(4))
    assert t1.children[0][1].value == res.value


def test_logistic():
    """
    Tests the logistic function
    """
    t1 = rev.Tensor(3)
    res = rev.logistic(t1)

    res.backward()

    expected_value = 1 / (1 + np.exp(-3))
    expected_deriv = np.exp(-3) / (np.exp(-3) + 1) ** 2

    assert math.isclose(res.value, expected_value, rel_tol=1e-12)
    assert math.isclose(t1.grad, expected_deriv, rel_tol=1e-12)


def test_loge():
    """
    Tests the log base e function
    """
    t1 = rev.Tensor(3)
    res = rev.log(t1)

    res.backward()

    expected_value = np.log(t1.value)
    expected_deriv = 1/t1.value

    assert res.value == expected_value
    assert t1.grad == expected_deriv


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
    assert t1 == t2
    assert t1 <= t2
    assert t1 >= t2
    assert t3 < t1
    assert t3 <= t1
    assert t4 > t1
    assert t4 >= t1
    assert t1 != t3

    # test against a non-Tensor
    assert t1 == 2
    assert t1 <= 2
    assert t1 >= 2
    assert t1 < 3
    assert t1 > 1
    assert t1 != 0


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


def test_tensor_build():
    """
    Test Tensor init
    """

    # Check that you can build a Tensor from another Tensor
    t0 = rev.Tensor([1, 2, 3])
    t1 = t0 * t0
    res = t1.sum()
    res.backward()

    assert np.allclose(t0.grad, [2, 4, 6])
    assert t0.children != []
    assert t0.grad_value is not None

    s0 = rev.Tensor(t0)
    assert s0.grad_value is None
    assert s0.children == []

    # Check that you can't build a > 2D tensor
    with pytest.raises(ValueError):
        rev.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])


def test_matmul():
    """
    Test matrix multiplication functionality
    """

    # test matrix x matrix
    t0 = rev.Tensor([[1, 2], [3, 4]])
    t1 = rev.Tensor([[5, 6], [7, 8]])
    t2 = rev.Tensor([[9, 10], [11, 12]])
    res = (t0 @ t1 @ t2).sum()

    res.backward()

    assert res.value == 2834
    assert np.allclose(t0.grad, [[233, 317], [233, 317]])
    assert np.allclose(t1.grad, [[76, 92], [114, 138]])
    assert np.allclose(t2.grad, [[62, 62], [72, 72]])

    # test matrix x vector
    t0 = rev.Tensor([[1, 2], [3, 4]])
    t1 = rev.Tensor([5, 6])
    res = (t0 @ t1).sum()

    res.backward()

    assert res.value == 56
    assert np.allclose(t0.grad, [[5, 6], [5, 6]])
    assert np.allclose(t1.grad, [4, 6])

    # test __rmatmul__
    t0 = [[1, 2], [3, 4]]
    t1 = rev.Tensor([5, 6])
    res = (t0 @ t1).sum()

    res.backward()
    assert res.value == 56
    assert np.allclose(t1.grad, [4, 6])


def test_broadcast():
    """
    Test gradients when broadcasting
    """

    t0 = rev.Tensor([[1], [2]])        # a 2 x 1 matrix
    t1 = rev.Tensor([[5, 6], [6, 7]])  # a 2 x 2 matrix
    res = (t0 * t1).sum()

    res.backward()

    assert res.value == 37
    assert np.allclose(t0.grad, [[11], [13]])
    assert np.allclose(t1.grad, [[1, 1], [2, 2]])


def test_reset_grad():
    """
    Test out reset_grad()
    """
    t0 = rev.Tensor(10)
    t1 = rev.Tensor(20)

    res = t0 * t1
    res.backward()

    assert t0.grad == 20
    assert t0.grad_value == 20
    assert t0.children != []

    assert t1.grad == 10
    assert t1.grad_value == 10
    assert t1.children != []

    t0.reset_grad()
    t1.reset_grad()

    assert t0.grad_value is None
    assert t0.children == []

    assert t1.grad_value is None
    assert t1.children == []


def test_aggregates():
    """
    Test out aggregates. Sums, products, means
    """

    # test aggregates for vectors
    t0 = rev.Tensor([3, 5, 7])

    res = t0.sum()
    res.backward()
    assert res.value == 15
    assert np.allclose(t0.grad, [1, 1, 1])

    t0.reset_grad()
    res = t0.mean()
    res.backward()
    assert res.value == 5
    assert np.allclose(t0.grad, [1/3, 1/3, 1/3])

    t0.reset_grad()
    res = t0.prod()
    res.backward()
    assert res.value == 105
    assert np.allclose(t0.grad, [35, 21, 15])

    # test aggregates for scalars
    t1 = rev.Tensor(5)

    res = t1.sum()
    res.backward()
    assert res.value == 5
    assert t1.grad == 1

    t1.reset_grad()
    res = t1.mean()
    res.backward()
    assert res.value == 5
    assert t1.grad == 1

    t1.reset_grad()
    res = t1.prod()
    res.backward()
    assert res.value == 5
    assert t1.grad == 1


def test_no_grad():
    """
    Check that the computational graph is not built while no_grad() is active
    """
    t0 = rev.Tensor(5)
    t1 = t0 * t0
    assert t0.children != []

    with rev.no_grad():
        t0 = rev.Tensor(5)
        t1 = t0 * t0
        assert t0.children == []


def test_get_set_item():
    """
    Test getting items in a Tensor
    """
    t0 = rev.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    s1 = t0[0, :]
    s2 = t0[1, :]
    res = (s1 * s2).sum()

    res.backward()
    assert np.allclose(s1.value, [1, 2, 3, 4])
    assert np.allclose(s2.value, [5, 6, 7, 8])

    assert res.value == 70
    assert np.allclose(s1.grad, [5, 6, 7, 8])
    assert np.allclose(s2.grad, [1, 2, 3, 4])
    assert np.allclose(t0.grad, [[5, 6, 7, 8], [1, 2, 3, 4]])


def test_set_item():
    """
    Test setting items in a Tensor
    """
    t0 = rev.Tensor([[1, 1, 1], [1, 1, 1]])
    s1 = rev.Tensor([1000, 2])
    t0[0, 1:3] = s1

    res = t0.prod()
    res.backward()

    assert res.value == 2000
    assert np.allclose(t0.grad, [[2000, 2, 1000], [2000, 2000, 2000]])
    assert np.allclose(s1.grad, [2, 1000])

import pyad.reverse_mode as rev
import numpy as np
import pytest
import matplotlib.axes._base as axbase

# Simple test suite explaining how Tensor works
#Initialization
def test_illu():
    x = rev.Tensor(0.5)
    y = rev.Tensor(4.2)
    z = rev.Tensor(3)
    f = x * y**3 + rev.sin(x) - rev.logistic(z)

    # set df seed
    f.backward()

    # f_val = f.value
    # x_val = x.value
    # x_grad = x.grad
    # y_grad = y.grad
    # z_grad = z.grad

    assert abs(f.value - (0.5*4.2**3+np.sin(0.5) - 1/(1+np.exp(-3)))) <= 1e-15
    assert abs(x.grad - (4.2**3 + np.cos(0.5))) <= 1e-15
    assert abs(y.grad - (3*0.5*4.2**2)) <= 1e-15
    assert abs(z.grad - (-np.exp(-3)/(1+np.exp(-3))**2)) <= 1e-15


def test_graph_mode():

    x = rev.Tensor(0.5)
    y = rev.Tensor(4.2)
    z = rev.Tensor(3)
    f = x * y**3 + rev.sin(x) - rev.logistic(z)

    #set df seed
    f.backward()

    rev_g = rev.rev_graph()
    plot = rev_g.plot_graph([x,y,z])

    assert type(plot).mro()[3] == axbase._AxesBase


def test_search_path():
    x = rev.Tensor(1)
    y = rev.Tensor(2)
    f = x + x*y

    #set df seed
    f.backward()

    rev_g = rev.rev_graph()
    rev_g.search_path(x)

    assert rev_g.connections == [[1, 2], [2, 3], [1, 3]]
    assert rev_g.formatted_connections == [['x1: 1.00', 'x2: 2.00'], ['x2: 2.00', 'x3: 3.00'], ['x1: 1.00', 'x3: 3.00']]


def test_graph_init():

    rev_g = rev.rev_graph()

    assert rev_g.connections == []
    assert rev_g.formatted_connections == []
    assert rev_g.unique_nodes == []
    assert rev_g.operations == []
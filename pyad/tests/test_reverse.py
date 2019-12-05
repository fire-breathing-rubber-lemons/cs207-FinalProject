import pyad.reverse_mode as rev
import numpy as np
import pytest

# Simple test suite explaining how Tensor works
#Initialization
def test_illu():
	x = rev.Tensor(0.5)
	y = rev.Tensor(4.2)
	z = rev.Tensor(3)
	f = x * y**3 + rev.sin(x) - rev.logistic(z)

	#set df seed
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
from reverse_mode import *
import numpy as np

#Initialization
x = Reverse(0.5)
y = Reverse(4.2)
z = Reverse(3)
f = x * y**3 + sin(x) - logistic(z)

#set df seed to be 1
f.grad_value = 1.0

f_val = f.value
x_val = x.value
x_grad = x.get_grad()
y_grad = y.get_grad()
z_grad = z.get_grad()

assert abs(f.value - (0.5*4.2**3+np.sin(0.5) - 1/(1+np.exp(-3)))) <= 1e-15
assert abs(x.get_grad() - (4.2**3 + np.cos(0.5))) <= 1e-15
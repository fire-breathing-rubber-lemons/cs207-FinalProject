import numpy as np

class Reverse:

	def __init__(self, value):
		self.value = value
		self.children = []
		self.grad_value = None

	def get_grad(self):
		if self.grad_value is None:
		    self.grad_value = sum(weight * var.get_grad() for weight, var in self.children)
		return self.grad_value

	def __neg__(self):
		return self.__mul__(-1)

	def __add__(self, other):
		if isinstance(other, Reverse):
			z = Reverse(self.value + other.value)
			self.children.append((1, z))
			other.children.append((1, z))
		else:
			z = Reverse(self.value + other)
			self.children.append((1, z))
		return z

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		sub = other.__neg__()
		return self.__add__(sub)

	def __rsub__(self, other):
		sub = self.__neg__()
		return sub + other

	def __mul__(self, other):
		if isinstance(other, Reverse):
			z = Reverse(self.value * other.value)
			self.children.append((other.value, z))
			other.children.append((self.value, z))
		else:
			z = Reverse(self.value*other)
			self.children.append((other, z))
		return z

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		if isinstance(other, Reverse):
			z = Reverse(self.value / other.value)
			self.children.append((1/other.value, z))
			other.children.append((-self.value/other.value**2, z))
		else:
			z = Reverse(self.value / other)
			self.children.append((1/other, z))
		return z

	def __rtruediv__(self, other):
		if isinstance(other, Reverse):
			z = Reverse(other.value / self.value)
			self.children.append((-other.value/self.value**2, z))
			other.children.append((1/self.value, z))
		else:
			z = Reverse(other / self.value)
			self.children.append((-other/self.value**2, z))
		return z

	def __pow__(self, other):
		if isinstance(other, Reverse):
			z = Reverse(self.value**other.value)
			self.children.append((other.value*self.value**(other.value-1), z))
			other.children.append((self.value**other.value*np.log(self.value), z))
		else:
			z = Reverse(self.value**other)
			self.children.append((other*self.value**(other-1), z))
		return z

	def __rpow__(self, other):
		z = Reverse(other**self.value)
		self.children.append((other**self.value*np.log(other), z))
		return z

# Elementary functions
def _elementary_op(obj, fn, deriv_fn):
	if isinstance(obj, Reverse):
		z = Reverse(fn(obj.value))
		obj.children.append((deriv_fn(obj.value), z))
		return z
	else:
		return fn(obj)

def sin(x):
	return _elementary_op(x, np.sin, np.cos)

def cos(x):
	return _elementary_op(x, np.cos, lambda x: -np.sin(x))

def tan(x):
	return _elementary_op(x, np.tan, lambda x: 1 / (np.cos(x) ** 2))

def arcsin(x):
	return _elementary_op(x, np.arcsin, lambda x: 1 / np.sqrt(1 - x ** 2))

def arccos(x):
	return _elementary_op(x, np.arccos, lambda x: -1 / np.sqrt(1 - x ** 2))

def arctan(x):
	return _elementary_op(x, np.arctan, lambda x: 1 / (1 + x ** 2))

def sinh(x):
	return _elementary_op(x, np.sinh, np.cosh)

def cosh(x):
	return _elementary_op(x, np.cosh, np.sinh)

def tanh(x):
	return _elementary_op(x, np.tanh, lambda x: 1/(np.cosh(x)**2))

def abs(x):
	return _elementary_op(x, np.abs, np.sign)

def exp(x):
	return _elementary_op(x, np.exp, np.exp)

def logistic(x):
	f = lambda x: 1/(1+np.exp(-x))
	df = lambda x: np.exp(-x)/(1+np.exp(-x))**2
	return _elementary_op(x, f, df)

def log(x, base = np.e):
	return _elementary_op(x, np.log, lambda x: 1 / x) / np.log(base)

def log2(x):
	return log(x, base = 2)

def log10(x):
	return log(x, base = 10)

def sqrt(x):
	return _elementary_op(x, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))

def cbrt(x):
	return _elementary_op(x, np.cbrt, lambda x: 1 / (3 * x**(2/3)))


# Simple test suite explaining how Reverse works
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

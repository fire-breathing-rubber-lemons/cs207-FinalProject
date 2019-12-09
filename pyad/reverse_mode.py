import numpy as np


class Tensor:

	def __init__(self, value):
		self.value = value
		self.children = []
		self.grad_value = None

	def backward(self):
		self.grad_value = 1.0

	@property
	def grad(self):
		if self.grad_value is None:
		    self.grad_value = sum(weight * var.grad for weight, var in self.children)
		return self.grad_value

	# Comparisons work like they do in numpy. Derivative information is ignored
    # numpy arrays are returned by comparisons
	def __lt__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		return self.value < other.value

	def __gt__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		return self.value > other.value

	def __le__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		return self.value <= other.value

	def __ge__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		return self.value >= other.value

	def __eq__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		return self.value == other.value

	def __ne__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		return self.value != other.value

	def __bool__(self):
		return bool(self.value)
	
	def __repr__(self):
		return f'Tensor({self.value}, D({self.grad_value}))'

	def __neg__(self):
		return self.__mul__(-1)

	def __add__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		z = Tensor(self.value + other.value)
		self.children.append((1.0, z))
		other.children.append((1.0, z))
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
		other = other if isinstance(other, Tensor) else Tensor(other)
		z = Tensor(self.value * other.value)
		self.children.append((other.value, z))
		other.children.append((self.value, z))
		return z

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		z = Tensor(self.value / other.value)
		self.children.append((1/other.value, z))
		other.children.append((-self.value/other.value**2, z))
		return z

	def __rtruediv__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		z = Tensor(other.value / self.value)
		self.children.append((-other.value/self.value**2, z))
		other.children.append((1/self.value, z))
		return z

	def __pow__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		z = Tensor(self.value**other.value)
		self.children.append((other.value*self.value**(other.value-1), z))
		other.children.append((self.value**other.value*np.log(self.value), z))
		return z

	def __rpow__(self, other):
		other = other if isinstance(other, Tensor) else Tensor(other)
		z = Tensor(other.value**self.value)
		self.children.append((other.value**self.value*np.log(other.value), z))
		return z


# Elementary functions
def _elementary_op(obj, fn, deriv_fn):
    obj = obj if isinstance(obj, Tensor) else Tensor(obj)
    z = Tensor(fn(obj.value))
    obj.children.append((deriv_fn(obj.value), z))
    return z


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
    return 1 / (1 + exp(-x))


def log(x, base=np.e):
    return _elementary_op(x, np.log, lambda x: 1 / x) / np.log(base)


def log2(x):
    return log(x, base=2)


def log10(x):
    return log(x, base=10)


def sqrt(x):
    return _elementary_op(x, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))


def cbrt(x):
    return _elementary_op(x, np.cbrt, lambda x: 1 / (3 * x**(2/3)))

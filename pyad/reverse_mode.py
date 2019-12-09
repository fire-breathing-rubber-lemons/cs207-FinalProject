import numpy as np


class Tensor:
    """
    Class for automatic differentiation reverse mode
    """

    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def backward(self):
        """
        A function that seeds in the derivative of a function with respect to itself, i.e. df/df = 1
        """
        self.grad_value = 1.0

    @property
    def grad(self):
        """
        A function that computes the gradient value using the chain rule
        """
        if self.grad_value is None:
            # recurse only if the value is not yet cached
            self.grad_value = sum(weight * var.grad for weight, var in self.children)
        return self.grad_value

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
        self.children.append((other.value**self.value*np.log(other), z))
        return z

# Elementary functions
def _elementary_op(obj, fn, deriv_fn):
    """
    A generic framework to allow for the chain rule of other elementary
    functions taken from the numpy module.
    Parameters
    ----------
    obj : Scalar or Tensor object which the elementary function is being
        differentiated at
    fn : np.function
        The elementary function from the numpy module
    deriv_fun:  np.function
        The prespecified derivative of the given numpy elementary function
    Returns
    -------
    Tensor: class
        Tensor object which contains the resulting value and result from the
        chain rule (new derivative)
    """
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

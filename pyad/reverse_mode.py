import numpy as np


class Tensor:
    def __init__(self, value):
        if isinstance(value, Tensor):
            self.value = np.array(value.value)
            self.children = value.children.copy()
            self.grad_value = None
        else:
            self.value = np.array(value)
            self.children = []
            self.grad_value = None

        if len(self.value.shape) > 2:
            raise ValueError('Can only support 0-D, 1-D, and 2-D Tensors')

    def backward(self):
        if self.value.shape != ():
            raise Exception('Cannot call .backward() on a non-scalar')
        self.grad_value = np.array(1.0)

    @property
    def grad(self):
        if self.grad_value is None:
            self.grad_value = 0
            for weight, node in self.children:
                if callable(weight):
                    self.grad_value += weight(node.grad)
                else:
                    self.grad_value += weight * node.grad
        return self.grad_value

    @property
    def shape(self):
        return self.value.shape

    def sum(self):
        z = Tensor(np.sum(self.value))
        self.children.append((np.ones(self.shape), z))
        return z

    def prod(self):
        if self.shape == ():
            return self

        forward_prod = np.cumprod(self.value.flatten())
        backward_prod = np.cumprod(self.value.flatten()[::-1])[::-1]
        result = np.ones(len(forward_prod))

        for i in range(len(forward_prod)):
            if i != 0:
                result[i] *= forward_prod[i - 1]
            if i != len(forward_prod) - 1:
                result[i] *= backward_prod[i + 1]

        z = Tensor(np.prod(self.value))
        self.children.append((result.reshape(self.shape), z))
        return z

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
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value * other.value)
        self.children.append((other.value, z))
        other.children.append((self.value, z))
        return z

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value / other.value)
        self.children.append((1 / other.value, z))
        other.children.append((-self.value / other.value**2, z))
        return z

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value ** other.value)
        self.children.append((other.value * self.value**(other.value - 1), z))
        other.children.append((self.value**other.value * np.log(self.value), z))
        return z

    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other ** self

    def _to_2d(self, x):
        """
        A helper function for reshaping numpy arrays as 2D matrices
        """
        if x.shape == ():
            return x.reshape(1, 1)
        return x.reshape(len(x), -1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def mm_backward_self(out_grad):
            other_mat = self._to_2d(other.value)
            out_grad_mat = self._to_2d(out_grad)
            return (out_grad_mat @ other_mat.T).reshape(self.value.shape)

        def mm_backward_other(out_grad):
            self_mat = self._to_2d(self.value)
            out_grad_mat = self._to_2d(out_grad)
            return (self_mat.T @ out_grad_mat).reshape(other.value.shape)

        z = Tensor(self.value @ other.value)
        self.children.append((mm_backward_self, z))
        other.children.append((mm_backward_other, z))
        return z

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other @ self


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
    return _elementary_op(x, np.tanh, lambda x: 1 / (np.cosh(x) ** 2))


def abs(x):
    return _elementary_op(x, np.abs, np.sign)


def exp(x):
    return _elementary_op(x, np.exp, np.exp)


def logistic(x):
    return 1 / (1 + exp(-x))


def log(x, base=np.e):
    if base == np.e:
        return _elementary_op(x, np.log, lambda x: 1 / x)
    return log(x) / log(base)


def log2(x):
    return log(x, base=2)


def log10(x):
    return log(x, base=10)


def sqrt(x):
    return _elementary_op(x, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))


def cbrt(x):
    return _elementary_op(x, np.cbrt, lambda x: 1 / (3 * x ** (2/3)))

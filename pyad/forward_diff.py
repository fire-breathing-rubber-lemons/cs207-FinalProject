import numpy as np

# we use this to get rid of disruptive runtime warnings
runtime_warning_filter = np.testing.suppress_warnings()
runtime_warning_filter.filter(RuntimeWarning)


class MultivariateDerivative:
    def __init__(self, variables=None):
        self.variables = variables or {}

    def __repr__(self):
        values = ', '.join(f'{k}={v:.3g}' for k, v in self.variables.items())
        return f'D({values})'

    def copy(self):
        return MultivariateDerivative(self.variables.copy())

    def __getitem__(self, key):
        return self.variables[key]

    # the only way to combine derivatives is with addition or multiplication
    def __add__(self, other):
        all_variables = set(self.variables.keys()) | set(other.variables.keys())
        result = {}
        for v in all_variables:
            result[v] = self.variables.get(v, 0) + other.variables.get(v, 0)
        return MultivariateDerivative(result)

    def mul(self, multiplier):
        result = {}
        for k, v in self.variables.items():
            result[k] = multiplier * v
        return MultivariateDerivative(result)


class Tensor:
    def __init__(self, value, d=None):
        if isinstance(value, Tensor):
            value, d = value.value, value.d.copy()

        self.value = np.array(value)
        self.d = d or MultivariateDerivative()

    @staticmethod
    def get_value_and_deriv(other):
        if isinstance(other, Tensor):
            return other.value, other.d
        return other, MultivariateDerivative()

    def norm(self):
        return np.linalg.norm(self.value)

    def __repr__(self):
        return f'Tensor({self.value:.3g}, {self.d})'

    def __neg__(self):
        return Tensor(-self.value, self.d.mul(-1))

    def __add__(self, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        return Tensor(self.value + other_v, self.d + other_d)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        return Tensor(self.value - other_v, self.d + other_d.mul(-1))

    def __rsub__(self, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        return Tensor(other_v - self.value, other_d + self.d.mul(-1))

    def __mul__(self, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        prod_rule = self.d.mul(other_v) + other_d.mul(self.value)
        return Tensor(self.value * other_v, prod_rule)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        quot_rule = self.d.mul(other_v) + other_d.mul(-self.value)
        quot_rule = quot_rule.mul(1 / (other_v ** 2))
        return Tensor(self.value / other_v, quot_rule)

    def __rtruediv__(self, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        quot_rule = other_d.mul(self.value) + self.d.mul(-other_v)
        quot_rule = quot_rule.mul(1 / (self.value ** 2))
        return Tensor(other_v / self.value, quot_rule)

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    # TODO: matrix operators like dot prod, matrix mult, inverse, tranpose, etc


class Variable(Tensor):
    def __init__(self, name, value):
        self.name = name
        self.value = np.array(value)
        self.d = MultivariateDerivative({
            name: np.ones(self.value.shape)
        })


# Elementary operations of a single variable. They all use chain rule.
def _elementary_op(obj, fn, deriv_fn):
    v, d = Tensor.get_value_and_deriv(obj)
    chain_rule = d.mul(deriv_fn(v))
    return Tensor(fn(v), chain_rule)


def sin(tensor):
    return _elementary_op(tensor, np.sin, np.cos)


def cos(tensor):
    return _elementary_op(tensor, np.cos, lambda x: -np.sin(x))


def tan(tensor):
    return _elementary_op(tensor, np.tan, lambda x: 1 / (np.cos(x) ** 2))


def arcsin(tensor):
    return _elementary_op(tensor, np.arcsin, lambda x: 1 / np.sqrt(1 - x ** 2))


def arccos(tensor):
    return _elementary_op(tensor, np.arccos, lambda x: -1 / np.sqrt(1 - x ** 2))


def arctan(tensor):
    return _elementary_op(tensor, np.arctan, lambda x: 1 / (1 + x ** 2))


def abs(tensor):
    # for simplicity, we just define D[abs(x)] == 0 when x == 0
    return _elementary_op(tensor, np.abs, np.sign)


def exp(tensor):
    return _elementary_op(tensor, np.exp, np.exp)


def log(tensor):
    return _elementary_op(tensor, np.log, lambda x: 1 / x)


def sqrt(tensor):
    return _elementary_op(tensor, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))


def cbrt(tensor):
    return _elementary_op(tensor, np.cbrt, lambda x: 1 / (3 * np.power(x, 2/3)))


@runtime_warning_filter
def power(base, exp):
    base_v, base_d = Tensor.get_value_and_deriv(base)
    exp_v, exp_d = Tensor.get_value_and_deriv(exp)

    result = base_v ** exp_v
    a = base_d.mul(exp_v * base_v ** (exp_v - 1.0))
    b = exp_d.mul(result * np.log(base_v))
    return Tensor(result, a + b)


# wrappers around Tensor and Variable constructors
def tensor(*args, **kwargs):
    return Tensor(*args, **kwargs)


def var(*args, **kwargs):
    return Variable(*args, **kwargs)

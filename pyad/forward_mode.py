import numpy as np
from collections import defaultdict

# we use this to get rid of disruptive runtime warnings
runtime_warning_filter = np.testing.suppress_warnings()
runtime_warning_filter.filter(RuntimeWarning)


class MultivariateDerivative:
    """
    Multivariate Derivative is a class called automatically by the Variable
    or Tensor classes to hold derivative information.

    Parameters
    ----------
    variables : dict
        Store the name and value of the derivative object allowing for multiple
        variable keys

    Attributes
    ----------
    variables : dict
        Internal storage of the named derivative values in dictionary format
    """

    def __init__(self, variables=None):
        variables = variables or {}
        self.variables = {k: np.array(v) for k, v in variables.items()}

    def __repr__(self):
        """
        Create a descriptive object for printing, D noting that the object is a
        derivative

        Returns
        -------
        str
            a string representing the derivative w.r.t all of the variables
        """
        values = ', '.join(f'{k}={v}' for k, v in self.variables.items())
        return f'D({values})'

    def copy(self):
        """
        Given an existing MultivariateDerivative object (self) create a new one
        as a copy

        Returns
        -------
        MultivariateDerivative
            a new MultivariateDerivative object
        """
        return MultivariateDerivative(self.variables.copy())

    def __getitem__(self, key):
        """
        Access the derivative of one of the variables in the object.
        instance.variables.keys() will reveal the existing keys within the
        instance

        Returns
        -------
        dict value
            Value of the dictionary of derivatives at the specified key
        """
        return self.variables[key]

    # the only way to combine derivatives is with addition or multiplication
    def __add__(self, other):
        """
        Adds this and another MultivariateDerivative object together and
        produces a new MultivariateDerivative object whose variable set is the
        union of the two individuals.

        Returns
        -------
        MultivariateDerivative
            a new MultivariateDerivative object
        """
        var_set = set(self.variables.keys()) | set(other.variables.keys())
        result = {}
        for v in var_set:
            a, b = self.variables.get(v, 0), other.variables.get(v, 0)
            a, b = self._broadcast(a, b)
            result[v] = a + b
        return MultivariateDerivative(result)

    def mul(self, multiplier):
        """
        Multiplies all of the values in this MultivariateDerivative by a
        scalar (i.e. non-Tensor and non-MultivariateDerivative) object and
        return a new MultivariateDerivative.

        Returns
        -------
        MultivariateDerivative
            a new MultivariateDerivative object
        """
        result = {}
        for k, v in self.variables.items():
            a, b = self._broadcast(multiplier, v)
            result[k] = a * b
        return MultivariateDerivative(result)

    def get_idx(self, i):
        """
        Returns a copy of the MultivariateDerivative with the desired index
        `i` from each variable selected

        Returns
        -------
        MultivariateDerivative
            a new MultivariateDerivative object
        """
        result = {k: v[i] for k, v in self.variables.items()}
        return MultivariateDerivative(result)

    def set_idx(self, i, other, tensor_value):
        """
        Modifies the MultivariateDerivative by updating with the desired index
        `i` in each variable. There cannot be any variables in `other` that
        are not already in the current variable set

        Returns
        -------
        None
        """
        for k, v in self.variables.items():
            if k not in other.variables:
                self.variables[k][i] *= 0

        for k, v in other.variables.items():
            if k not in self.variables:
                self.variables[k] = np.zeros(tensor_value.shape)
            self.variables[k][i] = other.variables[k]

    def _fix_shape(self, value):
        """
        If a vector Tensor and a scalar variable are added, then it's possible
        that the MultivariateDerivative that results won't have the right shape.
        This fixes any shapes based on the shape of the Tensor by broadcasting

        Returns
        -------
        None
        """
        for k, v in self.variables.items():
            if len(v.shape) < len(value.shape):
                a, b = self._broadcast(value, v)
                self.variables[k] = np.zeros(a.shape, dtype=b.dtype) + b

    def _broadcast(self, v1, v2):
        """
        Given two np.arrays, this function figures out the appropriate indexing
        for both variables which would allow the two operands to broadcast.
        """
        v1, v2 = np.array(v1), np.array(v2)
        if len(v1.shape) < len(v2.shape):
            idx = tuple(slice(None) for i in range(len(v1.shape)))
            idx = idx + (None,) * (len(v2.shape) - len(v1.shape))
            return v1[idx], v2
        elif len(v1.shape) > len(v2.shape):
            idx = tuple(slice(None) for i in range(len(v2.shape)))
            idx = idx + (None,) * (len(v1.shape) - len(v2.shape))
            return v1, v2[idx]
        else:
            return v1, v2


class Tensor:
    """
    Tensor (constant value) is the base class of the Variable and can be used to
    represent constant values where the derivative is not required for automatic
    computation during automatic differentiation. Derivative by default is set
    to an empty dictionary.

    Parameters
    ----------
    value : np.array
        Store the value of the constant

    d : MultivariateDerivative
        Stores the value of the derivative. The default is an empty
        MultivariateDerivative, but it can be manually set to a specific value

    Attributes
    ----------
    value : np.array
        Internal storage of the tensor value

    d : MultivariateDerivative
        Internal storage of the tensor derivative
    """
    def __init__(self, value, d=None):
        # If existing Tensor check for whether a derivative value was set
        if isinstance(value, Tensor):
            value, d = value.value, value.d.copy()

        self.value = np.array(value)
        self.d = d or MultivariateDerivative()
        self.d._fix_shape(self.value)

    @staticmethod
    def get_value_and_deriv(other):
        if isinstance(other, Tensor):
            return other.value, other.d
        return other, MultivariateDerivative()

    def sum(self):
        if self.shape == ():
            return Tensor(self)
        return sum(self, Tensor(0))

    def norm(self, d=2):
        if self.shape == ():
            return Tensor(self)
        return power(sum(x ** d for x in self), 1/d)

    def all(self):
        return bool(np.all(self.value))

    def any(self):
        return bool(np.any(self.value))

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        size = len(self)
        return (self[i] for i in range(size))

    @property
    def shape(self):
        return self.value.shape

    # Comparisons work like they do in numpy. Derivative information is ignored
    # numpy arrays are returned by comparisons
    def __lt__(self, other):
        other_v, _ = Tensor.get_value_and_deriv(other)
        return self.value < other_v

    def __gt__(self, other):
        other_v, _ = Tensor.get_value_and_deriv(other)
        return self.value > other_v

    def __le__(self, other):
        other_v, _ = Tensor.get_value_and_deriv(other)
        return self.value <= other_v

    def __ge__(self, other):
        other_v, _ = Tensor.get_value_and_deriv(other)
        return self.value >= other_v

    def __eq__(self, other):
        other_v, _ = Tensor.get_value_and_deriv(other)
        return self.value == other_v

    def __ne__(self, other):
        other_v, _ = Tensor.get_value_and_deriv(other)
        return self.value != other_v

    def __bool__(self):
        return bool(self.value)

    def __repr__(self):
        return f'Tensor({self.value}, {self.d})'

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
    def __getitem__(self, i):
        return Tensor(self.value[i], self.d.get_idx(i))

    def __setitem__(self, i, other):
        other_v, other_d = Tensor.get_value_and_deriv(other)
        self.d.set_idx(i, other_d, self.value)
        self.value[i] = other_v


class Variable(Tensor):
    """
    Variable (a variable value for the purposes of differentiation). A variable
    is used to represent any value which the user wishes to include in the
    partial derivative outputs and is built on top of the Tensor base class.

    As a variable a name parameter is required (example: x, y or z).

    Parameters
    ----------
    name: str
        Store the name of the variable

    value : np.array
        Store the value of the variable

    d : MultivariateDerivative
        Store the value of the derivative, default is the identity matrix
        which will result in the computation of the Jacobian

    Attributes
    ----------
    name: str
        Internal storage of the variable name

    value : np.array
        Internal storage of the variable value

    d : MultivariateDerivative
        Internal storage of the variable derivative
    """
    def __init__(self, name, value):
        self.name = name
        self.value = np.array(value)

        if len(self.value.shape) >= 2:
            raise ValueError('Variable does not support dimensions >= 2')

        if self.value.shape == (0,):
            raise ValueError('Cannot have an empty value for Variable')

        if self.value.shape == ():
            self.d = MultivariateDerivative({
                name: np.array(1)
            })
        else:
            self.d = MultivariateDerivative({
                name: np.eye(len(self.value))
            })


# Elementary operations of a single variable. They all use chain rule.
def _elementary_op(obj, fn, deriv_fn):
    """
    A generic framework to allow for the chain rule of other elementary
    functions taken from the numpy module.

    Parameters
    ----------
    obj : Class
        The Variable or Tensor which the elementary function is being
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
    v, d = Tensor.get_value_and_deriv(obj)
    chain_rule = d.mul(deriv_fn(v))
    return Tensor(fn(v), chain_rule)


def sin(tensor):
    """
    pyad sin - computes the sine function
        sine differentiates to cosine

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the sin function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.sin, np.cos)


def cos(tensor):
    """
    pyad cos - computes the cosine function
        cosine differentiates to minus sine

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the cosine function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.cos, lambda x: -np.sin(x))


def tan(tensor):
    """
    pyad tan - computes the tangent function
        The tangent of x differentiates to sec^2(x) (1/cos^2(x))

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the tangent function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.tan, lambda x: 1 / (np.cos(x) ** 2))


def arcsin(tensor):
    """
    pyad arcsin - computes the arcsine function
        The arcsine of x differentiates to 1/√(1-x^2)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the arcsine function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.arcsin, lambda x: 1 / np.sqrt(1 - x ** 2))


def arccos(tensor):
    """
    pyad arccos - computes the arccosine function
        The arccosine of x differentiates to -1/√(1-x^2)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the arccosine function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.arccos, lambda x: -1 / np.sqrt(1 - x ** 2))


def arctan(tensor):
    """
    pyad arctan - computes the arctangent function
        The arctangent of x differentiates to 1/(1+x^2)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the arctangent function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.arctan, lambda x: 1 / (1 + x ** 2))


def sinh(tensor):
    '''
    pyad sinh - computes the hyperbolic sine function
        The sinh of x differentiates to cosh(x)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the sinh function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Calls the _elementary_op function and returns the resulting Tensor
    '''
    return _elementary_op(tensor, np.sinh, np.cosh)


def cosh(tensor):
    '''
    pyad cosh - computes the hyperbolic cosine function
        The cosh of x differentiates to sinh(x)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the cosh function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Calls the _elementary_op function and returns the resulting Tensor
    '''
    return _elementary_op(tensor, np.cosh, np.sinh)


def tanh(tensor):
    '''
    pyad tanh - computes the hyperbolic tangent function
        The tanh of x differentiates to 1/cosh^2(x)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the cosh function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Calls the _elementary_op function and returns the resulting Tensor
    '''
    return _elementary_op(tensor, np.tanh, lambda x: 1/(np.cosh(x)**2))


def abs(tensor):
    """
    pyad abs - computes the absolute value function
        For simplicity, we have defined the absolute value derviative to be the
        sign of the argument:
            if x > 0:  return 1
            if x == 0: return 0
            if x < 0:  return -1

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the absolute value function
        is being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.abs, np.sign)


def exp(tensor):
    """
    pyad exp - computes the exponential function, e^x
        e^x differentiates to e^x

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the exponential function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.exp, np.exp)   


def log(tensor, base=np.e):
    """
    pyad log - computes the natural logarithm function
        The natural logarithm of x differentiates to 1/x

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the natural logarithm is
        being differentiated at

    base: float
        The base of the logarithm. Defaults to np.e.

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.log, lambda x: 1 / x) / np.log(base)


def log2(tensor):
    """
    pyad log2 - a wrapper of log to computes logarithm base 2

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return log(tensor, base=2)


def log10(tensor):
    """
    pyad log10 - a wrapper of log to computes logarithm base 10

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return log(tensor, base=10)


def sqrt(tensor):
    """
    pyad sqrt - computes the square root function
        The square root of x differentiates to 1/(2√x)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the square root is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))


def cbrt(tensor):
    """
    pyad cbrt - computes the cube root function
        The cube root of x differentiates to 1/(3x^(2/3))

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the cube root is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return _elementary_op(tensor, np.cbrt, lambda x: 1 / (3 * np.power(x, 2/3)))


@runtime_warning_filter
def power(base, exp):
    """
    pyad power - computes the power function
        The power function is differentiated using the generalized power rule:
            d/dx[a^b] = d/dx[exp(ln(a) * b)] = a^b * (b' * ln(a) + a'/a * b)

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the power function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    base_v, base_d = Tensor.get_value_and_deriv(base)
    exp_v, exp_d = Tensor.get_value_and_deriv(exp)

    result = base_v ** exp_v
    a = base_d.mul(exp_v * base_v ** (exp_v - 1.0))
    b = exp_d.mul(result * np.log(base_v))
    return Tensor(result, a + b)


def logistic(tensor):
    """
    pyad logistic - computes the logistic function: 1 / (1 + exp(-x))

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the logistic function is
        being differentiated at

    Returns
    -------
    Tensor: class
        Applies chain rule as appropriate and returns the resulting Tensor
    """
    return 1 / (1 + exp(-tensor))


# functions for vector operatoions
def stack(tensors):
    """
    pyad stack - combines multiple pyad Tensors into a single tensor by stacking

    Parameters
    ----------
    tensors : List[Tensor]
        The list of tensors that stack

    Returns
    -------
    Tensor: class
        The resulting derivative is computed by stacking the individual components
    """
    if len(tensors) == 0:
        raise ValueError('need at least one tensor to stack')

    values = []
    derivs = []
    var_set = set()
    for t in tensors:
        v, d = Tensor.get_value_and_deriv(t)
        values.append(v)
        derivs.append(d)
        var_set |= set(d.variables.keys())

    variables = defaultdict(list)
    for k in var_set:
        d_vals = [d.variables[k] for d in derivs if k in d.variables]
        if not all(v.shape == d_vals[0].shape for v in d_vals):
            raise ValueError('all input tensors must have the same shape')

        for d in derivs:
            if k not in d.variables:
                variables[k].append(np.zeros(d_vals[0].shape))
            else:
                variables[k].append(d.variables[k])

        variables[k] = np.array(variables[k])

    stacked_d = MultivariateDerivative(dict(variables))
    return Tensor(np.stack(values), stacked_d)


# wrappers around Tensor and Variable constructors
def tensor(*args, **kwargs):
    """
    tensor - a wrapper for the Tensor class constructor
    """
    return Tensor(*args, **kwargs)


def var(*args, **kwargs):
    """
    var - a wrapper for the Variable class constructor
    """
    return Variable(*args, **kwargs)

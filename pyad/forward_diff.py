import numpy as np

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
        self.variables = variables or {}

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
        all_variables = set(self.variables.keys()) | set(other.variables.keys())
        result = {}
        for v in all_variables:
            result[v] = self.variables.get(v, 0) + other.variables.get(v, 0)
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
            result[k] = multiplier * v
        return MultivariateDerivative(result)


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

    @staticmethod
    def get_value_and_deriv(other):
        if isinstance(other, Tensor):
            return other.value, other.d
        return other, MultivariateDerivative()

    def norm(self):
        return np.linalg.norm(self.value)

    def all(self):
        return bool(np.all(self.value))

    def any(self):
        return bool(np.any(self.value))

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
        Store the value of the derivative, default is an array of 1's
        (unit vectors) which will result in the computation of the Jacobian

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
        self.d = MultivariateDerivative({
            name: np.ones(self.value.shape)
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

import numpy as np

class MultivariateDerivative:
    '''
    Multivariate Derivative is a class called automatically by the Variable
    or Tensor classes to hold derivative information. 

    Parameters
    ----------
    variables : dict
        Store the name and value of the derivative object allowing for multiple variable keys
    
    Attributes
    ----------
    variables : dict
        Internal storage of the named derivative values in dictionary format
    '''
    def __init__(self, variables=None):
        self.variables = variables or {}

    def __repr__(self):
        '''
        Create a descriptive object for printing, D noting that the object is a derivative

        Returns
        -------
        str
            a string representing the value of the derivative of the MultivariateDerivative object
        '''
        values = ', '.join(f'{k}={v:.3g}' for k, v in self.variables.items())
        return f'D({values})'

    def copy(self):
        '''
        Given an existing MultivariateDerivative object (self) create a new one as a copy

        Returns
        -------
        MultivariateDerivative
            a new MultivariateDerivative object
        '''
        return MultivariateDerivative(self.variables.copy())

    def __getitem__(self, key):
        '''
        Access the derivative of one of the variables in the object, print(MultivariateDerivative)
        will reveal the existing keys within the instance.

        Returns
        -------
        dict value
            Value of the dictionary of derivatives at the specified key
        '''
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
    '''
    Tensor (constant value) is the base class of the Variable and can be used to represent 
    constant values where the derivative is not required for automatic computation during
    automatic differentiation. Derivative by default is set to an empty dictionary.

    Parameters
    ----------
    value : np.array
        Store the value of the constant

    d : MultivariateDerivative
        Store the value of the derivative (default is an empty MultivariateDerivative)
        but can be manually set to a specific value
    
    Attributes
    ----------
    value : np.array
        Internal storage of the tensor value

    d : MultivariateDerivative
        Internal storage of the tensor derivative
    '''
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
    '''
    Variable (a variable value for the purposes of differentiation). A variable is used to
    represent any value which the user wishes to include in the partial derivative outputs
    and is built on top of the Tensor base class.

    As a variable a name parameter is required (example: x, y or z).

    Parameters
    ----------
    name: str
        Store the name of the variable

    value : np.array
        Store the value of the variable

    d : MultivariateDerivative
        Store the value of the derivative, default is an array of 1's (unit vectors) which
        will result in the computation of the Jacobian
    
    Attributes
    ----------
    name: str
        Internal storage of the variable name

    value : np.array
        Internal storage of the variable value

    d : MultivariateDerivative
        Internal storage of the variable derivative
    '''
    def __init__(self, name, value):
        self.name = name
        self.value = np.array(value)
        self.d = MultivariateDerivative({
            name: np.ones(self.value.shape)
        })


# Elementary operations of a single variable. They all use chain rule.
def _elementary_op(obj, fn, deriv_fn):
    '''
    A generic framework to allow for the chain rule of other elementary functions 
    taken from the numpy module.

    Parameters
    ----------
    obj : Class
        The Variable or Tensor which the elementary function is being differentiated at

    fn : np.function
        The elementary function from the numpy module

    deriv_fun:  np.function
        The prespecified derivative of the given numpy elementary function


    Returns
    -------
    Tensor: class
        Tensor object which contains the resulting value and result from the
        chain rule (new derivative)
    '''
    v, d = Tensor.get_value_and_deriv(obj)
    chain_rule = d.mul(deriv_fn(v))
    return Tensor(fn(v), chain_rule)


def sin(tensor):
    '''
    pyad sin - to calculate a Tensor (value and derivative) of the sin function
        sin differentiates to cosine

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the sin function is being differentiated at

    Returns
    -------
    Tensor: class
        Calls the _elementary_op function and returns the resulting Tensor
    '''
    return _elementary_op(tensor, np.sin, np.cos)


def cos(tensor):
    '''
    pyad cos - to calculate a Tensor (value and derivative) of the cos function
        cosine differentiates to minus sin

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the cosine function is being differentiated at

    Returns
    -------
    Tensor: class
        Calls the _elementary_op function and returns the resulting Tensor
    '''
    return _elementary_op(tensor, np.cos, lambda x: -np.sin(x))


def tan(tensor):
    '''
    pyad cos - to calculate a Tensor (value and derivative) of the tan function
        tan differentiates to sec^2(x) (1/cos^2(x))

    Parameters
    ----------
    tensor : class
        The value of the variable or constant which the tan function is being differentiated at

    Returns
    -------
    Tensor: class
        Calls the _elementary_op function and returns the resulting Tensor
    '''
    return _elementary_op(tensor, np.tan, lambda x: 1 / (np.cos(x) ** 2))


def arcsin(tensor):
    return _elementary_op(tensor, np.arcsin, lambda x: 1 / np.sqrt(1 - x ** 2))


def arccos(tensor):
    return _elementary_op(tensor, np.arccos, lambda x: -1 / np.sqrt(1 - x ** 2))


def arctan(tensor):
    return _elementary_op(tensor, np.arctan, lambda x: 1 / (1 + x ** 2))


def abs(tensor):
    # for simplicity, we just define D[abs(x)] == 1 when x == 0
    return _elementary_op(tensor, np.abs, lambda x: (2 * (x >= 0)) - 1)


def exp(tensor):
    return _elementary_op(tensor, np.exp, np.exp)


def log(tensor):
    return _elementary_op(tensor, np.log, lambda x: 1 / x)


def sqrt(tensor):
    return _elementary_op(tensor, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))


def cbrt(tensor):
    return _elementary_op(tensor, np.cbrt, lambda x: 1 / (3 * np.pow(x, 2/3)))


def power(base, exp):
    base_v, base_d = Tensor.get_value_and_deriv(base)
    exp_v, exp_d = Tensor.get_value_and_deriv(exp)

    result = base_v ** exp_v
    a = base_d.mul(np.nan_to_num(exp_v * base_v ** (exp_v - 1.0)))
    b = exp_d.mul(result * np.nan_to_num(np.log(base_v)))
    return Tensor(result, a + b)


# wrappers around Tensor and Variable constructors
def tensor(*args, **kwargs):
    return Tensor(*args, **kwargs)


def var(*args, **kwargs):
    return Variable(*args, **kwargs)

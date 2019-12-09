import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# we use this to get rid of disruptive runtime warnings
runtime_warning_filter = np.testing.suppress_warnings()
runtime_warning_filter.filter(RuntimeWarning)


class Tensor:
    """
    Class for automatic differentiation reverse mode
    """
    _gradients_disabled = False

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
        """
        A function that seeds in the derivative of a function with respect to itself, i.e. df/df = 1
        """
        if self.value.shape != ():
            raise Exception('Cannot call .backward() on a non-scalar')
        self.grad_value = np.array(1.0)

    def add_child(self, weight, output_tensor):
        if not Tensor._gradients_disabled:
            self.children.append((weight, output_tensor))

    @property
    def grad(self):
        """
        A function that computes the gradient value using the chain rule
        """
        if self.grad_value is None:
            self.grad_value = np.array(0.0)
            for weight, node in self.children:
                if callable(weight):
                    update = weight(node.grad)
                else:
                    update = weight * node.grad

                # fix any issues with broadcasting by summing over
                # the broadcasted dimensions
                while (update.shape != ()
                       and update.shape != self.shape
                       and update.shape + (1,) != self.shape):
                    update = update.sum(axis=-1)

                if update.shape + (1,) == self.shape:
                    update = np.expand_dims(update, axis=-1)

                # add the corrected derivative to the gradient
                self.grad_value = self.grad_value + update

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

    def reset_grad(self):
        self.grad_value = None

    @property
    def shape(self):
        return self.value.shape

    def sum(self):
        z = Tensor(np.sum(self.value))
        self.add_child(np.ones(self.shape), z)
        return z

    def mean(self):
        return self.sum() / self.value.size

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
        self.add_child(result.reshape(self.shape), z)
        return z

    def __neg__(self):
        return self.__mul__(-1)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value + other.value)
        self.add_child(1.0, z)
        other.add_child(1.0, z)
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
        self.add_child(other.value, z)
        other.add_child(self.value, z)
        return z

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value / other.value)
        self.add_child(1 / other.value, z)
        other.add_child(-self.value / other.value**2, z)
        return z

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    @runtime_warning_filter
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value ** other.value)
        self.add_child(other.value * self.value**(other.value - 1), z)
        other.add_child(self.value**other.value * np.log(self.value), z)
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
        self.add_child(mm_backward_self, z)
        other.add_child(mm_backward_other, z)
        return z

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other @ self

    def __getitem__(self, idx):
        z = Tensor(self.value[idx])
        self.add_child(1, z)
        return z

    def __setitem__(self, idx, value):
        self.value[idx] = value.value if isinstance(Tensor, value) else value
        return self


class no_grad:
    def __enter__(self):
        Tensor._gradients_disabled = True

    def __exit__(self, typ, val, traceback):
        Tensor._gradients_disabled = False


# Elementary functions
@runtime_warning_filter
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
    obj.add_child(deriv_fn(obj.value), z)
    return z


def sin(x):
    return _elementary_op(x, np.sin, np.cos, 'sin')


def cos(x):
    return _elementary_op(x, np.cos, lambda x: -np.sin(x), 'cos')


def tan(x):
    return _elementary_op(x, np.tan, lambda x: 1 / (np.cos(x) ** 2), 'tan')


def arcsin(x):
    return _elementary_op(x, np.arcsin, lambda x: 1 / np.sqrt(1 - x ** 2), 'arcsin')


def arccos(x):
    return _elementary_op(x, np.arccos, lambda x: -1 / np.sqrt(1 - x ** 2), 'arccos')


def arctan(x):
    return _elementary_op(x, np.arctan, lambda x: 1 / (1 + x ** 2), 'arctan')


def sinh(x):
    return _elementary_op(x, np.sinh, np.cosh, 'sinh')


def cosh(x):
    return _elementary_op(x, np.cosh, np.sinh, 'cosh')


def tanh(x):
    return _elementary_op(x, np.tanh, lambda x: 1 / (np.cosh(x) ** 2))


def abs(x):
    return _elementary_op(x, np.abs, np.sign, 'abs')


def exp(x):
    return _elementary_op(x, np.exp, np.exp, 'exp')


def logistic(x):
    f = lambda x: 1/(1+np.exp(-x))
    df = lambda x: np.exp(-x)/(1+np.exp(-x))**2	
    return _elementary_op(x, f, df, 'logistic')


def log(x, base=np.e):
    if base == np.e:
        return _elementary_op(x, np.log, lambda x: 1 / x)
    return log(x) / log(base)


def log2(x):
    return log(x, base=2)


def log10(x):
    return log(x, base=10)


def sqrt(x):
    return _elementary_op(x, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)), 'sqrt')


def cbrt(x):
    return _elementary_op(x, np.cbrt, lambda x: 1 / (3 * x ** (2/3)))

# Graph mode
class rev_graph:
    
    def __init__(self):
        self.connections = []
        self.formatted_connections = []
        self.unique_nodes = []
        self.operations = []
        
    def append_connect(self, value):
        from_node = value[0]
        to_node = value[1]
        
        all_from_nodes = [x[0] for x in self.connections]
        all_to_nodes = [x[1] for x in self.connections]
        
        all_from_fnodes = [x[0] for x in self.formatted_connections]
        all_to_fnodes = [x[1] for x in self.formatted_connections]
        
        # FROM NODE
        if from_node not in self.unique_nodes:
            self.unique_nodes.append(from_node)
        
        if from_node in all_from_nodes:
            index = all_from_nodes.index(from_node)
            node_name_from = all_from_fnodes[index]
            
        elif from_node in all_to_nodes:
            index = all_to_nodes.index(from_node)
            node_name_from = all_to_fnodes[index]
        
        else:
            node_name_from = 'x{}: {:.2f}'.format(len(self.unique_nodes), from_node)
            
        # TO NODE
        if to_node not in self.unique_nodes:
            self.unique_nodes.append(to_node)
        
        if to_node in all_from_nodes:
            index = all_from_nodes.index(to_node)
            node_name_to = all_from_fnodes[index]
            
        elif to_node in all_to_nodes:
            index = all_to_nodes.index(to_node)
            node_name_to = all_to_fnodes[index]
        
        else:
            node_name_to = 'x{}: {:.2f}'.format(len(self.unique_nodes), to_node)
            
        self.connections.append(value)
        self.formatted_connections.append([node_name_from, node_name_to])

    def search_path(self, var):
        current_node = var
        for child in var.children:
            next_node = child[1] 
            self.append_connect([current_node.value, next_node.value])
            self.operations.append(child[2])
            self.search_path(next_node)

    def plot_graph(self, vars):
        
        # Refresh graph
        self.connections = []
        self.formatted_connections = []
        self.unique_nodes = []
        self.operations = []

        for var in vars:
            self.search_path(var)

        edges = self.formatted_connections
        ops = self.operations
            
        labels_dict = {}
        for key, value in zip(edges, ops):
            key_formatted = (key[0], key[1])
            labels_dict[key_formatted] = value

        _, graph = plt.subplots(figsize=(10,10))
        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, iterations=500)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_dict, label_pos=0.5)
        nx.draw_networkx(G, pos, with_labels=False, node_size = 200, ax = graph)

        for k,v in pos.items():
            x,y=v
            graph.text(x+0.01,y+0.03,s=k)
        
        return graph

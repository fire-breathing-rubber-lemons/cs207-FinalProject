import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
            self.grad_value = sum(weight * var.grad for weight, var, _ in self.children)
        return self.grad_value

    def __neg__(self):
        return self.__mul__(-1)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value + other.value)
        self.children.append((1.0, z, '+'))
        other.children.append((1.0, z, '+'))
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
        self.children.append((other.value, z, '*'))
        other.children.append((self.value, z, '*'))
        return z

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value / other.value)
        self.children.append((1/other.value, z, '/'))
        other.children.append((-self.value/other.value**2, z, '/'))
        return z

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(other.value / self.value)
        self.children.append((-other.value/self.value**2, z, '/'))
        other.children.append((1/self.value, z, '/'))
        return z

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(self.value**other.value)
        self.children.append((other.value*self.value**(other.value-1), z, '^'))
        other.children.append((self.value**other.value*np.log(self.value), z, '^'))
        return z

    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        z = Tensor(other.value**self.value)
        self.children.append((other.value**self.value*np.log(other), z, '^'))
        return z

# Elementary functions
def _elementary_op(obj, fn, deriv_fn, symbol):
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
    obj.children.append((deriv_fn(obj.value), z, symbol))
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
    return _elementary_op(x, np.tanh, lambda x: 1/(np.cosh(x)**2), 'tanh')


def abs(x):
    return _elementary_op(x, np.abs, np.sign, 'abs')


def exp(x):
    return _elementary_op(x, np.exp, np.exp, 'exp')


def logistic(x):
    f = lambda x: 1/(1+np.exp(-x))
    df = lambda x: np.exp(-x)/(1+np.exp(-x))**2	
    return _elementary_op(x, f, df, 'logistic')


def log(x, base=np.e):
    return _elementary_op(x, np.log, lambda x: 1 / x, 'log') / np.log(base)


def log2(x):
    return log(x, base=2)


def log10(x):
    return log(x, base=10)


def sqrt(x):
    return _elementary_op(x, np.sqrt, lambda x: 1 / (2 * np.sqrt(x)), 'sqrt')


def cbrt(x):
    return _elementary_op(x, np.cbrt, lambda x: 1 / (3 * x**(2/3)), 'cbrt')


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
        

        
        
    
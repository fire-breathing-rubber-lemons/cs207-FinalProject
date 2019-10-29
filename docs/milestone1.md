# `pyad`: CS207 Final Project Milestone 1


## Introduction
TODO
packa## Background
TODO

## How to use `pyad`
TODO

## Software Organization
TODO

## Implementation

### Data Structures
- scalar(integers or floats)
- dictionary(can be used to initialze/assign variables)
	
	e.g. we can initiate input values by calling: `initial_condition({x:1, y:2})`
- vector
	- A row vector of length n: `pyad.vector(n, 0)`
	- A column vector of length n: `pyad.vector(n, 1)`
- matrix
    
    e.g. `pyad.matrix(m,n)` creates an mxn matrix
- numpy array

	The package could make use of numpy arrays, but our build-in vectors and matrices make it easier for the users to recognize the sizes.
- tree(store the trace for easier access and visualization)

### Classes
We'll implement `forwardmode` class, which performs the forward pass of Automatic Differentiation; and `node` class, which stores the function and derivative values in a tree diagram. 

Demo:

```python

class forwardmode():
	'''Forward pass of AD'''

	def __init__(self, f):
		'''f = function to be differentiated'''

	def initial_conditions(*args):
		'''assign input vals'''		

	def derivative_seeds(*args):
		'''seed in derivative values df/dx_i, default is 1'''

	def compute_derivatives(self, f, n, trace=True):
		'''
		compute the nth derivative of f
		f = function to be differentiated
		n = nth derivative
		trace = boolean-whether to store the value, default is True
		'''

	def get_initial_conditions(self):

	def get_derivative(self):

	def get_trace(self):


class node():
	'''Store current function values and derivative values'''
	def __init__(self, val, parent):
		'''store values in its parent node'''

	def initilize_root(self, *args):

	def add_node(self, parent, *args):

```

### External Dependency

- `numpy` for elementary functions
- `pandas` for displaying the evaluation table
- `plotly` and `Graphviz` to visualize the computational graph

### Elementary Functions
We will deal with elementary functions using `numpy`. But we'll wrap `numpy` functions with `pyad` and overwrite some of the functions potentially. 

For example, user will call `pyad.sin(x)` .


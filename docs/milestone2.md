# `pyad`: CS207 Final Project Milestone 1


## Introduction

Derivatives and derivative arrays (Jacobians, Hessians, etc.) are ubiquitous in science and engineering. Example applications of derivatives include:

- __Data science:__ optimizing the parameters of a predictive model 
- __Electrical engineering:__ simulating circuits with semiconductor elements
- __Climatology:__ modeling changes in atmospheric conditions while assimilating data from sensors around the world 
- __Finance:__ calculating the price of financial instruments

In these cases (and many others involving simulation or optimization), it's necessary to compute derivatives and preferably not by hand.

So, how can derivatives be implemented in computer software?

One way is to create a numerical approximation (typically  using finite difference methods), however, this approach is prone to truncation error and is accurate only to a limited number of significant digits. 

Another way is to use a computer algebra system to symbolically compute the derivative, however, this has a high computational cost.

The method of __Automatic Differentiation__ solves both these issues!

It is able to compute derivatives to machine precision in a fast, efficient way.



## Background

To see how automatic differentiation works, consider the following simple example:

<img src=math1.png width="150">

Recall the chain rule, which states that if we have a function h(u(t)),  then

<img src=math2.png width="100">

We can differentiate f(x) symbolically using the chain rule:

<img src=math3.png width="450">
 
and evaluate the function and its derivative at, for example, a=1:

<img src=math4.png width="350">
 
The first derivative is rather ugly, and it will keep getting uglier if we take higher order derivatives. Consider the following similar but slighly modified approach:

We can represent f(x) as a graph consisting of a series of computations:

![](m1_graph.png)

The function is broken down into a sequence of __elementary operations.__ The value of $x_i$ at each step in the sequence is referred to as the __evaluation trace__.

We can create a table of the trace and its derivative at each step in the computation. We "seed" the derivative with a value of 1 and then proceed in steps:

![](m1_table.png)

This is conceptually the same thing that we did above when we differentiated symbolically using the chain rule and we find the same solution ___but___ notice that all we need to compute the the values of f(a) and f'(a) for each trace are the values in the row above (represented by corresponding colors in the table), and the differentiation rules for some simple elementary functions. There's no need to represent or store all those intermediate algebraic expressions symbolically. This is the beauty of automatic differentiation!

Although this example is very simple for illustrative purposes, the same idea can be generalized to multivariate and vector-valued functions. The matrix of first-order partial derivatives of a vector-valued function is referred to as the __Jacobian matrix__ of the functon, J<sub>f</sub>(_x_). Rather than creating a direct mapping between an input vector _x_ and full the Jacobian of a given function J<sub>f</sub>(_x_), automatic differentiation evaluates the action of the Jacobian on a vector _u_, computing the __Jacobian-vector product__ J<i>u</i>. When we represent our function of interest, f,  as a composite of N elementary functions, f = f<sub>N</sub>◦f<sub>N-1</sub>...◦f<sub>2</sub>◦f<sub>1</sub>, we can represent the Jacobian as J = J<sub>N</sub>·J<sub>N-1</sub>...·J<sub>2</sub>·J<sub>1</sub> due to the chain rule. We seed _u_ as a unit vector and then recursively compute matrix-vector products:

<p align="center"><i>u</i><sub>1</sub> = J<sub>1</sub><i>u</i></p align="center"> 
<p align="center"><i>u</i><sub>n</sub> = J<sub>n</sub><i>u</i><sub>n-1</sub> </p align="center"> 

The unit vector chosen for _u_ detemines which variable we are deriving with respect to. Consequently, we are able to compute only the partial derivatives we need rather than computing the full Jacobian.

This method of automatic differentiation (which is referred to as __forward mode__), can be very efficiently implemented as the previous evaluation trace can be overwritten at each step in the computation. We do not need to store the full sequence of evaluation traces. This is not true of other methods of automatic differentiation, such as reverse mode.
 
## How to use `pyad`

## TODO

### Set up

**pyad** will be self contained on Github and should be installable using pip and the github ssh address, or through the more formal approaches laid out in the next section.
```python
pip install git+ssh://git@github.com/fire-breathing-rubber-lemons/cs207-FinalProject.git
```

**pyad** will follow the standard Python packaging framework. To use **pyad** it will first need to be imported using.
```python
import pyad
```

Specific classes or functions can be called  individual and will operate independently, for example:
```python
from pyad import forwardmode
```

### Interaction Theory

In general the **pyad** package will work on an object oriented, class based approach similar to sklearn or other similar modules. **pyad** will contain a number of classes which can be instantiated - these will be classes such as `forwardmode`. The user will create a blank instance of the differentiator object which will then persist. By design this will be a blank slate and there will not be a specific set of default inputs as each user may have a very different use case (differentiating a single variable or multi-variable function for instance).

Simplicity for the user will be important. The idea is not to expose each stage of the AD process to the user but allow the user to very quickly get a result and then provide useful tools (methods) for the user to interrogate the results such as return a graph of the trace.

The user should be able to specify any differentiable function in the standard format, either a defined python function or a lambda function:
```python
def test_function(x,y):
    cos_x = pyad.cos(x)
    sin_y = pyad.sin(y)
    output = cos_x * sin_y
    return output

lambda x, y: pyad.cos(x) * pyad.sin(y)
```

**pyad** should be able to deal with either of these cases and end up with the same result, hence allowing the user to build functions of arbitrary complexity and not worry about having to change the implementation method.

The core operating principle will be:

1. Define a function to be differentiated (or alternatively use a lambda function inline) `def: test_function(x) return 2 * x` or `lambda x: 2 * x`
2. Instantiate a specific class of Automatic Differentiation from the **pyad** package (for instance `my_ad = pyad.forwardmode(test_function)` or `my_ad = pyad.forwardmode(lambda x: 2 * x)` 
3. Set up parameters with which to differentiate - this will need to be the seed of each of the variables and the initial derivative conditions (usually 1). `my_ad.initial_conditions(2.5), my_ad.derivative_seeds(1)` or `my_ad.initial_conditions({x:2.5, y:5.8}), my_ad.derivative_seeds({x:1, y:1)`
4. Compute the derivative - this should be an explicit step as the computational time may be non-negligible. `my_ad.compute_derivative()` - this computation could have a number of options
    * Allow the user control of whether or not to keep the full trace table for inspection (a must for reverse mode).
    * Utilise dual number implementation.
5. There should be a number of methods for interrogating the result once this is completed, these could include:
    * Get the output value for the derivative
    * View the initial conditions
    * View the output trace table


### Example Use Case

The following is an example of how to use the pyad package to differentiate a user defined function using the forward mode. The specific code is yet to be implemented but the operating process will be as follows:

```python
import pyad

def simple_function(x, y):
    return 2 * pyad.sin(x) + pyad.cos(y + 4)

ad_forward.pyad.forwardmode(simple_function)
ad_forward.initial_conditions({x:3,y:0.5})
ad_forward.derivative_seeds({x:1,y:1})

simple_derivative = ad_forward.compute_derivative(trace=True)

# Options to get information out of the object after computation
simple_derivative = ad.foward.get_derivative()
trace_df = ad.foward.get_trace()
initial_conditions = ad.foward.get_initial_conditions()

```


## Software Organization

## TODO

#### Directory Structure
The directory structure of the `pyad` package will be as follows where `cs207-FinalProject` is the name of the Github repository which hosts the package:

```
cs207-FinalProject/
    pyad/
        __init__.py
        forward_autodiff.py
        utilities/
            __init__.py
            ... (potential non-essential tools and extensions)
        tests/
            ... (tests for the core `forward_autodiff.py` as well as the utilities)
    docs/
        - ... (documentation about how to use pyad)
```

#### Modules
The only module we plan on including at this point is the `pyad` module which will initially just contain functionality for forward autodifferentiation.

#### Testing
Our test suite will be located in the `tests/` directory of the package. To run our tests, we are tentatively planning to use both `TravisCI` as well as `CodeCov`.

#### Distribution
We are tentatively planning to release our package on PyPI under the package name `pyad-207`.

#### Packaging
We will use [`setuptools`](https://packaging.python.org/tutorials/packaging-projects/) to package our software.

## Implementation details

## TODO

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


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

## TODO


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

Although this example is very simple for illustrative purposes, the same idea can be generalized to multivariate functions and higher order derivatives.

 

## How to use `pyad`

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
from pyad import forward_diff
```

### Interaction Theory

In general the **pyad** package will work on an object oriented, class based approach similar to sklearn or other similar modules. **pyad** will contain a number of classes which can be instantiated - these will be classes such as `MultivariateDerivative`. The user will create functions to be differentiated and initialize variables. There will not be a specific set of default inputs as each user may have a very different use case (differentiating a single variable or multi-variable function for instance).

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

### Demo

```python
import pyad

x = pyad.Variable('x', 1)
y = pyad.Variable('y', 2)
z = pyad.Variable('z', 3)

>>> x**2 + 2*y + z
Tensor(8, D(z=1, x=2, y=2))

def test_fun(x, y, z):
	return pyad.exp(pyad.cos(x) + pyad.sin(y))**z

>>> test_fun(x,y,z):
Tensor(77.4, D(x=-195, y=-96.6, z=112))

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

### Current Implementation

#### Core Data Structures: 
- tensor
- dictionary
- vector(to be implemented)
	- A row vector of length n: `pyad.vector(n, 0)`
	- A column vector of length n: `pyad.vector(n, 1)`

#### Core Classes
- `MultivariateDerivative`: a class to hold derivative information.
- `Tensor`: a class that takes in variable values, and compute and store the derivatives.
- `Variables`: a sub-class of `Tensor` that assigns variable values and initializes their derivatives to be 1. 

#### Important attributes
- Input variable names, such as x,y,z
- Variable values and derivative values

#### External dependency
`numpy` for elementary functions

#### Elementray functions
Add, Subtract, Multiply, Power, Trig functions, Inverse trig functions, Exponential function, Log function, Square root function, Cubic root function.

#### To be implemented
- Variables with vector/matrix format
- Matrix operators, such as dot product
- Modify the current implementation to support vector functions of vectors and scalar functions of vectors
- Maybe: A visualization tool to show the workflow

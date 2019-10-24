# `pyad`: CS207 Final Project Milestone 1


## Introduction
TODO

## Background
TODO

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
from pyad import forwardmode
```

### Interaction Theory

In general the **pyad** package will work on an object oriented, class based approach similar to sklearn or other similar modules. **pyad** will contain a number of classes which can be instantiated - these will be classes such as a `forward-mode-differentiator` or `reverse-mode-differentiator`. The user will create a blank instance of the differentator object which will then persist. By design this will be a blank slate and there will not be a specific set of default inputs as each user may have a very different use case (differentating a single variable or multi-variable function for instance).

Simplicity for the user will be important. The idea is not to expose each stage of the AD process to the user but allow the user to very quickly get a result and then provide useful tools (methods) for the user to interrogate the results such as return a graph of the trace.

The user should be able to specify any differentiable function in the standard format, either a defined python function or a lambda function:
```python
def test_function(x,y):
    cos_x = cos(x)
    sin_y = sin(y)
    output = cos_x * sin_y
    return output

lambda x, y: cos(x) * sin(y)
```

**pyad** should be able to deal with either of these cases and end up with the same result, hence allowing the user to build functions of arbitrary complexity and not worry about having to change the implementation method.

The core operating principle will be:
1. Instantiate a specific class of Automatic Differentiation from the **pyad** package (for instance `my_ad = pyad.forwardmode()`)
2. Define a function to be differentiated (or alternatively use a lambda function inline) `my_ad.function(test_function)` or `my_ad.function(lambda x: 2 * x)`
3. Set up parameters with which to differentiate - this will need to be the seed of each of the variables and the initial derivative conditions (usually 1). `my_ad.initial_conditions(2.5), my_ad.derivative_seeds(1)` or `my_ad.initial_conditions({x = 2.5, y = 5.8}), my_ad.derivative_seeds({x = 1, y = 1)`

### Attributes & Methods

#### Required Attributes

User will need to input a number of set up parameters - listed explicitly 

Gradient default values - must be set

#### Core Methods


#### Additional Methods


#### Operator Overloading

The user should be able to specify a function of any arbitrary complexity 


### Example Use Cases




## Software Organization
TODO

## Implementation
TODO

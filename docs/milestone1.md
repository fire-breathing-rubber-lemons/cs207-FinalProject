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

ad_forward = pyad.forwardmode()

def simple_function(x, y):
    return 2 * sin(x) + cos(y + 4)

ad_forward.function(simple_function)
ad_forward.initial_conditions({x:3,y:0.5})
ad_forward.derivative_seeds({x:1,y:1})

simple_derivative = ad_forward.compute_derivative(trace=True)

# Options to get information out of the object after computation
simple_derivative = ad.foward.get_derivative()
trace_df = ad.foward.get_trace()
initial_conditions = ad.foward.get_initial_conditions()

```


## Software Organization
TODO

## Implementation
TODO

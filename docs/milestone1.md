# `pyad`: CS207 Final Project Milestone 1


## Introduction
TODO

## Background
TODO

## How to use `pyad`
TODO

## Software Organization

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

## Implementation
TODO

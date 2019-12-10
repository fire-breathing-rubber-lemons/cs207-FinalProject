from setuptools import setup, find_packages

setup(
    name='pyad',
    description='A simple package for auto-differentiation.',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/fire-breathing-rubber-lemons/cs207-FinalProject',
    author='CS207 Project Group 10',
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.16.3',
        'networkx>=2.4',
        'matplotlib>=3.1.0'
    ],
)

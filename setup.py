
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the version string from the VERSION file
with open(path.join(here, 'VERSION'), 'r') as f:
    version = f.readline().strip()

setup(
    name='MXPTools',
    version=version,
    author='Michael Laraia',
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
)

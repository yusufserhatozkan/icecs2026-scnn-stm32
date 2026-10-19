from setuptools import setup, find_namespace_packages


setup(
    name='dpsnn',
    version='0.1',
    packages=find_namespace_packages(include=['dpsnn', 'dpsnn.*']),
)

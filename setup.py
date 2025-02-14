from setuptools import setup, find_packages

setup(
    name="heatwave",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'jax',
        'matplotlib',
    ],
) 
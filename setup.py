from setuptools import setup, find_packages

setup(
    name='tfe',
    packages=find_packages(),
    install_requires = [
        'numpy >= 1.22.3',
        'matplotlib <= 3.7.1'
    ])
'''
Run the doctests that live in src/dual.py and src/optim.py, so that
the small worked examples in each module docstring are checked by CI
too.
'''
import doctest

from src import dual, optim


def test_doctests_dual():
    assert doctest.testmod(dual, verbose=False).failed == 0


def test_doctests_optim():
    assert doctest.testmod(optim, verbose=False).failed == 0

'''
Run the doctests that live in src/poly.py, so that the small worked
examples in the module docstring are checked by CI too.
'''
import doctest

from src import poly


def test_doctests_poly():
    assert doctest.testmod(poly, verbose=False).failed == 0

'''
Run the doctests that live in the src/ modules, so that the small
worked examples in each module docstring are checked by CI too.
'''
import doctest

from src import dual, poly


def _run(module):
    return doctest.testmod(module, verbose=False).failed


def test_doctests_dual():
    assert _run(dual) == 0


def test_doctests_poly():
    assert _run(poly) == 0

'''
Property tests for src.dual.

The derivative checks are all "known values" rather than finite
differences, so the tests do not depend on a step size or on
machine precision.
'''
import math

from hypothesis import given
import hypothesis.strategies as st

from src.dual import Dual, sin, cos, exp, log
from tests.strategies import floats, pos_floats


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

@given(floats, floats, floats, floats)
def test_add(a, b, c, d):
    assert Dual(a, b) + Dual(c, d) == Dual(a + c, b + d)


@given(floats, floats, floats, floats)
def test_product_rule(a, b, c, d):
    assert Dual(a, b) * Dual(c, d) == Dual(a * c, b * c + a * d)


@given(floats, floats)
def test_scalar_mul(a, b):
    assert Dual(a, b) * 2 == Dual(2 * a, 2 * b)
    assert 2 * Dual(a, b) == Dual(2 * a, 2 * b)


@given(floats, floats)
def test_neg(a, b):
    assert -Dual(a, b) == Dual(-a, -b)


@given(floats, floats)
def test_sub(a, b):
    assert Dual(a, b) - Dual(a, b) == Dual(0.0, 0.0)


@given(floats)
def test_pow_int(x):
    assert Dual(x, 1.0) ** 3 == Dual(x ** 3, 3 * x ** 2)


def test_pow_zero():
    assert Dual(5.0, 3.0) ** 0 == Dual(1.0, 0.0)


# ---------------------------------------------------------------------------
# elementary functions
# ---------------------------------------------------------------------------

@given(floats)
def test_sin_deriv(x):
    assert abs(sin(Dual(x, 1.0)).d - math.cos(x)) < 1e-12


@given(floats)
def test_cos_deriv(x):
    assert abs(cos(Dual(x, 1.0)).d + math.sin(x)) < 1e-12


@given(floats)
def test_exp_deriv(x):
    assert abs(exp(Dual(x, 1.0)).d - math.exp(x)) < 1e-12


@given(pos_floats)
def test_log_deriv(x):
    assert abs(log(Dual(x, 1.0)).d - 1 / x) < 1e-12


# ---------------------------------------------------------------------------
# non-Dual arguments
# ---------------------------------------------------------------------------

def test_elementary_functions_accept_floats():
    assert sin(0.0) == 0.0
    assert cos(0.0) == 1.0
    assert exp(0.0) == 1.0
    assert log(1.0) == 0.0


def test_not_implemented_equality():
    assert not (Dual(1.0, 0.0) == 1)
    assert Dual(1.0, 0.0) != 1

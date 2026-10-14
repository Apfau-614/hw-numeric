'''
Property tests for src.dual.

The derivative checks are all "known values" rather than finite
differences, so the tests do not depend on a step size or on
machine precision.
'''
import math

from hypothesis import assume, given
import hypothesis.strategies as st

from src.dual import Dual, sin, cos, exp, log
from tests.strategies import floats, pos_floats, dual_st


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


@given(floats, floats, floats, floats)
def test_sub_between_distinct_duals(a, b, c, d):
    assert Dual(a, b) - Dual(c, d) == Dual(a - c, b - d)


@given(floats, floats, floats)
def test_rsub(a, b, c):
    assert c - Dual(a, b) == Dual(c - a, -b)


@given(floats, floats, floats)
def test_radd(a, b, c):
    assert c + Dual(a, b) == Dual(a + c, b)


@given(floats, floats, floats, floats)
def test_truediv(a, b, c, d):
    assume(abs(c) > 1e-3)
    assert Dual(a, b) / Dual(c, d) == Dual(a / c, (b * c - a * d) / (c ** 2))


@given(floats, floats)
def test_truediv_scalar(a, b):
    assert Dual(a, b) / 2 == Dual(a / 2, b / 2)


@given(floats, floats, floats)
def test_rtruediv(a, b, c):
    assume(abs(a) > 1e-3)
    assert c / Dual(a, b) == Dual(c / a, -c * b / (a ** 2))


@given(floats)
def test_pow_int(x):
    assert Dual(x, 1.0) ** 3 == Dual(x ** 3, 3 * x ** 2)


def test_pow_zero():
    assert Dual(5.0, 3.0) ** 0 == Dual(1.0, 0.0)


def test_pow_dual_known_value():
    d = Dual(2.0, 1.0) ** Dual(3.0, 0.0)
    assert abs(d.v - 8.0) < 1e-12
    assert abs(d.d - 12.0) < 1e-12


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


def test_eq_scalar_both_directions():
    assert not (1 == Dual(1.0, 0.0))
    assert 1 != Dual(1.0, 0.0)


@given(dual_st)
def test_hash_eq_contract(d):
    assert hash(d) == hash(Dual(d.v, d.d))


# ---------------------------------------------------------------------------
# repr round trip and str notation
# ---------------------------------------------------------------------------

@given(dual_st)
def test_repr_round_trips(d):
    assert eval(repr(d)) == d


def test_str_shows_value_and_derivative():
    assert str(Dual(1, 2)) == '1 + 2ε'
    assert str(Dual(3, 1) * Dual(2, 1)) == '6 + 5ε'

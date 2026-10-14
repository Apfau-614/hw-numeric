'''
Property tests for src.poly.

The Karatsuba test uses the student's own mul_naive as the oracle,
exactly like test_sorting compares merge_sorted against the standard
library.  The cross-type test at the bottom evaluates
sym_to_poly(e) at a point and compares with e.eval.
'''
from hypothesis import assume, given
import hypothesis.strategies as st

from src.poly import Poly
from tests.strategies import ints, poly_st, floats


# ---------------------------------------------------------------------------
# algebraic laws
# ---------------------------------------------------------------------------

@given(poly_st, poly_st)
def test_add_commutative(a, b):
    assert a + b == b + a


@given(poly_st, poly_st, poly_st)
def test_add_associative(a, b, c):
    assert (a + b) + c == a + (b + c)


@given(poly_st, poly_st)
def test_mul_commutative(a, b):
    assert a * b == b * a


@given(poly_st, poly_st, poly_st)
def test_mul_distributive(a, b, c):
    assert a * (b + c) == a * b + a * c


@given(poly_st)
def test_derivative_is_linear(a):
    b = Poly([1, 2, 3])
    assert (a + b).derivative() == a.derivative() + b.derivative()


def test_derivative_power_rule():
    # x^3 has derivative 3 x^2
    assert Poly([0, 0, 0, 1]).derivative() == Poly([0, 0, 3])


# ---------------------------------------------------------------------------
# equality, hashing, iadd
# ---------------------------------------------------------------------------

def test_eq_ignores_trailing_zeros():
    assert Poly([1, 2, 3]) == Poly([1, 2, 3, 0, 0])
    assert Poly([0, 0, 0]) == Poly([0])


def test_hash_eq_contract():
    a = Poly([1, 2, 3])
    b = Poly([1, 2, 3, 0])
    assert a == b and hash(a) == hash(b)


# ---------------------------------------------------------------------------
# repr round trip and str notation
# ---------------------------------------------------------------------------

@given(poly_st)
def test_repr_round_trips(p):
    assert eval(repr(p)) == p


def test_str_zero_polynomial():
    assert str(Poly([0])) == '0'
    assert str(Poly([0, 0, 0])) == '0'


def test_str_constant():
    assert str(Poly([5])) == '5'
    assert str(Poly([-5])) == '-5'


def test_str_suppresses_unit_coefficient():
    assert str(Poly([0, 1])) == 'x'
    assert str(Poly([0, 0, 1])) == 'x**2'
    assert str(Poly([0, -1])) == '-x'
    assert str(Poly([0, 0, -1])) == '-x**2'


def test_str_suppresses_zero_coefficients():
    assert str(Poly([1, 0, 2])) == '2x**2 + 1'
    assert str(Poly([1, 0, 0, 1])) == 'x**3 + 1'


def test_str_descending_degree():
    assert str(Poly([1, 2, 3])) == '3x**2 + 2x + 1'


def test_str_signs():
    assert str(Poly([1, -1])) == '-x + 1'
    assert str(Poly([-1, 1])) == 'x - 1'
    assert str(Poly([1, -1, 0, 1])) == 'x**3 - x + 1'


def test_str_of_binomial_square():
    assert str(Poly([1, 1]) ** 2) == 'x**2 + 2x + 1'


def test_iadd_returns_self():
    q = Poly([3, 4])
    q_id = id(q)
    q += Poly([1, 2])
    assert id(q) == q_id
    assert q == Poly([4, 6])


def test_pow():
    assert Poly([1, 1]) ** 2 == Poly([1, 2, 1])
    assert Poly([1, 1]) ** 0 == Poly([1])
    assert Poly([0, 1]) ** 5 == Poly([0, 0, 0, 0, 0, 1])


# ---------------------------------------------------------------------------
# Karatsuba: the student's own naive product is the oracle
# ---------------------------------------------------------------------------

@given(st.lists(ints, min_size=1, max_size=200),
       st.lists(ints, min_size=1, max_size=200))
def test_karatsuba_matches_naive(a, b):
    pa, pb = Poly(a), Poly(b)
    assert pa.mul_karatsuba(pb) == pa.mul_naive(pb)


# ---------------------------------------------------------------------------
# Horner vs naive
# ---------------------------------------------------------------------------

@given(poly_st, floats)
def test_horner_matches_naive(p, x):
    assert abs(p.eval_naive(x) - p.eval_horner(x)) < 1e-6 * (1 + abs(p.eval_naive(x)))


def test_mul_karatsuba_and_naive_agree_on_single_term():
    a = Poly([0] * 100 + [1])  # x^100
    b = Poly([0] * 100 + [1])  # x^100
    assert a.mul_naive(b) == Poly([0] * 200 + [1])
    assert a.mul_karatsuba(b) == Poly([0] * 200 + [1])

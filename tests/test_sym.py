'''
Property tests for src.sym.

The backward pass is checked against three independent oracles:

  1. the symbolic partial function in the same module,
  2. forward-mode Dual evaluation via sym_to_dual,
  3. hand-written derivatives for small cases.

The reachable expressions are restricted to the polynomial fragment
so that the symbolic partial does not blow up.
'''
import math

from hypothesis import assume, given
import hypothesis.strategies as st

from src.sym import (Var, Const, Add, Mul, Pow, Sin, Cos, Exp, Log,
                     partial, sym_to_dual)
from tests.strategies import floats, pos_floats, sym_st


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def test_eval_simple():
    x = Var('x')
    e = x ** 2 + 3 * x
    assert e.eval({'x': 2.0}) == 10.0


def test_eval_transcendental():
    x = Var('x')
    assert Sin(x).eval({'x': math.pi / 2}) == 1.0
    assert Cos(x).eval({'x': 0.0}) == 1.0
    assert Exp(x).eval({'x': 0.0}) == 1.0
    assert Log(x).eval({'x': 1.0}) == 0.0


@given(sym_st(3), floats)
def test_variables_match_names(e, x):
    assert e.variables() <= {'x'}


def test_variables_of_two_var_expression():
    e = Var('x') + Var('y')
    assert e.variables() == {'x', 'y'}


# ---------------------------------------------------------------------------
# backward vs partial vs dual
# ---------------------------------------------------------------------------

@given(sym_st(3), pos_floats)
def test_backward_matches_partial(e, x):
    env = {'x': x}
    g1 = e.backward(env).get('x', 0.0)
    g2 = partial(e, 'x').eval(env)
    assert abs(g1 - g2) < 1e-6 * (1 + abs(g2))


@given(sym_st(3), pos_floats)
def test_backward_matches_dual_forward(e, x):
    env = {'x': x}
    g1 = e.backward(env).get('x', 0.0)
    g2 = sym_to_dual(e, 'x', env).d
    assert abs(g1 - g2) < 1e-6 * (1 + abs(g2))


def test_backward_chain_rule_known_value():
    # d/dx sin(x) * x  = cos(x) * x + sin(x)
    x = Var('x')
    e = Sin(x) * x
    g = e.backward({'x': 1.0})['x']
    expected = math.cos(1.0) * 1.0 + math.sin(1.0)
    assert abs(g - expected) < 1e-12


def test_backward_of_constant_is_empty():
    assert Const(5).backward({}) == {}


# ---------------------------------------------------------------------------
# whole-network gradient
# ---------------------------------------------------------------------------

def test_backward_two_variables():
    # f(x, y) = x * y; df/dx = y, df/dy = x
    x, y = Var('x'), Var('y')
    g = (x * y).backward({'x': 3.0, 'y': 4.0})
    assert g['x'] == 4.0
    assert g['y'] == 3.0


def test_backward_accumulates_paths():
    # f(x) = x * x + x * x; df/dx = 2 * (x + x) = 4 x
    x = Var('x')
    e = x * x + x * x
    g = e.backward({'x': 5.0})['x']
    assert abs(g - 20.0) < 1e-12


# ---------------------------------------------------------------------------
# dunder behavior
# ---------------------------------------------------------------------------

def test_reflected_operators_build_nodes():
    assert 1 + Var('x') == Add(Const(1), Var('x'))
    assert 2 * Var('x') == Mul(Const(2), Var('x'))
    assert 1 - Var('x') == Add(Const(1), Mul(Const(-1), Var('x')))
    assert -Var('x') == Mul(Const(-1), Var('x'))


def test_add_is_not_commutative_syntactically():
    # Add preserves operand order, so these are two different nodes.
    a = Var('x') + Const(1)
    b = Const(1) + Var('x')
    assert a != b
    # but they evaluate to the same number
    assert a.eval({'x': 2.0}) == b.eval({'x': 2.0})

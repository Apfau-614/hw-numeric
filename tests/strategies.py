'''
Shared hypothesis strategies.

The `poly_st` and `sym_st` strategies generate small random values
that the property tests can check against algebraic laws (commutative
rings, chain rules, etc.) without an external oracle.  Anything the
poly and sym trees can be reduced to is compared against the same
operation on the other representation, so no two implementations of
the same algorithm are trusted unconditionally.
'''
import hypothesis.strategies as st

from src.poly import Poly
from src.dual import Dual
from src.sym import Var, Const, Add, Mul, Pow


# Bounded ints and floats so that hypothesis's shrinker produces
# readable counterexamples and so arithmetic does not overflow.
ints = st.integers(min_value=-100, max_value=100)
floats = st.floats(min_value=-100.0, max_value=100.0,
                   allow_nan=False, allow_infinity=False)
pos_floats = st.floats(min_value=1e-3, max_value=100.0,
                       allow_nan=False, allow_infinity=False)


poly_st = st.lists(ints, min_size=1, max_size=10).map(Poly)
dual_st = st.builds(Dual, floats, floats)


def sym_st(depth=3):
    '''
    A recursive strategy for small symbolic expression trees in the
    single variable 'x'.  At the top level we allow Add, Mul, and
    integer Pow, but no Sin/Cos/Exp/Log, so the resulting expression
    is a polynomial of moderate degree and compares cleanly against
    sym_to_poly.
    '''
    if depth <= 0:
        return st.one_of(st.just(Var('x')), ints.map(Const))
    leaf = st.one_of(st.just(Var('x')), ints.map(Const))
    children = sym_st(depth - 1)
    powers = st.tuples(children, st.integers(min_value=1, max_value=3))
    return st.one_of(
        leaf,
        st.builds(Add, children, children),
        st.builds(Mul, children, children),
        powers.map(lambda p: Pow(p[0], p[1])),
    )

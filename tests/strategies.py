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


# Bounded ints and floats so that hypothesis's shrinker produces
# readable counterexamples and so arithmetic does not overflow.
ints = st.integers(min_value=-100, max_value=100)
floats = st.floats(min_value=-100.0, max_value=100.0,
                   allow_nan=False, allow_infinity=False)
pos_floats = st.floats(min_value=1e-3, max_value=100.0,
                       allow_nan=False, allow_infinity=False)


poly_st = st.lists(ints, min_size=1, max_size=10).map(Poly)
dual_st = st.builds(Dual, floats, floats)

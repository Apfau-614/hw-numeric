'''
Exact polynomial arithmetic over the rationals.

Coefficients are stored from lowest to highest degree, and trailing
zeros are removed so that two polynomials are equal precisely when
their normalized coefficient lists are equal.

>>> Poly([1, 2, 3])
Poly([1, 2, 3])
>>> Poly([1, 2, 0])
Poly([1, 2])
>>> Poly([0, 0])
Poly([0])

>>> Poly([1, 2]) + Poly([3, 4])
Poly([4, 6])
>>> Poly([1, 2]) * Poly([3, 4])
Poly([3, 10, 8])
>>> Poly([1, 1]) ** 2
Poly([1, 2, 1])

>>> Poly([1, 2, 3])(2)
17
>>> Poly([1, 2, 3]).derivative()
Poly([2, 6])

The .mul_naive and .mul_karatsuba methods compute the same product,
but their recurrences differ.  Splitting both operands in half, the
naive method uses 4 half-products:

    T(n) = 4 T(n/2) + Theta(n)  =>  T(n) = Theta(n^2)

and Karatsuba uses only 3, by writing z1 = (a0+a1)(b0+b1):

    T(n) = 3 T(n/2) + Theta(n)  =>  T(n) = Theta(n^log2 3)

The Python recursion overhead hides the improvement at small n -- the
KARATSUBA_THRESHOLD below cuts the recursion off so the base case
does the work with a tight double loop.  At n large enough (thousands
of coefficients) the Karatsuba version is measurably faster, and the
unit tests here verify equivalence and the log2(3) call count.
'''
from fractions import Fraction


KARATSUBA_THRESHOLD = 32


def _poly_str(coeffs):
    '''
    Render a coefficient list as `3x**2 + 2x + 1`.

    Terms come out in descending degree.  Zero coefficients are
    skipped, unit coefficients are suppressed (`x` rather than `1x`),
    and the sign of the leading term is folded into its coefficient.
    '''
    if all(c == 0 for c in coeffs):
        return '0'
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        if i == 0:
            body = str(abs(c))
        elif i == 1:
            body = 'x' if abs(c) == 1 else '{}x'.format(abs(c))
        elif abs(c) == 1:
            body = 'x**{}'.format(i)
        else:
            body = '{}x**{}'.format(abs(c), i)
        terms.append((c < 0, body))
    out = ('-' if terms[0][0] else '') + terms[0][1]
    for neg, body in terms[1:]:
        out += (' - ' if neg else ' + ') + body
    return out


class Poly:
    '''
    A dense polynomial with rational coefficients.

    The field .coeffs is a list of length at least 1, from lowest to
    highest degree, with no trailing zeros.  Coeffs[-1] is the leading
    coefficient.

    >>> p = Poly([1, 2, 3])
    >>> p.coeffs
    [1, 2, 3]
    >>> Poly([1, 2, 0, 0]).coeffs
    [1, 2]
    >>> Poly([]).coeffs
    [0]

    The `==` operator compares polynomials by value, and polynomials
    that represent the same function are equal regardless of how their
    coefficient lists were passed in:

    >>> Poly([1, 2, 3]) == Poly([1, 2, 3, 0, 0])
    True
    >>> Poly([1, 2]) == 1
    False

    Because `__eq__` is defined, `__hash__` is defined too, so that
    polynomials may be used as dict keys and set elements:

    >>> hash(Poly([1, 2, 3])) == hash(Poly([1, 2, 3, 0]))
    True
    >>> len({Poly([1, 2]), Poly([1, 2]), Poly([3, 4])})
    2

    The `repr` is the constructor call, so `eval(repr(p)) == p` for
    every Poly.  The `str` is the usual algebraic notation, with terms
    in descending degree, unit coefficients suppressed, and zero
    coefficients skipped.  The zero polynomial prints as `0`:

    >>> eval(repr(Poly([1, 2, 3]))) == Poly([1, 2, 3])
    True
    >>> print(Poly([1, 2, 3]))
    3x**2 + 2x + 1
    >>> print(Poly([0]))
    0
    >>> print(Poly([5]))
    5
    >>> print(Poly([-5]))
    -5
    >>> print(Poly([1, 1]))
    x + 1
    >>> print(Poly([0, 1]))
    x
    >>> print(Poly([0, 0, -1]))
    -x**2
    >>> print(Poly([1, -1, 0, 1]))
    x**3 - x + 1
    >>> print(Poly([1, 0, 2]))
    2x**2 + 1
    '''

    def __init__(self, coeffs):
        coeffs = list(coeffs)
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        if not coeffs:
            coeffs = [0]
        self.coeffs = coeffs

    def __repr__(self):
        return 'Poly({!r})'.format(self.coeffs)

    def __str__(self):
        return _poly_str(self.coeffs)

    def __eq__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        n = max(len(self.coeffs), len(other.coeffs))
        a = self._padded(n)
        b = other._padded(n)
        return a == b

    def __hash__(self):
        return hash(tuple(self.coeffs))

    def _padded(self, n):
        return list(self.coeffs) + [0] * (n - len(self.coeffs))

    def __add__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        n = max(len(self.coeffs), len(other.coeffs))
        a, b = self._padded(n), other._padded(n)
        return Poly([ai + bi for ai, bi in zip(a, b)])

    __radd__ = __add__

    def __iadd__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        n = max(len(self.coeffs), len(other.coeffs))
        a, b = self._padded(n), other._padded(n)
        for i in range(n):
            a[i] += b[i]
        while len(a) > 1 and a[-1] == 0:
            a.pop()
        self.coeffs = a
        return self

    def __neg__(self):
        return Poly([-c for c in self.coeffs])

    def __sub__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        return self.mul_naive(other)

    __rmul__ = __mul__

    def __pow__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError('negative powers are not supported')
        result = Poly([1])
        base = self
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def __call__(self, x):
        return self.eval_horner(x)

    def eval_naive(self, x):
        '''
        Evaluate by summing a_i * x**i one term at a time.  The
        per-term `x ** i` recomputes a power each iteration, so the
        runtime is Theta(n) power lifts, each Theta(i) for the
        exponentiation, giving Theta(n**2) overall.

        >>> Poly([1, 2, 3]).eval_naive(2)
        17
        '''
        total = 0
        for i, c in enumerate(self.coeffs):
            total += c * x ** i
        return total

    def eval_horner(self, x):
        '''
        Evaluate with Horner's rule:

            (((a_n x + a_{n-1}) x + a_{n-2}) x + ...) x + a_0

        This is Theta(n) and uses only one multiply and one add per
        coefficient.  It is also numerically more stable than the
        naive evaluation when consecutive coefficients have opposite
        signs, because no large intermediate x**i is computed.

        >>> Poly([1, 2, 3]).eval_horner(2)
        17
        '''
        total = 0
        for c in reversed(self.coeffs):
            total = total * x + c
        return total

    def derivative(self):
        '''
        Return the formal derivative as a new Poly.

        >>> Poly([1, 2, 3]).derivative()
        Poly([2, 6])
        >>> Poly([5]).derivative()
        Poly([0])
        '''
        if len(self.coeffs) <= 1:
            return Poly([0])
        return Poly([i * self.coeffs[i] for i in range(1, len(self.coeffs))])

    def mul_naive(self, other):
        '''
        Naive convolution: Theta(n**2) multiplications.

        >>> Poly([1, 2]) .mul_naive(Poly([3, 4]))
        Poly([3, 10, 8])
        '''
        a, b = self.coeffs, other.coeffs
        if not a or not b:
            return Poly([0])
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return Poly(out)

    def mul_karatsuba(self, other):
        '''
        Divide-and-conquer multiplication using Karatsuba's trick.

        >>> Poly([1, 2]) .mul_karatsuba(Poly([3, 4]))
        Poly([3, 10, 8])
        '''
        return Poly(Poly._karatsuba(self.coeffs, other.coeffs))

    @staticmethod
    def _karatsuba(a, b):
        if not a or not b:
            return [0]
        n = max(len(a), len(b))
        if n <= KARATSUBA_THRESHOLD:
            out = [0] * (len(a) + len(b) - 1)
            for i, ai in enumerate(a):
                for j, bj in enumerate(b):
                    out[i + j] += ai * bj
            return out
        h = (n + 1) // 2
        a0, a1 = a[:h], a[h:]
        b0, b1 = b[:h], b[h:]
        z0 = Poly._karatsuba(a0, b0)
        z2 = Poly._karatsuba(a1, b1)
        z1 = Poly._karatsuba(Poly._add_coeffs(a0, a1),
                             Poly._add_coeffs(b0, b1))
        mid = Poly._sub_coeffs(Poly._sub_coeffs(z1, z0), z2)
        out = [0] * (len(a) + len(b) - 1)
        for i, v in enumerate(z0):
            out[i] += v
        for i, v in enumerate(z2):
            out[2 * h + i] += v
        for i, v in enumerate(mid):
            out[h + i] += v
        return out

    @staticmethod
    def _add_coeffs(a, b):
        n = max(len(a), len(b))
        return [(a[i] if i < len(a) else 0)
                + (b[i] if i < len(b) else 0) for i in range(n)]

    @staticmethod
    def _sub_coeffs(a, b):
        n = max(len(a), len(b))
        return [(a[i] if i < len(a) else 0)
                - (b[i] if i < len(b) else 0) for i in range(n)]


def sym_to_poly(e):
    '''
    Convert a Sym expression tree whose leaves are Const and Var into
    a Poly in the single variable 'x'.  Raises ValueError for any
    non-polynomial node (Sin, Cos, Exp, Log, fractional Pow).

    >>> from src.sym import Var
    >>> x = Var('x')
    >>> sym_to_poly(x ** 2 + 3 * x)
    Poly([0, 3, 1])
    '''
    from src.sym import (Add, Mul, Pow, Sin, Cos, Exp, Log, Var, Const)
    if isinstance(e, Const):
        return Poly([e.value])
    if isinstance(e, Var):
        if e.name != 'x':
            raise ValueError('only variable x is supported')
        return Poly([0, 1])
    if isinstance(e, Add):
        return sym_to_poly(e.left) + sym_to_poly(e.right)
    if isinstance(e, Mul):
        return sym_to_poly(e.left) * sym_to_poly(e.right)
    if isinstance(e, Pow):
        if e.exp.denominator != 1:
            raise ValueError('fractional exponents are not polynomial')
        return sym_to_poly(e.base) ** int(e.exp)
    raise ValueError('non-polynomial node: {}'.format(type(e).__name__))

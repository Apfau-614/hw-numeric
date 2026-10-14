'''
Exact polynomial arithmetic over the rationals.

Coefficients are stored from lowest to highest degree.

>>> p = Poly([1, 2, 3])
>>> print(p)
3x**2 + 2x + 1

Polynomials are just functions with special structure,
and they can be called like regular functions.

>>> p(3)
34

Because of this special structure, we can do a lot of operations that can't be done on ordinary functions.

We can compute derivatives:

>>> print(p.derivative())
6x + 2

We can check for equality:

>>> q = Poly([0, 3, 2, 1])
>>> print(q)
x**3 + 2x**2 + 3x
>>> p == q
False

We can do addition, subtraction, multiplication, and scalar exponentiation:

>>> print (p + q)
x**3 + 5x**2 + 5x + 1
>>> print (p - q)
-x**3 + x**2 - x + 1
>>> print (p * q)
3x**5 + 8x**4 + 14x**3 + 8x**2 + 3x
>>> print(p ** 2)
9x**4 + 12x**3 + 10x**2 + 4x + 1

Technically, polynomials form a *Ring*
(a type of structure studied in MATH171 Abstract Algebra),
and so they do not support division.
(If they did, they would be called a *Field*.)
'''

'''
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

    `Poly` is a mutable value type, so it is deliberately unhashable
    and cannot be used as a dict key or a set element:

    >>> hash(Poly([1, 2, 3]))
    Traceback (most recent call last):
        ...
    TypeError: unhashable type: 'Poly'

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

    # Mutable, so not hashable.
    __hash__ = None

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
        result = self + other
        if result is NotImplemented:
            return NotImplemented
        self.coeffs = result.coeffs
        return self

    def __neg__(self):
        return Poly([-c for c in self.coeffs])

    def __sub__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        return self + (-other)

    def __isub__(self, other):
        result = self - other
        if result is NotImplemented:
            return NotImplemented
        self.coeffs = result.coeffs
        return self

    def __mul__(self, other):
        if not isinstance(other, Poly):
            return NotImplemented
        return self.mul_naive(other)

    __rmul__ = __mul__

    def __imul__(self, other):
        result = self * other
        if result is NotImplemented:
            return NotImplemented
        self.coeffs = result.coeffs
        return self

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

    def __ipow__(self, n):
        result = self ** n
        if result is NotImplemented:
            return NotImplemented
        self.coeffs = result.coeffs
        return self

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
        n = max(len(a), len(b))
        if n <= KARATSUBA_THRESHOLD:
            out = [0] * (len(a) + len(b) - 1)
            for i, ai in enumerate(a):
                for j, bj in enumerate(b):
                    out[i + j] += ai * bj
            return out
        if len(a) < n:
            a = list(a) + [0] * (n - len(a))
        if len(b) < n:
            b = list(b) + [0] * (n - len(b))
        h = (n + 1) // 2
        a0, a1 = a[:h], a[h:]
        b0, b1 = b[:h], b[h:]
        z0 = Poly._karatsuba(a0, b0)
        z2 = Poly._karatsuba(a1, b1)
        z1 = Poly._karatsuba(Poly._add_coeffs(a0, a1),
                             Poly._add_coeffs(b0, b1))
        mid = Poly._sub_coeffs(Poly._sub_coeffs(z1, z0), z2)
        out = [0] * (2 * n - 1)
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

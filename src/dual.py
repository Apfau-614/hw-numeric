'''
Forward-mode automatic differentiation with dual numbers.

A dual number is a pair (v, d) representing the truncated Taylor
expansion v + d*epsilon with epsilon**2 == 0.  Arithmetic on dual
numbers computes both the value and first derivative of a function in
a single forward pass.

>>> Dual(3, 1) + Dual(2, 1)
Dual(v=5, d=2)
>>> Dual(3, 1) * Dual(2, 1)
Dual(v=6, d=5)
>>> Dual(3, 1) ** 2
Dual(v=9, d=6)

The module-level functions sin, cos, exp, and log work on both floats
and dual numbers, so the same code can be differentiated without
rewriting it:

>>> sin(Dual(0.0, 1.0))
Dual(v=0.0, d=1.0)
>>> exp(Dual(0.0, 1.0))
Dual(v=1.0, d=1.0)
>>> abs(cos(Dual(0.0, 1.0)).d - 0.0) < 1e-12
True

Nested duals differentiate twice.  That is how newton_step in
src/optim.py gets its second derivative.
'''
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Dual:
    '''
    The dual number v + d*epsilon with epsilon**2 == 0.

    The fields are .v (the value) and .d (the derivative with respect
    to whatever variable the caller has seeded with derivative 1).

    Every arithmetic method returns a new Dual and never mutates self.

    >>> Dual(1, 2) == Dual(1, 2)
    True
    >>> Dual(1, 2) == Dual(1, 3)
    False
    >>> Dual(1, 2) == 1
    False
    >>> hash(Dual(1, 2)) == hash(Dual(1, 2))
    True
    '''
    v: object
    d: object

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v + other.v, self.d + other.d)
        return Dual(self.v + other, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v - other.v, self.d - other.d)
        return Dual(self.v - other, self.d)

    def __rsub__(self, other):
        return Dual(other - self.v, -self.d)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v * other.v,
                        self.d * other.v + self.v * other.d)
        return Dual(self.v * other, self.d * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v / other.v,
                        (self.d * other.v - self.v * other.d)
                        / (other.v ** 2))
        return Dual(self.v / other, self.d / other)

    def __rtruediv__(self, other):
        return Dual(other / self.v, -other * self.d / (self.v ** 2))

    def __pow__(self, n):
        if isinstance(n, Dual):
            v = self.v ** n.v
            return Dual(v, v * (n.d * math.log(self.v)
                                 + n.v * self.d / self.v))
        return Dual(self.v ** n, n * self.v ** (n - 1) * self.d)


def sin(x):
    '''
    sin, differentiable if x is a Dual.

    >>> sin(0.0)
    0.0
    '''
    if isinstance(x, Dual):
        return Dual(math.sin(x.v), math.cos(x.v) * x.d)
    return math.sin(x)


def cos(x):
    '''
    cos, differentiable if x is a Dual.

    >>> cos(0.0)
    1.0
    '''
    if isinstance(x, Dual):
        return Dual(math.cos(x.v), -math.sin(x.v) * x.d)
    return math.cos(x)


def exp(x):
    '''
    exp, differentiable if x is a Dual.

    >>> exp(0.0)
    1.0
    '''
    if isinstance(x, Dual):
        e = math.exp(x.v)
        return Dual(e, e * x.d)
    return math.exp(x)


def log(x):
    '''
    log, differentiable if x is a Dual.

    >>> log(1.0)
    0.0
    '''
    if isinstance(x, Dual):
        return Dual(math.log(x.v), x.d / x.v)
    return math.log(x)

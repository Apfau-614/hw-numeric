'''
A symbolic expression tree with reverse-mode automatic differentiation.

Every leaf is a Const or a Var, and every internal node is one of the
arithmetic classes Add, Mul, Pow, Sin, Cos, Exp, Log.  Because the
arithmetic dunders are defined on the Sym base class, ordinary Python
expression syntax builds the tree:

>>> x = Var('x')
>>> e = x ** 2 + 3 * x
>>> e
Add(left=Pow(base=Var(name='x'), exp=Fraction(2, 1)), right=Mul(left=Const(value=3), right=Var(name='x')))

.eval(env) evaluates the tree at numeric values:

>>> e.eval({'x': 2.0})
10.0

.backward(env) returns the gradient of the root with respect to every
variable, computed in a single reverse pass:

>>> e.backward({'x': 2.0})['x']
7.0

The forward-mode equivalent is Dual in src/dual.py.  Forward-mode
computes one directional derivative per pass; reverse-mode computes
the full gradient in one pass.  For a function with n inputs and one
output, reverse-mode is n times cheaper.  That is why the deep
learning libraries build a symbolic graph and run backward on it.
'''
import math
from dataclasses import dataclass
from fractions import Fraction

from src.dual import Dual, sin as _sin, cos as _cos, exp as _exp, log as _log


def _wrap(x):
    return x if isinstance(x, Sym) else Const(x)


class Sym:
    '''
    Base class for every node in the expression tree.

    Subclasses must define .eval(env) and ._backward(grads, up, env).
    The dunders below build new nodes so that expressions like
    `x ** 2 + 3 * x` have their usual meaning:

    >>> Var('x') + 1
    Add(left=Var(name='x'), right=Const(value=1))
    >>> 1 + Var('x')
    Add(left=Const(value=1), right=Var(name='x'))
    >>> Var('x') * 2
    Mul(left=Var(name='x'), right=Const(value=2))
    '''

    def __add__(self, other):
        return Add(self, _wrap(other))

    def __radd__(self, other):
        return Add(_wrap(other), self)

    def __sub__(self, other):
        return Add(self, Mul(Const(-1), _wrap(other)))

    def __rsub__(self, other):
        return Add(_wrap(other), Mul(Const(-1), self))

    def __neg__(self):
        return Mul(Const(-1), self)

    def __mul__(self, other):
        return Mul(self, _wrap(other))

    def __rmul__(self, other):
        return Mul(_wrap(other), self)

    def __pow__(self, exp):
        if isinstance(exp, int):
            exp = Fraction(exp)
        return Pow(self, exp)

    def eval(self, env):
        raise NotImplementedError

    def _backward(self, grads, up, env):
        raise NotImplementedError

    def backward(self, env):
        '''
        Return a dict from variable name to the partial derivative of
        the root with respect to that variable, evaluated at env.

        The traversal is reverse mode: each node distributes its own
        upstream gradient (the "adjoint") to its children.

        >>> (Var('x') ** 2).backward({'x': 3.0})['x']
        6.0
        >>> (Var('x') ** 2 + Var('y')).backward({'x': 3.0, 'y': 0.0})
        {'y': 1.0, 'x': 6.0}
        '''
        grads = {}
        self._backward(grads, 1.0, env)
        return grads

    def variables(self):
        '''
        Return the set of variable names appearing anywhere in the tree.

        >>> (Var('x') + Var('y')).variables() == {'x', 'y'}
        True
        '''
        return set()


@dataclass(frozen=True)
class Const(Sym):
    '''A constant leaf.  Subclasses Sym.'''
    value: object

    def eval(self, env):
        return self.value

    def _backward(self, grads, up, env):
        pass


@dataclass(frozen=True)
class Var(Sym):
    '''A named variable leaf.  Subclasses Sym.'''
    name: str

    def eval(self, env):
        return env[self.name]

    def _backward(self, grads, up, env):
        grads[self.name] = grads.get(self.name, 0.0) + up

    def variables(self):
        return {self.name}


def _variables2(a, b):
    return a.variables() | b.variables()


@dataclass(frozen=True)
class Add(Sym):
    '''Addition node.  Subclasses Sym.'''
    left: Sym
    right: Sym

    def eval(self, env):
        return self.left.eval(env) + self.right.eval(env)

    def _backward(self, grads, up, env):
        self.left._backward(grads, up, env)
        self.right._backward(grads, up, env)

    def variables(self):
        return _variables2(self.left, self.right)


@dataclass(frozen=True)
class Mul(Sym):
    '''Multiplication node.  Subclasses Sym.'''
    left: Sym
    right: Sym

    def eval(self, env):
        return self.left.eval(env) * self.right.eval(env)

    def _backward(self, grads, up, env):
        lv = self.left.eval(env)
        rv = self.right.eval(env)
        self.left._backward(grads, up * rv, env)
        self.right._backward(grads, up * lv, env)

    def variables(self):
        return _variables2(self.left, self.right)


@dataclass(frozen=True)
class Pow(Sym):
    '''
    Power node.  The exponent is fixed at construction time and is a
    Fraction (so that `x ** 2` and `x ** Fraction(1, 2)` both work).
    '''
    base: Sym
    exp: Fraction

    def __post_init__(self):
        if isinstance(self.exp, int):
            object.__setattr__(self, 'exp', Fraction(self.exp))

    def eval(self, env):
        return self.base.eval(env) ** self.exp

    def _backward(self, grads, up, env):
        bv = self.base.eval(env)
        e = float(self.exp)
        self.base._backward(grads, up * e * bv ** (e - 1), env)

    def variables(self):
        return self.base.variables()


@dataclass(frozen=True)
class Sin(Sym):
    '''sin node.'''
    arg: Sym

    def eval(self, env):
        return _sin(self.arg.eval(env))

    def _backward(self, grads, up, env):
        self.arg._backward(grads, up * _cos(self.arg.eval(env)), env)

    def variables(self):
        return self.arg.variables()


@dataclass(frozen=True)
class Cos(Sym):
    '''cos node.'''
    arg: Sym

    def eval(self, env):
        return _cos(self.arg.eval(env))

    def _backward(self, grads, up, env):
        self.arg._backward(grads, -up * _sin(self.arg.eval(env)), env)

    def variables(self):
        return self.arg.variables()


@dataclass(frozen=True)
class Exp(Sym):
    '''exp node.'''
    arg: Sym

    def eval(self, env):
        return _exp(self.arg.eval(env))

    def _backward(self, grads, up, env):
        self.arg._backward(grads, up * self.eval(env), env)

    def variables(self):
        return self.arg.variables()


@dataclass(frozen=True)
class Log(Sym):
    '''log node.'''
    arg: Sym

    def eval(self, env):
        return _log(self.arg.eval(env))

    def _backward(self, grads, up, env):
        self.arg._backward(grads, up / self.arg.eval(env), env)

    def variables(self):
        return self.arg.variables()


def partial(e, name):
    '''
    Return a new Sym that is the symbolic partial derivative of e with
    respect to the variable `name`.  This is a small forward-mode
    symbolically-differentiating routine, so it can be used to
    cross-check the reverse-mode .backward method.

    >>> partial(Var('x') ** 2, 'x').eval({'x': 3.0})
    6.0
    >>> partial(Var('x') ** 2 + Var('y'), 'x').eval({'x': 1.0, 'y': 1.0})
    2.0
    >>> partial(Var('x') ** 2, 'y').eval({'x': 1.0})
    0
    '''
    if isinstance(e, Const):
        return Const(0)
    if isinstance(e, Var):
        return Const(1) if e.name == name else Const(0)
    if isinstance(e, Add):
        return Add(partial(e.left, name), partial(e.right, name))
    if isinstance(e, Mul):
        return Add(Mul(partial(e.left, name), e.right),
                   Mul(e.left, partial(e.right, name)))
    if isinstance(e, Pow):
        n = e.exp
        inner = Mul(Mul(Const(n), Pow(e.base, n - 1)),
                    partial(e.base, name))
        return inner
    if isinstance(e, Sin):
        return Mul(Cos(e.arg), partial(e.arg, name))
    if isinstance(e, Cos):
        return Mul(Mul(Const(-1), Sin(e.arg)), partial(e.arg, name))
    if isinstance(e, Exp):
        return Mul(e, partial(e.arg, name))
    if isinstance(e, Log):
        return Mul(Pow(e.arg, Fraction(-1)), partial(e.arg, name))
    raise TypeError('unknown node type: {}'.format(type(e).__name__))


def sym_to_dual(e, name, env):
    '''
    Evaluate e with Var(name) seeded as a Dual and every other Var
    read from env.  This is forward-mode differentiation, and is the
    counterpart to .backward's reverse mode.

    >>> x = Var('x')
    >>> sym_to_dual(x ** 2, 'x', {'x': 3.0})
    Dual(v=9.0, d=6.0)
    '''
    extended = {}
    for k, v in env.items():
        extended[k] = Dual(v, 0.0)
    extended[name] = Dual(env[name], 1.0)
    return e.eval(extended)

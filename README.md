# Numeric
[![](https://github.com/USER/numeric/workflows/tests/badge.svg)](https://github.com/USER/numeric/actions?query=workflow%3Atests)

A miniature numeric programming library built around three related
types, each implemented with the same Python dunder protocol.

<img src=img/meme.jpg width=300px />

## Learning objectives

1. Implement the Python data model (`__add__`, `__radd__`, `__mul__`,
   `__rmul__`, `__pow__`, `__eq__`, `__hash__`, `__repr__`).
1. Use `NotImplemented` and the `__eq__` / `__hash__` contract correctly.
1. See the master theorem applied to polynomial multiplication, and see
   why Karatsuba's recurrence `T(n) = 3 T(n/2) + O(n)` beats
   the naive `T(n) = 4 T(n/2) + O(n)`.
1. Understand forward-mode vs reverse-mode automatic differentiation and
   the runtime implications of each.
1. See why second-order methods reach the minimum of a quadratic in one
   step, and how stochastic gradient descent differs from the full-batch
   version.

## The three types

* `Poly` in `src/poly.py` is a dense polynomial over the rationals,
  stored as a list of coefficients from lowest to highest degree.
* `Dual` in `src/dual.py` is a dual number `v + d*epsilon` with
  `epsilon**2 == 0`.  It computes the value and first derivative of a
  function of one variable in a single forward pass.
* `Sym` in `src/sym.py` is a symbolic expression tree with `Var` leaves
  and arithmetic node classes.  Its `.eval(env)` evaluates at numeric
  values and its `.backward(env)` computes gradients by reverse-mode
  accumulation.

`Poly` and `Dual` are both *specializations* of `Sym`: converting a
symbol tree to `Poly` erases all non-polynomial nodes, and converting
to `Dual` truncates every non-linear node to first order.

## Optimization

`src/optim.py` implements `gd_step`, `sgd_step`, and `newton_step` on
top of `Dual`.  Each of these methods is a single line once the `Dual`
class is in place, which is the main practical payoff of automatic
differentiation.

## Tasks

Implement all functions and methods in `src/`.
All doctests and pytest cases must pass.

## Submission

Submit the URL of your forked repository to canvas.

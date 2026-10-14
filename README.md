# Numeric
[![](https://github.com/USER/numeric/workflows/tests/badge.svg)](https://github.com/USER/numeric/actions?query=workflow%3Atests)

A miniature numeric programming library built around two related
types, each implemented with the same Python dunder protocol.

<img src=img/meme.jpg width=300px />

## Learning objectives

1. Implement the Python data model (`__add__`, `__radd__`, `__mul__`,
   `__rmul__`, `__pow__`, `__eq__`, `__hash__`, `__repr__`).
1. Use `NotImplemented` and the `__eq__` / `__hash__` contract correctly.
1. See the master theorem applied to polynomial multiplication, and see
   why Karatsuba's recurrence `T(n) = 3 T(n/2) + O(n)` beats
   the naive `T(n) = 4 T(n/2) + O(n)`.

## The two types

* `Poly` in `src/poly.py` is a dense polynomial over the rationals,
  stored as a list of coefficients from lowest to highest degree.
* `Dual` in `src/dual.py` is a dual number `v + d*epsilon` with
  `epsilon**2 == 0`.  It computes the value and first derivative of a
  function of one variable in a single forward pass.

## Tasks

Implement all functions and methods in `src/`.
All doctests and pytest cases must pass.

## Submission

Submit the URL of your forked repository to canvas.

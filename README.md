# Numeric
[![tests-poly](https://github.com/mikeizbicki/numeric/actions/workflows/tests-poly.yml/badge.svg)](https://github.com/mikeizbicki/numeric/actions/workflows/tests-poly.yml)
[![tests-dual](https://github.com/mikeizbicki/numeric/actions/workflows/tests-dual.yml/badge.svg)](https://github.com/mikeizbicki/numeric/actions/workflows/tests-dual.yml)

A miniature numeric programming library built around two related types, each implemented with the same Python dunder protocol.
Types like these form the basis for machine learning libraries like pytorch.

<img src=img/meme.jpeg width=300px />

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

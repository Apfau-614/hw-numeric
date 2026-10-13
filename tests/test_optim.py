'''
Tests for src.optim.

The Newton tests confirm the character of the second-order method:
on a quadratic, one step reaches the minimum exactly.  The SGD test
checks that a batch of size len(xs) gives the same step as full-batch
gradient descent, and that repeated small batches make progress
toward the empirical minimizer.
'''
import random

from src.optim import gd_step, sgd_step, newton_step


# ---------------------------------------------------------------------------
# deterministic GD
# ---------------------------------------------------------------------------

def test_gd_single_sample_known_value():
    f = lambda w, x: (w - x) ** 2
    # loss is (w - 5)^2 at w=0, so grad is -10, step of 0.1 gives +1.0
    assert abs(gd_step(f, 0.0, 0.1, [5.0]) - 1.0) < 1e-12


def test_gd_full_batch_known_value():
    f = lambda w, x: (w - x) ** 2
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    # mean gradient at w=0 is 2*(0 - 3) = -6, step of 0.1 gives +0.6
    assert abs(gd_step(f, 0.0, 0.1, xs) - 0.6) < 1e-12


def test_gd_converges():
    f = lambda w, x: (w - x) ** 2
    w = 0.0
    for _ in range(500):
        w = gd_step(f, w, 0.1, [5.0])
    assert abs(w - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

def test_sgd_full_batch_matches_gd():
    f = lambda w, x: (w - x) ** 2
    xs = [1.0, 2.0, 3.0, 4.0]
    rng = random.Random(0)
    assert abs(sgd_step(f, 1.5, 0.1, xs, len(xs), rng)
               - gd_step(f, 1.5, 0.1, xs)) < 1e-12


def test_sgd_converges_roughly():
    f = lambda w, x: (w - x) ** 2
    xs = [float(i) for i in range(1, 11)]
    rng = random.Random(42)
    w = 0.0
    for _ in range(5000):
        w = sgd_step(f, w, 0.01, xs, 3, rng)
    assert abs(w - 5.5) < 0.2


# ---------------------------------------------------------------------------
# Newton
# ---------------------------------------------------------------------------

def test_newton_quadratic_one_step():
    f = lambda w: w ** 2
    assert abs(newton_step(f, 5.0)) < 1e-12


def test_newton_shifted_quadratic_one_step():
    f = lambda w: (w - 3.0) ** 2
    assert abs(newton_step(f, 10.0) - 3.0) < 1e-12


def test_newton_quartic_needs_multiple_steps():
    # (w - 2)^4 is convex but not quadratic, so Newton does not land
    # on the minimum in one step.
    f = lambda w: (w - 2.0) ** 4
    w = 10.0
    for _ in range(200):
        w = newton_step(f, w)
    assert abs(w - 2.0) < 1e-6

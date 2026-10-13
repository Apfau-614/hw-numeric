'''
First-order and second-order optimization on top of Dual.

Each function takes a scalar objective and returns the next iterate.
The derivative of the objective is computed by seeding a Dual with
derivative 1 and reading off the .d field, which is the practical
payoff of forward-mode automatic differentiation: the same code that
evaluates the loss also computes its gradient, with no hand-written
derivative rules.

The runtime of SGD, GD, and 2GD depends on three parameters:

    n    number of data points
    d    number of parameters
    T    number of iterations

    GD per step:                          O(n) loss evaluations
    SGD per step with batch size b:       O(b) loss evaluations
    2GD per step (using the Hessian):     O(d^2) extra evaluations,
                                          but quadratically fewer
                                          iterations on well-
                                          conditioned problems

Whether SGD, GD, or 2GD is best in practice is a statistical question
as much as a computational one.  Full-batch GD has lower variance per
step but does not scale to large datasets; SGD scales but adds noise;
Newton's method has superb convergence per step but each step is much
more expensive and requires the Hessian to be positive definite.
'''
from src.dual import Dual


def gd_step(f, w, lr, xs):
    '''
    One step of full-batch gradient descent.

    f(w, x) is the loss on sample x.  xs is the full data set.

    >>> f = lambda w, x: (w - x) ** 2
    >>> w = gd_step(f, 0.0, 0.1, [1.0, 2.0, 3.0, 4.0, 5.0])
    >>> abs(w - 0.6) < 1e-12
    True
    '''
    grad = 0.0
    for x in xs:
        grad += f(Dual(w, 1.0), x).d
    return w - lr * grad / len(xs)


def sgd_step(f, w, lr, xs, batch_size, rng):
    '''
    One step of stochastic gradient descent on a random minibatch.

    rng is a random.Random instance so that runs are reproducible.
    Samples are drawn without replacement, so a batch of size
    len(xs) gives exactly the full-batch gradient.

    >>> import random
    >>> f = lambda w, x: (w - x) ** 2
    >>> sgd_step(f, 0.0, 0.1, [1.0, 2.0, 3.0, 4.0, 5.0],
    ...          5, random.Random(0))
    0.6
    '''
    batch = rng.sample(xs, batch_size)
    grad = 0.0
    for x in batch:
        grad += f(Dual(w, 1.0), x).d
    return w - lr * grad / batch_size


def newton_step(f, w, lr=1.0):
    '''
    One Newton step on a scalar objective: w - lr * f'(w) / f''(w).

    The second derivative comes from a nested Dual.  Writing

        D2 = Dual(Dual(w, 1.0), Dual(1.0, 0.0))

    and evaluating f(D2) gives a Dual whose .v.d is f'(w) and whose
    .d.d is f''(w).  On a quadratic this reaches the minimum in one
    step, because f is exactly equal to its second-order Taylor
    expansion.

    >>> f = lambda w: w ** 2
    >>> abs(newton_step(f, 5.0)) < 1e-12
    True
    >>> g = lambda w: (w - 3.0) ** 2
    >>> abs(newton_step(g, 10.0) - 3.0) < 1e-12
    True
    '''
    d2 = Dual(Dual(w, 1.0), Dual(1.0, 0.0))
    result = f(d2)
    return w - lr * result.v.d / result.d.d

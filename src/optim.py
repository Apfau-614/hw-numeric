'''
Optimization algorithms and motivating models for Dual autodiff.

`src/dual.py` shows that a dual number carries a value and a single
directional derivative.  This module turns that primitive into the
thing machine learning actually needs: the *gradient* of a scalar loss
with respect to a list of parameters, and gradient descent on top.

>>> grad(lambda ws: ws[0] * ws[1] * ws[2], [1.0, 2.0, 3.0])
[6.0, 3.0, 2.0]

Forward mode gives one directional derivative per pass, so a gradient
with respect to p parameters costs p forward passes.  That is fine for
the toy models below, and is exactly why production libraries use
reverse mode instead.

The models are finite compositions of +, *, and tanh, so Dual handles
them unchanged.  Differentiating them by hand is error prone, and
their training trajectories have no pleasant closed form.

>>> xs, ys = [1.0, 1.0], [0.5, 0.5]
>>> loss = lambda ws: mse(rnn(xs, ws), ys)
>>> loss(descent(loss, [0.5], lr=0.5, steps=50)) < loss([0.5])
True
'''
from src.dual import Dual, exp, tanh


def grad(f, params):
    '''
    Gradient of the scalar f(params) with respect to each parameter.

    Parameter i is seeded with d=1 and the rest are held constant, so
    every entry of the result is a partial derivative and the whole
    gradient costs one forward pass per parameter.

    >>> grad(lambda ws: ws[0] * ws[1] * ws[2], [1.0, 2.0, 3.0])
    [6.0, 3.0, 2.0]
    >>> grad(lambda ws: ws[0] ** 2, [3.0])
    [6.0]
    >>> grad(lambda ws: sum(w ** 2 for w in ws), [1.0, 2.0, 3.0])
    [2.0, 4.0, 6.0]

    This is the bridge from Dual to the models below.  It costs one
    pass per weight, which is another reason the models stay small:

    >>> xs, ys = [1.0, 1.0], [0.5, 0.5]
    >>> loss = lambda ws: mse(rnn(xs, ws), ys)
    >>> [round(g, 4) for g in grad(loss, [0.5])]
    [0.0649]

    A central finite difference agrees, which is a good sanity check
    for any new function you feed to Dual:

    >>> g = grad(loss, [0.5])[0]
    >>> abs(g - (loss([0.5 + 1e-6]) - loss([0.5 - 1e-6])) / 2e-6) < 1e-8
    True
    '''
    return [
        f([Dual(p, 1.0) if j == i else p for j, p in enumerate(params)]).d
        for i in range(len(params))
    ]


def descent(f, params, lr=0.1, steps=100):
    '''
    Gradient descent on the scalar f, starting from params.

    Returns the updated parameters; call f on them to read the loss.
    This is plain SGD: fixed rate, no momentum, no line search.

    On a quadratic of curvature 2, a rate of 0.5 lands on the minimum
    in a single step:

    >>> descent(lambda ws: sum(w ** 2 for w in ws), [1.0, 2.0], lr=0.5, steps=1)
    [0.0, 0.0]

    A smaller rate still converges, just more slowly:

    >>> round(descent(lambda ws: (ws[0] - 3.0) ** 2, [0.0], lr=0.1, steps=100)[0], 4)
    3.0

    And it does train a real model.  Starting at w=0.5 the loss is
    about 0.1068; the optimum pins the hidden state at the target:

    >>> xs, ys = [1.0, 1.0], [0.5, 0.5]
    >>> loss = lambda ws: mse(rnn(xs, ws), ys)
    >>> loss(descent(loss, [0.5], lr=0.5, steps=50)) < loss([0.5])
    True
    '''
    for _ in range(steps):
        params = [p - lr * g for p, g in zip(params, grad(f, params))]
    return params


def momentum(f, params, lr=0.1, beta=0.9, steps=100):
    '''
    Gradient descent with heavy ball momentum.

    A velocity accumulates past gradients, so the iterates overshoot
    and ring around the minimum instead of creeping up to it.  On a
    quadratic the error still contracts geometrically:

    >>> round(momentum(lambda ws: (ws[0] - 3.0) ** 2, [0.0], lr=0.1, beta=0.9, steps=500)[0], 4)
    3.0

    beta=0 throws the velocity away and recovers plain descent, step
    for step and bit for bit:

    >>> a = momentum(lambda ws: (ws[0] - 3.0) ** 2, [0.0], lr=0.1, beta=0.0, steps=10)
    >>> b = descent(lambda ws: (ws[0] - 3.0) ** 2, [0.0], lr=0.1, steps=10)
    >>> a == b
    True
    '''
    v = [0.0] * len(params)
    for _ in range(steps):
        v = [beta * vi - lr * gi for vi, gi in zip(v, grad(f, params))]
        params = [p + vi for p, vi in zip(params, v)]
    return params


def mse(ys, ts):
    '''
    Mean squared error between predictions ys and targets ts.

    >>> mse([1.0, 2.0], [0.0, 0.0])
    2.5
    >>> mse([1.0, 2.0], [1.0, 2.0])
    0.0
    >>> mse([Dual(0.5, 1.0)], [0.0])
    Dual(v=0.25, d=1.0)
    '''
    return sum((y - t) ** 2 for y, t in zip(ys, ts)) / len(ys)


def rnn_layer(xs, w, h=0.0):
    '''
    One recurrent layer over the sequence xs.

    h_t = tanh(w * h_{t-1} + x_t) with h_0 = 0, returning every h_t.
    The recursion walks the sequence, not the parameters, so Dual
    flows through unchanged and the derivative arrives at the end of
    the unrolled tape.

    >>> [round(v, 3) for v in rnn_layer([1.0, 1.0], 0.5)]
    [0.762, 0.881]

    The first step is tanh(x_0), which does not involve w at all, so
    its derivative is exactly zero:

    >>> [round(v.d, 3) for v in rnn_layer([1.0, 1.0], Dual(0.5, 1.0))]
    [0.0, 0.17]
    '''
    if not xs:
        return []
    h = tanh(w * h + xs[0])
    return [h] + rnn_layer(xs[1:], w, h)


def rnn(xs, ws):
    '''
    Stack recurrent layers: layer d consumes the previous output.

    len(ws) == 1 is the single layer RNN from the `src/dual.py` module
    docstring, and it reproduces that file's 0.1703 exactly:

    >>> [round(v, 3) for v in rnn([1.0, 1.0], [0.5])]
    [0.762, 0.881]
    >>> [round(v.d, 3) for v in rnn([1.0, 1.0], [Dual(0.5, 1.0)])]
    [0.0, 0.17]

    More weights in the list makes it deep.  Layer two consumes layer
    one's output, so the values change even though w is the same:

    >>> [round(v, 3) for v in rnn([1.0, 1.0], [0.5, 0.5])]
    [0.363, 0.787]
    '''
    for w in ws:
        xs = rnn_layer(xs, w)
    return xs


def softmax(xs):
    '''
    Softmax over a list of values.

    >>> [round(v, 3) for v in softmax([0.0, 0.0])]
    [0.5, 0.5]
    >>> [round(v, 3) for v in softmax([0.0, 1.0])]
    [0.269, 0.731]

    Softmax is a ratio, so division has to be differentiable, and
    indeed the Jacobian diagonal s_i(1 - s_i) shows up in d:

    >>> [round(v.d, 4) for v in softmax([Dual(0.0, 1.0), Dual(0.0, 0.0)])]
    [0.25, -0.25]

    Real implementations subtract max(xs) before exponentiating to
    avoid overflow on large scores.  That shift is constant with
    respect to the weights, so it changes no derivative.
    '''
    es = [exp(x) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def attention(xs, wq, wk, wv):
    '''
    Single head attention with scalar features.

    Position i forms a query wq * x_i, compares it with every key
    wk * x_j, and returns the softmax weighted average of the values
    wv * x_j.  Real transformers scale the scores by 1 / sqrt(d); with
    d == 1 here that factor is 1 and is omitted.

    Identical inputs give identical scores, so the weights are uniform
    and each output is just the mean of the values:

    >>> [round(v, 3) for v in attention([1.0, 1.0], 0.3, 0.4, 0.5)]
    [0.5, 0.5]

    Unequal inputs put most of the weight on the better match, which
    drags each output toward the larger value:

    >>> [round(v, 3) for v in attention([1.0, 2.0], 0.3, 0.4, 0.5)]
    [0.765, 0.78]
    '''
    q = [wq * x for x in xs]
    k = [wk * x for x in xs]
    v = [wv * x for x in xs]
    out = []
    for i in range(len(xs)):
        weights = softmax([q[i] * k[j] for j in range(len(xs))])
        out.append(sum(weights[j] * v[j] for j in range(len(xs))))
    return out


def attn_block(xs, ws):
    '''
    One transformer block: attention, then a tiny feed forward network,
    added back to the input.

    ws is the 5-tuple (wq, wk, wv, w1, w2).  The residual keeps the
    gradient from vanishing as blocks are stacked, the same reason
    real transformers use them.

    >>> [round(v, 3) for v in attn_block([1.0, 1.0], [0.3, 0.4, 0.5, 1.0, 0.7])]
    [1.323, 1.323]

    With identical inputs every position gets the same attention, so
    the queries and keys only ever enter through a constant softmax.
    Their gradients are therefore exactly zero:

    >>> xs, ys = [1.0, 1.0], [0.5, 0.5]
    >>> ws = [0.3, 0.4, 0.5, 1.0, 0.7]
    >>> loss = lambda ws: mse(transformer(xs, ws), ys)
    >>> [round(g, 3) for g in grad(loss, ws)]
    [0.0, 0.0, 0.907, 0.453, 0.761]
    '''
    wq, wk, wv, w1, w2 = ws
    h = attention(xs, wq, wk, wv)
    return [x + w2 * tanh(w1 * hi) for x, hi in zip(xs, h)]


def transformer(xs, ws):
    '''
    Stack of transformer blocks.  ws holds 5 weights per block, so
    len(ws) == 5 is one block and len(ws) == 10 is two.

    >>> [round(v, 3) for v in transformer([1.0, 1.0], [0.3, 0.4, 0.5, 1.0, 0.7])]
    [1.323, 1.323]

    Depth compounds the block, just like the RNN stacks layers:

    >>> ws = [0.3, 0.4, 0.5, 1.0, 0.7] * 2
    >>> [round(v, 3) for v in transformer([1.0, 1.0], ws)]
    [1.729, 1.729]

    And it trains.  Note that with equal inputs the whole network
    collapses to 1 + w2 * tanh(w1 * wv), which is why only three of
    the five gradients were nonzero above:

    >>> xs, ys = [1.0, 1.0], [0.5, 0.5]
    >>> loss = lambda ws: mse(transformer(xs, ws), ys)
    >>> loss(descent(loss, [0.3, 0.4, 0.5, 1.0, 0.7], lr=0.01, steps=50)) < loss([0.3, 0.4, 0.5, 1.0, 0.7])
    True
    '''
    for i in range(0, len(ws), 5):
        xs = attn_block(xs, ws[i:i + 5])
    return xs

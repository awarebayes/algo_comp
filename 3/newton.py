import numpy as np


def y_newt(i, k, xs, ys) -> float:
    if i + k >= len(xs):
        return 0
    elif k == 0:
        return ys[i]
    else:
        return (y_newt(i + 1, k - 1, xs, ys) - y_newt(i, k - 1, xs, ys)) / (
            xs[i + k] - xs[i]
        )


def topk_closest(x, xs, k):
    closest = np.abs(xs - x).argsort()
    return np.sort(closest[:k])


def newton(x, xs, ys, power):
    topk = topk_closest(x, xs, k=power)
    xs = xs[topk]
    ys = ys[topk]

    lp = y_newt(0, 0, xs, ys)
    for k in range(1, power):
        lp += y_newt(0, k, xs, ys) * np.prod(x - xs[:k])
    return lp

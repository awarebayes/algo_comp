import numpy as np
import streamlit as st


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


def newton_3d(data, nx, ny, nz, xp, yp, zp):
    xs = np.array(data[0])
    ys = np.array(data[1])
    zs = np.array(data[2])
    tensor = np.array(data[3])

    z_values = np.zeros(len(zs))
    y_values = np.zeros(len(ys))

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            x_values = tensor[i][j]
            y_values[j] = newton(xp, xs, x_values, nx)
        z_values[i] = newton(yp, ys, y_values, ny)
    ans = newton(zp, zs, z_values, nz)
    return ans


def newton_3d_alt(axis, func, powers, point):
    xs, ys, zs = axis
    x_pow, y_pow, z_pow = powers
    x, y, z = point

    x_grid, y_grid, z_grid = np.meshgrid(*axis, indexing="ij")
    tensor = func(x_grid, y_grid, z_grid)
    tensor = np.swapaxes(tensor, 0, 2)

    z_values = np.zeros(tensor.shape[0])
    y_values = np.zeros(tensor.shape[1])

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            x_values = tensor[i][j]
            y_values[j] = newton(x, xs, x_values, x_pow)
        z_values[i] = newton(y, ys, y_values, y_pow)
    ans = newton(z, zs, z_values, z_pow)
    return ans

import numpy as np
import streamlit as st


def f(x, y, z):
    return x**2 + y**2 + z


def closest_range_in_sorted(sorted_array, k, needle):
    distance = np.abs(sorted_array - needle)
    closest = np.argsort(distance)
    topk_indices = closest[:k]

    min_bound, max_bound = topk_indices.min(), topk_indices.max()
    return min_bound, max_bound + 1


def div_diff(z, node):
    """
    Расчет разделенных разниц для полинома Ньютона
    """
    for i in range(node):
        pol = []
        for j in range(node - i):
            buf = (z[i + 1][j] - z[i + 1][j + 1]) / (z[0][j] - z[0][j + i + 1])
            pol.append(buf)
        z.append(pol)
    return z


def polinom_n(z, node, arg):
    """
    Расчет значение функции от заданного аргумента.
    Полином Ньютона.
    """
    pol = div_diff(z, node)
    y = 0
    buf = 1
    for i in range(node + 1):
        y += buf * pol[i + 1][0]
        buf *= arg - pol[0][i]
    return y


def multi_var_newton(axes, powers, point, tensor):
    n_vars = len(axes)
    # ищем границы для каждой оси
    bounds = []
    for axis, power, component in zip(axes, powers, point):
        bound = closest_range_in_sorted(axis, power, component)
        bounds.append(bound)

    # обрезаем оси
    for i in range(n_vars):
        axes[i] = axes[i][slice(*bounds[i])]

    # обрезаем тензор
    tensor = tensor[tuple(slice(*bound) for bound in bounds)]

    for dim in range(tensor.dim):
        pass


def main():
    x_axis, y_axis, z_axis = np.arange(0, 5, 1), np.arange(0, 5, 1), np.arange(0, 5, 1)
    z, y, x = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    f_tensor = f(x, y, z)
    multi_var_newton([x_axis, y_axis, z_axis], [2, 2, 2], [1, 2, 3], f_tensor)


main()

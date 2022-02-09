# Aboba

import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache
import pandas as pd
from numba import jit




df = pd.DataFrame({
    'x': np.arange(0, 1.05, 0.15),
    'y': np.array([1.000000, 0.838771, 0.655336, 0.450447, 0.225336, -0.018310, -0.278390, -0.552430]),
    'y`': np.array([-1.000000, -1.14944, -1.29552, -1.43497, -1.56464, -1.68164, -1.78333, -1.86742]),
})

func_dict = dict({k: v for k, v in zip(df['x'], df['y'])})

def y(*zs):
    if len(zs) == 1:
        point = zs[0]
        return func_dict[point]
    else:
        *starting_points, end = zs
        start, *ending_points = zs
        return (y(*starting_points) - y(*ending_points)) / (start - end)


def newton(x, n, xs):
    """
    Params:
        x: точка в которой ищем значение
        n: степерь полинома
        xs: подмножество аргуметов x, значения которых мы имеем
    Returns:
        Значение y в искомой точке
    """
    result = y(xs[0])
    for k in range(1, n):
        product = 1
        xs_from_0_to_k = xs[0:k]
        for x_i in xs_from_0_to_k:
            product *= (x - x_i)
        product *= y(*xs_from_0_to_k)
        result += product
    return result


x_0 = 0.525
comparison = {"x": [], "n": [], "newton":[]}

for n in range(4, 5):
    result = newton(x_0, n, df['x'])
    comparison['x'].append(x_0)
    comparison['n'].append(n)
    comparison['newton'].append(result)

comparison = pd.DataFrame(comparison)
comparison
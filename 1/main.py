import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numba import njit
import streamlit as st


def get_data(type):

    if type == "По задаче":
        x = np.arange(0, 1.05, 0.15)
        df = pd.DataFrame({
            'x': x,
            'y': [1.000000, 0.838771, 0.655336, 0.450447, 0.225336, -0.018310, -0.278390, -0.552430],
            'y`': [-1.000000, -1.14944, -1.29552, -1.43497, -1.56464, -1.68164, -1.78333, -1.86742],
            })
    elif type == "Sin(4*x)":
        x = np.linspace(0, 1.05, 20)
        df = pd.DataFrame({
            'x': x,
            'y': np.sin(4*x),
            'y`': 4 * np.cos(4 * x)
        })
    return df


def y(zs, xs, ys, cache):
    """
    Вычисление функции y.
    Params:
        zs: агрументы y(z1, z2, ..., zn)
        xs: аргументы исходной функции
        ys: значения исзодной функции
    Returns:
        значение функции y(z1, z2, ..., zn)
    """
    if len(zs) == 1:
        point = zs[0]
        index = xs.index(point)
        return ys[index]
    else:
        if zs in cache:
            return cache[zs]
        starting_points, end = zs[:-1], zs[-1]
        start, ending_points = zs[0], zs[1:]
        res = (y(starting_points, xs, ys, cache) - y(ending_points, xs, ys, cache)) / (start - end)
        cache[zs] = res
        return res


def newton(x, n, xs, ys):
    """
    Params:
        x: точка в которой ищем значение
        n: степерь полинома
        xs: подмножество аргуметов x, значения которых мы имеем
    Returns:
        Значение y в искомой точке
    """
    x_0 = (xs[0], )
    cache = {}
    result = y(x_0, xs, ys, cache)
    for k in range(1, n+1):
        product = 1
        for x_i in xs[0:k]:
            product *= (x - x_i)

        y_res = y(tuple(xs[0:k+1]), xs, ys, cache)
        product *= y_res
        result += product
    return result, cache


def pad(arr, target_len, value):
    n = len(arr)
    arr = list(arr)
    for i in range(target_len - n):
        arr.append(value)
    return arr


LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

@njit
def is_derivative(zs):
    for z in zs:
        if abs(z - zs[0]) > 0.001:
            return False
    return True

@njit
def look_up_derivative(zs, xs, y_derivatives):
    m = len(zs)
    fact = fast_factorial(m-1)
    index = xs.index(zs[0])
    derivative = y_derivatives[m-1][index]
    return derivative / fact


def y_with_lookup(zs, xs, y_derivatives, cache):
    """
    Вычисление функции y.
    Params:
        zs: агрументы y(z1, z2, ..., zn)
        xs: аргументы исходной функции
        ys: значения исзодной функции
    Returns:
        значение функции y(z1, z2, ..., zn)
    """
    if zs in cache:
        return cache[zs]

    derivative_is_in_table = len(zs) <= y_derivatives.shape[0]
    if is_derivative(zs) and derivative_is_in_table:
        res = look_up_derivative(zs, xs, y_derivatives)
    else:
        starting_points, end = zs[:-1], zs[-1]
        start, ending_points = zs[0], zs[1:]
        res = (y_with_lookup(starting_points, xs, y_derivatives, cache) -
                y_with_lookup(ending_points, xs, y_derivatives, cache)) \
                / (start - end)
    cache[zs] = res
    return res


def hermite(x, xs, n, y_derivatives):
    x_0 = [xs[0]]
    cache = dict()
    result = y_with_lookup(tuple(x_0), xs, y_derivatives, cache)

    for k in range(1, n+1):
        product = 1
        for x_i in xs[0:k]:
            product *= (x - x_i)
        product *= y_with_lookup(tuple(xs[0:k+1]), xs, y_derivatives, cache)
        result += product
    return result, cache


def main():

    st.title("Алгоритмы Ньютона и Эрмита")
    st.subheader("Постановка задачи, таблица")

    data_type = st.radio("Данные", ["По задаче", "Sin(4*x)"])

    df = get_data(data_type)
    st.dataframe(df)

    st.line_chart(df.set_index('x'))

    x_0 = st.slider("Значение x: ", min_value=float(df['x'].min()), max_value=float(df['x'].max()), value=float(df['x'].mean()))
    st.subheader('Метод Ньютона')
    newton_comparison = {"x": [], "n": [], "newton": []}

    cache = None
    for n in range(1, 7):
        result, cache = newton(x_0, n, df['x'].to_list(), df['y'].to_list())
        newton_comparison['x'].append(x_0)
        newton_comparison['n'].append(n)
        newton_comparison['newton'].append(result)

    newton_comparison = pd.DataFrame(newton_comparison)
    st.dataframe(newton_comparison)
    fig, ax = plt.subplots()

    ax.plot(df['x'], df['y'], label='y')
    ax.scatter(newton_comparison['x'], newton_comparison['newton'], c=newton_comparison['n'])
    ax.legend()
    st.pyplot(fig)


    args = list(cache.keys())
    max_args = len(max(args, key=len))
    args = [pad(arg, max_args, None) for arg in args]
    args = {f'z_{i}': [j[i] for j in args] for i in range(max_args)}

    st.write("Значения функции y")
    st.dataframe(pd.DataFrame({**args, "y(z1, ..., zn)": cache.values()}))

    st.subheader("Алгоритм Эрмита")

    y_derivatives = np.array([
        df['y'].values.repeat(2),
        df['y`'].values.repeat(2)
    ])

    x = df['x'].values.repeat(2)

    hermite_comparison = {"x": [], "n": [], "hermite": []}
    cache = None
    for n in range(1, 7):
        result, cache = hermite(x_0, list(x), n, y_derivatives)
        hermite_comparison['x'].append(x_0)
        hermite_comparison['n'].append(n)
        hermite_comparison['hermite'].append(result)

    hermite_comparison = pd.DataFrame(hermite_comparison)
    st.dataframe(hermite_comparison)

    fig, ax = plt.subplots()
    ax.scatter(hermite_comparison['x'], hermite_comparison['hermite'], c=hermite_comparison['n'])
    ax.plot(df['x'], df['y'], label='y')
    ax.legend()
    st.pyplot(fig)

    args = list(cache.keys())
    is_der = [is_derivative(arg) and len(arg) <= 2 for arg in args]
    max_args = len(max(args, key=len))
    args = [pad(arg, max_args, None) for arg in args]
    args = {f'z_{i}': [j[i] for j in args] for i in range(max_args)}

    st.write("Значения функции y")
    st.dataframe(pd.DataFrame({**args, "was looked up": is_der, "y(z1, ..., zn)": cache.values()}))

    st.subheader("Сравнение алгоритмов")

    merged = pd.merge(newton_comparison, hermite_comparison)
    st.dataframe(merged)



main()
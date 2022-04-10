import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

from random import uniform


def x_scalar_mul(k, m, xValues, rValues):
    N = len(xValues)

    res = 0
    for i in range(N):
        res += rValues[i] * (xValues[i] ** (k + m))

    return res


def xy_scalar_mul(k, xValues, yValues, rValues):
    N = len(xValues)

    res = 0
    for i in range(N):
        res += rValues[i] * yValues[i] * (xValues[i] ** k)

    return res


def onedim_mean_square(n, xValues, yValues, rValues):
    # La = M
    # Создаём матрицу L
    L = []
    for k in range(n + 1):
        row = []
        for m in range(n + 1):
            row.append(x_scalar_mul(k, m, xValues, rValues))
        L.append(row.copy())

    # Создаём матрицу M
    M = []
    for k in range(n + 1):
        M.append(xy_scalar_mul(k, xValues, yValues, rValues))

    # Решаем СЛАУ => получаем коэффициенты полинома
    res = lg.solve(L, M)

    return res


def polynomial(x, coef):
    res = 0
    for i in range(len(coef)):
        res += coef[i] * (x ** i)

    return res


def show_points(xValues, yValues, rValues):
    N = len(xValues)
    for i in range(N):
        I = rValues[i] / max(rValues)
        plt.plot(xValues[i], yValues[i], marker="o", markeredgecolor="grey",
                 markerfacecolor=(1, 0, 0, I))


def plot_graph(xValues, yValues, rValues, nval=(1, 2, 5)):
    colours = ['red', 'green', 'blue', 'yellow', 'orange']
    for n in nval:
        approximated_y = []
        coef = onedim_mean_square(n, xValues, yValues, rValues)
        x = min(xValues)
        step = 0.01
        buildx = []
        while x <= max(xValues):
            approximated_y.append(polynomial(x, coef))
            x += step
            buildx.append(x)

        plt.plot(buildx, approximated_y, color=colours[n % len(colours)],
                 label="n = {:1d}".format(n))


def build_approx_graph(xValues, yValues, rValues, nval=(1, 2, 5)):
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(7)

    plt.subplot(1, 2, 1)
    plot_graph(xValues, yValues, rValues, nval)
    show_points(xValues, yValues, rValues)
    plt.legend()
    plt.title("Веса равны")
    plt.grid()

    plt.subplot(1, 2, 2)
    plot_graph(xValues, yValues, rValues, nval)
    show_points(xValues, yValues, rValues)
    plt.legend()
    plt.title("Веса не равны")
    plt.grid()

    plt.show()


def func(x):
    return (x) ** 5


def create_random_data():
    xlb = -10
    xrb = 10

    ylb = -10
    yrb = 10

    xValues = []
    yValues = []
    rValues = []

    n = 20
    i = 0
    while i < n:
        x = uniform(xlb, xrb)
        y = func(x)

        if x not in xValues:
            xValues.append(x)
            yValues.append(y)
            rValues.append(1)
            i += 1

    return xValues, yValues, rValues


# xValues, yValues, rValues = create_random_data()

# tmp6
# xValues = [0, 2.5, 5.0, 7.5, 10.0]
# yValues = [1, -0.048, -0.178, 0.266, -0.246]
# rValues = [100, 7, 5, 0.1, 100]

# tmp7
xValues = [0, 0.524, 1.047, 1.571, 2.094, 2.618, 3.142]
yValues = [-0.648, -0.576, -0.074, 1.290, 3.946, 8.324, 14.855]
rValues = [1.300, 1.051, 8.261, 51.754, 10.285, 1.378, 2.233]
build_approx_graph(xValues, yValues, rValues)
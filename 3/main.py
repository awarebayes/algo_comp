import pandas as pd

from modules.newton import *
from modules.spline import *
from modules.io import *

import random
import numpy as np
import streamlit as st


def func(x):
    return np.sin(np.cos(x)) / (x + 50)


def generate_x_y(xstart, xend, xnum):
    x = list(np.linspace(xstart, xend, num=xnum))
    y = []
    for num in x:
        y.append(func(num))

    return x, y


# x, y = generate_x_y(-20, 20, 20)

x = [0, 1, 2]
y = [-1, 0, 7]

# Сплайн по трём точкам будет состоять из двух функций, поэтому всего неизвестных коэффициентов будет 8.

# 1, 2) ai = yi
# 3, 4) ai + bi(xi+1 - xi) + ci(xi+1 - xi)2 + d(xi+1 - xi)3 = yi+1, f1’(x2) = f2’(x2)
# 5) b1 + 2c1(x2 - x1) + 3d1(x2 - x1)2 = b2, f1’’(x2) = f2’’(x2)
# 6) c1 + 3d1(x2 - x1)2 = c2
# 7) f1’’(x1) = 0 => c1 = 0
# 8) f2’’(x3) = 0 => c2 + 3d2(x3 - x2) = 0


def main():

    i = st.number_input(
        "Введите аргумент x: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f"
    )

    if i < min(x) or i > max(x):
        st.error("Out of bounds!")
    elif len(x) != len(y):
        st.error("Broken data!")
    elif len(x) < 4:
        st.error("Not enought points for newton!")
        table = add_values_to_table(x, y)
        start = newton_interpolate_second_derivative(x[0], table, 2)
        end = newton_interpolate_second_derivative(x[-1], table, 2)
        st.write("Spline, boundaries = start, end", calculate_spline(i, x, y, 0, 12))
    else:
        table = add_values_to_table(x, y)

        start = newton_interpolate_second_derivative(x[0], table)
        end = newton_interpolate_second_derivative(x[-1], table)

        comparison = pd.DataFrame(
            {
                "name": [
                    "Newton, n=3",
                    "Spline, boundaries = 0:",
                    "Spline, right boundary = 0:",
                    "Spline, both boundaries are derivatives: ",
                ],
                "result": [
                    newton_interpolate(i, 3, table),
                    calculate_spline(i, x, y, 0, 0),
                    calculate_spline(i, x, y, start, 0),
                    calculate_spline(i, x, y, start, end),
                ],
                "start": [None, 0, start, start],
                "end": [None, 0, 0, end],
            }
        )

        st.dataframe(comparison)
        print_graphic(x, y, start, end, table)


if __name__ == "__main__":
    main()

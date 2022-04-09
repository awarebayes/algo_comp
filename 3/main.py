from spline import *
from readData import *
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from newton import newton


def main():

    table_type = st.selectbox("Тип функции", ["из таблицы", "своя"])
    point_table = None
    if table_type == "из таблицы":
        pointTable = read_table("./data.txt")
    else:
        xs = st.number_input(
            "Введите x_start: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f"
        )
        xe = st.number_input(
            "Введите x_end: ", -100.0, 100.0, 10.0, step=1.0, format="%.5f"
        )
        xst = int(st.number_input("Введите кол во узлов x: ", 1, 100, 10))
        expression = st.text_input("Введите выражение, f(x)=", "x**3")
        func = eval(f"lambda x: {expression}")
        pointTable = generateTable(xs, xe, xst, func)

    n = 3
    x = st.number_input("Введите x: ", -100.0, 100.0, 10.0, step=1.0, format="%.5f")

    start1 = 0
    end1 = 0
    start2 = 0
    end2 = 0
    start3 = 0
    end3 = 0

    xs = pointTable[:, 0]
    ys = pointTable[:, 1]

    yValues = [list(), list(), list(), list()]
    if n < len(pointTable):
        st.write("Ньютон 3-й степени:         ", newton(x, xs, ys, n + 1))
        end2 = newton(pointTable[-1][0], xs, ys, n + 1)
        start3 = newton(pointTable[0][0], xs, ys, n + 1)
        end3 = newton(pointTable[-1][0], xs, ys, n + 1)
    else:
        st.error(
            "Ньютон 3-й степени нельзя посчитать стпени",
            n,
            ", так как точек всего",
            len(pointTable),
        )

    st.write("Cплайн 0 and 0:             ", spline(x, xs, ys, (start1, end1)))
    st.write("Cплайн 0 and P''(xn):       ", spline(x, xs, ys, (start2, end2)))
    st.write("Cплайн P''(x0) and P''(xn): ", spline(x, xs, ys, (start3, end3)))

    xValues = np.linspace(pointTable[0][0], pointTable[-1][0], 100)

    if n < len(pointTable):
        for xi in xValues:
            yValues[3].append(newton(xi, pointTable[:, 0], pointTable[:, 1], n + 1))

    ranges = [(start1, end1), (start2, end2), (start3, end3)]
    for xi in xValues:
        yValues[0].append(spline(xi, xs, ys, (start1, end1)))

    for xi in xValues:
        yValues[1].append(spline(xi, xs, ys, (start2, end2)))

    for xi in xValues:
        yValues[2].append(spline(xi, xs, ys, (start3, end3)))

    fig, ax = plt.subplots()
    ax.plot(xValues, yValues[0], "-", color="r", label=f"spline ({ranges[0][0]: .2f}, {ranges[0][1]: .2f})")
    ax.plot(xValues, yValues[1], "-", color="b", label=f"spline ({ranges[1][0]: .2f}, {ranges[1][1]: .2f})")
    ax.plot(xValues, yValues[2], "-", color="g", label=f"spline ({ranges[2][0]: .2f}, {ranges[2][1]: .2f})")

    if n < len(pointTable):
        ax.plot(xValues, yValues[3], ":", color="black", label="newton")

    ax.legend()
    ax.set_title("Interpolation with splines")
    st.pyplot(fig)


if __name__ == "__main__":
    main()

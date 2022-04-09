import numpy as np

from io_table import *
import streamlit as st
import pandas as pd
from numpy_newton import newton_3d, newton_3d_alt


def main():

    table_type = st.selectbox("Тип функции", ["из таблицы", "своя"])
    func = None
    if table_type == "из таблицы":
        table = parse_table("data.txt")
    else:
        xs = st.number_input("Введите x_start: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f")
        xe = st.number_input("Введите x_end: ", -100.0, 100.0, 10.0, step=1.0, format="%.5f")
        xst = int(st.number_input("Введите кол во узлов x: ", 1, 100, 10))

        ys = st.number_input("Введите y_start: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f")
        ye = st.number_input("Введите y_end: ", -100.0, 100.0, 10.0, step=1.0, format="%.5f")
        yst = int(st.number_input("Введите кол во узлов y: ", 1, 100, 10))

        zs = st.number_input("Введите z_start: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f")
        ze = st.number_input("Введите z_end: ", -100.0, 100.0, 10.0, step=1.0, format="%.5f")
        zst = int(st.number_input("Введите кол во узлов z: ", 1, 100, 10))

        st.subheader("Expression")
        expression = st.text_input("Введите выражение", "x + y / z")
        func = eval(f"lambda x, y, z: {expression}")

        table = get_table(xs, xe, xst, ys, ye, yst, zs, ze, zst, expression)

    st.header("Table")

    #for k in range(len(table[2])):
    #    st.write("z =", int(table[2][k]))
    #    st.write(np.array(table[3][k]))

    st.header("Newton")
    x = st.number_input("Введите аргумент x: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f")
    y = st.number_input("Введите аргумент y: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f")
    z = st.number_input("Введите аргумент z: ", -100.0, 100.0, 1.0, step=1.0, format="%.5f")

    # 0.152, 0.131, 1.43

    nx = int(st.number_input("Введите степень аппроксимиляции nx: ", 0, 10, 3))
    ny = int(st.number_input("Введите степень аппроксимиляции ny: ", 0, 10, 3))
    nz = int(st.number_input("Введите степень аппроксимиляции nz: ", 0, 10, 3))

    if table_type == "Из таблицы":
       result = newton_3d(table, nx + 1, ny + 1, nz + 1, x, y, z)
    else:
        xs_ = np.linspace(xs, xe, xst)
        ys_ = np.linspace(ys, ye, yst)
        zs_ = np.linspace(zs, ze, zst)
        result = newton_3d_alt((xs_, ys_, zs_), func, (nx+1, ny+1, nz+1), (x, y, z))

    st.subheader("Result")
    st.write(f"Answer: {result}")

    if func:
        st.write("Actual:", func(x, y, z))
    st.subheader("All interpolations:")

    show_interpolations = st.checkbox("Show interpolations", False)

    if show_interpolations:
        for i_nz in range(1, nz + 1):
            st.write(f"nz = {i_nz}")
            df = {"nx": list(range(1, nx + 1))}
            for i_ny in range(1, ny + 1):
                df[f"ny={i_ny}"] = []
                for i_nx in range(1, nx + 1):
                    result = newton_3d(table, i_nx + 1, i_ny + 1, i_nz + 1, x, y, z)
                    df[f"ny={i_ny}"].append(result)
            st.dataframe(pd.DataFrame(df).set_index("nx"))


if __name__ == "__main__":
    main()

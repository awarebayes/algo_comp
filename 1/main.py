import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numba import njit, jit
import streamlit as st


@njit
def y_newt(i, k, xs, ys) -> float:
    """
    Args:
        i: индекс узла
        k: порядок
        xs: массив иксов
        ys: массив игреков

    Returns:
        значение функции y
    """
    if i + k >= len(xs):
        return 0
    elif k == 0:
        return ys[i]
    else:
        return (y_newt(i + 1, k - 1, xs, ys) - y_newt(i, k - 1, xs, ys)) / (
            xs[i + k] - xs[i]
        )


@njit
def y_herm(i, k, zs, ys, ys_der):
    if i + k >= len(zs):
        return 0
    elif k == 0:
        return ys[i // 2]
    elif k == 1 and i % 2 == 0:
        return ys_der[i // 2]
    else:
        return (
            y_herm(i + 1, k - 1, zs, ys, ys_der) - y_herm(i, k - 1, zs, ys, ys_der)
        ) / (zs[i + k] - zs[i])

@jit
def topk_closest(x, xs, k):
    closest = np.abs(xs - x).argsort()
    return np.sort(closest[:k])


@jit
def newton(x, xs, ys, power=None):
    if power is None:
        power = len(xs)

    topk = topk_closest(x, xs, power)
    xs = xs[topk]
    ys = ys[topk]

    lp = y_newt(0, 0, xs, ys)
    for k in range(1, power):
        lp += y_newt(0, k, xs, ys) * np.prod(x - xs[:k])
    return lp

@jit
def hermite(x, xs, ys, ys_der, power=None):
    if power is None:
        power = 2 * len(xs) + 2

    topk = topk_closest(x, xs, power)
    xs = xs[topk]
    ys = ys[topk]
    ys_der = ys_der[topk]

    zs = np.zeros(2 * len(xs))
    zs[::2] = xs
    zs[1::2] = xs

    hp = y_herm(0, 0, xs, ys, ys_der)
    for k in range(1, power):
        hp += y_herm(0, k, zs, ys, ys_der) * np.prod(x - zs[:k])
    return hp


def get_df(file):
    # get extension and read file
    extension = file.name.split(".")[1]
    if extension.upper() == "CSV":
        df = pd.read_csv(file, index_col=None)
    elif extension.upper() == "XLSX":
        df = pd.read_excel(file, engine="openpyxl")
    return df


def get_data(type):
    df = None
    if type == "По задаче":
        x = np.arange(0, 1.05, 0.15)
        df = pd.DataFrame(
            {
                "x": x,
                "y": [
                    1.000000,
                    0.838771,
                    0.655336,
                    0.450447,
                    0.225336,
                    -0.018310,
                    -0.278390,
                    -0.552430,
                ],
                "y`": [
                    -1.000000,
                    -1.14944,
                    -1.29552,
                    -1.43497,
                    -1.56464,
                    -1.68164,
                    -1.78333,
                    -1.86742,
                ],
            }
        )
    elif type == "Sin(4*x)":
        x = np.linspace(0, 1.05, 8)
        df = pd.DataFrame({"x": x, "y": np.sin(4 * x), "y`": 4 * np.cos(4 * x)})
    elif type == "Загрузить csv":
        file = st.file_uploader("Upload file", type=["csv", "xlsx"])
        if file is not None:
            return get_df(file)
    return df


def bin_search(xs, ys):
    for i in range(len(xs)-1):
        x_0 = xs[i]
        x_1 = xs[i + 1]
        y_0 = ys[i]
        y_1 = ys[i+1]
        if y_0 * y_1 <= 0:
            x_at_zero = (-y_0) * (x_1-x_0)/(y_1-y_0) + x_0
            if xs.min() <= x_at_zero <= xs.max():
                return x_at_zero
    return None


def backwards_newton(y_0, ys, xs, n):
    result = newton(y_0, ys, xs, power=n)
    if not ys.min() < result < ys.max():
        new_xs = np.linspace(xs.min(), xs.max(), 100)
        new_ys = []
        for new_x in new_xs:
            new_y = newton(new_x, xs, ys)
            new_ys.append(new_y)
        new_ys = np.array(new_ys)
        result = bin_search(new_xs, new_ys)
    return result


def main():
    st.title("Алгоритмы Ньютона и Эрмита")
    st.subheader("Постановка задачи, таблица")

    data_type = st.radio("Данные", ["По задаче", "Sin(4*x)", "Загрузить csv"])

    df = get_data(data_type)
    if df is None:
        return

    df = df.sort_values("x")

    st.dataframe(df)

    st.line_chart(df.set_index("x"))

    xs = df["x"].values
    ys = df["y"].values
    ys_der = df["y`"].values

    x_0 = st.slider(
        "Значение x: ",
        min_value=float(xs.min()),
        max_value=float(xs.max()),
        value=float(xs.mean())
    )

    topk = topk_closest(x_0, xs, 2)
    st.write("Ranked TopK closest")
    st.dataframe(pd.DataFrame({'index': topk, 'xs': xs[topk], 'ys': ys[topk]}))

    st.subheader("Метод Ньютона")
    newton_comparison = {"x": [], "n": [], "newton": []}

    newton_range = st.slider("n_newton", min_value=2, max_value=20, value=5)
    for n in range(2, newton_range):
        result = newton(x_0, xs, ys, power=n)
        newton_comparison["x"].append(x_0)
        newton_comparison["n"].append(n)
        newton_comparison["newton"].append(result)

    newton_comparison = pd.DataFrame(newton_comparison)
    st.dataframe(newton_comparison)
    fig, ax = plt.subplots()

    ax.plot(df["x"], df["y"], label="y")
    ax.scatter(
        newton_comparison["x"], newton_comparison["newton"], c=newton_comparison["n"]
    )
    ax.legend()
    st.pyplot(fig)

    st.subheader("Интерполирование методом ньютона")

    new_xs = np.linspace(xs.min(), xs.max(), 100)
    new_ys = []
    for new_x in new_xs:
        new_y = newton(new_x, xs, ys)
        new_ys.append(new_y)
    new_ys = np.array(new_ys)

    fig, ax = plt.subplots()
    ax.plot(new_xs, new_ys)
    st.pyplot(fig)

    st.subheader("Алгоритм Эрмита")

    hermite_comparison = {"x": [], "n": [], "hermite": []}
    hermite_range = st.slider("n_hermite", min_value=2, max_value=20, value=5)
    for n in range(2, hermite_range):
        result = hermite(x_0, xs, ys, ys_der, power=n)
        hermite_comparison["x"].append(x_0)
        hermite_comparison["n"].append(n)
        hermite_comparison["hermite"].append(result)



    hermite_comparison = pd.DataFrame(hermite_comparison)
    st.dataframe(hermite_comparison)

    fig, ax = plt.subplots()
    ax.scatter(
        hermite_comparison["x"],
        hermite_comparison["hermite"],
        c=hermite_comparison["n"],
    )
    ax.plot(df["x"], df["y"], label="y")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Сравнение алгоритмов")
    merged = pd.merge(newton_comparison, hermite_comparison)
    st.dataframe(merged)

    st.subheader("Обратная интерполяция")
    y_0 = 0
    backward_comparison = {"y": [], "n": [], "backward": []}

    backward_range = st.slider("n_backward", min_value=2, max_value=20, value=5)
    for n in range(2, backward_range):
        result = backwards_newton(y_0, ys, xs, n)
        backward_comparison["y"].append(y_0)
        backward_comparison["n"].append(n)
        backward_comparison["backward"].append(result)

    backward_comparison = pd.DataFrame(backward_comparison)
    st.dataframe(backward_comparison)
    fig, ax = plt.subplots()

    ax.plot(df["y"], df["x"], label="x")
    ax.axvline(x=0, c='k')
    ax.scatter(
        backward_comparison["y"], backward_comparison["backward"], c=backward_comparison["n"]
    )
    ax.legend()
    st.pyplot(fig)


main()

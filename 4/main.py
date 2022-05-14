import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go



def fit_data_np(x, y, w, n):
    return np.polyfit(x, y, w=w, deg=n)


def fit_data(x, y, w, n):
    a_mat = np.zeros((n+1, n+1))
    b_mat = np.zeros((n+1))

    for i in range(n+1):
        for j in range(n+1):
            a_mat[i][j] = np.sum(w * x ** (i + j))

    for i in range(n+1):
        b_mat[i] = np.sum(w * y * x ** i)

    coefs = np.linalg.solve(a_mat, b_mat)
    return coefs[::-1]


def load_data(filename):
    df = pd.read_csv(f"./data/{filename}")
    st.write("Here is the table")
    st.dataframe(df)
    x = None
    y = None
    w = None
    one_dim = False
    if filename in ["tmp6.csv", "tmp7.csv"]:
        x = df["x"].values
        y = df["y"].values
        w = df["weights"].values
        one_dim = True
    else:
        x = np.stack([df["x"].values, df["y"].values]).T
        y = df["z"].values
        w = df["weights"].values
    return x, y, w, one_dim


def norm(arr):
    pos = arr + min(arr)
    n = pos / max(pos)
    return n


def mse(poly, x_test, y_test):
    y_pred = poly(x_test)
    return mean_squared_error(y_test, y_pred)


def mae(poly, x_test, y_test):
    y_pred = poly(x_test)
    return mean_absolute_error(y_test, y_pred)


def one_dim_ls(x, y, w, n):
    coefs = fit_data(x, y, w, n)

    poly = np.poly1d(coefs)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = poly(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label=f"n={n}")
    norm_weights = np.clip(norm(w) * 100, 10, 100)
    ax.scatter(x, y, s=norm_weights)
    st.pyplot(fig)


def one_dim_ls_plot(x, y, w, n, ax):
    xs = np.linspace(x.min(), x.max(), 100)
    coefs = fit_data(x, y, w, n)
    poly = np.poly1d(coefs)
    ys = poly(xs)
    ax.plot(xs, ys, label=f"n={n}")

def multiple_ls_plot(x, y, w):
    fig, ax = plt.subplots()
    for i in [1, 2, 3, 5, 20]:
        one_dim_ls_plot(x, y, w, i, ax)
    norm_weights = np.clip(norm(w) * 100, 10, 100)
    ax.scatter(x, y, s=norm_weights)
    ax.legend()
    st.pyplot(fig)



def metric_table_one_dim(x, y, w):
    table = {'n': [], 'mse': [], 'mae': []}
    for i in [1, 2, 3, 5]:
        xs = np.linspace(x.min(), x.max(), 100)
        coefs = fit_data(x, y, w, i)
        poly = np.poly1d(coefs)
        ys = poly(xs)
        table['n'].append(i)
        table['mse'].append(mse(poly, x, y))
        table['mae'].append(mae(poly, x, y))
    st.subheader("Metrics")
    st.dataframe(pd.DataFrame(table))

def two_dim_slice_fit(x_fixed, x_interp, y, w, power, n_points=100):
    unique_x = np.unique(x_fixed)
    n_unique = len(unique_x)

    x_interp_test = np.linspace(x_interp.min(), x_interp.max(), n_points)

    x_interp_arr = np.zeros([n_unique, n_points])
    x_fixed_arr = np.zeros([n_unique, n_points])
    y_preds_arr = np.zeros([n_unique, n_points])

    for index, ux in enumerate(unique_x):

        mask = x_fixed == ux
        x_batch = x_interp[mask]
        y_batch = y[mask]
        weight_batch = w[mask]

        coefs = fit_data(x_batch, y_batch, weight_batch, power)
        poly = np.poly1d(coefs)
        y_preds = poly(x_interp_test)

        x_interp_arr[index] = x_interp_test
        x_fixed_arr[index] = np.ones(n_points) * ux
        y_preds_arr[index] = y_preds



    return x_fixed_arr, x_interp_arr, y_preds_arr


def two_dim_interp_weights(x_fixed, x_interp, weights, n_points=100):
    unique_x = np.unique(x_fixed)
    n_unique = len(unique_x)

    x_interp_test = np.linspace(x_interp.min(), x_interp.max(), n_points)
    w_interp_arr = np.zeros([n_unique, n_points])

    for index, ux in enumerate(unique_x):

        mask = x_fixed == ux
        x_batch = x_interp[mask]
        w_batch = weights[mask]
        w_preds = np.interp(x_interp_test, x_batch, w_batch)
        w_interp_arr[index] = w_preds

    return w_interp_arr


def plot_two_dim_fit(x_fixed, x_interp, y, w, power, n_points=100):
    unique_x = np.unique(x_fixed)
    n_unique = len(unique_x)

    x_interp_test = np.linspace(x_interp.min(), x_interp.max(), n_points)

    x_interp_arr = np.zeros([n_unique, n_points])
    x_fixed_arr = np.zeros([n_unique, n_points])
    y_preds_arr = np.zeros([n_unique, n_points])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for index, ux in enumerate(unique_x):
        mask = x_fixed == ux
        x_train = x_interp[mask]
        y_train = y[mask]
        weight_train = w[mask]

        coefs = np.polyfit(x_train, y_train, w=weight_train, deg=power)
        poly = np.poly1d(coefs)
        y_preds = poly(x_interp_test)

        x_interp_arr[index] = x_interp_test
        x_fixed_arr[index] = np.ones(n_points) * ux
        y_preds_arr[index] = y_preds
        ax.plot3D(x_interp_test, np.full_like(x_interp_test, ux), y_preds)
        ax.scatter3D(x_train, np.full_like(x_train, ux), y_train)
    st.pyplot(fig)


def two_dim_ls(x, y, w, n):
    x1s = x[:, 0]
    x2s = x[:, 1]

    plot_two_dim_fit(x2s, x1s, y, w, n)

    x2s_fixed, x1s_fixed, ys_fixed = two_dim_slice_fit(x2s, x1s, y, w, n, 100)
    ws_fixed = two_dim_interp_weights(x2s, x1s, w)

    plot_two_dim_fit(x1s_fixed, x2s_fixed, ys_fixed, ws_fixed, n)

    x1s_final, x2s_final, ys_final = two_dim_slice_fit(
        x1s_fixed.ravel(), x2s_fixed.ravel(), ys_fixed.ravel(), ws_fixed.ravel(), n, 100
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_wireframe(x1s_final, x2s_final, ys_final, rstride=10, cstride=10)
    ax.scatter3D(x1s, x2s, y, c=w)
    st.pyplot(fig)



def two_dim_ls_pyplot(x, y, w, n):
    x1s = x[:, 0]
    x2s = x[:, 1]

    x2s_fixed, x1s_fixed, ys_fixed = two_dim_slice_fit(x2s, x1s, y, w, n, 100)
    ws_fixed = two_dim_interp_weights(x2s, x1s, w)

    x1s_final, x2s_final, ys_final = two_dim_slice_fit(
        x1s_fixed.ravel(), x2s_fixed.ravel(), ys_fixed.ravel(), ws_fixed.ravel(), n, 100
    )


    lines = []
    line_marker = dict(color='#0066FF', width=2)
    for i, j, k in zip(x1s_final, x2s_final, ys_final):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
    scats = go.Scatter3d(x=x1s.ravel(), y=x2s.ravel(), z=y, mode='markers', marker=dict(color='#ff0000'))
    fig = go.Figure(data=[*lines, scats])
    st.plotly_chart(fig)



def main():
    st.header("Метод наименьших квадратов")
    data_type = st.radio(
        "Данные", ["tmp6.csv", "tmp7.csv", "tmp8.csv", "tmp9.csv", "custom.csv"]
    )
    x, y, w, one_dim = load_data(data_type)

    n = st.slider(
        "Степень: ",
        min_value=1,
        max_value=10,
        value=3,
    )

    if one_dim:
        one_dim_ls(x, y, w, n)
        multiple_ls_plot(x, y, w)
        metric_table_one_dim(x, y, w)
    else:
        try:
            two_dim_ls(x, y, w, n)
            two_dim_ls_pyplot(x, y, w, n)
        except np.linalg.LinAlgError:
            st.error("Степень слишком большая")


if __name__ == "__main__":
    main()

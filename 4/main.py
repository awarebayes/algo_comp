import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def fit_data(x, y, w, n):
    return np.polyfit(x, y, w=w, deg=n)

def fit_multidim(x, y, w, n):
    reg = LinearRegression().fit(x, y, sample_weight=w, )


def load_data(filename):
    df = pd.read_csv(f"./data/{filename}")
    st.write("Here is the table")
    st.dataframe(df)
    x = None
    y = None
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


def one_dim_ls(x, y, w, n):
    coefs = fit_data(x, y, w, n)

    poly = np.poly1d(coefs)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = poly(xs)

    fig, ax = plt.subplots()
    ax.plot(
        xs,
        ys
    )
    norm_weights = np.clip(norm(w)*100, 10, 100)
    ax.scatter(x, y, s=norm_weights)

    st.pyplot(fig)


def main():
    st.header("Метод наименьших квадратов")
    data_type = st.radio("Данные", ["tmp6.csv", "tmp7.csv", "tmp8.csv", "tmp9.csv", "custom.csv"])
    x, y, w, one_dim = load_data(data_type)

    n = st.slider(
        "Значение x: ",
        min_value=2,
        max_value=10,
        value=3,
    )

    if one_dim:
        one_dim_ls(x, y, w, n)



if __name__ == '__main__':
    main()
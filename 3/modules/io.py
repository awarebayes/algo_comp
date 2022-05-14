from .newton import *
from .spline import *
import streamlit as st

import matplotlib.pyplot as plt


def print_graphic(x, y, start, end, table):
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[j] < x[i]:
                x[i], x[j] = x[j], x[i]
                y[i], y[j] = y[j], y[i]

    cur = x[0]
    xg = []
    yg_1 = []
    yg_2 = []
    yg = []
    nwt = []
    step = 0.01
    while cur < x[-1]:
        xg.append(cur)
        yg.append(calculate_spline(cur, x, y, 0, 0))
        yg_1.append(calculate_spline(cur, x, y, start, 0))
        yg_2.append(calculate_spline(cur, x, y, start, end))
        nwt.append(newton_interpolate(cur, 3, table))
        cur += step

    fig, ax = plt.subplots()
    ax.plot(xg, yg, color="b", label="Spline, zeros")
    ax.plot(xg, yg_1, color="g", label="Spline, one derivative")
    ax.plot(xg, yg_2, color="r", label="Spline, two derivatives")
    ax.plot(xg, nwt, color="black", label="Newton")
    show_points(ax, x, y)

    ax.legend()
    ax.set_title("Approx comparison")
    st.pyplot(fig)


def show_points(ax, x, y):
    for i in range(len(x)):
        ax.plot(x[i], y[i], marker="o", markeredgecolor="grey", markerfacecolor="red")


def print_coefs_table(coefs, x, y):
    a, h, c, b, d = coefs

    print(
        "i"
        + "\t\t"
        + "x"
        + "\t\t\t"
        + "y"
        + "\t\t\t"
        + "h"
        + "\t\t\t"
        + "a"
        + "\t\t\t"
        + "c"
        + "\t\t\t"
        + "b"
        + "\t\t\t"
        + "d"
    )

    for i in range(len(c)):
        xt = "{:^10s}".format("None") if i >= len(x) else "{:^10.3f}".format(x[i])
        yt = "{:^10s}".format("None") if i >= len(y) else "{:^10.3f}".format(y[i])
        ht = (
            "{:^10s}".format("None")
            if i >= len(h) or h[i] == None
            else "{:^10.3f}".format(h[i])
        )
        at = (
            "{:^10s}".format("None")
            if i >= len(a) or a[i] == None
            else "{:^10.3f}".format(a[i])
        )
        ct = "{:^10s}".format("None") if c[i] == None else "{:^10.3f}".format(c[i])
        bt = (
            "{:^10s}".format("None")
            if i >= len(b) or b[i] == None
            else "{:^10.3f}".format(b[i])
        )
        dt = (
            "{:^10s}".format("None")
            if i >= len(d) or d[i] == None
            else "{:^10.3f}".format(d[i])
        )

        print(
            str(i)
            + "\t"
            + xt
            + "\t"
            + yt
            + "\t"
            + ht
            + "\t"
            + at
            + "\t"
            + ct
            + "\t"
            + bt
            + "\t"
            + dt
        )


def print_ksi_eta(yValues, hValues, ksi, eta, cValues):
    print("i" + "\t\t" + "y" + "\t\t\t" + "h" + "\t\t\t" + "ksi" + "\t\t\t" + "eta")

    for i in range(0, len(ksi)):
        ht = (
            "{:^10s}".format("None")
            if i >= len(hValues) or hValues[i] == None
            else "{:^10.5f}".format(hValues[i])
        )
        yt = (
            "{:^10s}".format("None")
            if i >= len(yValues) or yValues[i] == None
            else "{:^10.5f}".format(yValues[i])
        )
        ksit = (
            "{:^10s}".format("None")
            if i >= len(ksi) or ksi[i] == None
            else "{:^10.5f}".format(ksi[i])
        )
        etat = (
            "{:^10s}".format("None")
            if i >= len(eta) or eta[i] == None
            else "{:^10.5f}".format(eta[i])
        )
        ct = (
            "{:^10s}".format("None")
            if i >= len(cValues) or cValues[i] == None
            else "{:^10.5f}".format(cValues[i])
        )

        print(str(i) + "\t" + yt + "\t" + ht + "\t" + ksit + "\t" + etat)


def input_table_from_file(filename):
    x = []
    y = []

    if filename == None:
        print("Такого файла нет!")
    else:
        f = open(filename, "r")

        for line in f:
            row = line.strip().split()
            row = [float(x) for x in row]
            x.append(row[0])
            y.append(row[1])

        f.close()

    return x, y

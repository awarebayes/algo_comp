import numpy as np


EPS = 1e-6


def f(x, y, z, expression):
    try:
        return eval(expression)
    except ZeroDivisionError:
        return np.nan


def get_table(sx, ex, amx, sy, ey, amy, sz, ez, amz, expression):
    tab = [[], [], [], []]

    tab[0] = np.linspace(sx, ex, amx).tolist()
    tab[1] = np.linspace(sy, ey, amy).tolist()
    tab[2] = np.linspace(sz, ez, amz).tolist()

    for i in range(amz):
        tab[3].append([])
        for j in range(amy):
            tab[3][i].append([])
            for k in range(amx):
                tab[3][i][j].append(f(tab[0][k], tab[1][j], tab[2][i], expression))
    return tab


def parse_table(path):
    tab = [[], [], [], [[]]]
    file = open(path)
    flag_add_x = False
    flag_add_y = False
    z_index = 0
    y_index = 0
    for line in file.readlines():
        row = line.split("\n")[0].split("\t")

        if "z=" in row[0]:
            zStr = row[0].split("z=")
            tab[2].append(float(zStr[1]))
        elif "y\\x" in row[0]:
            if flag_add_x:
                continue
            for i in range(1, len(row)):
                tab[0].append(float(row[i]))
            flag_add_x = True
        else:
            if "end" in row[0]:
                continue
            if not row[0].isdigit():
                z_index += 1
                tab[3].append([])
                y_index = 0
                flag_add_y = True
                continue

            if not flag_add_y:
                tab[1].append(float(row[0]))

            tab[3][z_index].append([])
            for i in range(1, len(row)):
                tab[3][z_index][y_index].append(float(row[i]))
            y_index += 1

    file.close()
    return tab

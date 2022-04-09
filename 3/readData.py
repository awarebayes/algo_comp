import numpy as np

EPS = 1e-6


# генерация данных таблицы
def generateTable(sx, ex, amx, f):
    dataTable = list()
    xValues = np.linspace(sx, ex, amx)

    for i in range(amx):
        dataTable.append((xValues[i], f(xValues[i])))

    return np.array(dataTable)


# чтение файла в представленный массив данных
# криво, но лучше, чем ничего
def read_table(filename):

    dataTable = list()
    with open(filename, "r") as file:
        for line in file.readlines():
            row = list(map(float, line.split(" ")))
            dataTable.append((row[0], row[1]))

    return np.array(dataTable)

# Основной модуль для интерполяции
# функций методом Ньютона


def newton_table_is_valid(n, table):
    """
    Фунцкия проверки таблицы на валидность

    :param n: степень искомого полинома
    :param table: таблица входных точек
    :return: истина (таблица валидна) / ложь (таблица невалидна)
    """
    ans = True
    if len(table) < n + 1:
        ans = False

    return ans


def newton_find_closest_value_in_table(x, table):
    """
    Функция поиска индекса ближайшего к х элемента таблицы

    :param x: исходное значение
    :param table: таблица входных точек
    :return: индекс строки, содержащей ближайший икс
    """
    ind = 0
    min = abs(x - table[0][0])

    for i in range(1, len(table)):
        cur_min = abs(x - table[i][0])
        if cur_min <= min:
            min = cur_min
            ind = i

    return ind


def newton_init_closest_values_table(closest_i, n, table):
    """
    Функция составляет таблицу из n + 1 строк, находящихся рядом со
    строкой под номером closest_i

    :param closest_i: индекс строки, вокруг которой составляется таблица
    :param n: степень искомого полинома
    :param table: таблица пар икс - игрек
    :return: таблица из n + 1 строк окружения строки с номером closest_i
    """
    coefficient_table = []
    left = n + 1  # сколько строк осталось добавить
    i = closest_i  # справа
    j = i - 1  # слева

    while left > 0:
        # при добавлении таким образом сохранится изначальный порядок таблицы
        if i < len(table):
            coefficient_table.append([table[i][0], table[i][1]])
            i += 1
            left -= 1
        if left > 0 and j >= 0:
            coefficient_table.insert(0, [table[j][0], table[j][1]])
            j -= 1
            left -= 1

    return coefficient_table


def newton_count_coefficients(x, n, table):
    """
    Функция подсчёта таблицы коэффициентов для метода Ньютона

    :param x: исходное значение
    :param n: степень полинома
    :param table: таблица данных из пар икс-игрек
    :return: таблица коэффициентов
    """
    if not (newton_table_is_valid(n, table)):
        ret = None
    else:
        # Здесь таблица валидна
        # Нужно взять ближайшие n + 1 записей

        # Находим ближайший по иксу узел
        closest_i = newton_find_closest_value_in_table(x, table)
        # Формируем таблицу икс - игрек, в которой дальше будем считать коэффициенты
        coefficient_table = newton_init_closest_values_table(closest_i, n, table)

        # Заполняем таблицу
        for j in range(2, n + 2):
            cur_i = 0
            while cur_i + j - 1 < len(coefficient_table):
                # берём элемент той же строки и строки ниже из левого столбца
                new_coefficient = (
                    coefficient_table[cur_i][j - 1]
                    - coefficient_table[cur_i + 1][j - 1]
                ) / (coefficient_table[cur_i][0] - coefficient_table[cur_i + j - 1][0])
                coefficient_table[cur_i].append(new_coefficient)
                cur_i += 1

        ret = coefficient_table

    return ret


def newton_interpolate(x, n, table):
    """
    Основная функция интерполяции методом Ньютона

    :param x: исходное значение икса
    :param n: степень полинома
    :param table: таблица узлов
    :return: значение функции в точке x
    """
    if n < 0:
        return "Степень полинома не может быть отрицательной"
    else:
        coefficient_table = newton_count_coefficients(x, n, table)
        if coefficient_table == None:
            return "Недостаточно данных для построения полинома"
        else:
            coefficients = coefficient_table[0][2:]  # коэффициенты
            # print("---")
            # for line in coefficient_table:
            #     print(line)
            # print("---")

            val = coefficient_table[0][1]  # искомое значение
            common_x = 1  # общий множитель иксов
            for i in range(2, n + 2):
                common_x *= x - coefficient_table[i - 2][0]
                val = val + coefficients[i - 2] * common_x

            return val


def add_values_to_table(xValues, yValues):
    table = []
    for i in range(len(xValues)):
        table.append([xValues[i], yValues[i]])

    return table


def newton_interpolate_second_derivative(x, table, n=3):
    """
    Функция для вычисления производной полинома Ньютона

    :param x: исходное значение икса
    :param n: степень полинома
    :param table: таблица узлов
    :return: значение функции в точке x
    """
    if n == 2:
        coefficient_table = newton_count_coefficients(x, n, table)
        coefficients = coefficient_table[0][2:]  # коэффициенты
        return 2 * coefficients[-1]
    if n < 0:
        return "Степень полинома не может быть отрицательной"
    else:
        coefficient_table = newton_count_coefficients(x, n, table)
        if coefficient_table == None:
            return "Недостаточно данных для построения полинома"
        else:
            coefficients = coefficient_table[0][2:]  # коэффициенты

            val = 2 * coefficients[-2] + 2 * coefficients[-1] * (
                3 * x
                - (
                    coefficient_table[0][0]
                    + coefficient_table[1][0]
                    + coefficient_table[2][0]
                )
            )
            return val

def a(y):
    return float(y)

def calc_a_coefs(yValues):
    val = []

    val.append(None)
    for i in range(1, len(yValues) + 1):
        val.append(a(yValues[i - 1]))

    return val

def h(x1, x2):
    return x2 - x1

def calc_h_coefs(xValues):
    val = []

    val.append(None)
    for i in range(1, len(xValues)):
        val.append(h(xValues[i - 1], xValues[i]))
        # hi = xi - xi-1
        # Первый элемент None, т.к. вычисление с 1 коэф.

    return val

def c(ksi_i1, c_i1, eta_i1):
    return ksi_i1 * c_i1 + eta_i1

def ksi_next(h, hprev, ksi):
    return - (h / (hprev * ksi + 2 * (hprev + h)))

def f(y, yprev, yprevprev, h, hprev):
    return 3 * ((y - yprev) / h - (yprev - yprevprev) / hprev)

def eta_next(h, hprev, ksi, eta, y, yprev, yprevprev):
    return (f(y, yprev, yprevprev, h, hprev) - hprev * eta) / \
           (hprev * ksi + 2 * (hprev + h))

def calc_ksi(ksi_start, hValues):
    val = []
    val.append(0)
    val.append(0)
    val.append(ksi_start)

    for i in range(2, len(hValues)):
        new = ksi_next(hValues[i], hValues[i - 1], val[-1])
        val.append(new)

    return val

def calc_eta(eta_start, hValues, yValues, ksiValues):
    val = []
    val.append(0)
    val.append(0)
    val.append(eta_start)

    for i in range(2, len(hValues)):
        new = eta_next(hValues[i], hValues[i - 1], ksiValues[i], val[-1], yValues[i], yValues[i - 1], yValues[i - 2])
        val.append(new)

    return val

def calc_c_coefs(start, end, hValues, yValues):
    # start, end - начальные и конечные C
    val = []
    val.insert(0, end) # cN + 1

    ksi_start = 0
    eta_start = start

    ksi = calc_ksi(ksi_start, hValues)
    eta = calc_eta(eta_start, hValues, yValues, ksi)

    for i in range(len(ksi) - 2, 1, -1):
        val.insert(0, c(ksi[i + 1], val[0], eta[i + 1]))

    val.insert(0, start) # c1
    val.insert(0, None)

    return val

def b_coef(y, yprev, h, cnext, c):
    return ((y - yprev) / h - h * (cnext + 2 * c) / 3)

def calc_b_coefs(yValues, hValues, cValues):
    val = []
    val.append(None)

    for i in range(1, len(yValues)):
        val.append(b_coef(yValues[i], yValues[i - 1], hValues[i], cValues[i + 1], cValues[i]))

    val.append(None)

    return val

def d_coef(cnext, c, h):
    return (cnext - c) / (3 * h)

def calc_d_coefs(cValues, hValues):
    val = []

    val.append(None)
    for i in range(1, len(cValues) - 1):
        val.append(d_coef(cValues[i + 1], cValues[i], hValues[i]))

    val.append(None)

    return val

def calc_all_coefs(x0_deriv, xn_deriv, xValues, yValues):
    c0 = x0_deriv / 2
    cN = xn_deriv / 2

    a = calc_a_coefs(yValues)
    h = calc_h_coefs(xValues)
    c = calc_c_coefs(c0, cN, h, yValues)
    b = calc_b_coefs(yValues, h, c)
    d = calc_d_coefs(c, h)

    return a, h, c, b, d

def find_closest_interval(x, xValues):
    i = 1
    while not(xValues[i - 1] <= x <= xValues[i] or xValues[i] <= x <= xValues[i - 1]):
        i += 1
    return i

def calculate_spline(x, xValues, yValues, x0_deriv, xn_deriv):
    a, h, c, b, d = calc_all_coefs(x0_deriv, xn_deriv, xValues, yValues)
    #print_coefs_table(calc_all_coefs(x0_deriv, xn_deriv, xValues, yValues), xValues, yValues)

    i = find_closest_interval(x, xValues)
    diff = x - xValues[i - 1]
    val = a[i] + b[i] * diff + c[i] * (diff ** 2) + d[i] * (diff ** 3)

    return val
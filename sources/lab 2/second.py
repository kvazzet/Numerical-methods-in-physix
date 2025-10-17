import math

def f(h, R, rs, rw):
    # f(h) = h**2*(3*R - h) - 4*R**3*(rs/rw)
    return h**2 * (3*R - h) - 4*R**3 * (rs / rw)

def df(h, R, rs, rw):
    # Производная по h: 2h*(3R-h) + h^2*(-1)
    return 2*h*(3*R - h) - h**2

# 1. Метод бисекции
def bisection(f, a, b, args, eps=1e-6, max_iter=100):
    n = 0
    while abs(b-a) > eps and n < max_iter:
        c = (a+b)/2
        if f(c, *args) == 0 or abs(f(c, *args)) < eps:
            return c, n+1
        if f(a, *args) * f(c, *args) < 0:
            b = c
        else:
            a = c
        n += 1
    return (a+b)/2, n+1

# 2. Метод простой итерации
def g(h, R, rs, rw, lmbd):
    # h_new = h - лямбда*f(h)
    return h - lmbd*f(h, R, rs, rw)

def simple_iteration(f, h0, args, eps=1e-6, max_iter=100, lmbd=0.01):
    n = 0
    h = h0
    while n < max_iter:
        h_new = g(h, *args, lmbd)
        if abs(h_new - h) < eps:
            return h_new, n+1
        h = h_new
        n += 1
    return h, n

# 3. Метод Ньютона
def newton(f, df, h0, args, eps=1e-6, max_iter=100):
    n = 0
    h = h0
    while abs(f(h, *args)) > eps and n < max_iter:
        h_new = h - f(h, *args) / df(h, *args)
        if abs(h_new - h) < eps:
            return h_new, n+1
        h = h_new
        n += 1
    return h, n

if __name__ == "__main__":
    # Пример: R=1.0, rs=0.6, rw=1.0 (шарик легче воды)
    R = 1.0    # радиус
    rs = 0.6   # плотность шарика
    rw = 1.0   # плотность жидкости

    # Решаем уравнение на интервале h ∈ [0, 2*R] (шар плавает — погружён частично)
    a, b = 0.0, 2*R
    h0 = R           # начальное приближение для итераций
    lmbd = 0.01      # параметр для устойчивой итерации
    args = (R, rs, rw)

    root_bis, n_bis = bisection(f, a, b, args)
    print(f"Бисекция: h = {root_bis:.6f}, итераций: {n_bis}")

    root_iter, n_iter = simple_iteration(f, h0, args, lmbd=lmbd)
    print(f"Итерации: h = {root_iter:.6f}, итераций: {n_iter}")

    root_newt, n_newt = newton(f, df, h0, args)
    print(f"Ньютон: h = {root_newt:.6f}, итераций: {n_newt}")

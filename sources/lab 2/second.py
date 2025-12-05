import math
import matplotlib.pyplot as plt
import numpy as np


def f(h, R, rs, rw):
    return h ** 2 * (3 * R - h) - 4 * R ** 3 * (rs / rw)


def df(h, R, rs, rw):
    return 2 * h * (3 * R - h) - h ** 2


def bisection(f, a, b, args, eps=1e-6, max_iter=100):
    n = 0
    while abs(b - a) > eps and n < max_iter:
        c = (a + b) / 2
        if f(c, *args) == 0 or abs(f(c, *args)) < eps:
            return c, n + 1
        if f(a, *args) * f(c, *args) < 0:
            b = c
        else:
            a = c
        n += 1
    return (a + b) / 2, n + 1


def g(h, R, rs, rw, lmbd):
    return h - lmbd * f(h, R, rs, rw)


def simple_iteration(f, h0, args, eps=1e-6, max_iter=100, lmbd=0.01):
    n = 0
    h = h0
    while n < max_iter:
        h_new = g(h, *args, lmbd)
        if abs(h_new - h) < eps:
            return h_new, n + 1
        h = h_new
        n += 1
    return h, n


def newton(f, df, h0, args, eps=1e-6, max_iter=100):
    n = 0
    h = h0
    while abs(f(h, *args)) > eps and n < max_iter:
        h_new = h - f(h, *args) / df(h, *args)
        if abs(h_new - h) < eps:
            return h_new, n + 1
        h = h_new
        n += 1
    return h, n


if __name__ == "__main__":
    R = 1.0
    rw = 1.0

    # Для графика: варьируем плотность шарика от 0 до 2*rw
    rs_values = np.linspace(0.1, 2.0, 100)
    distances = []

    for rs in rs_values:
        args = (R, rs, rw)
        # Используем метод Ньютона (можно любой другой)
        if rs < rw:
            # Шар легче воды, плавает: ищем корень в [0, 2R]
            h0 = R
            root, _ = newton(f, df, h0, args)
        else:
            # Шар тяжелее или равен воде, тонет: h = 2R (полностью погружён)
            root = 2 * R

        # Расстояние от центра до поверхности: d = R - h
        distance = R - root
        distances.append(distance)

    # Построение графика
    plt.figure(figsize=(8, 5))
    plt.plot(rs_values, distances, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=-R, color='r', linestyle='--', alpha=0.3, label='Полное погружение (центр на глубине R)')
    plt.axvline(x=rw, color='g', linestyle='--', alpha=0.3, label=f'ρₛ = ρ_воды = {rw}')

    plt.xlabel('Плотность шарика ρₛ', fontsize=12)
    plt.ylabel('Расстояние от центра до поверхности d (R - h)', fontsize=12)
    plt.title('Зависимость расстояния от центра шара до поверхности воды от плотности', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Пример расчета для конкретной плотности
    rs = 0.6
    args = (R, rs, rw)
    a, b = 0.0, 2 * R
    h0 = R
    lmbd = 0.01

    root_bis, n_bis = bisection(f, a, b, args)
    print(f"Бисекция: h = {root_bis:.6f}, итераций: {n_bis}, d = {R - root_bis:.6f}")

    root_iter, n_iter = simple_iteration(f, h0, args, lmbd=lmbd)
    print(f"Итерации: h = {root_iter:.6f}, итераций: {n_iter}, d = {R - root_iter:.6f}")

    root_newt, n_newt = newton(f, df, h0, args)
    print(f"Ньютон: h = {root_newt:.6f}, итераций: {n_newt}, d = {R - root_newt:.6f}")

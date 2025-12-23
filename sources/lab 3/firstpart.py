import numpy as np
from time import perf_counter
from scipy.integrate import quad

def f(x):
    return np.arctan(x) / x

a, b = 1.0, 3.7
I_true, _ = quad(f, a, b)

def central_rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)   # n узлов
    return h * np.sum(f(x)), len(x)

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = a + h * np.arange(n + 1)       # n+1 узел
    y = f(x)
    I = h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])
    return I, len(x)

def simpson_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = a + h * np.arange(n + 1)       # n+1 узел
    y = f(x)
    I = (h/3) * (y[0] + 2*np.sum(y[2:n:2]) + 4*np.sum(y[1:n:2]) + y[n])
    return I, len(x), n                # вернём фактическое n

n = 40
print(f"Интервал [{a}, {b}], n = {n}\n")

# Прямоугольники
t0 = perf_counter()
I_rect, cnt_rect = central_rectangle_rule(f, a, b, n)
t1 = perf_counter()
print("Центральные прямоугольники:")
print("  Кол-во узлов:", cnt_rect)
print("  Значение:", I_rect)
print("  Ошибка:", abs(I_rect - I_true))
print("  Время, с:", t1 - t0, "\n")

# Трапеции
t0 = perf_counter()
I_trap, cnt_trap = trapezoidal_rule(f, a, b, n)
t1 = perf_counter()
print("Трапеции:")
print("  Кол-во узлов:", cnt_trap)
print("  Значение:", I_trap)
print("  Ошибка:", abs(I_trap - I_true))
print("  Время, с:", t1 - t0, "\n")

# Симпсон
t0 = perf_counter()
I_simp, cnt_simp, n_eff = simpson_rule(f, a, b, n)
t1 = perf_counter()
print("Симпсон:")
print("  Фактическое n (чётное):", n_eff)
print("  Кол-во узлов:", cnt_simp)
print("  Значение:", I_simp)
print("  Ошибка:", abs(I_simp - I_true))
print("  Время, с:", t1 - t0)


import numpy as np

def f(x):
    return np.arctan(x) / x

# Центральные прямоугольники
def central_rectangle_rule(f, a, b, n):
    h = (b - a) / n
    xi = a + h * np.arange(n) + h / 2
    return h * np.sum(f(xi))

# Трапеции
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    xi = a + h * np.arange(n)
    xi1 = xi + h
    return np.sum((f(xi) + f(xi1)) * h / 2)

# Симпсон
def simpson_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    xi0 = a
    xiN = b
    xi = a + h * np.arange(1, n, 2)
    xj = a + h * np.arange(2, n, 2)
    return (h/3) * (
        f(xi0) + 2 * np.sum(f(xj)) + 4 * np.sum(f(xi)) + f(xiN)
    )

# Вычисление

a, b = 2.5, 3.7
n = 1000  # для теста, потом увеличь до 50, 100

central = central_rectangle_rule(f, a, b, n)
trapez = trapezoidal_rule(f, a, b, n)
simpson = simpson_rule(f, a, b, n)

print('Промежуток:', a, 'до', b)
print('Центральные прямоугольники:', central)
print('Трапеции:', trapez)
print('Симпсон:', simpson)

import numpy as np
import matplotlib.pyplot as plt

# ----- ПАРАМЕТРЫ ЗАДАЧИ -----
nu = 10.0          # ν
a, b = 1.0, 3.0   # границы отрезка (a > 0!)
N = 100           # число шагов по x
ua, ub = 1.0, 0.0 # граничные условия u(a), u(b)

# ----- СЕТКА -----
h = (b - a) / N
x = np.linspace(a, b, N + 1)

# ----- КОЭФФИЦИЕНТЫ УРАВНЕНИЯ -----
# x^2 u'' + x u' + (x^2 - ν^2) u = 0
# Приводим к дивергентному виду:
# (x^2 u')' + (x^2 - ν^2) u = 0

def p(x):
    return x**2

def q(x):
    return x**2 - nu**2

# ----- ПОСТРОЕНИЕ КОЭФФИЦИЕНТОВ ТРЁХДИАГОНАЛЬНОЙ МАТРИЦЫ -----
# Используется классическая центральная аппроксимация дивергентной формы:
# (p u')' ~ (1/h) [ p_{i+1/2}(u_{i+1}-u_i)/h - p_{i-1/2}(u_i-u_{i-1})/h ]

A = np.zeros(N - 1)  # поддиагональ
B = np.zeros(N - 1)  # главная диагональ
C = np.zeros(N - 1)  # наддиагональ
F = np.zeros(N - 1)  # правая часть (0, т.к. однородное)

for i in range(1, N):
    xi = x[i]
    p_plus  = p(xi + 0.5 * h)
    p_minus = p(xi - 0.5 * h)
    qi = q(xi)

    A[i-1] =  p_minus / h**2          # коэффициент при u_{i-1}
    C[i-1] =  p_plus  / h**2          # коэффициент при u_{i+1}
    B[i-1] = -(p_plus + p_minus) / h**2 + qi  # при u_i

# Учёт граничных условий u(a)=ua, u(b)=ub
F[0]   -= A[0]         * ua   # первый узел зависит от u_0
F[-1]  -= C[-1]        * ub   # последний узел зависит от u_N

# границы теперь считаем перенесёнными в правую часть
A[0]   = 0.0
C[-1]  = 0.0

# ----- МЕТОД ПРОГОНКИ (THOMAS) -----
alpha = np.zeros(N - 1)
beta  = np.zeros(N - 1)

alpha[0] = -C[0] / B[0]
beta[0]  =  F[0] / B[0]

for i in range(1, N - 1):
    denom = B[i] + A[i] * alpha[i-1]
    alpha[i] = -C[i] / denom
    beta[i]  = (F[i] - A[i] * beta[i-1]) / denom

u = np.zeros(N + 1)
u[0]  = ua
u[-1] = ub

u[N-1] = beta[-1]
for i in range(N-2, 0, -1):
    u[i] = alpha[i] * u[i+1] + beta[i]

# ----- ВИЗУАЛИЗАЦИЯ -----
plt.plot(x, u, '-o', ms=3)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid(True)
plt.title('Численное решение краевой задачи')
plt.show()

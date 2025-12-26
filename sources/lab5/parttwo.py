import numpy as np
import matplotlib.pyplot as plt

# ----- ПАРАМЕТРЫ ЗАДАЧИ -----
Ts = 420.0         # температура стенки, K
T_inf = 290.0      # температура окружающей среды, K
L = 0.1            # длина пластины, м (пример)
h = 50.0           # коэффициент теплоотдачи, Вт/(м^2 K)
k = 20.0           # коэффициент теплопроводности, Вт/(м K)

Bi = h * L / k     # число Био
N = 5            # число шагов по x

# ----- СЕТКА ПО ξ = x/L -----
xi = np.linspace(0.0, 1.0, N + 1)
h_xi = 1.0 / N

# ----- СБОРКА ТРЁХДИАГОНАЛЬНОЙ СИСТЕМЫ ДЛЯ θ -----
# Уравнение: θ'' - Bi*θ = 0
# θ(0)=1 (Дирихле), θ'(1)=0 (Нейман: (θ_N - θ_{N-1})/h = 0 -> θ_N = θ_{N-1})

M = N + 1                    # всего узлов
A = np.zeros(M)              # поддиагональ
B = np.zeros(M)              # диагональ
C = np.zeros(M)              # наддиагональ
F = np.zeros(M)              # правая часть

# Внутренние узлы 1..N-1
for i in range(1, N):
    A[i] = 1.0 / h_xi**2
    C[i] = 1.0 / h_xi**2
    B[i] = -2.0 / h_xi**2 - Bi
    F[i] = 0.0

# Граничное условие слева: θ(0) = 1
B[0] = 1.0
C[0] = 0.0
F[0] = 1.0

# Правый край: θ'(1) = 0 -> (θ_N - θ_{N-1})/h = 0 -> θ_N - θ_{N-1} = 0
A[N] = -1.0
B[N] = 1.0
C[N] = 0.0
F[N] = 0.0

# ----- МЕТОД ПРОГОНКИ -----
alpha = np.zeros(M)
beta = np.zeros(M)

alpha[0] = -C[0] / B[0]
beta[0] = F[0] / B[0]

for i in range(1, M):
    denom = B[i] + A[i] * alpha[i - 1]
    if i < M - 1:
        alpha[i] = -C[i] / denom
    beta[i] = (F[i] - A[i] * beta[i - 1]) / denom

theta = np.zeros(M)
theta[-1] = beta[-1]
for i in range(M - 2, -1, -1):
    theta[i] = alpha[i] * theta[i + 1] + beta[i]

# Возврат к температуре T(x)
T = theta * (Ts - T_inf) + T_inf
x = xi * L

# ----- ГРАФИК -----
plt.plot(x, T, '-o', ms=3, label='T(x)')
plt.xlabel('x, м')
plt.ylabel('T, K')
plt.title('Распределение температуры в пластине')
plt.grid(True)
plt.legend()
plt.show()

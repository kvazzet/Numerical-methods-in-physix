import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

def f(x):
    return np.arctan(x) / x

def rect_rule(f, a, b, n):
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    return h * np.sum(f(x))

def trapz_rule(f, a, b, n):
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = f(x)
    return (h/3) * (y[0] + 2*np.sum(y[2:n:2]) + 4*np.sum(y[1:n:2]) + y[n])

# Контроль по заданной точности

def adaptive_integrate(method, f, a, b, eps, gamma, n_start=32, n_max=100000):
    Rs, Ns, Is = [], [], []
    n = n_start
    I0 = method(f, a, b, n)
    while n < n_max:
        n2 = n * 2
        I1 = method(f, a, b, n2)
        R = (I1 - I0) / (2**gamma - 1)
        Rs.append(np.abs(R)); Ns.append(np.log(n2)); Is.append(I1)
        if np.abs(R) < eps:
            I_runget = I1 + R
            return I_runget, n2, Rs, Ns, Is
        n = n2
        I0 = I1
    return I1, n, Rs, Ns, Is

# Интервал интегрирования
a, b = 2.5, 3.7

eps = 1e-10  # требуемая точность
# Применяем все три метода:
I_rect, nR, RsR, NsR, IsR = adaptive_integrate(rect_rule, f, a, b, eps, gamma=2)
I_trapz, nT, RsT, NsT, IsT = adaptive_integrate(trapz_rule, f, a, b, eps, gamma=2)
I_simp, nS, RsS, NsS, IsS = adaptive_integrate(simpson_rule, f, a, b, eps, gamma=4)

print("Результаты с точностью", eps)
print(" Центральные прямоугольники:", I_rect, ", n =", nR)
print(" Трапеции:", I_trapz, ", n =", nT)
print(" Симпсон:", I_simp, ", n =", nS)

# Метод Гаусса-Лежандра (5 узлов, таблица из методички по SciPy)
I_gauss5, _ = fixed_quad(f, a, b, n=5)
print(" Гаусс-Лежандр (n=5):", I_gauss5)

# Визуализация ошибок и порядка
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(NsR, np.log(RsR), label='Rectangles')
plt.plot(NsT, np.log(RsT), label='Trapezoids')
plt.plot(NsS, np.log(RsS), label='Simpson')
plt.xlabel('ln n'); plt.ylabel('ln |R_k|')
plt.legend(); plt.title('Сходимость ошибки')
plt.subplot(1,2,2)
plt.plot(NsR[1:], -np.diff(np.log(RsR))/np.diff(NsR), label='Rectangles')
plt.plot(NsT[1:], -np.diff(np.log(RsT))/np.diff(NsT), label='Trapezoids')
plt.plot(NsS[1:], -np.diff(np.log(RsS))/np.diff(NsS), label='Simpson')
plt.xlabel('ln n'); plt.ylabel('gamma_k — порядок')
plt.legend(); plt.title('Вычисляемый порядок метода')
plt.tight_layout(); plt.show()


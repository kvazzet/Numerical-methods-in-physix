import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad, quad
from time import perf_counter

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

a, b = 1, 3.7
eps = 1e-10
I_true, _ = quad(f, a, b)

t0 = perf_counter()
I_rect, nR, RsR, NsR, IsR = adaptive_integrate(rect_rule,  f, a, b, eps, gamma=2)
t_rect = perf_counter() - t0

t0 = perf_counter()
I_trapz, nT, RsT, NsT, IsT = adaptive_integrate(trapz_rule, f, a, b, eps, gamma=2)
t_trap = perf_counter() - t0

t0 = perf_counter()
I_simp, nS, RsS, NsS, IsS = adaptive_integrate(simpson_rule, f, a, b, eps, gamma=4)
t_simp = perf_counter() - t0

n_gauss = 5
t0 = perf_counter()
I_gauss, _ = fixed_quad(f, a, b, n=n_gauss)
t_gauss = perf_counter() - t0

print("Результаты с точностью", eps)
print(f" Центральные прямоугольники: {I_rect:.15f}, n = {nR:7d}, |err| = {abs(I_rect - I_true):.3e}, time = {t_rect:.3e} s")
print(f" Трапеции:                   {I_trapz:.15f}, n = {nT:7d}, |err| = {abs(I_trapz - I_true):.3e}, time = {t_trap:.3e} s")
print(f" Симпсон:                    {I_simp:.15f}, n = {nS:7d}, |err| = {abs(I_simp - I_true):.3e}, time = {t_simp:.3e} s")
print(f" Гаусс–Лежандр :             {I_gauss:.15f}, n =      {n_gauss:2d}, |err| = {abs(I_gauss - I_true):.3e}, time = {t_gauss:.3e} s")

plt.figure(figsize=(12,4))

# 1) ln|R_k| от ln n_k
plt.subplot(1, 2, 1)
plt.plot(NsR, np.log(RsR), 'o-', label='Rectangles')
plt.plot(NsT, np.log(RsT), 'o-', label='Trapezoids')
plt.plot(NsS, np.log(RsS), 'o-', label='Simpson')
plt.xlabel('ln n_k')
plt.ylabel('ln |R_k|')
plt.title('Сходимость ошибки')
plt.legend()
plt.grid(True)

# 2) gamma_k от ln n_k, gamma_k ≈ -Δln|R_k| / Δln n_k

gammaR = -np.diff(np.log(RsR)) / np.diff(NsR)
gammaT = -np.diff(np.log(RsT)) / np.diff(NsT)
gammaS = -np.diff(np.log(RsS)) / np.diff(NsS)

plt.subplot(1, 2, 2)
plt.plot(NsR[1:], gammaR, 'o-', label='Rectangles')
plt.plot(NsT[1:], gammaT, 'o-', label='Trapezoids')
plt.plot(NsS[1:], gammaS, 'o-', label='Simpson')
plt.xlabel('ln n_k')
plt.ylabel(r'$\gamma_k$')
plt.title('Вычисляемый порядок метода')
plt.legend()
plt.grid(True)

# расширяем диапазон по X
all_N = np.array(NsR + NsT + NsS)
plt.xlim(all_N.min() - 0.5, all_N.max() + 0.5)

plt.tight_layout()
plt.show()

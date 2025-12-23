import numpy as np
from scipy.integrate import quad

# Подынтегральная функция
f = lambda x: np.exp(-x**3) * np.sin(x) * np.log(x)

# Ограничиваем низ 0 -> a (например, 1e-8), верх 10 или 20
A, a = 12, 1e-8

# Проверка через scipy (это эталонное значение)
I_scipy, err = quad(f, 0, np.inf, limit=200)
print(f'SciPy result: {I_scipy:.12f}\nabs.err={err}')

# Классический Симпсон на [a, A]
def simpson_rule(f, a, b, n):
    if n % 2:
        n += 1
    h = (b - a) / n
    x = a + h * np.arange(n+1)
    y = f(x)
    return (h/3)*(y[0] + 2*np.sum(y[2:-1:2]) + 4*np.sum(y[1::2]) + y[-1])

# Подбор оптимального n до сходимости
n = 128
I_prev = simpson_rule(f, a, A, n)
while True:
    n *= 2
    I = simpson_rule(f, a, A, n)
    if abs(I - I_prev) < 1e-7:
        break
    I_prev = I
print(f'Simpson [a={a},A={A},n={n}]: {I:.10f}')

# Оценка остатка R = интеграл(A,inf)
def tail_estimate(A):
    # sin(x) осциллирует, но e^{-x^3} быстро падает
    # Грубая оценка: максимум |sin(x)|=1, ln(x) почти ln(A) на этом участке
    return quad(lambda x: np.exp(-x**3)*abs(np.log(x)), A, np.inf, limit=200)[0]
Rtail = tail_estimate(A)
print('Upper tail estimate:', Rtail)

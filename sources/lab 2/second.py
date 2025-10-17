import math

def f(x):
    # 2*ln(x) + sin(ln(x)) - cos(ln(x))
    return 2 * math.log(x) + math.sin(math.log(x)) - math.cos(math.log(x))

def df(x):
    # Производная f(x):
    # 2/x + cos(ln(x))/x + sin(ln(x))/x
    ln_x = math.log(x)
    return (2 + math.cos(ln_x) + math.sin(ln_x)) / x

def bisection(f, a, b, eps=1e-6, max_iter=100):
    n = 0
    while (b - a) / 2 > eps and n < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, n+1
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        n += 1
    return (a + b) / 2, n+1

# Функция-итератор для метода простой итерации (relaxation).
def g(x, lambda_=0.3):
    # x_new = x - lambda_ * f(x)
    return x - lambda_ * f(x)

def simple_iteration(f, x0, eps=1e-6, max_iter=100, lambda_=0.3):
    n = 0
    x = x0
    while n < max_iter:
        x_new = g(x, lambda_)
        if abs(x_new - x) < eps:
            return x_new, n + 1
        x = x_new
        n += 1
    return x, n

def newton(f, df, x0, eps=1e-6, max_iter=100):
    n = 0
    x = x0
    while abs(f(x)) > eps and n < max_iter:
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < eps:
            return x_new, n + 1
        x = x_new
        n += 1
    return x, n

if __name__ == "__main__":
    a, b = 1, 3
    x0 = 2.0
    eps = 1e-6

    root_bis, n_bis = bisection(f, a, b, eps)
    print(f"Бисекция: x = {root_bis:.7f}, итераций: {n_bis}")

    root_iter, n_iter = simple_iteration(f, x0, eps, lambda_=0.3)
    print(f"Простая итерация: x = {root_iter:.7f}, итераций: {n_iter}")

    root_newton, n_newton = newton(f, df, x0, eps)
    print(f"Ньютон: x = {root_newton:.7f}, итераций: {n_newton}")


import math
import time

def f(x):
    # 2*ln(x) + sin(ln(x)) - cos(ln(x))
    return 2 * math.log(x) + math.sin(math.log(x)) - math.cos(math.log(x))

def df(x):
    # Производная f(x):
    # 2/x + cos(ln(x))/x + sin(ln(x))/x
    ln_x = math.log(x)
    return (2 + math.cos(ln_x) + math.sin(ln_x)) / x

def bisection(f, a, b, eps=1e-12, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Метод дихотомии требует fa * fb < 0 на [a, b]")
    n = 0
    left, right = a, b
    while (right - left) / 2 > eps and n < max_iter:
        c = (left + right) / 2
        fc = f(c)
        if fc == 0.0:
            left = right = c
            n += 1
            break
        if fa * fc < 0:
            right = c
            fb = fc
        else:
            left = c
            fa = fc
        n += 1
    x = (left + right) / 2
    err = abs(f(x))
    return x, n, err

def simple_iteration(f, x0, eps=1e-12, max_iter=1000, lambda_=0.3):
    n = 0
    x = x0
    while n < max_iter:
        x_new = x - lambda_ * f(x)
        if abs(x_new - x) < eps:
            x = x_new
            n += 1
            break
        x = x_new
        n += 1
    err = abs(f(x))
    return x, n, err

def newton(f, df, x0, eps=1e-12, max_iter=200):
    n = 0
    x = x0
    while n < max_iter:
        fx = f(x)
        if abs(fx) <= eps:
            break
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("df(x) == 0 в методе Ньютона")
        x_new = x - fx / dfx
        if abs(x_new - x) < eps:
            x = x_new
            n += 1
            break
        x = x_new
        n += 1
    err = abs(f(x))
    return x, n, err

def timed_call(func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    return (*result, t1 - t0)  # (root, iters, err, elapsed_seconds)

if __name__ == "__main__":
    a, b = 1.0, 3.0
    x0 = 2.0
    eps = 1e-12  # ужесточаем допуск

    root_bis, n_bis, err_bis, t_bis = timed_call(bisection, f, a, b, eps=eps)
    print(f"Дихотомия: x = {root_bis:.15f}, невязка = {err_bis:.3e}, итераций = {n_bis}, время = {t_bis*1e3:.3f} мс")

    root_iter, n_iter, err_iter, t_iter = timed_call(simple_iteration, f, x0, eps=eps, lambda_=0.3)
    print(f"Простая итерация: x = {root_iter:.15f}, невязка = {err_iter:.3e}, итераций = {n_iter}, время = {t_iter*1e3:.3f} мс")

    root_newton, n_newton, err_newton, t_newton = timed_call(newton, f, df, x0, eps=eps)
    print(f"Ньютон: x = {root_newton:.15f}, невязка = {err_newton:.3e}, итераций = {n_newton}, время = {t_newton*1e3:.3f} мс")

import numpy as np
import time
import matplotlib.pyplot as plt

# Быстрая реализация SOR для трёхдиагональной матрицы
def sor_tridiag(a, b, c, d, omega, x0=None, tol=1e-8, maxiter=5000):
    n = len(b)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy().astype(float)
    for k in range(1, maxiter+1):
        if n == 1:
            x_new0 = (d[0]) / b[0]
            x[0] = (1-omega)*x[0] + omega * x_new0
        else:
            s = c[0] * x[1]
            x_new0 = (d[0] - s) / b[0]
            x[0] = (1-omega)*x[0] + omega * x_new0
            for i in range(1, n-1):
                s = a[i-1]*x[i-1] + c[i]*x[i+1]
                x_new_i = (d[i] - s) / b[i]
                x[i] = (1-omega)*x[i] + omega * x_new_i
            s = a[-1]*x[-2]
            x_new_n = (d[-1] - s) / b[-1]
            x[-1] = (1-omega)*x[-1] + omega * x_new_n
        Ax = np.empty_like(d)
        if n == 1:
            Ax[0] = b[0]*x[0]
        else:
            Ax[0] = b[0]*x[0] + c[0]*x[1]
            for i in range(1, n-1):
                Ax[i] = a[i-1]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
            Ax[-1] = a[-1]*x[-2] + b[-1]*x[-1]
        relres = np.linalg.norm(d - Ax) / np.linalg.norm(d)
        if relres < tol:
            return x, k, relres
    return x, maxiter, relres

# Генератор SPD трёхдиагонали
def generate_spd_tridiag(n, off=1.0, delta=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.full(n-1, -off, dtype=float)
    c = np.full(n-1, -off, dtype=float)
    b = np.full(n, 2*off + delta, dtype=float)
    return a, b, c

# Эксперимент с выводом в консоль и построением графика
def run_experiment_with_plot(sizes=[200, 500, 1000], omegas=None, tols=None, maxiter=5000):
    if omegas is None:
        omegas = np.linspace(0.6, 1.9, 28)
    if tols is None:
        tols = [1e-4, 1e-8, 1e-12]
    for n in sizes:
        print(f"\nРазмер матрицы n = {n}")
        a, b, c = generate_spd_tridiag(n, off=1.0, delta=1.0)
        x_true = np.arange(n, dtype=float)
        Ax = np.empty(n, dtype=float)
        if n == 1:
            Ax[0] = b[0]*x_true[0]
        else:
            Ax[0] = b[0]*x_true[0] + c[0]*x_true[1]
            for i in range(1, n-1):
                Ax[i] = a[i-1]*x_true[i-1] + b[i]*x_true[i] + c[i]*x_true[i+1]
            Ax[-1] = a[-1]*x_true[-2] + b[-1]*x_true[-1]
        d = Ax.copy()
        for tol in tols:
            print(f"  Точность tol = {tol}")
            iters_list = []
            for omega in omegas:
                x_approx, iters, relres = sor_tridiag(a, b, c, d, omega, tol=tol, maxiter=maxiter)
                iters_list.append(iters)
            iters_arr = np.array(iters_list)
            best_idx = np.argmin(iters_arr)
            best_omega = omegas[best_idx]
            best_iters = iters_arr[best_idx]
            print(f"    Лучшее omega: {best_omega:.3f}, число итераций: {best_iters}")
            # Построение графика
            plt.figure(figsize=(7,4))
            plt.plot(omegas, iters_arr, marker='o')
            plt.xlabel('omega')
            plt.ylabel('Число итераций до сходимости')
            plt.title(f'n={n}, tol={tol}: итерации vs omega')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    sizes = [200, 500, 1000]
    omegas = np.linspace(0.6, 1.9, 28)
    tols = [1e-4, 1e-8, 1e-12]
    run_experiment_with_plot(sizes=sizes, omegas=omegas, tols=tols, maxiter=5000)


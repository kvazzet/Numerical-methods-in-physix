import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- optional: pyamg ---
try:
    from pyamg.gallery import linear_elasticity
    HAVE_PYAMG = True
except Exception:
    HAVE_PYAMG = False

def relres(A, x, b):
    r = b - A.dot(x)
    return float(np.linalg.norm(r) / np.linalg.norm(b))

# Fallback: SPD 2D Poisson (2 DOF)
def generate_spd_block_poisson(nx, ny):
    N = nx * ny
    data, rows, cols = [], [], []

    def idx(i, j): return i * ny + j

    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)
            diag = 0.0
            if i > 0:
                q = idx(i - 1, j); rows.append(p); cols.append(q); data.append(-1.0); diag += 1.0
            if i < nx - 1:
                q = idx(i + 1, j); rows.append(p); cols.append(q); data.append(-1.0); diag += 1.0
            if j > 0:
                q = idx(i, j - 1); rows.append(p); cols.append(q); data.append(-1.0); diag += 1.0
            if j < ny - 1:
                q = idx(i, j + 1); rows.append(p); cols.append(q); data.append(-1.0); diag += 1.0
            rows.append(p); cols.append(p); data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = A + 1e-3 * sp.eye(N, format="csr")
    return sp.block_diag((A, A), format="csr")

# Elasticity system (pyamg or fallback)
def generate_elasticity_system(grid=(30, 30), spacing=(1.0, 1.0), E=1e5, nu=0.3):
    nx, ny = grid
    if HAVE_PYAMG:
        # spacing must be a 2-tuple (dx, dy)
        A, B = linear_elasticity(grid, spacing=spacing, E=E, nu=nu, format="csr")
        A = sp.csr_matrix(A)
    else:
        A = generate_spd_block_poisson(nx, ny)
    n = A.shape[0]
    x_true = np.arange(n, dtype=float)
    b = A.dot(x_true)
    return A, b, x_true

# Iterative methods
def jacobi_sparse(A, b, x0=None, maxiter=2000, tol=1e-8, omega=2/3):
    n = b.size
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    D = A.diagonal()
    if np.any(D == 0):
        raise ValueError("Нулевой элемент на диагонали — Jacobi невозможен")
    invD = 1.0 / D
    normb = np.linalg.norm(b)
    history = []
    last = np.inf
    worsen = 0

    for k in range(maxiter):
        r = b - A.dot(x)
        x = x + omega * (invD * r)
        rel = np.linalg.norm(r) / (normb if normb != 0 else 1.0)
        history.append(rel)

        # критерий сходимости
        if rel < tol:
            return x, history

        # защита от дивергенции/переполнения
        if not np.isfinite(rel):
            print("Jacobi diverged: residual is not finite at iter", k+1)
            return x, history
        worsen = worsen + 1 if rel > last else 0
        last = rel
        if worsen >= 20:  # 20 подряд итераций без улучшения — останавливаем
            print("Jacobi likely diverging: stopping early")
            return x, history

    return x, history


def gauss_seidel_sparse(A, b, x0=None, maxiter=2000, tol=1e-8):
    A_csr = A.tocsr()
    n = b.size
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    history = []
    for _ in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            cols = A_csr.indices[start:end]
            vals = A_csr.data[start:end]
            a_ii = 0.0; s = 0.0
            for jj, col in enumerate(cols):
                if col == i:
                    a_ii = vals[jj]
                else:
                    s += vals[jj] * x[col]
            if a_ii == 0.0:
                raise ValueError(f"Zero diagonal at row {i}")
            x[i] = (b[i] - s) / a_ii
        history.append(relres(A, x, b))
        if np.linalg.norm(x - x_old) < tol:
            return x, history
    return x, history

def sor_sparse(A, b, omega=1.2, x0=None, maxiter=2000, tol=1e-8):
    A_csr = A.tocsr()
    n = b.size
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    history = []
    for _ in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            cols = A_csr.indices[start:end]
            vals = A_csr.data[start:end]
            a_ii = 0.0; s = 0.0
            for jj, col in enumerate(cols):
                if col == i:
                    a_ii = vals[jj]
                else:
                    s += vals[jj] * x[col]
            if a_ii == 0.0:
                raise ValueError(f"Zero diagonal at row {i}")
            x_i = (b[i] - s) / a_ii
            x[i] = (1.0 - omega) * x[i] + omega * x_i
        history.append(relres(A, x, b))
        if np.linalg.norm(x - x_old) < tol:
            return x, history
    return x, history

def compare_methods(grid=(30, 30), spacing=(1.0, 1.0), omega=1.2, maxiter=2000, tol=1e-8):
    print("Генерация системы A...")
    A, b, x_true = generate_elasticity_system(grid=grid, spacing=spacing)
    n = A.shape[0]
    print(f"Размерность системы n = {n}")

    # SciPy >=1.14: use rtol/atol instead of tol
    t0 = time.perf_counter()
    x_cg, info = spla.cg(A, b, rtol=1e-10, atol=0.0, maxiter=2000)
    t_cg = time.perf_counter() - t0
    print(f"CG (scipy) время: {t_cg:.3f} с, info={info}, relres={relres(A, x_cg, b):.2e}")

    results = {}

    t0 = time.perf_counter()
    x_j, hist_j = jacobi_sparse(A, b, maxiter=maxiter, tol=tol)
    t_j = time.perf_counter() - t0
    results["Jacobi"] = (t_j, hist_j)
    print(f"Jacobi: итераций={len(hist_j)}, финальная невязка={hist_j[-1]:.2e}, время={t_j:.3f}s")

    t0 = time.perf_counter()
    x_gs, hist_gs = gauss_seidel_sparse(A, b, maxiter=maxiter, tol=tol)
    t_gs = time.perf_counter() - t0
    results["Gauss-Seidel"] = (t_gs, hist_gs)
    print(f"Gauss-Seidel: итераций={len(hist_gs)}, финальная невязка={hist_gs[-1]:.2e}, время={t_gs:.3f}s")

    t0 = time.perf_counter()
    x_sor, hist_sor = sor_sparse(A, b, omega=omega, maxiter=maxiter, tol=tol)
    t_sor = time.perf_counter() - t0
    results[f"SOR (ω={omega:.2f})"] = (t_sor, hist_sor)
    print(f"SOR ω={omega}: итераций={len(hist_sor)}, финальная невязка={hist_sor[-1]:.2e}, время={t_sor:.3f}s")

    # Plot
    plt.figure(figsize=(8, 5))
    for name, (t, h) in results.items():
        plt.semilogy(np.arange(1, len(h) + 1), h, label=f"{name} ({len(h)} it)")
    plt.xlabel("Итерация")
    plt.ylabel("||Ax - b|| / ||b|| (log-scale)")
    plt.title(f"Сходимость методов, grid={grid}, n={n}")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

if __name__ == "__main__":
    res = compare_methods(grid=(30, 30), spacing=(1.0, 1.0), omega=1.2, maxiter=2000, tol=1e-8)


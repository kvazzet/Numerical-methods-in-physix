# newton_basins_z3_minus_1.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Полином и производная: f(z)=z^3-1, f'(z)=3 z^2
def newton_step(z):
    denom = 3.0 * z * z
    # защитимся от деления на 0 (если z=0 в процессе)
    denom = np.where(denom == 0, 1e-12 + 0j, denom)
    return z - (z**3 - 1.0) / denom

def main():
    # Сетка: -2..2 с шагом 0.004 → 1001×1001 = 1_002_001 точка
    x = np.linspace(-2.0, 2.0, 1001)
    y = np.linspace(-2.0, 2.0, 1001)
    X, Y = np.meshgrid(x, y)              # форма (1001, 1001)
    Z = X + 1j * Y                        # стартовые точки z^(0)

    # Три корня z^3=1
    roots = np.array([
        1.0 + 0j,
        np.exp(2j * np.pi / 3.0),
        np.exp(-2j * np.pi / 3.0),
    ], dtype=np.complex128)

    # 30 итераций Ньютона
    for _ in range(30):
        Z = newton_step(Z)

    # К какому корню ближе после 30-й итерации
    # labels[i,j] ∈ {0,1,2} — индекс ближайшего корня
    dists = np.stack([np.abs(Z - r) for r in roots], axis=2)  # (H,W,3)
    labels = np.argmin(dists, axis=2).astype(np.uint8)

    # Палитра для 3 корней и изображение
    cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.figure(figsize=(7, 7), dpi=120)
    plt.imshow(labels, cmap=cmap, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation="nearest")
    plt.xlabel("Re z")
    plt.ylabel("Im z")
    plt.title("Basins of attraction for z^3 - 1 under Newton's method")
    plt.tight_layout()
    plt.show()
    # при необходимости можно сохранить картинку:
    # plt.savefig("newton_basins_z3_minus_1.png", bbox_inches="tight", dpi=200)

if __name__ == "__main__":
    main()

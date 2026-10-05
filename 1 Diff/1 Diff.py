import numpy as np
import matplotlib.pyplot as plt

x0 = 2.0                       # точка дифференцирования
n = np.arange(1, 22)           # n = 1..21
h = 2.0 / 2.0**n               # h_n = 2 / 2^n

# функции и их аналитические производные
funcs = [
    (r"$\sin(x^2)$",
     lambda x: np.sin(x**2),
     lambda x: 2*x*np.cos(x**2)),
    (r"$\cos(\sin x)$",
     lambda x: np.cos(np.sin(x)),
     lambda x: -np.sin(np.sin(x))*np.cos(x)),
    (r"$\exp(\sin(\cos x))$",
     lambda x: np.exp(np.sin(np.cos(x))),
     lambda x: -np.exp(np.sin(np.cos(x)))*np.cos(np.cos(x))*np.sin(x)),
    (r"$\ln(x+3)$",
     lambda x: np.log(x + 3),
     lambda x: 1/(x + 3)),
    (r"$(x+3)^{0.5}$",
     lambda x: np.sqrt(x + 3),
     lambda x: 0.5/np.sqrt(x + 3)),
]

# численные методы
methods = [
    ("1: правая разность",
     lambda f, x, h: (f(x+h) - f(x))/h),
    ("2: левая разность",
     lambda f, x, h: (f(x) - f(x-h))/h),
    ("3: центральная разность",
     lambda f, x, h: (f(x+h) - f(x-h))/(2*h)),
    ("4: 4-й порядок",
     lambda f, x, h: 4/3*(f(x+h) - f(x-h))/(2*h)
                   - 1/3*(f(x+2*h) - f(x-2*h))/(4*h)),
    ("5: 6-й порядок",
     lambda f, x, h: 3/2*(f(x+h) - f(x-h))/(2*h)
                   - 3/5*(f(x+2*h) - f(x-2*h))/(4*h)
                   + 1/10*(f(x+3*h) - f(x-3*h))/(6*h)),
]

eps = np.finfo(float).eps


def slope(h, err, f0):
    """Наклон p в err ~ C h^p по участку, где доминирует погрешность метода:
    h <= 0.25 (работает разложение Тейлора) и err >> ошибки округления eps*|f|/h."""
    mask = (h <= 0.25) & (err > 100 * eps * abs(f0) / h)
    if mask.sum() < 3:
        return np.nan, 0
    p = np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)[0]
    return p, mask.sum()


for i, (name, f, df) in enumerate(funcs, 1):
    exact = df(x0)
    print(f"\n{name.strip('$')}:  f'(x0) = {exact:.15g}")
    plt.figure(figsize=(8, 6))          # отдельное окно для каждой функции
    for mname, D in methods:
        err = np.abs(D(f, x0, h) - exact)
        p, k = slope(h, err, f(x0))
        print(f"  {mname:<26s} наклон p = {p:6.3f}  (по {k} точкам)")
        plt.plot(np.log10(h), np.log10(err), "o-", ms=4, label=f"{mname}, $p \\approx {p:.2f}$")
    plt.xlabel(r"$\log_{10} h$")
    plt.ylabel(r"$\log_{10}\,\mathrm{err}$")
    plt.title(f"log err (log h):  {name},  $x_0 = {x0}$")
    plt.grid(True, ls=":")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"err_{i}.png", dpi=120)

plt.show()
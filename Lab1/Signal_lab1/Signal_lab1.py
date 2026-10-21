import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

T = np.pi
W0 = 2

def exact_function(x, n=21):
    return n * np.sin(np.pi * n * x)

def get_fourier_coefficients(n, N):
    def calc_a0():
        integrand = lambda x: exact_function(x, n)
        res, _ = spi.quad(integrand, 0, np.pi, limit=1000)
        return (2 / T) * res

    def calc_ak(k):
        integrand = lambda x: exact_function(x, n) * np.cos(W0 * k * x)
        res, _ = spi.quad(integrand, 0, np.pi, limit=1000)
        return (2 / T) * res

    def calc_bk(k):
        integrand = lambda x: exact_function(x, n) * np.sin(W0 * k * x)
        res, _ = spi.quad(integrand, 0, np.pi, limit=1000)
        return (2 / T) * res

    a0 = calc_a0()
    ak = [calc_ak(k) for k in range(1, N + 1)]
    bk = [calc_bk(k) for k in range(1, N + 1)]
    return a0, ak, bk

def fourier_series(x, N, a0, ak_list, bk_list):
    y = np.full_like(x, a0 / 2, dtype=float)
    for k in range(1, N + 1):
        y += ak_list[k - 1] * np.cos(W0 * k * x) + bk_list[k - 1] * np.sin(W0 * k * x)
    return y

def print_and_save_results(filename, n, N, a0, ak, bk, error):
    error_pct = error * 100
    output = []
    
    output.append(f"Лабораторна робота №1. Варіант {n} (n = {n})")
    output.append(f"Функція: f(x) = {n} * sin({n} * pi * x), x in [0, pi]")
    output.append(f"Порядок наближення N: {N}")
    output.append(f"Постійна складова a0: {a0:+.6f}\n")
    output.append(f"{'k':<4} {'ak (косинусний)':<21} {'bk (синусний)'}")
    output.append("-" * 45)
    
    for k in range(N):
        output.append(f"{k+1:<4} {ak[k]:<21.6f} {bk[k]:.6f}")
        
    output.append(f"\nВідносна похибка наближення (N={N}): {error_pct:.2f}%\n")
    output.append(f"Результати збережено у файл '{filename}'")
    
    full_text = "\n".join(output)
    print(full_text)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_text)

def plot_results(x_vals, y_exact, y_approx, ak, bk, error, n, N):
    plt.figure(figsize=(10, 10))
    error_pct = error * 100
    
    plt.subplot(3, 1, 1)
    plt.xlim(0, 0.5) 
    plt.plot(x_vals, y_exact, label=f"f(x)={n}sin({n}πx)", color="lightblue", linewidth=2)
    plt.plot(x_vals, y_approx, label=f"Сума Фур'є (N={N})", color="red", linestyle="--", linewidth=2)
    plt.title(f"Наближення функції (масштаб x: 0..0.5). Похибка = {error_pct:.2f}%")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=8)
    
    plt.subplot(3, 1, 2)
    plt.xlim(0, 1) 
    plt.plot(x_vals, y_approx, label=f"Сума Фур'є (N={N})", color="red", linestyle="--", linewidth=2.5, zorder=10)
    for k in range(1, N + 1): 
        harmonic = ak[k - 1] * np.cos(W0 * k * x_vals) + bk[k - 1] * np.sin(W0 * k * x_vals)
        plt.plot(x_vals, harmonic, label=f"k={k}", alpha=0.7)
        
    plt.title(f"Окремі гармоніки та сума ряду (масштаб x: 0..1)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=8, ncol=6) 
    
    plt.subplot(3, 1, 3)
    plt.ylim(0, 1) 
    k_vals = np.arange(1, N + 1)
    plt.stem(k_vals - 0.1, ak, linefmt='b-', markerfmt='bo', basefmt='k-', label='a_k (косинуси)')
    plt.stem(k_vals + 0.1, bk, linefmt='r-', markerfmt='ro', basefmt='k-', label='b_k (синуси)')
    plt.title(f"Спектр коефіцієнтів Фур'є (масштаб y: 0..1)")
    plt.xlabel("Номер гармоніки k")
    plt.ylabel("Амплітуда")
    plt.xticks(k_vals)
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.show()

def main():
    n = 21
    N = 33
    filename = f"fourier_results_var{n}.txt"
    
    a0, ak, bk = get_fourier_coefficients(n, N)
    
    x_vals = np.linspace(0, np.pi, 2000)
    y_exact = exact_function(x_vals, n)
    y_approx = fourier_series(x_vals, N, a0, ak, bk)
    
    norm_exact = np.linalg.norm(y_exact)
    error = np.linalg.norm(y_exact - y_approx) / norm_exact if norm_exact != 0 else 0.0
    
    print_and_save_results(filename, n, N, a0, ak, bk, error)
    plot_results(x_vals, y_exact, y_approx, ak, bk, error, n, N)

if __name__ == "__main__":
    main()
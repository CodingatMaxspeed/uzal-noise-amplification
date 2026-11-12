"""Figure 5: what 10 % observational noise does, Sec. V A and Fig. 13(a).

Without noise the Mackey-Glass optimum is a low-dimensional delay
reconstruction, m = 4 at t_w = 30. Sec. V A reports that this stops being true
once the series is noisy:

    "the higher the number of delayed components considered inside the
     selected time window, the lower the value of L_k ... the profiles of L_k
     vs t_w do not reach their minimum for a small dimension m, and the
     minimum of L_k corresponds to the case of including every possible
     delayed value inside the time window"

that is, the optimum moves onto the fDC line. The two panels here are the same
computation on the same series, once clean and once with noise, so the switch
is visible rather than asserted.

This matters beyond reproducing a figure. Real measurements are noisy, and the
low-dimensional answer that the noise-free benchmarks give does not survive
contact with them: the reconstruction that minimises L_k is then the one that
keeps every sample in the window. Anything built on top of this estimator has
to know which regime it is in.

Panel (b) of the original Fig. 13, Legendre coordinates, is not reproduced --
that transformation does not exist here yet. It is the reason the paper can
recover a low-dimensional optimum (n = 4, t_w = 113) even under noise.

Noise: i.i.d. Gaussian with standard deviation 10 % of the series' own, added
after normalisation, with a fixed seed. The paper says "10 % (amplitude)"
without further detail.

Takes several minutes. Run: python demo/fig5_noise_moves_to_fdc.py
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from uzal.sweep import cost_at, separation_horizon
from uzal.systems import mackey_glass

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
DIMS = [2, 3, 4, 5, 11, 21]  # the dimension set of Fig. 13(a)
T_W_MAX, DENSE_TO, N_SPARSE = 1000, 100, 16
K, NOISE, SEED = 3, 0.10, 0


def window_grid(step):
    """Every admissible window up to DENSE_TO, log-spaced beyond it."""
    dense = np.arange(step, DENSE_TO + 1, step)
    sparse = np.geomspace(DENSE_TO, T_W_MAX, N_SPARSE)
    sparse = np.maximum(1, np.round(sparse / step)).astype(int) * step
    return np.unique(np.concatenate([dense, sparse]))


def profile(ax, x, title):
    """One panel: L_k vs t_w for every m in DIMS, plus the fDC line."""
    T_M, _, _ = separation_horizon(x, max_T=300)
    best = (None, None, np.inf)

    for m in DIMS:
        t_ws = window_grid(m - 1)
        ls = np.array([cost_at(x, m, t // (m - 1), T_M, k=K, N=None).L for t in t_ws])
        ax.plot(t_ws, ls, "-", lw=1.4, label=f"$m={m}$")
        i = int(np.argmin(ls))
        if ls[i] < best[2]:
            best = (f"m={m}", int(t_ws[i]), float(ls[i]))
        print(f"    m={m:>2}: min L={ls[i]:+.4f} at t_w={t_ws[i]:>4}", flush=True)

    t_ws = window_grid(1)
    ls = np.array([cost_at(x, t + 1, 1, T_M, k=K, N=None).L for t in t_ws])
    ax.plot(t_ws, ls, "--", lw=3.0, color="0.45", zorder=1, label="fDC")
    i = int(np.argmin(ls))
    if ls[i] < best[2]:
        best = ("fDC", int(t_ws[i]), float(ls[i]))
    print(f"    fDC : min L={ls[i]:+.4f} at t_w={t_ws[i]:>4} (m={t_ws[i] + 1})", flush=True)

    ax.scatter([best[1]], [best[2]], s=190, facecolor="none", edgecolor="k", lw=1.8,
               zorder=6)
    ax.annotate(rf"{best[0]}, $t_w={best[1]}$" "\n" rf"$L_k={best[2]:.3f}$",
                xy=(best[1], best[2]), xytext=(0.05, 0.10),
                textcoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xscale("log")
    ax.set_xlim(1, T_W_MAX)
    ax.set_xlabel(r"$t_w$")
    ax.set_title(rf"{title}   ($T_M={T_M}$)", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    return best


def main():
    FIGURES.mkdir(exist_ok=True)
    raw = mackey_glass()
    clean = raw / raw.std()
    noisy = clean + NOISE * np.random.default_rng(SEED).standard_normal(len(clean))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)

    print("(a) noise free:", flush=True)
    t0 = time.perf_counter()
    best_clean = profile(ax_a, clean, "(a)  noise free")
    print(f"    [{time.perf_counter() - t0:.0f}s]")

    print(f"({chr(98)}) with {NOISE:.0%} noise:", flush=True)
    t0 = time.perf_counter()
    best_noisy = profile(ax_b, noisy, rf"(b)  ${NOISE:.0%}$ i.i.d. noise".replace("%", r"\%"))
    print(f"    [{time.perf_counter() - t0:.0f}s]")

    ax_a.set_ylabel(r"$L_k$")
    ax_a.legend(fontsize=9, loc="upper right", framealpha=0.9, ncols=2)
    fig.suptitle(r"Mackey-Glass: noise moves the optimum off the low-dimensional "
                 r"reconstruction onto fDC", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = FIGURES / "fig5_noise_moves_to_fdc.png"
    fig.savefig(out, dpi=170)

    print(f"\nwrote {out}")
    print(f"  noise free : {best_clean[0]}, t_w={best_clean[1]}, L={best_clean[2]:.4f}"
          "   (paper: m=4, t_w=30)")
    print(f"  with noise : {best_noisy[0]}, t_w={best_noisy[1]}, L={best_noisy[2]:.4f}"
          "   (paper Sec. V A: fDC)")


if __name__ == "__main__":
    main()

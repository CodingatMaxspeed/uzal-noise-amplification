"""Figure 4: a reproduction of Fig. 1 of the paper.

Fig. 1 profiles L_k over the embedding window t_w for the noise-free
Mackey-Glass series, one curve per embedding dimension m, plus the full delay
coordinate reconstruction as a thick dashed line. Its caption:

    "Profiles of the proposed cost function as a function of the dimension m
     and time window t_w for the Mackey-Glass time series. The thick dashed
     line corresponds to the case of using all delay coordinates within the
     time window t_w (i.e., using tau = 1 and m = t_w + 1). The cost function
     has an absolute minimum at m = 4, t_w = 30."

The fDC line is the reason Sec. II C can call any reconstruction from the
interval [t - t_w, t] a transformation of one object: it contains every sample
in the window, so PCA, Legendre coordinates and ordinary delay coordinates are
all projections of it. Here it serves as the lower envelope one gets by
spending dimensions freely, against which the low-dimensional curves are read.

Choices the paper leaves open, made explicit:

  * Normalisation. The series is divided by its standard deviation, for the
    reasons set out in demo/fig3_paper_fig7.py: alpha_k * sigma_bar_k carries
    the units of x, and the published axes only make sense normalised. The
    shift is a constant and moves no minimum.
  * k, N and the Theiler window are not stated anywhere in the paper. Here
    k = 3 (Sec. IV A suggests 2 or 3), w = t_w, and T_M comes from the
    space-time separation plot.
  * No reference points are sampled -- every valid one is used, as in the
    other figures, so the curves carry no seed. The cost is one expensive
    point: at t_w = 1000 the fDC vector has m = 1001 components and that
    single evaluation peaks around 1.6 GB. Sampling N = 2000 with a seed
    brings it to 0.4 GB and moves L by less than 0.01 over most of the range,
    but it also shifts the m = 4 minimum from the published t_w = 30 to 27,
    so the deterministic route is worth the memory.

Takes several minutes, most of it in the fDC line at large t_w.
Run: python demo/fig4_paper_fig1.py
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from uzal.sweep import cost_at, separation_horizon
from uzal.systems import mackey_glass

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
DIMS = [2, 3, 4, 5, 6]
T_W_MIN, T_W_MAX = 1, 1000
DENSE_TO, N_SPARSE = 100, 16
K, N_REF, SEED = 3, None, None  # N=None: every valid reference point
PAPER = (4, 30)  # m, t_w -- the absolute minimum named in the caption


def window_grid(step, dense_to=DENSE_TO, hi=T_W_MAX, n_sparse=N_SPARSE):
    """Every admissible window up to dense_to, log-spaced beyond it.

    A purely log-spaced grid is too coarse where it matters: for m = 4 it
    steps over t_w = 30, the published minimum, and settles on 27 with a
    slightly larger L. The structure lives below t_w = 100, so that range is
    scanned exhaustively and the long flat tail is sampled logarithmically to
    fill out the decade the published axis shows.
    """
    dense = np.arange(step, dense_to + 1, step)
    sparse = np.geomspace(dense_to, hi, n_sparse)
    sparse = np.maximum(1, np.round(sparse / step)).astype(int) * step
    return np.unique(np.concatenate([dense, sparse]))


def main():
    FIGURES.mkdir(exist_ok=True)
    raw = mackey_glass()
    x = raw / raw.std()
    T_M, _, _ = separation_horizon(x, max_T=300)
    print(f"Mackey-Glass: std(raw)={raw.std():.4f}, shift={-np.log10(raw.std()):+.3f}, "
          f"T_M={T_M}, k={K}, N={N_REF or 'all'}")

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    best = (None, None, np.inf)

    for m in DIMS:
        t0 = time.perf_counter()
        t_ws = window_grid(m - 1)
        ls = np.array([cost_at(x, m, t_w // (m - 1), T_M, k=K, N=N_REF, seed=SEED).L
                       for t_w in t_ws])
        ax.plot(t_ws, ls, "-", lw=1.5, label=f"$m={m}$")
        i = int(np.argmin(ls))
        if ls[i] < best[2]:
            best = (m, int(t_ws[i]), float(ls[i]))
        print(f"  m={m}: min L={ls[i]:+.4f} at t_w={t_ws[i]:>4} "
              f"(tau={t_ws[i] // (m - 1):>4}), {len(t_ws)} points "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)

    # the full delay coordinate line: tau = 1, m = t_w + 1
    t0 = time.perf_counter()
    t_ws = window_grid(1)
    ls = np.array([cost_at(x, t_w + 1, 1, T_M, k=K, N=N_REF, seed=SEED).L
                   for t_w in t_ws])
    ax.plot(t_ws, ls, "--", lw=3.0, color="0.45", zorder=1,
            label=r"fDC  ($\tau=1$, $m=t_w+1$)")
    i = int(np.argmin(ls))
    print(f"  fDC: min L={ls[i]:+.4f} at t_w={t_ws[i]:>4} (m={t_ws[i] + 1}), "
          f"{len(t_ws)} points [{time.perf_counter() - t0:.0f}s]", flush=True)

    ax.scatter([best[1]], [best[2]], s=190, facecolor="none", edgecolor="k",
               lw=1.8, zorder=6)
    ax.annotate(rf"$m={best[0]}$, $t_w={best[1]}$" "\n" rf"$L_k={best[2]:.3f}$",
                xy=(best[1], best[2]), xytext=(0.05, 0.10),
                textcoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3,
                                connectionstyle="arc3,rad=0.25"))

    ax.set_xscale("log")
    ax.set_xlim(1, T_W_MAX)
    ax.set_ylim(-2.05, 0.05)
    ax.set_xlabel(r"$t_w$")
    ax.set_ylabel(r"$L_k$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncols=2)
    ax.set_title(r"Reproduction of Fig. 1: Mackey-Glass, $L_k$ vs $t_w$"
                 "\n"
                 rf"$k={K}$, $T_M={T_M}$, all reference points, series normalised to unit variance",
                 fontsize=12)
    fig.tight_layout()
    out = FIGURES / "fig4_paper_fig1.png"
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")
    print(f"  absolute minimum : m={best[0]}, t_w={best[1]}, L={best[2]:.4f}")
    print(f"  paper caption    : m={PAPER[0]}, t_w={PAPER[1]}")


if __name__ == "__main__":
    main()

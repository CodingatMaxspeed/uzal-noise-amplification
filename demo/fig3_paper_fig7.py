"""Figure 3: a reproduction of Fig. 7 of the paper, panels (a) to (c).

Fig. 7 shows L_k against the embedding window t_w for k = 1 ... 5 at a fixed
embedding dimension, for three benchmark series:

    (a) Lorenz,       m = 3
    (b) Roessler,     m = 3
    (c) Mackey-Glass, m = 4

Its point is the sentence in the caption: "for all time series studied in this
work the structure of maxima and minima converges for low values of k". The
curves shift vertically with k but the position of the minimum does not move,
which is why Sec. IV A can recommend k = 2 or 3 for general use.

Panel (d) of the original, Mackey-Glass with 10 % noise, is left out: the noisy
case needs the fDC and Legendre machinery of Sec. V, which does not exist here
yet.

Two choices the paper does not pin down, made visible rather than hidden:

  * Normalisation. alpha_k * sigma_bar_k carries the units of x, so L_k moves
    by log10(c) when the series is scaled by c. The paper never states a
    normalisation for L_k -- the only one it mentions is for Eq. (5) -- but on
    the raw series the Lorenz curves run above zero, while Fig. 7(a) sits
    entirely below it. Dividing each series by its standard deviation puts all
    three panels on the published scale. The shift is a constant per panel and
    moves no minimum; both values are printed.
  * T_M and the Theiler window. T_M comes from the space-time separation plot
    as in Sec. IV C; the Theiler window is w = t_w. No reference points are
    sampled, every valid one is used, so the curves carry no seed.

Takes a few minutes. Run: python demo/fig3_paper_fig7.py
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from uzal.sweep import cost_sweep, separation_horizon
from uzal.systems import lorenz, mackey_glass, roessler

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
KS = [1, 2, 3, 4, 5]
T_W_MAX = 100
NORMALISE = True  # divide each series by its standard deviation

PANELS = [
    ("(a)", "Lorenz", lorenz, 3, (-1.65, 0.05)),
    ("(b)", "Rössler", roessler, 3, (-3.05, 0.05)),
    ("(c)", "Mackey-Glass", mackey_glass, 4, (-2.55, 0.05)),
]


def main():
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), sharex=True)

    for ax, (tag, name, gen, m, ylim) in zip(axes, PANELS):
        t0 = time.perf_counter()
        raw = gen()
        scale = raw.std() if NORMALISE else 1.0
        x = raw / scale
        T_M, _, _ = separation_horizon(x, max_T=300)

        print(f"\n{tag} {name}: m={m}, T_M={T_M}, std(raw)={raw.std():.4f}, "
              f"shift={-np.log10(raw.std()):+.3f}")

        lo, hi = np.inf, -np.inf
        for k in KS:
            rec = cost_sweep(x, dims=[m], t_w_max=T_W_MAX, T_M=T_M, k=k, N=None)[m]
            ax.plot(rec["t_w"], rec["L"], "-", lw=1.5, label=f"$k={k}$")
            i = int(np.argmin(rec["L"]))
            lo, hi = min(lo, rec["L"].min()), max(hi, rec["L"].max())
            print(f"    k={k}: min L={rec['L'][i]:+.4f} at t_w={rec['t_w'][i]:>3} "
                  f"(tau={rec['tau'][i]:>2}), range [{rec['L'].min():+.3f}, "
                  f"{rec['L'].max():+.3f}], {len(rec)} points")

        if lo < ylim[0] or hi > ylim[1]:
            print(f"    NOTE: data [{lo:.2f}, {hi:.2f}] exceeds ylim {ylim}")

        ax.set_xlim(0, T_W_MAX)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_xlabel(r"$t_w$")
        ax.set_title(rf"{tag} {name},  $m={m}$", fontsize=11)
        ax.grid(alpha=0.2, which="both")
        ax.text(0.97, 0.05, rf"$T_M={T_M}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="0.4")
        print(f"    [{time.perf_counter() - t0:.0f}s]")

    axes[0].set_ylabel(r"$L_k$")
    axes[0].legend(fontsize=9, loc="upper left", framealpha=0.9)

    sub = "series normalised to unit variance" if NORMALISE else "raw series"
    fig.suptitle(rf"Reproduction of Fig. 7(a)–(c): $L_k$ vs $t_w$ for "
                 rf"$k=1\ldots5$   ({sub})", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out = FIGURES / "fig3_paper_fig7.png"
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Figure 2: the cost function on the Lorenz series, against the window size.

Panel (a) is the result the paper reports in Appendix A: scanning L_k over the
embedding window t_w for several embedding dimensions m, the global minimum
sits at m = 3, tau = 6, that is t_w = 12.

Panel (b) takes the m = 3 curve apart into the two terms of Eq. (22),
L_k = R_k + I_k with R_k = log10(sigma_bar_k) and I_k = log10(alpha_k). Over
the range that decides the optimum, t_w up to about 30, R_k falls in 93 % of
its steps -- more window means less redundance -- while I_k rises in every
one of them, irrelevance stretching the attractor. Neither is monotone over
the full range plotted, but neither has a minimum near t_w = 12 either: only
their sum turns there, and that is the whole construction.

The horizon T_M is not guessed: it comes from the space-time separation plot
of the raw series, the criterion the paper itself uses. All valid reference
points are used, so the curves carry no sampling noise and no seed.

Takes about a minute. Run: python demo/fig2_lorenz_sweep.py
"""

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from uzal.sweep import cost_sweep, separation_horizon, sweep_minimum
from uzal.systems import lorenz

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
DIMS = [2, 3, 4, 5, 6]
T_W_MAX, K = 100, 3
PAPER = (3, 6, 12)  # m, tau, t_w -- Appendix A


def main():
    FIGURES.mkdir(exist_ok=True)
    x = lorenz()
    T_M, _, _ = separation_horizon(x, max_T=300)
    print(f"T_M from the separation plot: {T_M} sampling steps")

    t0 = time.perf_counter()
    sweep = cost_sweep(x, dims=DIMS, t_w_max=T_W_MAX, T_M=T_M, k=K, N=None)
    print(f"swept {sum(len(r) for r in sweep.values())} reconstructions "
          f"in {time.perf_counter() - t0:.0f}s")

    m_best, t_w_best, tau_best, L_best = sweep_minimum(sweep)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # ---- (a) L against t_w, one curve per m -----------------------------
    for m in DIMS:
        rec = sweep[m]
        ax_a.plot(rec["t_w"], rec["L"], "-o", ms=3.2, lw=1.4, label=f"$m={m}$")
    ax_a.scatter([t_w_best], [L_best], s=170, facecolor="none", edgecolor="k",
                 lw=1.8, zorder=5)
    ax_a.annotate(
        rf"$m={m_best}$, $\tau={tau_best}$, $t_w={t_w_best}$"
        "\n"
        rf"$L_k={L_best:.3f}$",
        xy=(t_w_best, L_best), xytext=(0.30, 0.13), textcoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.3,
                        connectionstyle="arc3,rad=0.25"),
    )
    ax_a.set_xlabel(r"embedding window $t_w$  [sampling steps]")
    ax_a.set_ylabel(r"$L_k$")
    ax_a.set_title(rf"(a)  Lorenz, $k={K}$, $T_M={T_M}$, all reference points")
    ax_a.legend(ncols=2, fontsize=9, framealpha=0.9)
    ax_a.grid(alpha=0.25)
    ax_a.set_xlim(0, T_W_MAX)

    # ---- (b) the two competing terms, Eq. (22) --------------------------
    rec = sweep[PAPER[0]]
    ax_b.plot(rec["t_w"], rec["R"], "-o", ms=3.2, lw=1.4, color="C3",
              label=r"$R_k=\log_{10}\bar\sigma_k$   (redundance)")
    ax_b.plot(rec["t_w"], rec["I"], "-o", ms=3.2, lw=1.4, color="C2",
              label=r"$I_k=\log_{10}\alpha_k$   (irrelevance)")
    ax_b.plot(rec["t_w"], rec["L"], "-o", ms=3.6, lw=2.0, color="C0",
              label=r"$L_k=R_k+I_k$")
    i = int(np.argmin(rec["L"]))
    ax_b.axvline(rec["t_w"][i], color="0.6", ls="--", lw=1.1)
    ax_b.text(rec["t_w"][i] + 2, ax_b.get_ylim()[1], rf"$t_w={rec['t_w'][i]}$",
              va="top", fontsize=9.5, color="0.4")
    ax_b.set_xlabel(r"embedding window $t_w$  [sampling steps]")
    ax_b.set_ylabel("term")
    ax_b.set_title(r"(b)  $m=3$: the two competing terms of Eq. (3.40)")
    ax_b.legend(fontsize=9, framealpha=0.9)
    ax_b.grid(alpha=0.25)
    ax_b.set_xlim(0, T_W_MAX)

    # No figure-level title: the panels carry (a) and (b), and what the
    # suptitle used to say belongs in the thesis caption, not in the image.
    fig.tight_layout()
    out = FIGURES / "fig2_lorenz_sweep.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    print(f"\n  global minimum : m={m_best}, tau={tau_best}, t_w={t_w_best}, "
          f"L={L_best:.4f}")
    print(f"  paper          : m={PAPER[0]}, tau={PAPER[1]}, t_w={PAPER[2]}")
    print("  best t_w per m :", "  ".join(
        f"m={m}:{int(r['t_w'][np.argmin(r['L'])])}" for m, r in sweep.items()))


if __name__ == "__main__":
    main()

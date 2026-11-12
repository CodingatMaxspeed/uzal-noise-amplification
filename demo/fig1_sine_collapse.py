"""Figure 1: what the cost function measures, on a sampled sine.

x[n] = sin(2 pi n / 12), twelve samples per period.

  m = 2, tau = 3  ->  x-bar(t) = (sin th, -cos th), the unit circle
  m = 2, tau = 6  ->  x-bar(t) = (x[t], -x[t]), a collapsed line

A strictly periodic series repeats exactly, x-bar(t) = x-bar(t + 12), and a
Theiler window of w = t_w does not exclude a separation of 12. So the
neighbourhood radius eps_k of Eq. (16) collapses in *both* panels. The
difference is what the numerator does meanwhile:

  circle:  coincident points also share their future, E_k -> 0 as well.
           sigma_k = E_k / eps_k is 0/0 -- they are the same state.
  line:    coincident points have different futures, E_k stays finite.
           sigma_k = finite / 0 -- two states share one place, and the
           estimator diverges. That is the failure Eq. (17) is built to see.

x-bar(8) and x-bar(10) are the marked pair: the same point of the plane,
with futures x[11] = -0.5 and x[13] = +0.5 three steps later.

Run: python demo/fig1_sine_collapse.py
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from uzal.estimator import E2, delay_coordinate_embedding, eps2
from uzal.estimator import nearest_neighbour_theiler as neighbours

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
PERIOD, N_PERIODS, T_M, K = 12, 8, 3, 1
ZERO = 1e-25  # below this a squared radius is numerically zero


def diagnose(x, tau):
    """eps^2 and E^2 over the reconstruction, with the default Theiler window."""
    X, times = delay_coordinate_embedding(x, m=2, tau=tau)
    U, _ = neighbours(X, times, n_samples=len(x), k=K, w=tau, T_M=T_M)
    return X, times, eps2(X, U), E2(x, times, U, T_M)


def panel(ax, X, tau, title):
    ax.axhline(0, color="0.92", lw=0.8, zorder=0)
    ax.axvline(0, color="0.92", lw=0.8, zorder=0)
    ax.plot(X[:, 0], X[:, 1], "-", color="0.82", lw=1.0, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], s=70, facecolor="white", edgecolor="C0",
               linewidth=1.7, zorder=3)
    ax.set_xlabel(r"$x[t]$")
    ax.set_ylabel(rf"$x[t-{tau}]$")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)


def main():
    FIGURES.mkdir(exist_ok=True)
    x = np.sin(2 * np.pi * np.arange(PERIOD * N_PERIODS) / PERIOD)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.8))

    # ---- (a) the injective reconstruction -------------------------------
    Xa, _, eps_a, E_a = diagnose(x, 3)
    panel(ax_a, Xa, 3, r"(a)  $m=2$, $\tau=3$:  the unit circle")
    deg_a = eps_a <= ZERO
    ax_a.text(0.035, 0.035,
              rf"$\varepsilon_k^2 \approx 0$ for all {eps_a.size} neighbourhoods,"
              "\n"
              rf"but $E_k^2 \leq 10^{{{int(np.floor(np.log10(E_a[deg_a].max())))}}}$"
              r" there as well"
              "\n"
              r"$\Rightarrow$ same place, same future",
              transform=ax_a.transAxes, ha="left", va="bottom", fontsize=9.5,
              color="0.25",
              bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.85"))

    # ---- (b) the collapsed reconstruction -------------------------------
    Xb, times_b, eps_b, E_b = diagnose(x, 6)
    panel(ax_b, Xb, 6, r"(b)  $m=2$, $\tau=6$:  collapsed onto a line")
    deg_b = eps_b <= ZERO

    p = Xb[times_b == 8][0]
    ax_b.scatter(*p, s=300, facecolor="none", edgecolor="C3", lw=2.3, zorder=4)
    ax_b.annotate(
        rf"$\bar x(8) = \bar x(10) = ({p[0]:.3f},\ {p[1]:.3f})$"
        "\n"
        rf"$x[11] = {x[11]:+.1f}$   vs   $x[13] = {x[13]:+.1f}$",
        xy=p, xytext=(0.97, 0.97), textcoords="axes fraction",
        ha="right", va="top", fontsize=9.5, color="C3",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="C3", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="C3", lw=1.5,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax_b.text(0.035, 0.035,
              rf"$\varepsilon_k^2 \approx 0$ for all {eps_b.size} "
              rf"({int((eps_b == 0).sum())} exactly zero),"
              "\n"
              rf"but $E_k^2$ reaches ${E_b[deg_b].max():.2f}$ there"
              "\n"
              r"$\Rightarrow$ $\sigma_k = E_k/\varepsilon_k$ diverges",
              transform=ax_b.transAxes, ha="left", va="bottom", fontsize=9.5,
              color="C3",
              bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="C3", alpha=0.95))

    fig.suptitle(
        r"$x[n]=\sin(2\pi n/12)$: $\varepsilon_k$ collapses in both windows, "
        r"only $E_k$ tells them apart",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = FIGURES / "fig1_sine_collapse.png"
    fig.savefig(out, dpi=160)

    print(f"wrote {out}")
    for tau, eps, E, deg in [(3, eps_a, E_a, deg_a), (6, eps_b, E_b, deg_b)]:
        print(f"  tau={tau}: eps^2 numerically zero for {deg.sum():>2}/{eps.size},"
              f"  max E^2 there = {E[deg].max():.3e}")


if __name__ == "__main__":
    main()

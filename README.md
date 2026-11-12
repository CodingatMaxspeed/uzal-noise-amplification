# Uzal cost function

Implementation of the state space reconstruction cost function of

> Uzal, L. C., Grinblat, G. L. & Verdes, P. F. (2011).
> *Optimal reconstruction of dynamical systems: A noise amplification approach.*
> Physical Review E **84**, 016223. https://doi.org/10.1103/PhysRevE.84.016223

`L_k` scores a reconstruction of a dynamical system from a scalar time series,
so that the embedding dimension and the delay are read off its minimum instead
of being chosen by eye. This repository implements it equation by equation,
pins each equation with its own test, and reproduces the optima and figures the
paper publishes.

The library needs **numpy alone**. matplotlib is used only by the scripts under
`demo/`.

## The method

Given a scalar series `x`, a delay reconstruction uses the vectors
`x̄(t) = (x[t], x[t-τ], …, x[t-(m-1)τ])` with embedding window `t_w = (m-1)·τ`.
The cost function scores such a reconstruction, so that the optimal `m` and `τ`
are read off its minimum.

| Eq. | Object | Meaning |
|-----|--------|---------|
| (13) | `E²_k(T,x̄)` | variance of the future values of the raw observable over `U_k(x̄)`, divisor `k+1` |
| (15) | `E²_k(x̄)` | (13) averaged over horizons `T_j ∈ [0, T_M]` |
| (16) | `ε²_k(x̄)` | mean squared pairwise distance inside `U_k(x̄)` |
| (17) | `σ²_k(x̄) = E²_k/ε²_k` | the noise amplification estimator |
| (18) | `σ̄²_k` | mean over `N` reference points |
| (21) | `α²_k` | normalisation against irrelevance |
| (26) | `L_k = log₁₀(α_k σ̄_k)` | the cost function |

The asymmetry between numerator and denominator is what makes the method general:
`ε_k` is measured in the **reconstruction space**, `E_k` on the **raw time series**,
via the time stamp of each neighbour rather than any component of its vector. A
transformation of the reconstruction therefore changes only the denominator, so
`L_k` can be evaluated on any point cloud — delay coordinates, PCA, Legendre.

## What is here

`src/uzal/estimator.py` — the method:

- `delay_coordinate_embedding` — delay vectors **and** their time stamps
- `nearest_neighbour_theiler` — `U_k(x̄)` by BLAS-backed brute force, Theiler window
- `eps2` — Eq. (16), the denominator, in the reconstruction space
- `E2` — Eqs. (13)–(15), the numerator, on the raw series
- `cost_function` / `Cost` — Eqs. (17), (18), (21), (23)–(26)

`src/uzal/systems.py` — Lorenz, Rössler and Mackey-Glass per Appendices B–D.
`src/uzal/sweep.py` — `T_M` from the space-time separation plot, and the
`(m, t_w)` scan. `tests/` — 148 tests, one per equation-level component plus
the reference results below. `demo/` — six figure scripts.

Not implemented: the Legendre and PCA coordinates of Sec. V B and V C. `L_k`
itself is already coordinate-free — `cost_function` takes any point cloud with
time stamps, see `tests/test_cost_of_cloud.py` — so what is missing is the
construction of those coordinates, not the scoring of them.

## Setup

Requires Python ≥ 3.11.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"             # numpy and pytest
pytest                              # 148 tests, about 55 s
```

To run the figures as well:

```sh
pip install -e ".[dev,demo]"        # adds matplotlib
python demo/fig1_sine_collapse.py   # instant
python demo/fig2_lorenz_sweep.py    # about a minute
python demo/fig3_paper_fig7.py      # reproduces Fig. 7(a)-(c)
```

Figures are written to `figures/`, which is not tracked.

## Using it

Two inputs: a scalar series and the horizon `T_M` of Eq. (15), the latter read
off the series rather than guessed. Everything is in sampling steps, not in
time units.

```python
from uzal.sweep import separation_horizon, cost_sweep, sweep_minimum

T_M, _, _ = separation_horizon(x, max_T=300)   # 74 for the Lorenz series
sweep = cost_sweep(x, dims=[2, 3, 4, 5], t_w_max=40, T_M=T_M, N=None)
m, t_w, tau, L = sweep_minimum(sweep)          # (3, 12, 6, -0.9614)
```

`cost_at(x, m, tau, T_M)` scores a single reconstruction, and
`cost_of_cloud(x, X, times, T_M)` an arbitrary point cloud: `eps_k` reads the
cloud while `E_k` reads the raw series through the time stamps, so a
non-uniform lag set (`uzal.embedding.lag_embedding`), PCA or Legendre
coordinates are all scored by the same call. Both return
`Cost(L, R, I, sigma_bar, alpha)` with `L = R + I`, and the minimum over the
grid is the answer.

Two things that bite. The Theiler window is not optional -- without it the
nearest neighbours of a delay vector are its own predecessors, `eps_k -> 0`
and the estimator collapses; the default is `w = t_w`. And `N=None` uses every
valid reference point instead of sampling 1000 of them: slower, but the curves
then carry no sampling noise and no seed.

## Deviations

Every point at which this implementation departs from, or fills a gap in, the
paper is listed in [DEVIATIONS.md](DEVIATIONS.md), with what the source states,
what is done here, and what it changes. The Theiler window is the one to read
first: the paper prescribes that one is used and gives no value for it, and
without one the estimator collapses.

## Layout

```
src/uzal/    the method: embeddings (vectors and time stamps), estimator
tests/       one test per equation-level component, plus the reference results
demo/        scripts producing figures; output goes to figures/ (gitignored)
```

## Reference results

Reproducing the published results is a permanent goal. Two anchors are
reproduced, both without any sampling — every valid reference point is used, so
these numbers carry no seed:

| Series | Paper | Reproduced |
|---|---|---|
| Lorenz, `T_M` from the separation plot | `m=3`, `τ=6`, `t_w=12` (App. A) | **`m=3`, `τ=6`, `t_w=12`** |
| Rössler | `m=3`, `τ=5`, `t_w=10` (App. A) | **`t_w=10` at `m=3`, for every `k`** |
| Mackey-Glass, noise free | `m=4`, `τ=10`, `t_w=30` (App. A, Fig. 1) | **`m=4`, `τ=10`, `t_w=30`** |
| Mackey-Glass, `T_M` | 44 (Sec. V) | **43** |

### Fig. 7 reproduced curve for curve

`demo/fig3_paper_fig7.py` rebuilds panels (a)–(c) of Fig. 7: `L_k` against `t_w`
for `k = 1 … 5` at fixed `m`. Every admissible window is evaluated with every
valid reference point, so the curves carry no seed and every feature in them is
real.

| `k` | Lorenz `m=3` | Rössler `m=3` | Mackey-Glass `m=4` |
|---|---|---|---|
| 1 | `t_w=8`, `L=−1.547` | `t_w=10`, `L=−2.519` | `t_w=30`, `L=−2.215` |
| 2 | `t_w=12`, `L=−1.117` | `t_w=10`, `L=−2.124` | `t_w=27`, `L=−1.857` |
| 3 | `t_w=12`, `L=−0.961` | `t_w=10`, `L=−1.830` | `t_w=30`, `L=−1.739` |
| 4 | `t_w=12`, `L=−0.895` | `t_w=10`, `L=−1.691` | `t_w=30`, `L=−1.666` |
| 5 | `t_w=12`, `L=−0.844` | `t_w=10`, `L=−1.609` | `t_w=30`, `L=−1.604` |

The published optimum comes out for nearly every `k`, which is the point the
caption makes — *"the structure of maxima and minima converges for low values
of `k`"* — and the reason Sec. IV A can recommend `k = 2` or `3` without
further argument.

**The secondary maxima agree too.** The curves carry pronounced bumps at
`t_w ≈ 46` for Lorenz and `t_w ≈ 50` for Rössler, and both appear at the same
place in the published figure (checked by eye against the original). Hitting a
minimum can be luck; reproducing the local maxima between the minima cannot be,
and it is the strongest single piece of evidence that the implementation
computes the same quantity as the paper.

Panel (d) of Fig. 7, Mackey-Glass with 10 % noise, is not reproduced in that
form: it needs the Legendre machinery of Sec. V. The noise *behaviour* is
reproduced separately, below.

### What 10 % noise does, Sec. V A

`demo/fig5_noise_moves_to_fdc.py` runs the same profile twice, clean and with
i.i.d. noise at 10 % of the series' standard deviation:

| | noise free | 10 % noise |
|---|---|---|
| `m=2` | −0.957 at `t_w=18` | −0.669 at `t_w=19` |
| `m=4` | **−1.739 at `t_w=30`** | −0.924 at `t_w=30` |
| `m=11` | −1.729 at `t_w=30` | −0.973 at `t_w=40` |
| `m=21` | −1.721 at `t_w=40` | −0.993 at `t_w=40` |
| fDC | −1.726 at `t_w=34` | **−1.013 at `t_w=117`** |

Clean, the optimum is the low-dimensional `m = 4`. Noisy, `L_k` falls
monotonically with the number of delayed components and the minimum lands on
the fDC line — the reconstruction that keeps every sample in the window. That
is exactly the claim of Sec. V A, and it is a qualitative prediction rather
than another number, so reproducing it is worth more than one more optimum.

It also matters for anything built on this estimator: real measurements are
noisy, and the low-dimensional answer the clean benchmarks give does not carry
over. Note where the noisy optimum sits — `t_w = 117`, against the `t_w = 113`
the paper reaches with Legendre coordinates at `n = 4`. The optimal *window*
is the same; what Legendre buys is reaching it with four components instead of
118.

**Normalisation.** The series are divided by their standard deviation.
`α_k · σ̄_k` carries the units of `x`, and the paper documents no normalisation
for `L_k` — the only one it states is for Eq. (5). Two independent signatures
say it normalised anyway: on raw series the Lorenz curves run above zero while
Fig. 7(a) lies entirely below it, and the published Rössler axis carries four
integer ticks from 0 to −3, which the raw range (down to −1.62) would leave
half empty. The shift is a constant per panel and moves no minimum.

`T_M` is not fitted: it is read off the first maximum of the 95th-percentile
space-time separation curve, the criterion the paper itself states. That gives
74 sampling steps for Lorenz, for which no published value exists, and 43 for
Mackey-Glass against the paper's 44.

Still to reproduce: Legendre coordinates, which recover `n=4`, `t_w=113` for
noisy Mackey-Glass (Sec. V B), and Chua with Legendre, `n=3`, `t_w=81`
(Sec. V C).

### How sharp is the optimum

The window `t_w` is well determined; the dimension `m` is not, by a wide margin.
At the optimum the curves for `m = 3, 4, 5` all minimise at `t_w = 12` and their
`L_k` differ by about 0.01, which is the same size as the scatter between
different random draws of reference points. Choosing `m` by the global minimum
is therefore only meaningful once sampling noise is removed. `T_M` matters more
than it looks: below about 60 sampling steps the Lorenz optimum drifts to larger
windows, exactly the instability the paper reports in Fig. 12.

## Implementation notes

- **The Theiler window is mandatory.** Without excluding temporally close
  neighbours, `ε_k → 0` and the estimator collapses. Default `w ≥ t_w`.
- **Factor `N`.** Eqs. (18), (20) and (21) as printed are inconsistent by a factor
  `N`: Eq. (19) is a mean, while Eq. (21) is the reciprocal of a *sum*. Both
  readings are exposed; the discrepancy is `log₁₀√N` and does not move the
  minimum of `L_k`.
- **`T` and `T_M` are in sampling steps**, not time units.
- **Index bounds.** Valid time stamps are `t_w ≤ t ≤ len(x) - 1 - T_M`, and this
  must hold for the neighbours too, since their futures are needed.
- **High dimensions.** For fDC reconstructions (`τ=1`, `m=t_w+1`) a k-d tree
  degenerates, so distances are computed by BLAS-backed brute force.

## The paper

The paper is not redistributed here — it is copyrighted and obtainable through
its DOI, https://doi.org/10.1103/PhysRevE.84.016223. All equation numbers in
the code and in this README refer to it.

## Citation

If this implementation is useful to you, please cite the paper above. The
repository itself can be cited through [CITATION.cff](CITATION.cff), which
GitHub renders as a ready-made entry.

## License

MIT, see [LICENSE](LICENSE). The paper's results belong to its authors; the
code here is mine.

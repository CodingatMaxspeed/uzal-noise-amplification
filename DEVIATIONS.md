# Deviations

Every point at which this implementation departs from, or fills a gap in, the
paper. Each entry says what the source states, what is done here, why, and what
it changes.

Kept because a reproduction is only worth as much as its list of differences.

---

## 1. The Theiler window

**What the paper states.** Nothing numerical. Footnote [49] in full:

> "The k nearest neighbors are chosen from different orbit segments by avoiding
> points inside the same Theiler window [J. Theiler, Phys. Rev. A 34, 2427
> (1986)], as is the usual practice in time series analysis [2]."

Reference [2] is Kantz & Schreiber, *Nonlinear Time Series Analysis* (1997).
The paper therefore prescribes that a window is used and leaves its size to
"usual practice".

**What this implementation does.** `w = t_w` by default: a neighbour must be
separated by at least one full embedding window. `nearest_neighbour_theiler`
takes `w` as an argument, so any other rule can be passed in.

**Why it matters.** Without a window `eps_k -> 0` and the estimator collapses,
because the nearest neighbours of a delay vector are its own immediate
predecessors and successors. This is not a refinement, it decides whether the
quantity means anything.

**What it changes.** The window is not neutral between short and long `t_w`:
tying `w` to `t_w` gives a short window a smaller exclusion zone than a long
one, so the two are not scored under identical conditions. A floor
`w = max(t_w, w0)`, with `w0` the 1/e decay time of the autocorrelation,
removes that asymmetry. Both reproduced optima below survive either rule
unchanged, which is why the simpler one is kept here.

## 2. The factor `N` in Eqs. (18), (20) and (21)

As printed these are inconsistent by a factor `N`: Eq. (19) is a mean while
Eq. (21) is the reciprocal of a *sum*. Both readings are implemented
(`convention="mean"` and `"sum"`); the default is `"mean"`, which is the only
one under which `L_k` does not depend on `N` and under which `alpha^2 sigma^2`
is the weighted average the text below Eq. (20) describes. The discrepancy is
`log10(sqrt(N))` and moves no minimum.
Pinned by `tests/test_cost.py::test_the_two_readings_differ_by_log_sqrt_n`.

## 3. Normalisation of the series

The paper documents a normalisation only for Eq. (5), not for `L_k`, yet two
signatures say the published figures were produced on normalised series: on raw
data the Lorenz curves run above zero while Fig. 7(a) lies entirely below it,
and the published Roessler axis carries four integer ticks from 0 to -3 where
the raw range reaches only -1.62. Series are divided by their standard
deviation. The shift is constant per panel and moves no minimum.

## 4. The pair convention in Eq. (16)

Eq. (16) as printed sums over `x'` and `x''` in `U_k(x)` with the single
condition `x'' != x'`, that is over *ordered* pairs, `k(k+1)` terms, while
carrying the prefactor `2 / (k(k+1))`. Read literally that is twice the mean
squared pairwise distance. This implementation sums each *unordered* pair once,
`k(k+1)/2` terms, so `eps_k^2` is the mean itself -- the reading the surrounding
text describes and the one that gives `eps^2 = d^2`, not `2 d^2`, for two points
a distance `d` apart.

The ambiguity does not reach `L_k`. A constant factor `c` in `eps_k^2` divides
`sigma_k^2` by `c` and multiplies `alpha_k^2` by `c` through Eq. (21), so it
cancels exactly in the product; `R_k` and `I_k` are each shifted by
`+/- log10(sqrt(c))`.

Both halves of this are pinned by tests: the convention by
`tests/test_eps.py::test_k_one_is_the_squared_distance` and
`::test_number_of_pairs`, the cancellation by
`tests/test_cost.py::test_invariant_under_a_rescaled_reconstruction`.

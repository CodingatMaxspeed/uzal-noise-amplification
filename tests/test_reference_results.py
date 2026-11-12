"""The published results, as end-to-end regressions.

These are the anchors of CLAUDE.md. They exercise every component at once:
generator, embedding, neighbour search, both halves of the estimator and the
cost function. If one of them drifts, one of these fails.

They sample N reference points rather than using all of them, to stay within
a few seconds. The deterministic full sweep run by demo/fig2_lorenz_sweep.py
reaches the same optimum with no seed at all.
"""

import numpy as np
import pytest

from uzal.sweep import cost_at, cost_sweep, separation_horizon, sweep_minimum
from uzal.systems import lorenz, mackey_glass


@pytest.fixture(scope="module")
def lorenz_series():
    return lorenz()


def test_lorenz_horizon_from_the_separation_plot(lorenz_series):
    """T_M is derived by the paper's own criterion, not chosen by hand."""
    T_M, _, _ = separation_horizon(lorenz_series, max_T=300)
    assert T_M == 74


def test_lorenz_optimum_is_m3_tau6(lorenz_series):
    """Appendix A: the method gives m = 3, tau = 6, that is t_w = 12."""
    sweep = cost_sweep(lorenz_series, dims=[2, 3, 4, 5], t_w_max=40, T_M=74,
                       k=3, N=3000, seed=0)
    m, t_w, tau, _ = sweep_minimum(sweep)
    assert (m, tau, t_w) == (3, 6, 12)


def test_lorenz_optimum_survives_the_other_reading_of_alpha(lorenz_series):
    """The factor-N discrepancy shifts L_k by log10(sqrt(N)) and must not
    move the minimum."""
    kw = dict(dims=[2, 3, 4, 5], t_w_max=40, T_M=74, k=3, N=3000, seed=0)
    mean = cost_sweep(lorenz_series, convention="mean", **kw)
    total = cost_sweep(lorenz_series, convention="sum", **kw)
    assert sweep_minimum(mean)[:3] == sweep_minimum(total)[:3]
    for m in mean:
        assert np.allclose(mean[m]["L"] - total[m]["L"], 0.5 * np.log10(3000))


def test_lorenz_window_is_stable_across_dimensions(lorenz_series):
    """Fig. 7(a): once m is large enough to unfold the attractor, the optimal
    window stops depending on m."""
    sweep = cost_sweep(lorenz_series, dims=[3, 4, 5], t_w_max=40, T_M=74,
                       k=3, N=3000, seed=0)
    best = [int(rec["t_w"][np.argmin(rec["L"])]) for rec in sweep.values()]
    assert all(abs(b - 12) <= 2 for b in best)


@pytest.fixture(scope="module")
def mackey_glass_series():
    return mackey_glass()


def test_mackey_glass_horizon_matches_the_paper(mackey_glass_series):
    """Sec. V reports T_M = 44 from the same construction."""
    T_M, _, _ = separation_horizon(mackey_glass_series, max_T=300)
    assert abs(T_M - 44) <= 2


def test_mackey_glass_optimum_is_m4_tw30(mackey_glass_series):
    """Fig. 1 and Appendix A: m = 4, tau = 10, that is t_w = 30.

    Run without sampling. The minimum is shallow here -- L_k differs by about
    0.01 between t_w = 24 and t_w = 30 -- so with N reference points drawn at
    random the argmin wanders by a grid step. Using every valid point removes
    that ambiguity, at the cost of about ten seconds.
    """
    sweep = cost_sweep(mackey_glass_series, dims=[3, 4, 5], t_w_max=40,
                       T_M=43, k=3, N=None)
    m, t_w, tau, _ = sweep_minimum(sweep)
    assert (m, tau, t_w) == (4, 10, 30)


# --- Sec. V A: what 10 % observational noise does -------------------------

NOISE, NOISE_SEED, MG_T_M = 0.10, 0, 43


@pytest.fixture(scope="module")
def mg_clean(mackey_glass_series):
    return mackey_glass_series / mackey_glass_series.std()


@pytest.fixture(scope="module")
def mg_noisy(mg_clean):
    """The same series with i.i.d. noise at 10 % of its standard deviation."""
    rng = np.random.default_rng(NOISE_SEED)
    return mg_clean + NOISE * rng.standard_normal(len(mg_clean))


def best_dc(x, m, t_w_max=60):
    """The lowest L_k over windows, at fixed embedding dimension."""
    return min(cost_at(x, m, t_w // (m - 1), MG_T_M, k=3, N=1500, seed=0).L
               for t_w in range(m - 1, t_w_max + 1, m - 1))


def best_fdc(x, t_w_max=100, step=10):
    """The lowest L_k along the full delay coordinate line, tau = 1."""
    return min(cost_at(x, t_w + 1, 1, MG_T_M, k=3, N=1500, seed=0).L
               for t_w in range(step, t_w_max + 1, step))


def test_noise_does_not_move_the_horizon(mg_clean, mg_noisy):
    """T_M comes from the raw series, and 10 % noise does not disturb it."""
    assert separation_horizon(mg_clean, 300)[0] == MG_T_M
    assert separation_horizon(mg_noisy, 300)[0] == MG_T_M


def test_noise_moves_the_optimum_onto_fdc(mg_clean, mg_noisy):
    """Sec. V A: without noise a low-dimensional reconstruction wins; with it
    the minimum is the one that keeps every sample in the window."""
    assert best_fdc(mg_clean) > best_dc(mg_clean, 4)
    assert best_fdc(mg_noisy) < best_dc(mg_noisy, 4)


def test_noise_orders_the_dimensions_monotonically(mg_noisy):
    """Sec. V A: "the higher the number of delayed components considered
    inside the selected time window, the lower the value of L_k"."""
    best = [best_dc(mg_noisy, m) for m in (2, 3, 4, 5, 6)]
    assert np.all(np.diff(best) < 0)


def test_without_noise_the_ordering_stops_at_m4(mg_clean):
    """The contrast that makes the previous test mean something: on the clean
    series L_k stops falling at m = 4, which is why Fig. 1 has a minimum
    there rather than an envelope."""
    best = [best_dc(mg_clean, m) for m in (2, 3, 4, 5, 6)]
    assert not np.all(np.diff(best) < 0)
    assert int(np.argmin(best)) == 2  # m = 4 is the third entry

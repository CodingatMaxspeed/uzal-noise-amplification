"""Choosing T_M, and the grid over (m, t_w)."""

import numpy as np
import pytest

from uzal.sweep import cost_sweep, separation_curve, separation_horizon, sweep_minimum
from uzal.systems import lorenz, mackey_glass


def sine(n=4000, period=100):
    return np.sin(2 * np.pi * np.arange(n) / period)


def walk(n=2000, seed=0):
    """An aperiodic series. A pure sine will not do for a sweep: it repeats
    exactly, so eps_k collapses to zero and Eq. (17) has no value."""
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def test_separation_curve_of_a_sine_peaks_at_half_a_period():
    """|sin(t+T) - sin(t)| is largest when T is half a period, wherever t is."""
    T, curve = separation_curve(sine(), max_T=200)
    assert T[int(np.argmax(curve))] == 50
    assert np.isclose(curve.max(), 2.0, atol=0.01)


def test_separation_horizon_finds_that_peak():
    T_M, _, _ = separation_horizon(sine(), max_T=200)
    assert T_M == 50


@pytest.mark.parametrize("smooth", [None, 11, 21, 41])
def test_horizon_does_not_depend_on_the_smoothing_width(smooth):
    """The raw Lorenz curve reverses direction repeatedly, so the first
    descent has to be located on a smoothed copy; widening the window past
    the ripple scale must not move the answer."""
    assert separation_horizon(lorenz(), 300, smooth=smooth)[0] == 74


def test_horizon_of_mackey_glass_matches_the_paper():
    """Sec. V reports T_M = 44 for this series."""
    T_M, _, _ = separation_horizon(mackey_glass(), 300)
    assert abs(T_M - 44) <= 2


def test_horizon_complains_when_the_curve_still_rises():
    with pytest.raises(ValueError):
        separation_horizon(sine(period=10_000), max_T=50)


def test_sweep_grid_keeps_tau_integral():
    """t_w runs over multiples of m-1 so that tau = t_w / (m-1) is an integer."""
    sw = cost_sweep(walk(), dims=[2, 3, 4], t_w_max=24, T_M=30, k=2, N=100, seed=0)
    for m, rec in sw.items():
        assert np.all(rec["t_w"] % (m - 1) == 0)
        assert np.array_equal(rec["tau"] * (m - 1), rec["t_w"])
        assert rec["t_w"][0] == m - 1
        assert rec["t_w"][-1] <= 24


def test_sweep_records_the_decomposition():
    sw = cost_sweep(walk(), dims=[3], t_w_max=12, T_M=30, k=2, N=100, seed=0)
    rec = sw[3]
    assert np.allclose(rec["L"], rec["R"] + rec["I"])
    assert np.allclose(rec["R"], np.log10(rec["sigma_bar"]))
    assert np.allclose(rec["I"], np.log10(rec["alpha"]))


def test_sweep_is_reproducible():
    kw = dict(dims=[3], t_w_max=12, T_M=30, k=2, N=100, seed=5)
    a = cost_sweep(walk(), **kw)[3]
    b = cost_sweep(walk(), **kw)[3]
    assert np.array_equal(a["L"], b["L"])


def test_sweep_minimum_picks_the_global_one():
    sw = cost_sweep(walk(), dims=[2, 3, 4], t_w_max=24, T_M=30, k=2, N=100, seed=0)
    m, t_w, tau, L = sweep_minimum(sw)
    assert L == min(rec["L"].min() for rec in sw.values())
    assert sw[m]["L"][sw[m]["t_w"] == t_w][0] == L
    assert tau * (m - 1) == t_w


def test_sweep_rejects_dimension_one():
    """m = 1 has no window to scan: t_w is identically zero."""
    with pytest.raises(ValueError):
        cost_sweep(walk(500), dims=[1], t_w_max=10, T_M=10, N=50, seed=0)

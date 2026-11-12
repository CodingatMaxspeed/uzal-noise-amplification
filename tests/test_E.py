"""E_k, Eqs. (13)-(15): the future-value variance, measured on the raw series."""

import inspect

import numpy as np
import pytest

from uzal.estimator import E2, eps2
from uzal.estimator import delay_coordinate_embedding as embed
from uzal.estimator import nearest_neighbour_theiler as neighbours

L, M, TAU, K, T_M = 2000, 3, 6, 3, 40
T_W = (M - 1) * TAU


@pytest.fixture
def setup():
    x = np.cumsum(np.random.default_rng(0).standard_normal(L))
    X, times = embed(x, m=M, tau=TAU)
    U, _ = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=200, rng=42)
    return x, X, times, U


def test_shape_and_sign(setup):
    x, _, times, U = setup
    out = E2(x, times, U, T_M)
    assert out.shape == (U.shape[0],)
    assert np.all(out >= 0.0)


def test_constant_series_has_no_spread(setup):
    _, _, times, U = setup
    assert np.allclose(E2(np.ones(L), times, U, T_M), 0.0)


def test_horizon_zero_is_the_spread_of_the_raw_values(setup):
    x, X, times, _ = setup
    U, _ = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=0, N=100, rng=2)
    assert np.allclose(E2(x, times, U, 0), np.var(x[times[U]], axis=1))


def test_k_one_closed_form(setup):
    """Two members: the population variance is (a - b)^2 / 4."""
    x, X, times, _ = setup
    U, _ = neighbours(X, times, n_samples=L, k=1, w=T_W, T_M=T_M, N=100, rng=1)
    t0, t1 = times[U[:, 0]], times[U[:, 1]]
    horizons = np.arange(T_M + 1)
    expected = np.mean((x[t0[:, None] + horizons] - x[t1[:, None] + horizons]) ** 2 / 4, axis=1)
    assert np.allclose(E2(x, times, U, T_M), expected)


def test_divisor_is_k_plus_one(setup):
    """Eq. (13) divides by k+1, i.e. ddof=0, not by k."""
    x, _, times, U = setup
    stamps = times[U]
    horizons = np.arange(T_M + 1)
    futures = x[stamps[:, :, None] + horizons[None, None, :]]
    by_hand = ((futures - futures.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    by_hand = (by_hand / U.shape[1]).mean(axis=-1)
    assert np.allclose(E2(x, times, U, T_M), by_hand)


def test_shift_invariant_and_scales_quadratically(setup):
    x, _, times, U = setup
    base = E2(x, times, U, T_M)
    assert np.allclose(E2(x + 17.0, times, U, T_M), base)
    for c in [0.5, 3.0]:
        assert np.allclose(E2(c * x, times, U, T_M), c ** 2 * base)


def test_does_not_depend_on_the_reconstruction(setup):
    """The asymmetry of CLAUDE.md: transforming X moves eps_k but not E_k.

    E_k reaches the observable through the time stamp of each neighbour, which
    is not a component of its vector, so it cannot see X at all. This is why
    L_k can be evaluated on any point cloud.

    The independence is structural rather than numerical: there is no X to
    pass to E2, so it is asserted on the signature. What can be measured is
    the other half of the asymmetry -- the same transformation does move
    eps_k.
    """
    x, X, times, U = setup
    assert list(inspect.signature(E2).parameters) == ["x", "times", "U", "T_M"]
    rng = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rng.standard_normal((M, M)))
    transformed = 5.0 * (X @ Q.T)
    assert not np.allclose(eps2(transformed, U), eps2(X, U))


def test_larger_horizons_average_over_more(setup):
    x, _, times, U = setup
    short, long = E2(x, times, U, 5), E2(x, times, U, 40)
    assert short.shape == long.shape
    assert not np.allclose(short, long)


def test_reports_a_horizon_beyond_the_series(setup):
    x, _, times, U = setup
    with pytest.raises(ValueError):
        E2(x, times, U, 10 ** 5)


def test_rejects_a_negative_horizon(setup):
    x, _, times, U = setup
    with pytest.raises(ValueError):
        E2(x, times, U, -1)

"""Embeddings with arbitrary lag sets.

The anchor is the first test: the uniform embedding that reproduced the
published optima is the special case lags = (0, tau, ..., (m-1) tau) of this
one. Anything the new function gets wrong about component order, window size
or time stamps shows up there first.
"""

import numpy as np
import pytest

from uzal.embedding import lag_embedding
from uzal.estimator import delay_coordinate_embedding as dce
from uzal.estimator import eps2, E2
from uzal.estimator import nearest_neighbour_theiler as neighbours


def series(n=200):
    return np.sin(2 * np.pi * np.arange(n) / 17) + 0.3 * np.cos(np.arange(n) / 3)


@pytest.mark.parametrize("m, tau", [(1, 5), (2, 1), (3, 6), (4, 10), (5, 1), (4, 7)])
def test_uniform_case_is_bitwise_the_existing_function(m, tau):
    """lags = (0, tau, ..., (m-1) tau) must reproduce delay_coordinate_embedding
    exactly -- not close, equal. m = 3, tau = 6 is the published Lorenz optimum."""
    X, times = lag_embedding(series(), np.arange(m) * tau)
    Xb, tb = dce(series(), m, tau)
    assert np.array_equal(X, Xb)
    assert np.array_equal(times, tb)


@pytest.mark.parametrize("lags", [[0, 7, 32], [0, 1, 2, 3], [0, 13], [0, 5, 9, 40]])
def test_columns_are_the_definition(lags):
    """X[i, j] == x[times[i] - lags[j]], every entry, no vectorisation."""
    x = series()
    X, times = lag_embedding(x, lags)
    for i in range(len(times)):
        for j, lag in enumerate(lags):
            assert X[i, j] == x[times[i] - lag]


@pytest.mark.parametrize("lags, t_w", [([0], 0), ([0, 13], 13), ([0, 7, 32], 32),
                                       ([0, 1, 199], 199)])
def test_window_is_max_of_lags_not_the_uniform_formula(lags, t_w):
    """t_w = max(lags). For (0, 7, 32) that is 32, where (m-1)*tau would say 64."""
    x = series()
    X, times = lag_embedding(x, lags)
    assert X.shape == (len(x) - t_w, len(lags))
    assert times[0] == t_w and times[-1] == len(x) - 1


def test_m_one_is_the_series_itself():
    x = series()
    X, times = lag_embedding(x, [0])
    assert np.array_equal(X[:, 0], x)
    assert times[0] == 0


def test_nonuniform_is_genuinely_a_different_cloud():
    """(0, 7, 32) must not coincide with any uniform set sharing its window."""
    x = series()
    X, _ = lag_embedding(x, [0, 7, 32])
    for tau in (16, 32):                       # t_w = 32 at m = 3 and m = 2
        Xu, _ = lag_embedding(x, np.arange(33 // tau + 1) * tau)
        assert X.shape != Xu.shape or not np.array_equal(X, Xu)


def test_the_lag_set_moves_eps_but_not_E():
    """The asymmetry that makes method B possible at all: two lag sets with the
    same window share their time stamps, so the same U indexes the same raw
    futures -- E_k is identical while eps_k is not. A different reconstruction
    therefore moves only the denominator of Eq. (17)."""
    x = np.cumsum(np.random.default_rng(0).standard_normal(2000))
    Xa, ta = lag_embedding(x, [0, 7, 32])
    Xb, tb = lag_embedding(x, [0, 16, 32])     # same t_w, so the same stamps
    assert np.array_equal(ta, tb)
    U, _ = neighbours(Xa, ta, n_samples=len(x), k=3, w=32, T_M=40, N=100, rng=0)
    assert np.array_equal(E2(x, ta, U, 40), E2(x, tb, U, 40))
    assert not np.allclose(eps2(Xa, U), eps2(Xb, U))


def test_accepts_a_list_and_returns_float():
    X, _ = lag_embedding([0, 1, 2, 3, 4], [0, 2])
    assert X.dtype == np.float64


@pytest.mark.parametrize("lags, match", [
    ([], "non-empty"),
    (5, "non-empty"),                          # a scalar has no shape to speak of
    ([0, 7.9], "integers"),                    # silent truncation would hide a bug
    ([0, 7, 7], "strictly increase"),
    ([0, 9, 4], "strictly increase"),
    ([3, 7], "strictly increase"),             # must start at 0
    ([0, 200], "too short"),                   # t_w == L leaves no rows
])
def test_rejects_malformed_lags(lags, match):
    with pytest.raises(ValueError, match=match):
        lag_embedding(series(), lags)


def test_rejects_a_non_scalar_series():
    with pytest.raises(ValueError):
        lag_embedding([[1.0, 2.0], [3.0, 4.0]], [0, 1])

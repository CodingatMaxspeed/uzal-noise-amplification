"""L_k on an arbitrary point cloud.

The anchor is the first test: cost_at, which reproduced the published optima,
must come out of cost_of_cloud unchanged. The rest pins the properties that
method B relies on -- permutation invariance above all, since it is what lets
an exhaustive lag scan restrict itself to tau1 < tau2.
"""

import numpy as np
import pytest

from uzal.embedding import lag_embedding
from uzal.estimator import delay_coordinate_embedding as dce
from uzal.sweep import cost_at, cost_of_cloud
from uzal.systems import lorenz

T_M = 74


@pytest.fixture(scope="module")
def x():
    return lorenz()


@pytest.mark.parametrize("m, tau", [(2, 1), (3, 6), (4, 10), (5, 8), (11, 1)])
@pytest.mark.parametrize("convention", ["mean", "sum"])
def test_reproduces_cost_at_exactly(x, m, tau, convention):
    """Not close -- equal. cost_at is the path that produced m=3, tau=6."""
    X, times = dce(x, m, tau)
    assert cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1,
                         convention=convention) == cost_at(
        x, m, tau, T_M, k=3, N=1500, seed=1, convention=convention)


def test_w_none_is_the_window_off_the_first_time_stamp(x):
    """The documented default: w = t_w, read off times[0] rather than from m."""
    X, times = dce(x, 3, 6)
    assert times[0] == (3 - 1) * 6
    assert cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1) == cost_of_cloud(
        x, X, times, T_M, k=3, N=1500, seed=1, w=int(times[0]))


def test_theiler_zero_is_passed_through_not_treated_as_missing(x):
    """w = 0 is a legal window -- it excludes only the reference point. It is
    also falsy, so a `if not w` default would silently turn it into t_w."""
    X, times = dce(x, 3, 6)
    zero = cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1, w=0)
    auto = cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1, w=None)
    assert zero.L != auto.L


def test_permutation_of_the_coordinates_leaves_L_alone(x):
    """Why the lag scan may restrict itself to tau1 < tau2: the columns of X
    enter only through distances, which do not care about their order.

    Invariant in exact arithmetic, but not bit for bit: permuting the columns
    permutes the summation order inside eps2, so alpha can move by one ulp.
    Measured at about 6e-17, fourteen orders of magnitude below the 0.0028 that
    separates the best twenty lag sets, so the restriction is safe.
    """
    X, times = lag_embedding(x, [0, 12, 39])
    base = cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1)
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        other = cost_of_cloud(x, X[:, perm], times, T_M, k=3, N=1500, seed=1)
        assert np.allclose(other, base, rtol=1e-12, atol=0.0)
        assert abs(other.L - base.L) < 1e-12


def test_orthonormal_transformation_leaves_L_alone(x):
    """alpha_k removes a global scale; an orthonormal map changes no distance
    at all, so L_k is untouched. The counterpart of test_eps.py."""
    X, times = lag_embedding(x, [0, 12, 39])
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))
    base = cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1)
    rotated = cost_of_cloud(x, X @ Q.T, times, T_M, k=3, N=1500, seed=1)
    assert np.isclose(rotated.L, base.L)


def test_isotropic_rescaling_leaves_L_alone(x):
    """eps_k -> c eps_k with E_k untouched, which is exactly what alpha_k cancels."""
    X, times = lag_embedding(x, [0, 12, 39])
    base = cost_of_cloud(x, X, times, T_M, k=3, N=1500, seed=1)
    for c in (0.01, 7.0, 1000.0):
        assert np.isclose(cost_of_cloud(x, c * X, times, T_M, k=3, N=1500,
                                        seed=1).L, base.L)


def test_a_nonuniform_cloud_differs_from_the_uniform_one(x):
    """The point of the whole exercise: same window, different lag set,
    different L_k. Otherwise there would be nothing to scan over."""
    Xn, tn = lag_embedding(x, [0, 12, 39])
    Xu, tu = lag_embedding(x, [0, 19, 38])
    assert not np.isclose(cost_of_cloud(x, Xn, tn, T_M, k=3, N=1500, seed=1).L,
                          cost_of_cloud(x, Xu, tu, T_M, k=3, N=1500, seed=1).L)


def test_scores_a_cloud_that_is_not_an_embedding(x):
    """E_k reaches the series through the time stamps alone, so the cloud may
    be anything -- here two delay coordinates plus a running mean."""
    X, times = lag_embedding(x, [0, 12])
    smooth = np.convolve(x, np.ones(5) / 5, mode="same")[times]
    cloud = np.column_stack([X, smooth])
    c = cost_of_cloud(x, cloud, times, T_M, k=3, N=1500, seed=1)
    assert np.isfinite(c.L) and np.isclose(c.L, c.R + c.I)


@pytest.mark.parametrize("X, times, match", [
    (np.zeros(5), np.arange(5), "2-D point cloud"),
    (np.zeros((5, 2)), np.arange(3), "length 5"),
    (np.zeros((5, 2)), np.arange(5) + 0.5, "integers"),
])
def test_rejects_malformed_clouds(x, X, times, match):
    with pytest.raises(ValueError, match=match):
        cost_of_cloud(x, X, times, 10)


def test_rejects_a_non_scalar_series():
    with pytest.raises(ValueError, match="scalar series"):
        cost_of_cloud(np.zeros((5, 2)), np.zeros((5, 2)), np.arange(5), 10)

"""Neighbourhoods U_k(x-bar): Theiler window, index bounds, reproducibility."""

import numpy as np
import pytest

from uzal.estimator import delay_coordinate_embedding as embed
from uzal.estimator import nearest_neighbour_theiler as neighbours

L, M, TAU, K, T_M = 2000, 3, 6, 3, 40
T_W = (M - 1) * TAU


@pytest.fixture
def series():
    return np.cumsum(np.random.default_rng(0).standard_normal(L))


@pytest.fixture
def embedded(series):
    return (series, *embed(series, m=M, tau=TAU))


def test_shapes_and_reference_column(embedded):
    _, X, times = embedded
    U, ref = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=200, rng=42)
    assert U.shape == (200, K + 1)
    assert ref.shape == (200,)
    assert np.array_equal(U[:, 0], ref)


def test_theiler_window_is_respected(embedded):
    _, X, times = embedded
    for w in [0, 5, T_W, 50]:
        U, ref = neighbours(X, times, n_samples=L, k=K, w=w, T_M=T_M, N=100, rng=1)
        dt = np.abs(times[ref][:, None] - times[U[:, 1:]])
        assert dt.min() > w


def test_reference_point_is_never_its_own_neighbour(embedded):
    _, X, times = embedded
    U, ref = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=200, rng=42)
    assert not (U[:, 1:] == ref[:, None]).any()
    assert all(len(set(row)) == K + 1 for row in U)


def test_futures_of_every_member_stay_inside_the_series(embedded):
    """The index bound must hold for the neighbours, not only the references."""
    _, X, times = embedded
    for T_max in [0, 10, T_M, 200]:
        U, _ = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_max, N=100, rng=3)
        assert times[U].max() + T_max <= L - 1


def test_same_seed_reproduces_and_different_seed_does_not(embedded):
    _, X, times = embedded
    kw = dict(n_samples=L, k=K, w=T_W, T_M=T_M, N=200)
    a, _ = neighbours(X, times, rng=42, **kw)
    b, _ = neighbours(X, times, rng=42, **kw)
    c, _ = neighbours(X, times, rng=7, **kw)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_block_size_does_not_change_the_result(embedded):
    _, X, times = embedded
    kw = dict(n_samples=L, k=K, w=T_W, T_M=T_M, N=200, rng=42)
    base, _ = neighbours(X, times, **kw)
    for block in [1, 7, 64, 10_000]:
        other, _ = neighbours(X, times, block=block, **kw)
        assert np.array_equal(base, other)


def test_matches_a_naive_search(embedded):
    """Brute force with the Gram identity against straightforward distances."""
    _, X, times = embedded
    U, ref = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=40, rng=5)
    pool = np.flatnonzero(times <= L - 1 - T_M)
    for i, r in enumerate(ref):
        d = np.linalg.norm(X[pool] - X[r], axis=1)
        d[np.abs(times[pool] - times[r]) <= T_W] = np.inf
        assert set(U[i, 1:]) == set(pool[np.argsort(d, kind="stable")[:K]])


def test_none_uses_every_valid_point(embedded):
    _, X, times = embedded
    U, ref = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M)
    assert np.array_equal(ref, np.flatnonzero(times <= L - 1 - T_M))
    assert U.shape[0] == ref.size


def test_references_are_a_subset_of_the_candidates(embedded):
    _, X, times = embedded
    pool = np.flatnonzero(times <= L - 1 - T_M)
    U, ref = neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=200, rng=9)
    assert np.isin(ref, pool).all()
    assert np.isin(U[:, 1:], pool).all()
    assert not np.isin(pool, ref).all()  # a proper subset when N < len(pool)


def test_sampling_without_a_seed_is_refused(embedded):
    _, X, times = embedded
    with pytest.raises(ValueError):
        neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=200)


def test_too_large_a_theiler_window_is_reported(embedded):
    _, X, times = embedded
    with pytest.raises(ValueError):
        neighbours(X, times, n_samples=L, k=K, w=L, T_M=T_M, N=10, rng=0)


def test_too_many_reference_points_is_reported(embedded):
    _, X, times = embedded
    with pytest.raises(ValueError):
        neighbours(X, times, n_samples=L, k=K, w=T_W, T_M=T_M, N=10**6, rng=0)

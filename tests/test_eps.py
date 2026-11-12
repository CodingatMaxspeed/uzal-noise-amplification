"""eps_k, Eq. (16): the neighbourhood radius in the reconstruction space."""

import numpy as np
import pytest

from uzal.estimator import eps2


def cloud(n_points, m, seed=0):
    return np.random.default_rng(seed).standard_normal((n_points, m))


def neighbourhoods(n_points, n_sets, q, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(n_points, size=q, replace=False) for _ in range(n_sets)])


@pytest.mark.parametrize("k, m", [(1, 2), (2, 3), (3, 1), (5, 4), (10, 7)])
def test_matches_the_explicit_double_sum(k, m):
    """Eq. (16) written out pair by pair, without any vectorisation."""
    q, n_sets, n_points = k + 1, 6, 200
    X = cloud(n_points, m)
    U = neighbourhoods(n_points, n_sets, q)
    expected = np.array([
        sum(np.sum((X[U[n, a]] - X[U[n, b]]) ** 2)
            for a in range(q) for b in range(a + 1, q)) * 2.0 / (k * q)
        for n in range(n_sets)
    ])
    assert np.allclose(eps2(X, U), expected)


@pytest.mark.parametrize("k, m", [(1, 2), (2, 3), (3, 5), (7, 4)])
def test_variance_identity(k, m):
    """sum over pairs equals q times the spread about the centroid, so
    eps^2 = 2(k+1)/k times the population variance of the neighbourhood."""
    q, n_points = k + 1, 200
    X = cloud(n_points, m, seed=2)
    U = neighbourhoods(n_points, 8, q, seed=3)
    identity = 2.0 * q / k * np.var(X[U], axis=1).sum(-1)
    assert np.allclose(eps2(X, U), identity)


def test_k_one_is_the_squared_distance():
    X = cloud(10, 3, seed=4)
    U = np.array([[0, 1], [2, 3]])
    expected = [np.sum((X[0] - X[1]) ** 2), np.sum((X[2] - X[3]) ** 2)]
    assert np.allclose(eps2(X, U), expected)


def test_number_of_pairs():
    """The prefactor is the reciprocal of the pair count, so a neighbourhood
    of identical spacing returns exactly that spacing squared."""
    X = np.array([[0.0], [1.0], [2.0]])
    # pairs: 1, 4, 1  -> mean 2
    assert np.isclose(eps2(X, np.array([[0, 1, 2]]))[0], 2.0)


def test_collapsed_neighbourhood_is_zero():
    assert np.allclose(eps2(np.zeros((5, 3)), np.array([[0, 1, 2, 3]])), 0.0)


def test_scales_quadratically():
    """eps^2(cX) = c^2 eps^2(X): the scale that alpha_k later removes."""
    X = cloud(30, 4, seed=5)
    U = neighbourhoods(30, 3, 4, seed=6)
    base = eps2(X, U)
    for c in [0.01, 3.0, 1000.0]:
        assert np.allclose(eps2(c * X, U), c ** 2 * base)


def test_orthonormal_transformations_leave_it_alone():
    X = cloud(30, 4, seed=7)
    U = neighbourhoods(30, 3, 4, seed=8)
    Q, _ = np.linalg.qr(np.random.default_rng(9).standard_normal((4, 4)))
    assert np.allclose(Q @ Q.T, np.eye(4))
    assert np.allclose(eps2(X @ Q.T, U), eps2(X, U))


def test_merely_orthogonal_is_not_enough():
    """CLAUDE.md: rows must be orthonormal, not merely orthogonal. alpha_k
    removes a global scale, not an anisotropic stretch."""
    X = cloud(30, 4, seed=7)
    U = neighbourhoods(30, 3, 4, seed=8)
    Q, _ = np.linalg.qr(np.random.default_rng(9).standard_normal((4, 4)))
    # Scaling the *rows* keeps them orthogonal while destroying their unit
    # norm. Q @ diag(d) would not: it scales the columns, and the rows of the
    # result are no longer orthogonal at all.
    stretched = np.diag([1.0, 5.0, 0.2, 2.0]) @ Q
    rows = stretched @ stretched.T
    assert np.allclose(rows - np.diag(np.diag(rows)), 0.0)  # orthogonal
    assert not np.allclose(np.linalg.norm(stretched, axis=1), 1.0)  # not normal
    assert not np.allclose(eps2(X @ stretched.T, U), eps2(X, U))


def test_rejects_malformed_inputs():
    X = cloud(10, 2)
    with pytest.raises(ValueError):
        eps2(X.ravel(), np.array([[0, 1]]))
    with pytest.raises(ValueError):
        eps2(X, np.array([[0], [1]]))  # fewer than two columns

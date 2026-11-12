"""Delay coordinate embedding, the definition of x-bar(t)."""

import numpy as np
import pytest

from uzal.estimator import delay_coordinate_embedding as embed


def sine(n=48, period=12):
    return np.sin(2 * np.pi * np.arange(n) / period)


@pytest.mark.parametrize("m, tau", [(1, 1), (2, 3), (3, 4), (5, 1), (4, 7)])
def test_shapes_and_stamps(m, tau):
    x = sine(60)
    X, times = embed(x, m=m, tau=tau)
    t_w = (m - 1) * tau
    assert X.shape == (len(x) - t_w, m)
    assert times.shape == (X.shape[0],)
    assert times[0] == t_w
    assert times[-1] == len(x) - 1


@pytest.mark.parametrize("m, tau", [(2, 3), (3, 4), (4, 2)])
def test_columns_are_the_definition(m, tau):
    """X[i, j] == x[t - j*tau]: the paper's order, present first, oldest last."""
    x = sine(60)
    X, times = embed(x, m=m, tau=tau)
    for j in range(m):
        assert np.allclose(X[:, j], x[times - j * tau])


def test_sine_reconstructs_the_unit_circle():
    """m=2, tau=3 on 12 samples per period gives (sin th, -cos th)."""
    x = sine(48)
    X, times = embed(x, m=2, tau=3)
    theta = 2 * np.pi * times / 12
    assert np.allclose(X, np.column_stack([np.sin(theta), -np.cos(theta)]))
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_sine_collapses_at_half_a_period():
    """tau=6 makes the second component -x[t], so the cloud is a line."""
    x = sine(48)
    X, times = embed(x, m=2, tau=6)
    assert np.allclose(X[:, 0], -X[:, 1])


def test_non_injective_pair_of_figure_one():
    """x-bar(8) and x-bar(10) coincide while their futures at T=3 differ."""
    x = sine(48)
    X, times = embed(x, m=2, tau=6)
    a = X[times == 8][0]
    b = X[times == 10][0]
    assert np.allclose(a, b)
    assert np.allclose(a, [-np.sqrt(3) / 2, np.sqrt(3) / 2])
    assert not np.isclose(x[8 + 3], x[10 + 3])


def test_m_one_is_the_series_itself():
    x = sine(20)
    X, times = embed(x, m=1, tau=5)
    assert X.shape == (20, 1)
    assert np.allclose(X[:, 0], x)
    assert times[0] == 0


def test_accepts_a_list_and_returns_float():
    X, _ = embed([0, 1, 2, 3, 4], m=2, tau=1)
    assert X.dtype == np.float64


@pytest.mark.parametrize("kwargs", [
    dict(m=0, tau=1),
    dict(m=2, tau=0),
    dict(m=2, tau=1000),
])
def test_rejects_impossible_parameters(kwargs):
    with pytest.raises(ValueError):
        embed(sine(20), **kwargs)


def test_rejects_a_non_scalar_series():
    with pytest.raises(ValueError):
        embed([[1.0, 2.0], [3.0, 4.0]], m=2, tau=1)

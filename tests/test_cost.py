"""L_k, Eqs. (17), (18), (21), (23)-(26), and the factor-N discrepancy."""

import numpy as np
import pytest

from uzal.estimator import Cost, cost_function


@pytest.fixture
def halves():
    rng = np.random.default_rng(0)
    return rng.gamma(2.0, size=500), rng.gamma(2.0, size=500)


def test_fields_and_additivity(halves):
    E2, eps2 = halves
    c = cost_function(E2, eps2)
    assert isinstance(c, Cost)
    assert c.L == c.R + c.I  # Eq. (25), bit for bit
    assert np.isclose(c.R, np.log10(c.sigma_bar))
    assert np.isclose(c.I, np.log10(c.alpha))


def test_matches_the_equations_written_out(halves):
    E2, eps2 = halves
    c = cost_function(E2, eps2)
    sigma2_bar = np.mean(E2 / eps2)  # Eqs. (17), (18)
    alpha2 = 1.0 / np.mean(1.0 / eps2)  # Eq. (21), mean reading
    assert np.isclose(c.sigma_bar, np.sqrt(sigma2_bar))
    assert np.isclose(c.alpha, np.sqrt(alpha2))
    assert np.isclose(c.L, np.log10(np.sqrt(alpha2) * np.sqrt(sigma2_bar)))


def test_sigma_bar_averages_the_squares(halves):
    """Eq. (18) is a mean of sigma^2, so sigma_bar is sqrt(mean), not mean(sqrt)."""
    E2, eps2 = halves
    c = cost_function(E2, eps2)
    assert np.isclose(c.sigma_bar, np.sqrt(np.mean(E2 / eps2)))
    assert not np.isclose(c.sigma_bar, np.mean(np.sqrt(E2 / eps2)))


def test_alpha_is_the_harmonic_mean_not_the_arithmetic_one(halves):
    """Eq. (19) averages eps^2 arithmetically; Eq. (21) does not."""
    _, eps2 = halves
    c = cost_function(*halves)
    assert np.isclose(c.alpha ** 2, 1.0 / np.mean(1.0 / eps2))
    assert not np.isclose(c.alpha, np.sqrt(np.mean(eps2)))


def test_weighted_average_interpretation(halves):
    """The text below Eq. (20): alpha^2 sigma_bar^2 is the weighted average of
    E^2 with weights proportional to eps^-2 and summing to one. Only the mean
    reading of Eq. (21) satisfies this."""
    E2, eps2 = halves
    c = cost_function(E2, eps2, "mean")
    weighted = np.sum(E2 / eps2) / np.sum(1.0 / eps2)
    assert np.isclose(c.alpha ** 2 * c.sigma_bar ** 2, weighted)


@pytest.mark.parametrize("n", [10, 100, 500])
def test_the_two_readings_differ_by_log_sqrt_n(n):
    rng = np.random.default_rng(1)
    E2, eps2 = rng.gamma(2.0, size=n), rng.gamma(2.0, size=n)
    mean = cost_function(E2, eps2, "mean")
    total = cost_function(E2, eps2, "sum")
    assert np.isclose(mean.L - total.L, 0.5 * np.log10(n))
    assert np.isclose(mean.R, total.R)  # only alpha differs


def test_invariant_under_a_rescaled_reconstruction(halves):
    """The purpose of alpha_k: scaling only the reconstruction leaves L_k alone.

    eps^2 -> c^2 eps^2 while E^2 is untouched, so sigma_bar -> sigma_bar / c
    and alpha -> c alpha.
    """
    E2, eps2 = halves
    base = cost_function(E2, eps2).L
    for c in [0.001, 3.0, 1000.0]:
        assert np.isclose(cost_function(E2, c ** 2 * eps2).L, base)


def test_scaling_the_observable_shifts_by_log_c(halves):
    """x -> c x scales both halves, so L_k -> L_k + log10(c)."""
    E2, eps2 = halves
    base = cost_function(E2, eps2).L
    for c in [2.0, 10.0]:
        shifted = cost_function(c ** 2 * E2, c ** 2 * eps2).L
        assert np.isclose(shifted, base + np.log10(c))


def test_collapsed_neighbourhood_is_reported_not_propagated(halves):
    """A single zero denominator makes alpha 0 and sigma_bar inf at once,
    whose product is NaN. It must not travel silently through a sweep."""
    E2, eps2 = halves
    broken = eps2.copy()
    broken[3] = 0.0
    with pytest.raises(ValueError, match="eps"):
        cost_function(E2, broken)


def test_rejects_unknown_convention_and_mismatched_shapes(halves):
    E2, eps2 = halves
    with pytest.raises(ValueError):
        cost_function(E2, eps2, "harmonic")
    with pytest.raises(ValueError):
        cost_function(E2, eps2[:10])

"""The benchmark generators of Appendices B to D."""

import numpy as np
import pytest

from uzal.systems import _rk4, lorenz, mackey_glass, roessler


def test_rk4_integrates_exponential_decay():
    """ds/dt = -s from s=1 is exp(-t); RK4 at dt=0.01 should be very close."""
    out = _rk4(lambda s: -s, [1.0], 0.01, 500)
    t = np.arange(501) * 0.01
    assert out.shape == (501, 1)  # n_steps + 1 rows, row i at time i*dt
    assert np.allclose(out[:, 0], np.exp(-t), atol=1e-10)


def test_rk4_conserves_the_energy_of_a_harmonic_oscillator():
    out = _rk4(lambda s: np.array([s[1], -s[0]]), [1.0, 0.0], 0.01, 2000)
    assert out.shape == (2001, 2)
    energy = out[:, 0] ** 2 + out[:, 1] ** 2
    assert np.allclose(energy, 1.0, atol=1e-8)


@pytest.mark.parametrize("gen", [lorenz, roessler, mackey_glass])
def test_length_and_determinism(gen):
    a = gen(n=500, transient=100)
    b = gen(n=500, transient=100)
    assert a.shape == (500,)
    assert np.array_equal(a, b)  # no randomness anywhere
    assert np.all(np.isfinite(a))


def test_lorenz_matches_appendix_d():
    """sigma=10, rho=28, beta=8/3 from (1, 1, 50), dt=0.01, 1000 discarded."""
    x = lorenz()
    assert x.shape == (10000,)
    # the classical attractor: x roughly within +-20, two wings, zero-ish mean
    assert 15.0 < np.abs(x).max() < 22.0
    assert abs(x.mean()) < 2.0
    assert 6.0 < x.std() < 10.0
    # both wings are visited
    assert (x > 5).mean() > 0.2 and (x < -5).mean() > 0.2


def test_lorenz_transient_is_actually_discarded():
    """The initial z=50 is far off the attractor; the first kept sample must
    already be on it.

    The transient is quiet, not wild: z = 50 drives x towards zero, so the
    first 50 samples reach only |x| = 1.0, while the stillest 50-sample window
    the attractor itself ever produces still reaches 1.45. Not one of its 1950
    windows is as quiet as the transient, which is what "off the attractor"
    amounts to for the observable alone.
    """
    full = lorenz(n=2000, transient=0)
    kept = lorenz(n=2000, transient=1000)
    win = 50
    quietest = min(np.abs(kept[i:i + win]).max() for i in range(len(kept) - win))
    assert np.abs(full[:win]).max() < quietest
    assert np.array_equal(full[1000:], kept[:1000])


def test_roessler_matches_appendix_c():
    x = roessler()
    assert x.shape == (10000,)
    assert 10.0 < np.abs(x).max() < 25.0
    assert np.all(np.isfinite(x))


def test_mackey_glass_matches_appendix_b():
    x = mackey_glass()
    assert x.shape == (10000,)
    # the tau=17 attractor lives in a narrow positive band
    assert 0.2 < x.min() < 0.7
    assert 1.1 < x.max() < 1.5
    assert np.all(x > 0)


def test_mackey_glass_refuses_an_incommensurate_step():
    """tau must be an exact multiple of the integration step, otherwise the
    history lookup would need interpolation at every stage.

    With the paper's tau = 17 and dt = 0.5 the ratio is 34 * substeps and can
    never fail, so the guard has to be provoked with a delay that does not
    divide the step.
    """
    with pytest.raises(ValueError, match="multiple"):
        mackey_glass(n=10, transient=0, tau=17.005, dt=0.5, substeps=50)


def test_mackey_glass_is_insensitive_to_the_substep_count():
    """Halving the integration step must not change the attractor band."""
    coarse = mackey_glass(n=2000, transient=500, substeps=25)
    fine = mackey_glass(n=2000, transient=500, substeps=100)
    assert abs(coarse.min() - fine.min()) < 0.05
    assert abs(coarse.max() - fine.max()) < 0.05
    assert abs(coarse.std() - fine.std()) < 0.02

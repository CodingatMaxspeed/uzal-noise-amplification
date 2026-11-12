"""Benchmark systems of the paper, Appendices B to D.

Each generator returns the scalar observable exactly as the paper specifies:
the x coordinate, sampled at the stated step, with the stated transient
discarded and 10000 samples kept by default.

The series are regenerated rather than stored. They are fully determined by
the parameters below plus the initial condition, so a stored copy could only
drift away from the recipe it claims to implement.
"""

import numpy as np


def _rk4(deriv, state0, dt, n_steps):
    """n_steps of classical RK4 on ds/dt = deriv(s).

    Returns the initial state followed by the state after each step, so the
    result has n_steps + 1 rows and row i is the state at time i*dt. Every
    step that is computed is also returned.
    """
    state = np.asarray(state0, dtype=float)
    out = np.empty((n_steps + 1, state.size))
    out[0] = state
    for i in range(n_steps):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = state
    return out


def lorenz(n=10000, transient=1000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0,
           state0=(1.0, 1.0, 50.0)):
    """x coordinate of the Lorenz system, Appendix D.

    Appendix D: sigma = 10, rho = 28, beta = 8/3, initialised at
    x = y = 1, z = 50, fourth-order Runge-Kutta with step dt = 0.01, the
    first 1000 data points discarded as a transient, 10000 samples kept.

    The optimum reported in Appendix A for this series is m = 3, tau = 6,
    that is t_w = 12.

    Returns
    -------
    x : (n,) ndarray
    """
    def deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    # transient + n rows are wanted, and n_steps steps give n_steps + 1 rows.
    return _rk4(deriv, state0, dt, transient + n - 1)[transient:, 0]


def roessler(n=10000, transient=8000, dt=0.125, a=0.15, b=0.2, c=10.0,
             state0=(1.0, 1.0, 1.0)):
    """x coordinate of the Roessler system, Appendix C.

    Appendix C: alpha = 0.15, beta = 0.2, gamma = 10, initialised at
    x = y = z = 1, fourth-order Runge-Kutta, the x coordinate sampled with a
    step of 0.125, the first 8000 data points discarded, 10000 kept.

    The paper does not separate integration step from sampling step here, so
    both are taken to be dt. The optimum reported in Appendix A is m = 3,
    tau = 5, that is t_w = 10.

    Returns
    -------
    x : (n,) ndarray
    """
    def deriv(s):
        x, y, z = s
        return np.array([-(y + z), x + a * y, b + z * (x - c)])

    # transient + n rows are wanted, and n_steps steps give n_steps + 1 rows.
    return _rk4(deriv, state0, dt, transient + n - 1)[transient:, 0]


def mackey_glass(n=10000, transient=2000, dt=0.5, a=0.2, b=0.1, c=10.0, tau=17.0,
                 substeps=50, x0=1.2):
    """Mackey-Glass series, Appendix B.

    dx/dt = a x(t - tau) / (1 + x(t - tau)^c) - b x, with a = 0.2, b = 0.1,
    c = 10, tau = 17, history x(t < 0) = 0 and x(0) = 1.2, sampled every
    dt = 0.5 after discarding 2000 points as a transient.

    This is a delay differential equation, so RK4 needs the state one delay
    in the past at each of its stages. The integration step is dt/substeps,
    chosen so that tau is an exact multiple of it; the two full-step stages
    then land on stored history and only the half step is interpolated,
    linearly. With the default substeps that step is 0.01, three orders of
    magnitude below the delay.

    The optimum reported in Sec. V for the noise-free series is m = 4,
    t_w = 30.

    Returns
    -------
    x : (n,) ndarray
    """
    h = dt / substeps
    lag = tau / h
    if abs(lag - round(lag)) > 1e-9:
        raise ValueError(f"tau={tau} is not a multiple of the step dt/substeps={h}")
    lag = int(round(lag))

    # The last sample sits at step (transient + n - 1) * substeps, so
    # anything beyond that would be computed and thrown away.
    total = (transient + n - 1) * substeps
    hist = np.zeros(lag + total + 1)  # history x(t < 0) = 0 sits in [0, lag)
    hist[lag] = x0

    def delayed(i, half=False):
        """x at time (i - lag) steps, linearly interpolated for a half step."""
        j = i - lag
        return 0.5 * (hist[j] + hist[j + 1]) if half else hist[j]

    def rate(xi, xd):
        """The right-hand side, with the delayed value supplied per stage."""
        return a * xd / (1.0 + xd ** c) - b * xi

    for i in range(lag, lag + total):
        x = hist[i]
        # The delayed term is frozen per stage at the time that stage asks
        # for; only the two mid-stages need the interpolated half step.
        d0, dh, d1 = delayed(i), delayed(i, half=True), delayed(i + 1)
        k1 = rate(x, d0)
        k2 = rate(x + 0.5 * h * k1, dh)
        k3 = rate(x + 0.5 * h * k2, dh)
        k4 = rate(x + h * k3, d1)
        hist[i + 1] = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return hist[lag::substeps][transient:]

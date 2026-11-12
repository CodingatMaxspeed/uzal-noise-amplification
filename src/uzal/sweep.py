"""Choosing T_M, and scanning L_k over a grid of reconstructions.

Neither belongs to the estimator itself: these are the two things one has to
do around it to turn L_k into the answer to "which m and tau".
"""

import numpy as np

from .estimator import (
    E2,
    Cost,
    cost_function,
    delay_coordinate_embedding,
    eps2,
    nearest_neighbour_theiler,
)


def separation_curve(x, max_T, percentile=95.0):
    """Upper curve of a space-time separation plot, Sec. V and Fig. 12(c).

    For each horizon T the given percentile of |x(t + T) - x(t)|. The paper
    uses the 95th percentile, which is "almost an upper bound" for that
    distribution.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series, without any reconstruction.
    max_T : int
        Largest horizon to evaluate, in sampling steps.
    percentile : float
        Which percentile to follow.

    Returns
    -------
    T : (max_T,) ndarray
        Horizons 1 ... max_T.
    curve : (max_T,) ndarray
        The percentile at each horizon.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be a scalar series, got shape {x.shape}")
    if not 1 <= max_T < len(x):
        raise ValueError(f"need 1 <= max_T < {len(x)}, got {max_T}")

    T = np.arange(1, max_T + 1)
    curve = np.array([np.percentile(np.abs(x[t:] - x[:-t]), percentile) for t in T])
    return T, curve


def separation_horizon(x, max_T, percentile=95.0, smooth=None):
    """T_M as the paper picks it: the first maximum of the separation curve.

    "We use this curve to choose T_M as the time of its first maximum in
    order to ensure that nearby orbits are given enough time to diverge."
    For Mackey-Glass this reproduces T_M = 43 against the 44 of the paper.

    The curve is not smooth at single-step resolution -- for the Lorenz
    series it reverses direction eleven times before T = 150 -- so the first
    descent of the raw curve fires on a ripple rather than on the maximum.
    It is therefore smoothed by a centred moving average before the descent
    is located, and the maximum is then refined on the raw curve within one
    smoothing width. Widening `smooth` past the ripple scale changes nothing.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series.
    max_T : int
        Largest horizon to evaluate.
    percentile : float
        Which percentile of |x(t+T) - x(t)| to follow.
    smooth : int or None
        Width of the moving average, in sampling steps. None uses one tenth
        of max_T, at least 3.

    Returns
    -------
    T_M : int
        The horizon, in sampling steps.
    T, curve : ndarray
        The raw curve, for plotting.
    """
    T, curve = separation_curve(x, max_T, percentile)
    width = max(3, max_T // 10) if smooth is None else int(smooth)
    if width % 2 == 0:
        width += 1  # keep the average centred
    pad = width // 2
    padded = np.pad(curve, pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(width) / width, mode="valid")

    falling = np.flatnonzero(np.diff(smoothed) < 0.0)
    if falling.size == 0:
        raise ValueError(
            f"the separation curve still rises at T={max_T}; increase max_T"
        )
    # Refine on the raw curve: the smoothing displaces the peak by up to
    # half a window, so take the true maximum in that neighbourhood.
    centre = falling[0]
    lo, hi = max(0, centre - width), min(len(curve), centre + width + 1)
    return int(T[lo + int(np.argmax(curve[lo:hi]))]), T, curve


def cost_at(x, m, tau, T_M, k=3, N=1000, seed=0, convention="mean", theiler=None,
            block=512):
    """L_k for one single reconstruction (m, tau).

    The whole pipeline for one point of a sweep, so that grids which are not
    of the form "fixed m, t_w over multiples of m-1" -- the fDC line of Fig. 1
    is one, with tau = 1 and m = t_w + 1 -- do not have to repeat it.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series.
    m, tau : int
        The reconstruction. The window is t_w = (m - 1) * tau.
    T_M : int
        Prediction horizon of Eq. (15).
    k, N, seed, convention, block :
        As for cost_sweep. theiler defaults to w = t_w.

    Returns
    -------
    Cost
    """
    x = np.asarray(x, dtype=float)
    X, times = delay_coordinate_embedding(x, m=m, tau=tau)
    return cost_of_cloud(x, X, times, T_M, k=k, w=theiler, N=N, seed=seed, convention=convention, block=block)


def cost_sweep(x, dims, t_w_max, T_M, k=3, N=1000, seed=0, convention="mean",
               theiler=None):
    """L_k over a grid of delay reconstructions.

    For every embedding dimension m the window t_w runs over the multiples of
    m - 1 up to t_w_max, so that tau = t_w / (m - 1) stays an integer. That
    is the grid the paper plots in Figs. 1, 7 and 18.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series.
    dims : iterable of int
        Embedding dimensions m to scan, each >= 2.
    t_w_max : int
        Largest embedding window, in sampling steps.
    T_M : int
        Prediction horizon of Eq. (15), e.g. from separation_horizon.
    k : int
        Neighbours per reference point. Sec. IV A suggests 2 or 3.
    N : int
        Reference points per reconstruction, Eq. (18).
    seed : int
        Seed for that draw. Each (m, t_w) uses the same seed, so the sweep is
        reproducible; the points themselves still differ between windows
        because the valid pool does.
    convention : {"mean", "sum"}
        Reading of Eq. (21), passed to cost_function.
    theiler : int or None
        Theiler window. None follows the default of CLAUDE.md, w = t_w.

    Returns
    -------
    dict
        m -> record array with fields t_w, tau, L, R, I, sigma_bar, alpha.
    """
    x = np.asarray(x, dtype=float)
    dims = [int(m) for m in dims]
    if any(m < 2 for m in dims):
        raise ValueError(f"need m >= 2 for a window sweep, got {dims}")

    fields = [("t_w", int), ("tau", int), ("L", float), ("R", float),
              ("I", float), ("sigma_bar", float), ("alpha", float)]
    out = {}
    for m in dims:
        rows = []
        for t_w in range(m - 1, t_w_max + 1, m - 1):
            tau = t_w // (m - 1)
            c = cost_at(x, m, tau, T_M, k=k, N=N, seed=seed,
                        convention=convention, theiler=theiler)
            rows.append((t_w, tau, c.L, c.R, c.I, c.sigma_bar, c.alpha))
        out[m] = np.array(rows, dtype=fields)
    return out


def sweep_minimum(sweep):
    """The (m, t_w) at which L_k is smallest across a whole sweep.

    Returns
    -------
    m, t_w, tau, L
    """
    best = None
    for m, rec in sweep.items():
        i = int(np.argmin(rec["L"]))
        if best is None or rec["L"][i] < best[3]:
            best = (m, int(rec["t_w"][i]), int(rec["tau"][i]), float(rec["L"][i]))
    return best


def cost_of_cloud(x, X, times, T_M, k=3, w=None, N=None, seed=0,
                  convention="mean", block=512) -> Cost:
    """L_k for an arbitrary point cloud, Eqs. (17), (18), (21), (23)-(26).

    The cloud need not be a delay reconstruction. eps_k reads X, E_k reads the
    raw series through the time stamps, so delay coordinates, an arbitrary lag
    set, PCA or Legendre coordinates are all scored by the same three calls.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series. E_k is measured on this, never on X.
    X : (M, d) array_like
        The point cloud, e.g. from lag_embedding or delay_coordinate_embedding.
    times : (M,) array_like of int
        Time stamp of each row of X.
    T_M : int
        Prediction horizon of Eq. (15).
    w : int or None
        Theiler window. None reads t_w off times[0], which both embedding
        functions guarantee to be the window; a cloud assembled some other way
        has to say what it means.
    k, N, seed, convention, block :
        As for cost_sweep.

    Returns
    -------
    Cost
        L, R, I, sigma_bar and alpha.
    """
    x = np.asarray(x, dtype=float)
    X = np.asarray(X, dtype=float)
    times = np.asarray(times)
    if x.ndim != 1:
        raise ValueError(f"x must be a scalar series, got shape {x.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D point cloud, got shape {X.shape}")
    if times.ndim != 1 or times.size != X.shape[0]:
        raise ValueError(
            f"times must be 1-D of length {X.shape[0]}, got shape {times.shape}"
        )
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError(f"times must be integers, got dtype {times.dtype}")

    # t_w, not the column count. For tau > 1 the window is far wider than m,
    # and it is the window that decides which neighbours are merely temporal
    # ones. At tau = 1 the two coincide, which is where the mistake hides.
    w = int(times[0]) if w is None else w
    U, _ = nearest_neighbour_theiler(
        X, times, n_samples=len(x), k=k, w=w, T_M=T_M, N=N, rng=seed, block=block
    )
    return cost_function(E2(x, times, U, T_M), eps2(X, U), convention)

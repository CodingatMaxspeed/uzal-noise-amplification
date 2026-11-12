"""Reconstructions from an arbitrary set of lags.

Uzal's own reconstruction is uniform, and estimator.delay_coordinate_embedding
builds exactly that. Sec. V E drops the restriction: on the torus the paper
picks lags that are not multiples of one another. This module is that
generalisation, with the uniform case as the special case
lags = (0, tau, ..., (m-1) tau).

The estimator needs no change to score these clouds. eps_k reads the point
cloud, E_k reads the raw series through the time stamps, so a different lag set
moves only the denominator of Eq. (17).
"""

import numpy as np


def lag_embedding(x, lags) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruction of a scalar series from an arbitrary set of lags.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series.
    lags : (m,) array_like of int
        Backward lags, strictly increasing and starting at 0, so column 0 holds
        the present and column m-1 the oldest value. Uniform delay coordinates
        are lags = (0, tau, ..., (m-1) tau).

    Returns
    -------
    X : (L - t_w, m) ndarray
        X[i, j] = x[times[i] - lags[j]], in the component order of the paper:
        present first, oldest last.
    times : (L - t_w,) ndarray
        Time stamps t_w <= t <= L - 1, one per row of X, where t_w = lags[-1].
        Not a component of X, but Eq. (13) needs them: the future value of a
        neighbour is x[t' + T] with t' its time stamp.
    """
    x = np.asarray(x, dtype=float)
    lags = np.asarray(lags)
    if x.ndim != 1:
        raise ValueError(f"x must be a scalar series, got shape {x.shape}")
    if lags.ndim != 1 or lags.size == 0:
        raise ValueError(f"lags must be a non-empty 1-D array, got shape {lags.shape}")
    if not np.issubdtype(lags.dtype, np.integer):
        raise ValueError(f"lags must be integers, got dtype {lags.dtype}")
    if lags[0] != 0 or np.any(np.diff(lags) <= 0):
        raise ValueError(f"lags must start at 0 and strictly increase, got {lags}")

    t_w = int(lags[-1])  # embedding window: max(lags), not (m - 1) * tau
    L = len(x)
    if L <= t_w:
        raise ValueError(f"series of length {L} too short for t_w={t_w}")

    # Column by column rather than x[times[:, None] - lags[None, :]]: that
    # broadcast builds an index array as large as X itself. Measured at
    # m = 1001, L = 10000 it peaks at 144 MB against 72 MB here, and at the
    # small m a lag scan hammers it is also about 5x slower.
    times = np.arange(t_w, L)
    X = np.empty((L - t_w, lags.size))
    for j, lag in enumerate(lags):
        X[:, j] = x[t_w - lag:L - lag]
    return X, times

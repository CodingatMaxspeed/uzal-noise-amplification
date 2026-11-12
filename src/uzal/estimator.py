from typing import NamedTuple
import numpy as np

def delay_coordinate_embedding(x: np.ndarray, m: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """Delay coordinate reconstruction of a scalar time series.

    Returns the delay vectors together with the time stamps they belong to.
    The time stamp is not a component of the vector, but Eq. (13) needs it:
    the future value x'(T) of a neighbour is x[t' + T], where t' is that
    neighbour's time stamp.

    Parameters
    ----------
    x : (L,) array_like
        Scalar time series.
    m : int
        Embedding dimension, m >= 1.
    tau : int
        Delay in sampling steps, tau >= 1.

    Returns
    -------
    X : (L - t_w, m) ndarray
        X[i] = (x[t], x[t - tau], ..., x[t - (m-1) tau]) for t = times[i],
        in the component order of the paper: present first, oldest last.
    times : (L - t_w,) ndarray
        Time stamps t_w <= t <= L - 1, one per row of X.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be a scalar series, got shape {x.shape}")
    if m < 1 or tau < 1:
        raise ValueError(f"need m >= 1 and tau >= 1, got m={m}, tau={tau}")

    t_w = (m - 1) * tau  # embedding window
    if len(x) <= t_w:
        raise ValueError(f"series of length {len(x)} too short for t_w={t_w}")
    times = np.arange(t_w, len(x))
    # Column j holds x[t - j*tau]: (M, 1) time stamps against (1, m) delays.
    idx = times[:, None] - np.arange(m)[None, :] * tau
    return x[idx], times


def nearest_neighbour_theiler(
    X: np.ndarray,
    times: np.ndarray,
    n_samples: int,
    k: int,
    w: int,
    T_M: int = 0,
    N: int | None = None,
    rng: np.random.Generator | int | None = None,
    block: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbourhoods U_k(x̄) with a Theiler window, by brute force.

    Reference points and candidates are drawn from the same pool
    t_w <= t <= n_samples - 1 - T_M: Eq. (13) evaluates x[t' + T] up to
    T = T_M for the *neighbours* too, not only for the reference point.

    Distances are computed by BLAS-backed brute force rather than with a k-d
    tree, which degenerates for fDC reconstructions (tau = 1, m = t_w + 1).

    Parameters
    ----------
    X : (M, m) ndarray
        Delay vectors, as returned by delay_coordinate_embedding.
    times : (M,) ndarray
        Their time stamps.
    n_samples : int
        Length L of the underlying scalar series; with T_M it fixes the
        latest usable time stamp.
    k : int
        Number of neighbours, k >= 1. Sec. IV A suggests k = 2 or 3.
    w : int
        Theiler window. A neighbour is admissible iff |t - t'| > w, so w = 0
        excludes only the reference point itself. Default choice: w >= t_w.
    T_M : int
        Largest prediction horizon Eq. (15) will use, in sampling steps.
    N : int or None
        Number of reference points to draw. None uses every valid point.
    rng : Generator, int or None
        Seed or generator for that draw. Required whenever N is not None.
    block : int
        Reference points per distance block; peak memory is block * M * 32 B.

    Returns
    -------
    U : (N, k + 1) ndarray
        Indices into X. Column 0 is the reference point, columns 1: are its
        neighbours, unordered. This is U_k(x̄) of Eq. (13), k + 1 elements.
    ref : (N,) ndarray
        Indices into X of the reference points, identical to U[:, 0].
    """
    X = np.asarray(X, dtype=float)
    times = np.asarray(times)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if times.ndim != 1:
        raise ValueError(f"times must be 1-D, got shape {times.shape}")
    if len(X) != len(times):
        raise ValueError(f"X and times disagree: {len(X)} rows vs {len(times)} stamps")
    if k < 1:
        raise ValueError(f"need k >= 1, got {k}")
    if w < 0 or T_M < 0:
        raise ValueError(f"need w >= 0 and T_M >= 0, got w={w}, T_M={T_M}")

    # Index bound of CLAUDE.md, applied to reference points and candidates
    # alike, since the futures of the neighbours are needed as well.
    pool = np.flatnonzero(times <= n_samples - 1 - T_M)
    if pool.size <= k:
        raise ValueError(
            f"only {pool.size} vectors satisfy t <= {n_samples - 1 - T_M}, "
            f"need more than k={k}"
        )

    if N is None:
        ref = pool
    else:
        if rng is None:
            raise ValueError("sampling N reference points requires a seed (rng)")
        if N > pool.size:
            raise ValueError(f"N={N} exceeds the {pool.size} valid vectors")
        ref = np.sort(np.random.default_rng(rng).choice(pool, size=N, replace=False))

    Xp, tp = X[pool], times[pool]
    norm_p = np.einsum("ij,ij->i", Xp, Xp)  # (P,) row norms, no temporary

    U = np.empty((ref.size, k + 1), dtype=np.intp)
    U[:, 0] = ref  # U_k contains x̄ itself, Eq. (13) averages over k + 1

    for lo in range(0, ref.size, block):
        hi = min(lo + block, ref.size)
        Xr, tr = X[ref[lo:hi]], times[ref[lo:hi]]
        # ||a - b||^2 via the Gram matrix; squared throughout, because
        # ranking is unaffected and Eq. (16) wants squares anyway.
        d2 = norm_p[None, :] - 2.0 * (Xr @ Xp.T)
        d2 += np.einsum("ij,ij->i", Xr, Xr)[:, None]
        np.maximum(d2, 0.0, out=d2)  # the identity dips below 0 on round-off
        d2[np.abs(tr[:, None] - tp[None, :]) <= w] = np.inf
        nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
        if not np.isfinite(np.take_along_axis(d2, nn, axis=1)).all():
            raise ValueError(f"Theiler window w={w} leaves fewer than k={k} neighbours")
        U[lo:hi, 1:] = pool[nn]

    return U, ref


  
def eps2(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Squared neighbourhood radii of U_k(x̄), Eq. (16).

    Mean squared distance over all k(k+1)/2 unordered pairs inside U_k(x̄),
    which has k + 1 members including the reference point itself.

    Measured in the *reconstruction* space, unlike its counterpart E_k of
    Eqs. (13)-(15), which is measured on the raw series. That asymmetry is
    what lets L_k be evaluated on any point cloud.

    Parameters
    ----------
    X : (M, m) ndarray
        Delay vectors, as returned by delay_coordinate_embedding.
    U : (N, k + 1) ndarray
        Indices into X of the reference points and their neighbours, as
        returned by nearest_neighbour_theiler. k is read off U.

    Returns
    -------
    eps2 : (N,) ndarray
        eps^2_k(x̄_i). Squared, because Eq. (17) is a ratio of squares.
        Exactly 0 where a neighbourhood collapses; Eq. (17) decides what
        to do about that.
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if U.ndim != 2 or U.shape[1] < 2:
        raise ValueError(f"U must be 2-D with at least 2 columns, got {U.shape}")

    q = U.shape[1]  # k + 1 members of U_k
    k = q - 1
    pts = X[U]  # (N, q, m)

    # Every unordered pair once. The 1 is triu_indices' diagonal offset,
    # not the neighbour count: it drops the diagonal i == j.
    i, j = np.triu_indices(q, 1)
    diff = pts[:, i, :] - pts[:, j, :]  # (N, k(k+1)/2, m)
    return 2.0 / (k * q) * np.einsum("npm,npm->n", diff, diff)



def E2(x: np.ndarray, times: np.ndarray, U: np.ndarray, T_M: int) -> np.ndarray:
    """Variance of the future values over U_k(x̄), Eqs. (13)-(15).

    Eq. (13) together with Eq. (14) is the population variance of the k + 1
    future values x'(T) inside U_k(x̄); Eq. (15) averages it over the
    horizons T_j = 0 ... T_M.

    Measured on the *raw series*, unlike eps2, which is measured in the
    reconstruction space. x'(T) is x[t' + T], where t' is the neighbour's
    time stamp and not a component of its vector. E_k therefore does not
    depend on X at all: a transformation of the reconstruction changes only
    the denominator of Eq. (17). That is why L_k can be evaluated on any
    point cloud.

    Parameters
    ----------
    x : (L,) array_like
        The raw scalar series the embedding was built from.
    times : (M,) ndarray
        Time stamps of the delay vectors, from delay_coordinate_embedding.
    U : (N, k + 1) ndarray
        Neighbourhood indices, from nearest_neighbour_theiler.
    T_M : int
        Largest prediction horizon in sampling steps. Eq. (15) averages over
        the p = T_M + 1 integer horizons in [0, T_M]. Must not exceed the
        T_M the neighbourhoods were built with.

    Returns
    -------
    E2 : (N,) ndarray
        E^2_k(x̄_i), Eq. (15). Squared, because Eq. (17) is a ratio of squares.
    """
    x = np.asarray(x, dtype=float)
    times = np.asarray(times)
    U = np.asarray(U)
    if x.ndim != 1:
        raise ValueError(f"x must be a scalar series, got shape {x.shape}")
    if U.ndim != 2 or U.shape[1] < 2:
        raise ValueError(f"U must be 2-D with at least 2 columns, got {U.shape}")
    if T_M < 0:
        raise ValueError(f"need T_M >= 0, got {T_M}")

    stamps = times[U]  # (N, q) time stamps of the k + 1 members
    horizons = np.arange(T_M + 1)  # (p,) the T_j of Eq. (15)
    if stamps.max() + T_M > len(x) - 1:
        raise ValueError(
            f"x[{stamps.max() + T_M}] needed but the series ends at "
            f"{len(x) - 1}; build the neighbourhoods with T_M >= {T_M}"
        )

    # x'(T) = x[t' + T]: (N, q, 1) stamps broadcast against (1, 1, p) horizons.
    futures = x[stamps[:, :, None] + horizons[None, None, :]]  # (N, q, p)

    # var over the q members is Eqs. (13)+(14) — ddof=0 gives the divisor
    # k + 1. The mean then runs over the p horizons, Eq. (15). Different
    # axes: after var the shape is (N, p).
    return futures.var(axis=1).mean(axis=-1)

class Cost(NamedTuple):
    """Eq. (26) together with its two competing terms, Eqs. (22)-(25)."""

    L: float  # Eq. (26), log10(alpha * sigma_bar)
    R: float  # Eq. (23), log10(sigma_bar), penalises redundance
    I: float  # Eq. (24), log10(alpha), penalises irrelevance
    sigma_bar: float  # Eq. (18)
    alpha: float  # Eq. (21)


def cost_function(E2: np.ndarray, eps2: np.ndarray, convention: str = "mean") -> Cost:
    """The cost function L_k of Eq. (26), from the two halves of the estimator.

    Eq. (17) forms the ratio per reference point, Eq. (18) averages it over
    the N of them, Eq. (21) normalises against irrelevance, and Eq. (26)
    takes the logarithm of the product.

    Eqs. (18), (20) and (21) as printed are inconsistent by a factor N:
    Eq. (19) is a mean, while Eq. (21) is the reciprocal of a *sum*. Both
    readings are offered.

    - "mean": alpha^2 = 1 / mean_i eps^-2. This makes alpha^2 * sigma_bar^2
      the weighted average of E^2 with weights summing to one, exactly as
      the text below Eq. (20) requires, and it is the only reading under
      which L_k does not depend on N.
    - "sum": alpha^2 = 1 / sum_i eps^-2, literally as printed. Yields
      L_k - log10(sqrt(N)). The minimum sits in the same place.

    Parameters
    ----------
    E2 : (N,) ndarray
        E^2_k(x-bar_i), Eq. (15).
    eps2 : (N,) ndarray
        eps^2_k(x-bar_i), Eq. (16). Must be strictly positive.
    convention : {"mean", "sum"}
        Which reading of Eq. (21) to apply.

    Returns
    -------
    Cost
        L, R, I, sigma_bar and alpha, with L = R + I exactly, Eq. (25).
    """
    E2 = np.asarray(E2, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    if E2.ndim != 1 or E2.shape != eps2.shape:
        raise ValueError(f"E2 and eps2 must be 1-D and alike, got {E2.shape}, {eps2.shape}")
    if convention not in ("mean", "sum"):
        raise ValueError(f"convention must be 'mean' or 'sum', got {convention!r}")

    collapsed = int(np.count_nonzero(eps2 <= 0.0))
    if collapsed:
        raise ValueError(
            f"{collapsed} of {eps2.size} neighbourhoods have eps^2 <= 0, where "
            "Eq. (17) is undefined. Footnote [50] adds noise below the "
            "resolution of x to avoid this."
        )

    sigma2 = E2 / eps2  # Eq. (17)
    sigma2_bar = sigma2.mean()  # Eq. (18)

    inv = 1.0 / eps2
    alpha2 = 1.0 / (inv.sum() if convention == "sum" else inv.mean())  # Eq. (21)

    # Eq. (26) via the squares: one sqrt fewer, and R + I stays exactly
    # additive as Eq. (25) demands.
    R = 0.5 * np.log10(sigma2_bar)  # Eq. (23)
    I = 0.5 * np.log10(alpha2)  # Eq. (24)
    return Cost(R + I, R, I, np.sqrt(sigma2_bar), np.sqrt(alpha2))

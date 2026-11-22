"""Gaffke confidence interval for the multinomial weighted mean.

For X ~ Mult(n, p) and mu = v^T p, the interval is built from two
Dirichlet posteriors that each add one pseudo-count at an extreme
category:

    P_- ~ Dir(x + e_min),   P_+ ~ Dir(x + e_max)
    CI  = [ Q_{alpha/2}(v^T P_-),  Q_{1-alpha/2}(v^T P_+) ]

where e_min / e_max are the indicators of the categories carrying the
smallest / largest value.  There is no optimisation over the simplex
and no enumeration of multinomial outcomes: each endpoint is one
scalar quantile of a Dirichlet-weighted sum.

Categories with a zero Dirichlet parameter contribute no mass and are
dropped, which is the correct limit of Dir(alpha) as alpha_i -> 0.
Categories with equal values are merged.  See gaffke.dirichlet_sum for
the computation of the quantile.
"""

from __future__ import annotations

import numpy as np

from gaffke.dirichlet_sum import dirichlet_sum_ppf


def gaffke_ci(counts, values, alpha=0.05, *, tol=1e-12):
    """Gaffke confidence interval for the multinomial weighted mean.

    Parameters
    ----------
    counts : array-like of int
        Observed category counts.
    values : array-like of float
        Numerical value assigned to each category.
    alpha : float
        Significance level (default 0.05 for a 95% CI).
    tol : float
        Tolerance of the root finder, relative to max(values) - min(values).

    Returns
    -------
    dict with keys: lower, upper.
    """
    counts = np.asarray(counts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if counts.ndim != 1 or len(counts) != len(values) or len(counts) == 0:
        raise ValueError("counts and values must be 1-d with the same nonzero length")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite and nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    a_lo = counts.copy()
    a_lo[int(np.argmin(values))] += 1.0
    a_hi = counts.copy()
    a_hi[int(np.argmax(values))] += 1.0

    lower = dirichlet_sum_ppf(a_lo, values, alpha / 2.0, xtol=tol)
    upper = dirichlet_sum_ppf(a_hi, values, alpha / 2.0, upper=True, xtol=tol)

    return {"lower": float(lower), "upper": float(upper)}

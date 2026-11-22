"""Distribution of a weighted sum of Dirichlet coordinates.

Y = w^T P with P ~ Dirichlet(a), a_i > 0.  This module gives the CDF,
the density and the quantile function of Y, with a small relative
error also far in the tails.

Reduction.  With independent G_i ~ Gamma(a_i, 1), P = G / sum(G), so
P(Y <= y) = P(Z <= 0) with Z = sum_i lam_i G_i and lam_i = w_i - y.
The moment generating function of Z is

    M(s) = E[exp(s Z)] = prod_i (1 - s lam_i)^(-a_i),

finite on the strip 1/min(lam) < Re s < 1/max(lam).

Inversion on a shifted line.  For a real c in that strip,

    P(Z <= 0) = -(1/pi) int_0^inf Re[ M(c + i t) / (c + i t) ] dt   (c < 0),
    P(Z >  0) = +(1/pi) int_0^inf Re[ M(c + i t) / (c + i t) ] dt   (c > 0).

For c -> 0 this is the Gil-Pelaez formula.  We take c at the saddle
point of log M (the root of K'(c) = 0, K = log M), on the side of the
smaller tail.  On this line the integrand does not oscillate near
t = 0 and its size is exp(K(c)), the size of the tail.  Thus the
error is small relative to the tail probability, not only in absolute
value.  The density of Y follows from d/dy log M(s) = -s sum a_i /
(1 - s lam_i):

    f(y) = (1/pi) int_0^inf Re[ M(c + i t) sum_i a_i / (1 - (c + i t) lam_i) ] dt.

Quadrature.  We scale s by sd(Z) so the integrand varies on a scale
of order one.  All singularities of the integrand (branch points of
M and the pole at s = 0) are on the imaginary axis of t.  We cut the
half line at T, where the integrand is below e^-40 times its value at
t = 0, and we split [0, T] into panels [0, h0], [h0, 2 h0],
[2 h0, 4 h0], ...  A point i b of the imaginary axis is at a distance
of at least c from the panel [c, 2 c], which is the panel length.
Thus a 12-point Gauss-Legendre rule converges at the same fast
geometric rate on each panel, for every position of the singularities
(y close to a value w_i, small parameters with slow decay, large
parameters with a narrow peak).  The phase of the integrand can still
turn fast far from t = 0 when the parameters have very different sizes
(for example 1 and 10^4); we then split each panel so that the phase
turns by at most 3 radians on each half of a panel.

Quantile.  Newton's method on log(tail) - log(q) as a function of
log(d), with d the distance from y to the end of the tail, and with
the density for the derivative.  It runs inside a bracket that always
contains the root (bisection when a Newton step leaves the bracket).
The start point is the quantile of the Beta law with the same mean
and variance on [min w, max w].  Two special cases have closed forms:
one distinct value (Y is constant) and two distinct values (Y is an
affine image of a Beta variable).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betainccinv, betaincinv

__all__ = ["dirichlet_sum_cdf", "dirichlet_sum_sf", "dirichlet_sum_pdf", "dirichlet_sum_ppf"]

_N_GL = 12
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_N_GL)
_gl_t = 0.5 * (_gl_x + 1.0)  # nodes on [0, 1]
_gl_w = 0.5 * _gl_w  # weights on [0, 1]
_LOG_CUT = 40.0  # truncate the integral where the integrand < e^-40
_MAX_HALF_PHASE = 3.0  # max phase change on half a panel (radians)


def _reduce(a, w):
    """Drop null parameters and merge equal values (sum of parameters).

    The merge is exact: coordinates of a Dirichlet vector that share a
    value can be added, and the sum is again Dirichlet (aggregation).
    Returns the sorted distinct values and their parameters.
    """
    a = np.asarray(a, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    keep = a > 0
    a, w = a[keep], w[keep]
    if len(w) == 0:
        raise ValueError("all Dirichlet parameters are zero")
    wu, inv = np.unique(w, return_inverse=True)
    if len(wu) < len(w):
        a = np.bincount(inv, weights=a, minlength=len(wu))
    else:
        a = a[np.argsort(w, kind="stable")]
    return a, wu


def _saddle(a, ls, z, bound, s0=None):
    """Approximate root of K'(s) = sum a ls / (1 - s ls) between 0 and bound.

    K' is increasing, with a pole at bound.  Bracketed Newton; we stop
    when |K'(s)| <= 0.1 sqrt(K''(s)): then the phase of the integrand
    turns by less than 0.1 radian over the width of its peak.  More
    accuracy is not useful, since any c in the strip gives the exact
    integral.
    """
    # the side of the root is the side of the branch point (bound < 0 for
    # the lower tail), also when z = 0 (y at the mean of Y)
    if bound < 0:
        b_lo, b_hi = bound, 0.0
    else:
        b_lo, b_hi = 0.0, bound
    s = -z if s0 is None else s0  # -z is exact for a normal law
    if not (b_lo < s < b_hi):
        s = 0.5 * bound
    g_prev = math.inf
    for _ in range(100):
        d = 1.0 / (1.0 - s * ls)
        al = a * ls * d
        g = float(al.sum())
        dg = float(al @ (ls * d))
        if g * g <= 0.01 * dg:
            return s
        if g > 0.0:
            b_hi = s
        else:
            b_lo = s
        s_new = s - g / dg
        # bisection when Newton leaves the bracket or converges slowly
        # (close to the pole, K' is far from linear)
        if not (b_lo < s_new < b_hi) or abs(g) > 0.5 * g_prev:
            s_new = 0.5 * (b_lo + b_hi)
        g_prev = abs(g)
        s = s_new
    return s


def _cut_point(a, lp):
    """T with 1/2 sum a log(1 + T^2 lp^2) ~= _LOG_CUT (Newton in log T)."""
    a2 = lp * lp
    t = math.sqrt(2.0 * _LOG_CUT / float(a @ a2))  # exact for large a
    for _ in range(60):
        x = t * t * a2
        g = 0.5 * float(a @ np.log1p(x)) - _LOG_CUT
        if -2.0 < g < 4.0:  # e^-44 < |integrand| < e^-38
            break
        dg = float(a @ (x / (1.0 + x)))  # d g / d log T
        t *= math.exp(-max(min(g / dg, 3.0), -3.0))
    return t


_POW2 = np.concatenate(([0.0], 2.0 ** np.arange(80)))


def _edges(t_max, t_sing):
    """Panel edges 0, h0, 2 h0, ..., h0 2^(m-1), t_max.

    t_sing is the distance from 0 to the nearest singularity.
    """
    h0 = min(t_max, 0.5 * t_sing, 1.0)
    m = max(0, math.ceil(math.log2(t_max / h0) - 1e-9))
    edges = np.empty(m + 2)
    edges[:-1] = h0 * _POW2[: m + 1]
    edges[-1] = t_max
    return edges


def _nodes(edges):
    """Gauss-Legendre nodes and weights on the panels."""
    h = edges[1:] - edges[:-1]
    t = (edges[:-1, None] + h[:, None] * _gl_t).ravel()
    wt = (h[:, None] * _gl_w).ravel()
    return t, wt, h


def _split(edges, h, dtheta):
    """Split the panels on which the phase turns too fast.

    dtheta is the derivative of the phase at the nodes.  A panel of
    length h with |dtheta| <= W has at most W h / 2 radians of phase on
    each side of its center; we keep this below _MAX_HALF_PHASE, where
    the 12-point rule integrates exp(i W t) with an error below 1e-14.
    Returns None when no panel must be split.
    """
    w_max = np.abs(dtheta).reshape(len(h), _N_GL).max(axis=1)
    parts = np.ceil(w_max * h / (2.0 * _MAX_HALF_PHASE))
    if parts.max() <= 1.0:
        return None
    parts = np.maximum(parts, 1.0).astype(np.int64)
    sub = np.repeat(h / parts, parts)
    start = np.repeat(edges[:-1], parts)
    idx = np.arange(len(sub)) - np.repeat(np.cumsum(parts) - parts, parts)
    out = np.empty(len(sub) + 1)
    out[:-1] = start + idx * sub
    out[-1] = edges[-1]
    return out


def _tail_pdf(a, w, y, need_pdf=True, s0=None):
    """Smaller tail of Y at y, and the density.

    y must be strictly inside (min w, max w).  Returns
    (upper, log_tail, log_pdf, s): upper is False when log_tail is
    log P(Y <= y) and True when it is log P(Y > y); s is the saddle
    point (a start point for the next call at a close y).
    """
    lam = w - y
    sd = math.sqrt(float(a @ (lam * lam)))
    ls = lam / sd
    z = float(a @ ls)  # E[Z] / sd(Z), > 0 when y is below the mean
    upper = z < 0.0
    bound = 1.0 / float(ls.max() if upper else ls.min())  # branch point
    s = _saddle(a, ls, z, bound, s0)
    c_min = min(1.0, 0.5 * abs(bound))
    c = s if abs(s) >= c_min else math.copysign(c_min, bound)
    p = 1.0 - c * ls  # > 0 on the strip
    lp = ls / p
    # K(c) = log M(c). log1p(-c ls), not log(p): p = 1 - c ls is rounded to
    # about 1e-16 absolute, and a (up to 1e8 or more) would multiply that
    # error into a relative error of 1e-9 on the tail.
    k_c = -float(a @ np.log1p(-c * ls))
    t_max = _cut_point(a, lp)
    edges = _edges(t_max, min(abs(c), 1.0 / float(np.max(np.abs(lp)))))
    t, wt, h = _nodes(edges)
    x = np.multiply.outer(t, lp)  # (nodes, k)
    x2 = x * x
    d = 1.0 / (1.0 + x2)
    # The phase theta(t) = sum a atan(t lp) has derivative d @ (a lp).
    # Its linear part is zero at the saddle point, but with parameters
    # of very different sizes it can turn fast away from t = 0.
    new_edges = _split(edges, h, d @ (a * lp))
    if new_edges is not None:
        t, wt, h = _nodes(new_edges)
        x = np.multiply.outer(t, lp)
        x2 = x * x
        d = 1.0 / (1.0 + x2)
    theta = np.arctan(x) @ a
    r = np.exp(-0.5 * (np.log1p(x2) @ a))
    cs, sn = np.cos(theta), np.sin(theta)
    integral = float(wt @ (r * (c * cs + t * sn) / (c * c + t * t)))
    tail = integral if upper else -integral
    log_tail = k_c + math.log(tail / math.pi) if tail > 0.0 else -math.inf
    if not need_pdf:
        return upper, log_tail, None, s
    ap = a / p
    s_re = d @ ap
    s_im = (x * d) @ ap
    dens = float(wt @ (r * (cs * s_re - sn * s_im)))
    log_pdf = k_c + math.log(dens / (math.pi * sd)) if dens > 0.0 else -math.inf
    return upper, log_tail, log_pdf, s


def _log_tails(upper, log_tail):
    """(log P(Y <= y), log P(Y > y)) from the smaller tail."""
    other = math.log1p(-math.exp(log_tail)) if log_tail < 0.0 else -math.inf
    return (other, log_tail) if upper else (log_tail, other)


def _cdf_sf(a, w, y):
    a, w = _reduce(a, w)
    y = float(y)
    if len(w) == 1:
        return (0.0, 1.0) if y < w[0] else (1.0, 0.0)
    if y <= w[0]:
        return 0.0, 1.0
    if y >= w[-1]:
        return 1.0, 0.0
    lo, R = float(w[0]), float(w[-1] - w[0])
    upper, lt, _, _ = _tail_pdf(a, (w - lo) / R, (y - lo) / R, need_pdf=False)
    lf, ls = _log_tails(upper, lt)
    return math.exp(lf), math.exp(ls)


def dirichlet_sum_cdf(a, w, y):
    """P(w^T P <= y) for P ~ Dirichlet(a)."""
    return _cdf_sf(a, w, y)[0]


def dirichlet_sum_sf(a, w, y):
    """P(w^T P > y) for P ~ Dirichlet(a)."""
    return _cdf_sf(a, w, y)[1]


def dirichlet_sum_pdf(a, w, y):
    """Density of w^T P at y for P ~ Dirichlet(a), with 2+ distinct values."""
    a, w = _reduce(a, w)
    if len(w) < 2 or not (w[0] < y < w[-1]):
        return 0.0
    lo, R = float(w[0]), float(w[-1] - w[0])
    return math.exp(_tail_pdf(a, (w - lo) / R, (y - lo) / R)[2]) / R


def dirichlet_sum_ppf(a, w, q, *, upper=False, xtol=1e-12, return_info=False):
    """Quantile of Y = w^T P for P ~ Dirichlet(a).

    Returns y with P(Y <= y) = q, or with P(Y > y) = q if upper is True
    (use this form for upper quantiles with a small q: it keeps the
    relative accuracy of q).  xtol is the tolerance on the root,
    relative to max(w) - min(w): the iteration stops when a Newton step
    is smaller than xtol, or when 100 times the predicted error after
    the step (from the curvature) is smaller than xtol.
    """
    a, w = _reduce(a, w)
    info = {"evals": 0, "method": "const"}

    def done(res):
        return (float(res), info) if return_info else float(res)

    if not upper and q > 0.5:
        q, upper = 1.0 - q, True
    elif upper and q > 0.5:
        q, upper = 1.0 - q, False
    lo, hi = float(w[0]), float(w[-1])
    if len(w) == 1:
        return done(lo)
    if q <= 0.0:
        return done(hi if upper else lo)
    R = hi - lo
    w01 = (w - lo) / R
    w01[-1] = 1.0
    if len(w) == 2:
        # Y = lo + R * B with B ~ Beta(a_2, a_1)
        info["method"] = "beta"
        b = betainccinv(a[1], a[0], q) if upper else betaincinv(a[1], a[0], q)
        return done(lo + R * float(b))

    info["method"] = "newton"
    A = float(a.sum())
    m = float(a @ w01) / A
    var = float(a @ (w01 - m) ** 2) / (A * (A + 1.0))
    nu = m * (1.0 - m) / var - 1.0  # Beta(m nu, (1 - m) nu): same moments
    y = float(betainccinv(m * nu, (1.0 - m) * nu, q) if upper
              else betaincinv(m * nu, (1.0 - m) * nu, q))
    if not (0.0 < y < 1.0):
        y = m
    # Work with the distance d to the end of the tail: d = y (lower
    # tail) or d = 1 - y (upper tail).  The tail is increasing in d and
    # close to C d^a near d = 0, so Newton's method runs on
    # log(tail) - log(q) as a function of log(d).
    d = 1.0 - y if upper else y
    log_q = math.log(q)
    d_lo, d_hi = 0.0, 1.0  # the root is in (d_lo, d_hi)
    s0 = None
    u_prev = slope_prev = None
    for _ in range(200):
        up, lt, lpdf, s0 = _tail_pdf(a, w01, 1.0 - d if upper else d, s0=s0)
        info["evals"] += 1
        lf, ls = _log_tails(up, lt)
        lt = ls if upper else lf
        g = lt - log_q
        if g == 0.0:
            break
        if g < 0.0:
            d_lo = d
        else:
            d_hi = d
        # Newton step in u = log(d): d g / d u = d * pdf / tail
        u = math.log(d)
        slope = d * math.exp(lpdf - lt) if lpdf > -math.inf else 0.0
        du = -g / slope if slope > 0.0 else math.nan
        d_new = d * math.exp(max(-50.0, min(50.0, du))) if math.isfinite(du) else math.inf
        inside = d_lo <= d_new <= d_hi
        astep = abs(d_new - d)
        # Error after a Newton step: about |g''/(2 g')| du^2 (in u).  We
        # estimate g'' by the change of the slope since the last point
        # and stop when 100 times this error is below xtol.
        fast = False
        if u_prev is not None and u != u_prev and inside:
            curv = abs(slope - slope_prev) / abs(u - u_prev) / (2.0 * slope)
            fast = 100.0 * d_new * curv * du * du <= xtol
        if astep <= xtol or fast:
            if inside:
                d = d_new
            break
        u_prev, slope_prev = u, slope
        if not inside or d_new in (d_lo, d_hi):
            d_new = 0.5 * (d_lo + d_hi)
        d = d_new
        if d_hi - d_lo <= xtol:
            break
    y = 1.0 - d if upper else d
    return done(lo + R * y)

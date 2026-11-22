"""Exact coverage of the Gaffke interval and of published intervals.

X ~ Mult(n, p) on known values v, scaled to [0, 1]; target mu = v^T p.
For each design (v, n), every interval is computed once for every outcome x,
and the coverage C(p) = sum_x P(x | p) 1[L(x) <= mu(p) <= U(x)] is exact
(full enumeration) on a regular lattice of the simplex, boundary included.

Methods (all published; alpha = 0.05):
  gaffke     Gaffke (2005); valid (Vlassis and Thomas 2026)
  wald       mean +- z * sd / sqrt(n)                (sd with 1/n)
  student    mean +- t_{n-1} * s / sqrt(n)           (s with 1/(n-1))
  hoeffding  mean +- sqrt(log(2/alpha) / (2n))       (Hoeffding 1963)
  bernstein  empirical Bernstein (Maurer and Pontil 2009, Thm 4), delta = alpha/2 per side
  anderson   Anderson (1969): DKW band on the empirical CDF, alpha/2 per side
  betting    hedged capital (Waudby-Smith and Ramdas 2024), order of the trials
             drawn at random: the coverage is averaged over R_PERM orders
  profile    profile likelihood with the chi2_1 cutoff (Wilks)
  jeffreys   Bayesian, equal-tail credible interval, Dirichlet(1/2) prior

Results: experiments/results/coverage.csv (one row per design and method).
Usage: python experiments/coverage.py   (needs the gaffke package)
"""

from __future__ import annotations

import csv
import itertools
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import norm, t as student_t

from gaffke import dirichlet_sum_ppf, gaffke_ci
from profile_ci import profile_ci


def jeffreys_ci(counts, values, alpha=0.05):
    """Equal-tail credible interval with the prior Dirichlet(1/2)."""
    a = np.asarray(counts, dtype=float) + 0.5
    return {"lower": dirichlet_sum_ppf(a, values, alpha / 2),
            "upper": dirichlet_sum_ppf(a, values, alpha / 2, upper=True)}

ALPHA = 0.05
R_PERM = 8  # random orders per outcome for the betting interval
GRID = np.linspace(0.0, 1.0, 1001)  # candidate means for the betting interval
OUT = Path(__file__).parent / "results" / "coverage.csv"

DESIGNS = [  # (name, values, n, lattice resolution)
    ("k3", [0.0, 0.5, 1.0], 5, 80),
    ("k3", [0.0, 0.5, 1.0], 10, 80),
    ("k3", [0.0, 0.5, 1.0], 20, 80),
    ("k3", [0.0, 0.5, 1.0], 30, 80),
    ("k3skew", [0.0, 0.1, 1.0], 10, 80),
    ("k3skew", [0.0, 0.1, 1.0], 30, 80),
    ("k5", [0.0, 0.25, 0.5, 0.75, 1.0], 5, 16),
    ("k5", [0.0, 0.25, 0.5, 0.75, 1.0], 10, 16),
    ("k5", [0.0, 0.25, 0.5, 0.75, 1.0], 20, 16),
]


def compositions(total, k):
    """All x in N^k with sum total."""
    for bars in itertools.combinations(range(total + k - 1), k - 1):
        prev, x = -1, []
        for b in bars:
            x.append(b - prev - 1)
            prev = b
        x.append(total + k - 2 - prev)
        yield x


# ── intervals, vectorized over outcomes (rows of X) ────────────────────────

def moments(X, v):
    n = X.sum(1)
    m = X @ v / n
    var = np.maximum(X @ (v * v) / n - m * m, 0.0)
    return n, m, var


def wald(X, v):
    n, m, var = moments(X, v)
    h = norm.ppf(1 - ALPHA / 2) * np.sqrt(var / n)
    return m - h, m + h


def student(X, v):
    n, m, var = moments(X, v)
    s2 = var * n / (n - 1)
    h = student_t.ppf(1 - ALPHA / 2, n - 1) * np.sqrt(s2 / n)
    return m - h, m + h


def hoeffding(X, v):
    n, m, _ = moments(X, v)
    h = np.sqrt(math.log(2 / ALPHA) / (2 * n))
    return m - h, m + h


def bernstein(X, v):
    n, m, var = moments(X, v)
    s2 = var * n / (n - 1)
    lg = math.log(4 / ALPHA)
    h = np.sqrt(2 * s2 * lg / n) + 7 * lg / (3 * (n - 1))
    return m - h, m + h


def _anderson_upper(X, w):
    """Upper bound on E[W], W in [0, 1] with values w: E W = 1 - int_0^1 F,
    and F >= F_hat - eps with probability >= 1 - alpha/2 (one-sided DKW)."""
    n = X.sum(1)[0]
    eps = math.sqrt(math.log(2 / ALPHA) / (2 * n))
    order = np.argsort(w)
    cum = np.cumsum(X[:, order], 1) / n  # F_hat on [w_(j), w_(j+1))
    gaps = np.diff(np.append(w[order], 1.0))
    return 1.0 - (np.maximum(cum - eps, 0.0) * gaps).sum(1)


def anderson(X, v):
    """Anderson (1969): the DKW band on the empirical CDF, alpha/2 per side."""
    return 1.0 - _anderson_upper(X, 1.0 - v), _anderson_upper(X, v)


def betting_one(z):
    """Hedged capital interval of Waudby-Smith and Ramdas (2024) for a fixed
    n: predictable plug-in bets lambda_t = sqrt(2 log(2/alpha) / (n
    sigma_{t-1}^2)), truncated at 1/2m and 1/2(1-m); capital
    max(K+/2, K-/2); the interval is the set of m whose capital stays
    below 1/alpha at every t (Ville). z: data in [0, 1], in trial order."""
    n = len(z)
    t = np.arange(1, n + 1)
    csum = np.cumsum(z)
    mu_hat = (0.5 + csum) / (t + 1)
    sig2 = (0.25 + np.cumsum((z - mu_hat) ** 2)) / (t + 1)
    sig2_prev = np.concatenate([[0.25], sig2[:-1]])
    lam = np.sqrt(2 * math.log(2 / ALPHA) / (n * sig2_prev))
    m = GRID[:, None]
    lp = np.minimum(lam[None, :], 0.5 / np.maximum(m, 1e-12))
    lm = np.minimum(lam[None, :], 0.5 / np.maximum(1 - m, 1e-12))
    kp = np.cumsum(np.log1p(lp * (z[None, :] - m)), 1)
    km = np.cumsum(np.log1p(-lm * (z[None, :] - m)), 1)
    k = np.maximum(kp, km) + math.log(0.5)
    keep = (k < math.log(1 / ALPHA)).all(1)
    if not keep.any():
        return z.mean(), z.mean()
    return GRID[keep].min(), GRID[keep].max()


def betting(X, v, rng):
    """R_PERM random orders per outcome: returns arrays (N, R_PERM)."""
    N = len(X)
    L = np.empty((N, R_PERM))
    U = np.empty((N, R_PERM))
    for r, x in enumerate(X):
        z = np.repeat(v, x)
        for j in range(R_PERM):
            L[r, j], U[r, j] = betting_one(rng.permutation(z))
    return L, U


def per_outcome(fn, X, v):
    L = np.empty(len(X))
    U = np.empty(len(X))
    for r, x in enumerate(X):
        res = fn(x, v)
        L[r], U[r] = res["lower"], res["upper"]
    return L, U


# ── exact coverage on a lattice of p ──────────────────────────────────────

def coverage(X, v, L, U, n, res):
    """Coverage and expected width at every lattice point p = j / res."""
    P = np.array(list(compositions(res, len(v))), dtype=float) / res
    mu = P @ v
    logc = gammaln(n + 1) - gammaln(X + 1).sum(1)  # (N,)
    width = (U - L).mean(1) if U.ndim == 2 else U - L
    cov = np.empty(len(P))
    wid = np.empty(len(P))
    for s in range(0, len(P), 256):
        Pc = P[s:s + 256]
        lp = logc[:, None] + xlogy(X[:, :, None], Pc.T[None, :, :]).sum(1)
        w = np.exp(lp)  # (N, chunk)
        m = mu[s:s + 256][None, :]
        if L.ndim == 2:  # several random orders: average of the indicators
            ins = ((L[:, :, None] <= m[:, None, :] + 1e-12)
                   & (m[:, None, :] - 1e-12 <= U[:, :, None])).mean(1)
        else:
            ins = (L[:, None] <= m + 1e-12) & (m - 1e-12 <= U[:, None])
        cov[s:s + 256] = (w * ins).sum(0)
        wid[s:s + 256] = w.T @ width
    return P, cov, wid


def main():
    rng = np.random.default_rng(1)
    rows = []
    for name, v, n, res in DESIGNS:
        v = np.asarray(v)
        X = np.array(list(compositions(n, len(v))), dtype=float)
        Xi = X.astype(int)
        methods = {
            "gaffke": lambda: per_outcome(lambda x, vv: gaffke_ci(x, vv, ALPHA), Xi, v),
            "wald": lambda: wald(X, v),
            "student": lambda: student(X, v),
            "hoeffding": lambda: hoeffding(X, v),
            "bernstein": lambda: bernstein(X, v),
            "anderson": lambda: anderson(X, v),
            "betting": lambda: betting(Xi, v, rng),
            "profile": lambda: per_outcome(lambda x, vv: profile_ci(x, vv, ALPHA), Xi, v),
            "jeffreys": lambda: per_outcome(lambda x, vv: jeffreys_ci(x, vv, ALPHA), Xi, v),
        }
        for meth, fn in methods.items():
            t0 = time.perf_counter()
            L, U = fn()
            L = np.clip(L, 0.0, 1.0)
            U = np.clip(U, 0.0, 1.0)
            sec = time.perf_counter() - t0
            P, cov, wid = coverage(X, v, L, U, n, res)
            interior = (P > 0).all(1)
            row = dict(design=name, k=len(v), n=n, method=meth,
                       min_cov=cov.min(), min_cov_interior=cov[interior].min(),
                       mean_cov=cov.mean(), frac_below=(cov < 1 - ALPHA - 1e-9).mean(),
                       mean_width=wid.mean(), sec=sec, points=len(P))
            rows.append(row)
            print({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in row.items()},
                  flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

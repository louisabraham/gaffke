"""Tests for the Gaffke interval and the Dirichlet weighted-sum quantile.

References used here:
- exact CDF for integer Dirichlet parameters (residue formula, rational
  arithmetic): Y = v^T P is a B-spline with knot v_i of multiplicity a_i;
- Beta law for two distinct values (mpmath, 40 digits);
- Gil-Pelaez integral in mpmath (40 digits) for large n;
- Monte Carlo with 2e6 Dirichlet samples;
- exact enumeration of the multinomial law for the coverage.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from math import comb

import mpmath as mp
import numpy as np
import pytest
from scipy.special import gammaln, xlogy

from gaffke import gaffke_ci
from gaffke.dirichlet_sum import (
    dirichlet_sum_cdf,
    dirichlet_sum_pdf,
    dirichlet_sum_ppf,
    dirichlet_sum_sf,
)

# ── Exact references ─────────────────────────────────────────────────────────


def _series_pow(d, e, L):
    """Taylor coefficients of (d + s)^e at s = 0, up to s^(L-1)."""
    out = []
    for r in range(L):
        if e >= 0:
            c = comb(e, r) if r <= e else 0
        else:
            c = (-1) ** r * comb(-e + r - 1, r)
        out.append(c * d ** (e - r) if c else Fraction(0))
    return out


def _mul(a, b, L):
    out = [Fraction(0)] * L
    for i, x in enumerate(a):
        if x:
            for j in range(L - i):
                out[i + j] += x * b[j]
    return out


def exact_sf(a, v, y, power=None):
    """P(v^T P > y), P ~ Dir(a), integer a, distinct v (exact rationals).

    P(Y > y) = sum over v_j > y of Res_{t = v_j} (t - y)^m / prod_l (t - v_l)^a_l
    with m = sum(a) - 1 (Hermite-Genocchi formula for the divided
    difference of (t - y)_+^m).  power = m - 1 gives the density / m.
    """
    a = [int(t) for t in a]
    v = [Fraction(t) for t in v]
    y = Fraction(y)
    m = sum(a) - 1 if power is None else power
    tot = Fraction(0)
    for j, (aj, vj) in enumerate(zip(a, v)):
        if vj <= y:
            continue
        s = _series_pow(vj - y, m, aj)
        for l, (al, vl) in enumerate(zip(a, v)):
            if l != j:
                s = _mul(s, _series_pow(vj - vl, -al, aj), aj)
        tot += s[aj - 1]
    return tot


def exact_pdf(a, v, y):
    m = sum(int(t) for t in a) - 1
    return m * exact_sf(a, v, y, power=m - 1)


def mp_cdf(a, w, y, dps=40):
    """P(w^T P <= y) by the Gil-Pelaez integral in mpmath."""
    with mp.workdps(dps):
        a = [mp.mpf(float(t)) for t in a]
        lam = [mp.mpf(float(t)) - mp.mpf(float(y)) for t in w]
        sd = mp.sqrt(sum(ai * li * li for ai, li in zip(a, lam)))
        ls = [li / sd for li in lam]

        def f(v):
            th = sum(ai * mp.atan(v * li) for ai, li in zip(a, ls))
            lr = -sum(ai * mp.log1p(v * v * li * li) for ai, li in zip(a, ls)) / 2
            return mp.sin(th) * mp.exp(lr) / v

        pts = [0] + [mp.mpf(2) ** j / 64 for j in range(40)]
        return mp.mpf(1) / 2 - mp.quad(f, pts) / mp.pi


def _random_integer_case(rng, kmin=3, kmax=7, amax=8):
    k = int(rng.integers(kmin, kmax + 1))
    a = [int(t) for t in rng.integers(1, amax + 1, k)]
    num = np.sort(rng.choice(np.arange(0, 50), k, replace=False))
    v = [Fraction(int(t), 49) for t in num]
    return a, v, np.array([float(t) for t in v])


# ── CDF, density and quantile against exact values ───────────────────────────


@pytest.mark.parametrize("seed", range(6))
def test_cdf_matches_exact_formula(seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):
        a, v, vf = _random_integer_case(rng)
        ys = list(rng.uniform(vf[0], vf[-1], 3))
        ys += [vf[1] + 1e-9, vf[1] - 1e-6, vf[0] + 1e-3, vf[-1] - 1e-3]
        for y in ys:
            sf = exact_sf(a, v, Fraction(y))
            F, S = float(1 - sf), float(sf)
            tail, got = (F, dirichlet_sum_cdf(a, vf, y)) if F < 0.5 else (S, dirichlet_sum_sf(a, vf, y))
            # relative error on the smaller tail
            assert abs(got - tail) <= 1e-10 * tail + 1e-300, (a, vf, y, tail, got)


@pytest.mark.parametrize("seed", range(3))
def test_pdf_matches_exact_formula(seed):
    rng = np.random.default_rng(100 + seed)
    for _ in range(30):
        a, v, vf = _random_integer_case(rng)
        for y in rng.uniform(vf[0], vf[-1], 3):
            ref = float(exact_pdf(a, v, Fraction(y)))
            got = dirichlet_sum_pdf(a, vf, y)
            assert got == pytest.approx(ref, rel=1e-8, abs=1e-300)


@pytest.mark.parametrize("seed", range(4))
def test_quantile_error_below_1e12_of_range(seed):
    """Error of the root, from the exact CDF and density at the root."""
    rng = np.random.default_rng(200 + seed)
    for _ in range(25):
        a, v, vf = _random_integer_case(rng)
        for q in [1e-10, 1e-6, 0.025, 0.3, 0.5]:
            for upper in [False, True]:
                y = dirichlet_sum_ppf(a, vf, q, upper=upper)
                sf = exact_sf(a, v, Fraction(y))
                target = Fraction(q)
                g = (target - sf) if upper else (1 - sf - target)
                err = float(g / exact_pdf(a, v, Fraction(y)))
                assert abs(err) <= 1e-12 * (vf[-1] - vf[0]), (a, vf, q, upper, err)


# ── Two distinct values: Beta law ────────────────────────────────────────────


def _beta_tails(a, b, x, big):
    """(P(B <= x), P(B > x), density) for B ~ Beta(a, b)."""
    if not big:
        with mp.workdps(40):
            xm = mp.mpf(x)
            lo = mp.betainc(a, b, 0, xm, regularized=True)
            dens = xm ** (a - 1) * (1 - xm) ** (b - 1) / mp.beta(a, b)
            return float(lo), float(1 - lo), float(dens)
    from scipy.special import betainc, betaincc
    from scipy.stats import beta as beta_law

    return float(betainc(a, b, x)), float(betaincc(a, b, x)), float(beta_law.pdf(x, a, b))


@pytest.mark.parametrize(
    "counts",
    [[0, 3], [3, 0], [1, 1], [7, 2], [999, 1], [10**6, 3], [12345, 67890], [10**9, 10**9]],
)
@pytest.mark.parametrize("alpha", [0.2, 0.05, 1e-6, 1e-12])
def test_two_values_beta(counts, alpha):
    """Reference: mpmath (40 digits) for counts <= 1000, else scipy betainc."""
    v = [-1.5, 2.0]
    r = gaffke_ci(counts, v, alpha)
    R = v[1] - v[0]
    big = max(counts) > 1000
    c0, c1 = counts
    # lower: P ~ Dir(c0 + 1, c1), Y = v0 + R B with B ~ Beta(c1, c0 + 1)
    if c1 > 0:
        F, _, dens = _beta_tails(c1, c0 + 1, (r["lower"] - v[0]) / R, big)
        assert abs(F - alpha / 2) / dens <= 1e-12
    else:
        assert r["lower"] == v[0]
    # upper: P ~ Dir(c0, c1 + 1), Y = v0 + R B with B ~ Beta(c1 + 1, c0)
    if c0 > 0:
        _, S, dens = _beta_tails(c1 + 1, c0, (r["upper"] - v[0]) / R, big)
        assert abs(S - alpha / 2) / dens <= 1e-12
    else:
        assert r["upper"] == v[1]


# ── Large n: high-precision quadrature and Monte Carlo ───────────────────────


@pytest.mark.parametrize("n", [10**4, 10**6, 10**8])
@pytest.mark.parametrize("q", [5e-7, 0.025])
def test_large_n_against_mpmath(n, q):
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a = n * np.array([0.1, 0.1, 0.2, 0.3, 0.3])
    a[0] += 1.0
    y = dirichlet_sum_ppf(a, w, q)
    F = mp_cdf(a, w, y)
    err = float((F - q)) / dirichlet_sum_pdf(a, w, y)
    assert abs(err) <= 1e-12 * (w[-1] - w[0])
    # relative accuracy of the CDF itself in the tail
    assert dirichlet_sum_cdf(a, w, y) == pytest.approx(float(F), rel=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_monte_carlo(seed):
    rng = np.random.default_rng(300 + seed)
    k = int(rng.integers(3, 12))
    v = np.sort(rng.normal(size=k))
    n = int(10 ** rng.uniform(0.5, 3))
    x = rng.multinomial(n, rng.dirichlet(np.ones(k)))
    a = x.astype(float)
    a[0] += 1.0
    m = 2_000_000
    sample = rng.dirichlet(a + 1e-300, m) @ v  # 1e-300: zero parameters
    for q in [0.001, 0.025, 0.5]:
        for upper in [False, True]:
            y = dirichlet_sum_ppf(a, v, q, upper=upper)
            frac = np.mean(sample > y) if upper else np.mean(sample <= y)
            assert abs(frac - q) <= 5 * math.sqrt(q * (1 - q) / m) + 1e-6


# Hard cases found by random search: parameters of very different sizes
# (fast phase away from t = 0) and a saddle point close to the pole.
HARD_CASES = [
    ([21342.0, 2.0, 1.0], [0.0, 0.2285115, 1.0], 5.693413709578831e-06, True),
    ([11.0, 18.0, 35.0, 243.0, 124.0], [0.0, 1.0, 2.0, 3.0, 4.0], 7.867670246976742e-11, False),
    ([1.0, 630.0, 11.0, 1.0, 2.0], [0.0, 0.07010751, 0.27765248, 0.60529453, 1.0], 2.562e-08, False),
    ([1.0, 292.0, 16.0, 1.0], [0.0, 0.66917395, 0.67871415, 1.0], 2.868e-09, True),
    ([366.0, 1.0, 5.0, 1.0], [0.0, 0.29750274, 0.37047708, 1.0], 3.184e-12, True),
]


def _check_against_mpmath(a, w, q, upper):
    a, w = np.asarray(a, float), np.asarray(w, float)
    y = dirichlet_sum_ppf(a, w, q, upper=upper)
    F = mp_cdf(a, w, y)
    tail = float(1 - F) if upper else float(F)
    got = dirichlet_sum_sf(a, w, y) if upper else dirichlet_sum_cdf(a, w, y)
    assert got == pytest.approx(tail, rel=1e-9)
    err = (tail - q) / dirichlet_sum_pdf(a, w, y)
    assert abs(err) <= 1e-12 * (w.max() - w.min())


@pytest.mark.parametrize("case", HARD_CASES)
def test_hard_cases_against_mpmath(case):
    _check_against_mpmath(*case)


@pytest.mark.parametrize("seed", range(20))
def test_random_skewed_cases_against_mpmath(seed):
    rng = np.random.default_rng(900 + seed)
    k = int(rng.integers(3, 8))
    a = np.round(10 ** rng.uniform(0, 5, k))
    a[rng.integers(k)] = 1.0
    w = np.sort(rng.uniform(size=k))
    q = 10 ** rng.uniform(-12, -0.5)
    _check_against_mpmath(a, w, q, bool(rng.integers(2)))


# ── Invariances ──────────────────────────────────────────────────────────────


def _close(x, y, scale):
    return abs(x - y) <= 1e-11 * scale


@pytest.mark.parametrize("seed", range(5))
def test_affine_invariance(seed):
    rng = np.random.default_rng(400 + seed)
    k = int(rng.integers(3, 15))
    v = rng.normal(size=k)
    x = rng.multinomial(int(rng.integers(1, 500)), rng.dirichlet(np.ones(k)))
    r = gaffke_ci(x, v, 0.05)
    R = v.max() - v.min()
    for b, c in [(1e-9, 0.0), (1e9, 0.0), (3.0, 1e6), (2.5, -7.0)]:
        s = gaffke_ci(x, b * v + c, 0.05)
        # b * v + c is rounded: allow a few ulps of |c| on top of 1e-11 R
        tol = b * R + 1e-4 * abs(c)
        assert _close(s["lower"], b * r["lower"] + c, tol)
        assert _close(s["upper"], b * r["upper"] + c, tol)
    # a negative factor exchanges the two endpoints
    s = gaffke_ci(x, -2.0 * v, 0.05)
    assert _close(s["lower"], -2.0 * r["upper"], 2 * R)
    assert _close(s["upper"], -2.0 * r["lower"], 2 * R)


@pytest.mark.parametrize("seed", range(5))
def test_permutation_invariance(seed):
    rng = np.random.default_rng(500 + seed)
    k = int(rng.integers(3, 15))
    v = rng.uniform(size=k)
    x = rng.multinomial(int(rng.integers(0, 300)), rng.dirichlet(np.ones(k)))
    r = gaffke_ci(x, v, 0.01)
    perm = rng.permutation(k)
    s = gaffke_ci(x[perm], v[perm], 0.01)
    assert s["lower"] == r["lower"] and s["upper"] == r["upper"]


@pytest.mark.parametrize("seed", range(5))
def test_split_category_with_equal_value(seed):
    """Two categories with the same value act as one (Dirichlet aggregation)."""
    rng = np.random.default_rng(600 + seed)
    k = int(rng.integers(3, 10))
    v = rng.uniform(size=k)
    x = rng.multinomial(int(rng.integers(1, 300)), rng.dirichlet(np.ones(k)))
    j = int(rng.integers(k))
    part = int(rng.integers(0, x[j] + 1))
    x2 = np.append(x, part)
    x2[j] -= part
    v2 = np.append(v, v[j])
    r, s = gaffke_ci(x, v, 0.05), gaffke_ci(x2, v2, 0.05)
    assert s == r


# ── Monotonicity ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_monotone_in_alpha(seed):
    rng = np.random.default_rng(700 + seed)
    k = int(rng.integers(3, 12))
    v = rng.uniform(size=k)
    x = rng.multinomial(int(10 ** rng.uniform(0, 5)), rng.dirichlet(np.ones(k)))
    alphas = [1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.999]
    rs = [gaffke_ci(x, v, al) for al in alphas]
    lo = [r["lower"] for r in rs]
    hi = [r["upper"] for r in rs]
    assert all(b >= a for a, b in zip(lo, lo[1:]))
    assert all(b <= a for a, b in zip(hi, hi[1:]))
    assert all(a <= b for a, b in zip(lo, hi))
    assert all(v.min() <= t <= v.max() for t in lo + hi)


@pytest.mark.parametrize("seed", range(4))
def test_monotone_in_counts(seed):
    """Moving one count to a category with a larger value moves both endpoints up."""
    rng = np.random.default_rng(800 + seed)
    k = int(rng.integers(3, 10))
    v = np.sort(rng.uniform(size=k))
    x = rng.multinomial(int(rng.integers(2, 200)), rng.dirichlet(np.ones(k)))
    r = gaffke_ci(x, v)
    for i in range(k - 1):
        if x[i] == 0:
            continue
        x2 = x.copy()
        x2[i] -= 1
        x2[i + 1] += 1
        s = gaffke_ci(x2, v)
        assert s["lower"] >= r["lower"] - 1e-12
        assert s["upper"] >= r["upper"] - 1e-12


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_no_data_gives_full_range():
    assert gaffke_ci([0, 0, 0, 0], [1, 2, 3, 4]) == {"lower": 1.0, "upper": 4.0}


def test_single_category():
    assert gaffke_ci([5], [2.5]) == {"lower": 2.5, "upper": 2.5}
    assert gaffke_ci([0], [2.5]) == {"lower": 2.5, "upper": 2.5}


def test_all_values_equal():
    assert gaffke_ci([3, 4, 5], [7.0, 7.0, 7.0]) == {"lower": 7.0, "upper": 7.0}


def test_one_observation():
    # x = e_2: lower is a quantile of Dir(1, 1, 0) -> Beta; upper of Dir(0, 1, 1)
    r = gaffke_ci([0, 1, 0], [0.0, 0.5, 1.0], 0.1)
    assert r["lower"] == pytest.approx(0.5 * 0.05, rel=1e-12)  # Uniform on [0, 0.5]
    assert r["upper"] == pytest.approx(1.0 - 0.5 * 0.05, rel=1e-12)


def test_all_mass_on_one_inner_category():
    n = 50
    r = gaffke_ci([0, n, 0], [0.0, 0.5, 1.0], 0.05)
    # lower: 0.5 * B, B ~ Beta(n, 1), quantile 0.025 is 0.025^(1/n)
    assert r["lower"] == pytest.approx(0.5 * 0.025 ** (1 / n), rel=1e-12)
    assert r["upper"] == pytest.approx(1.0 - 0.5 * 0.025 ** (1 / n), rel=1e-12)


def test_ties_at_the_extremes():
    r = gaffke_ci([2, 3, 4, 1], [0.0, 0.0, 1.0, 1.0], 0.05)
    s = gaffke_ci([5, 5], [0.0, 1.0], 0.05)
    assert r == s


@pytest.mark.parametrize("n", [10**7, 10**9, 10**12])
def test_huge_n(n):
    v = np.array([0.0, 0.2, 0.7, 1.0])
    p = np.array([0.1, 0.4, 0.3, 0.2])
    x = np.round(n * p)
    r = gaffke_ci(x, v, 1e-6)
    mu = float(p @ v)
    sd = math.sqrt(float(p @ (v - mu) ** 2) / n)
    assert r["lower"] < mu < r["upper"]
    # normal limit: half width about 4.89 sd
    assert (mu - r["lower"]) / sd == pytest.approx(4.89, rel=0.02)
    assert (r["upper"] - mu) / sd == pytest.approx(4.89, rel=0.02)


@pytest.mark.parametrize("alpha", [1e-15, 1e-10, 0.5, 0.999])
def test_extreme_alpha(alpha):
    x, v = [3, 0, 5, 2, 0, 1], [0.1, 0.3, 0.35, 0.6, 0.9, 1.0]
    r = gaffke_ci(x, v, alpha)
    assert 0.1 <= r["lower"] <= r["upper"] <= 1.0
    a_lo = np.array(x, float)
    a_lo[0] += 1
    assert dirichlet_sum_cdf(a_lo, v, r["lower"]) == pytest.approx(alpha / 2, rel=1e-9)
    a_hi = np.array(x, float)
    a_hi[-1] += 1
    assert dirichlet_sum_sf(a_hi, v, r["upper"]) == pytest.approx(alpha / 2, rel=1e-9)


def test_invalid_input():
    with pytest.raises(ValueError):
        gaffke_ci([1, 2], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gaffke_ci([1, -2, 3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gaffke_ci([1, 2], [1.0, np.nan])
    with pytest.raises(ValueError):
        gaffke_ci([1, 2], [1.0, 2.0], alpha=0.0)


# ── Coverage by exact enumeration ────────────────────────────────────────────


def _compositions(n, k):
    for c in itertools.combinations(range(n + k - 1), k - 1):
        b = (-1,) + c + (n + k - 1,)
        yield [b[i + 1] - b[i] - 1 for i in range(k)]


def _worst_one_sided_coverage(n, v, alpha, n_grid=60):
    """Minimum over p of P(L <= mu) and of P(U >= mu), k = 3, v = (0, v1, 1).

    The coverage of L is constant in p on {mu(p) in (L_(j), L_(j+1))}
    except through P_p, so we scan the level sets mu(p) = t for t just
    above and below each endpoint value, along the segment of the
    simplex with that mean.
    """
    X = np.array(list(_compositions(n, 3)))
    ci = [gaffke_ci(x, v, alpha) for x in X]
    L = np.array([c["lower"] for c in ci])
    U = np.array([c["upper"] for c in ci])
    logc = gammaln(n + 1) - gammaln(X + 1).sum(1)
    eps = 1e-9
    mus = np.concatenate([L - eps, L + eps, U - eps, U + eps, np.linspace(0, 1, 41)])
    mus = np.unique(np.clip(mus, 0.0, 1.0))
    worst_lo = worst_hi = 1.0
    v1 = v[1]
    for mu in mus:
        # p = (p0, p1, p2) with p1 v1 + p2 = mu, p0 + p1 + p2 = 1
        p1_max = min(1.0, mu / v1 if v1 > 0 else 1.0, (1 - mu) / (1 - v1) if v1 < 1 else 1.0)
        for p1 in np.linspace(0.0, p1_max, n_grid):
            p2 = mu - p1 * v1
            p0 = 1.0 - p1 - p2
            p = np.clip([p0, p1, p2], 0.0, 1.0)
            pr = np.exp(logc + xlogy(X, p).sum(1))
            worst_lo = min(worst_lo, pr[L <= mu].sum())
            worst_hi = min(worst_hi, pr[U >= mu].sum())
    return worst_lo, worst_hi


@pytest.mark.parametrize("v1", [0.5, 0.1, 0.9])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_coverage_exact_enumeration(v1, n, alpha):
    v = np.array([0.0, v1, 1.0])
    lo, hi = _worst_one_sided_coverage(n, v, alpha)
    assert lo >= 1 - alpha / 2 - 1e-9
    assert hi >= 1 - alpha / 2 - 1e-9


def test_cdf_at_the_mean():
    # regression: y exactly at the mean of Y (z = 0) chose the wrong side
    # for the saddle point search
    from scipy.stats import beta
    assert dirichlet_sum_cdf([3.0, 3.0], [0.0, 1.0], 0.5) == pytest.approx(0.5, abs=1e-14)
    assert dirichlet_sum_cdf([2.0, 4.0], [0.0, 1.0], 4 / 6) == pytest.approx(beta.cdf(4 / 6, 4, 2), rel=1e-12)
    assert dirichlet_sum_cdf([2.0, 2.0, 2.0], [0.0, 0.5, 1.0], 0.5) == pytest.approx(0.5, abs=1e-14)

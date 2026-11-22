"""Gaffke's test and confidence interval for the mean of bounded data.

Draft for scipy.stats (the module and function names follow the stats
conventions; the axis/nan_policy decorator and the array API declaration are
to be added when the code moves into a SciPy checkout).
"""

from dataclasses import dataclass

import numpy as np

from _dirichlet_sum import _cdf_sf, dirichlet_sum_ppf

__all__ = ["gaffke_test"]


@dataclass
class ConfidenceInterval:
    low: float
    high: float


@dataclass
class GaffkeTestResult:
    """Result of `gaffke_test`.

    Attributes
    ----------
    statistic : float
        The sample mean.
    pvalue : float
        The p-value of the test.
    """

    statistic: float
    pvalue: float
    _x: np.ndarray
    _bounds: tuple
    _alternative: str

    def confidence_interval(self, confidence_level=0.95):
        """Confidence interval for the mean.

        The interval is the set of values ``popmean`` that the test does not
        reject at level ``1 - confidence_level`` (with the same
        ``alternative``). Its coverage is at least ``confidence_level`` for
        every distribution on ``bounds`` and every sample size.

        Parameters
        ----------
        confidence_level : float, optional
            Default 0.95.

        Returns
        -------
        ci : ConfidenceInterval
            Object with attributes ``low`` and ``high``.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("`confidence_level` must be in (0, 1).")
        lb, ub = self._bounds
        alpha = 1 - confidence_level
        if self._alternative == "two-sided":
            alpha = alpha / 2
        low, high = lb, ub
        if self._alternative in ("two-sided", "greater"):
            a, w = _weights(self._x, lb)
            low = dirichlet_sum_ppf(a, w, alpha)
        if self._alternative in ("two-sided", "less"):
            a, w = _weights(self._x, ub)
            high = dirichlet_sum_ppf(a, w, alpha, upper=True)
        return ConfidenceInterval(low=float(low), high=float(high))


def _weights(x, extra):
    """Dirichlet(1, ..., 1) weights on the n observations and on one more
    value `extra` (a bound), merged by value: Y = sum_i D_i x_i + D_0 extra."""
    return np.ones(len(x) + 1), np.append(x, extra)


def gaffke_test(x, popmean, *, bounds=(0, 1), alternative="two-sided"):
    r"""Gaffke's test for the mean of a bounded variable.

    For independent observations in a known interval ``bounds = (lb, ub)``,
    test the null hypothesis that their mean is ``popmean``. The test makes
    no other assumption on the distribution, and its level is exact in the
    sense of the Clopper--Pearson interval: the probability of a false
    rejection is at most the nominal level for every sample size.

    Parameters
    ----------
    x : array_like
        One-dimensional sample, with values in ``bounds``.
    popmean : float
        Mean under the null hypothesis.
    bounds : tuple of two floats, optional
        Lower and upper bounds of the support, ``lb < ub``. Default (0, 1).
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis:

        * 'two-sided': the mean is not ``popmean``.
        * 'less': the mean is less than ``popmean``.
        * 'greater': the mean is greater than ``popmean``.

    Returns
    -------
    result : GaffkeTestResult
        Object with attributes ``statistic`` (the sample mean) and
        ``pvalue``, and the method ``confidence_interval``.

    Notes
    -----
    Let :math:`D = (D_0, \dots, D_n)` follow the uniform Dirichlet law,
    independent of the sample. For ``alternative='greater'`` the p-value is

    .. math::

        p = \Pr_D\Big(D_0\,\mathrm{lb} + \sum_{i=1}^n D_i x_i \le \mu_0\Big),

    the probability that a random average of the sample, with one more
    observation at the lower bound, is at most :math:`\mu_0`. For
    ``alternative='less'`` the extra observation is at the upper bound and
    the inequality is reversed. The two-sided p-value is twice the smaller
    one-sided p-value, at most 1.

    The test was proposed by Gaffke [1]_ with the conjecture that it is
    valid; the conjecture was proved by Vlassis and Thomas [2]_. For
    observations in :math:`\{0, 1\}` the confidence interval is the
    Clopper--Pearson interval. Learned-Miller and Thomas [3]_ study the
    interval, and Ming et al. [4]_ show that its width is asymptotically
    that of the normal interval.

    The probabilities above are tail probabilities of a weighted sum of
    Dirichlet variables. They are computed by a Fourier inversion integral
    on a line through the saddle point, which gives them with a small
    relative error, also far in the tails [5]_.

    References
    ----------
    .. [1] N. Gaffke, "Three test statistics for a nonparametric one-sided
           hypothesis on the mean of a nonnegative variable", Mathematical
           Methods of Statistics 14(4), 451-467, 2005.
    .. [2] N. Vlassis and P. S. Thomas, "An exact distribution-free test for
           means of nonnegative random variables", arXiv:2607.08415, 2026.
    .. [3] E. Learned-Miller and P. S. Thomas, "A new confidence interval for
           the mean of a bounded random variable", arXiv:1905.06208, 2019.
    .. [4] J. Ming, A. Ramdas, Y. Shen, R. Wang and I. Waudby-Smith,
           "Gaffke's confidence interval for the mean of bounded data is
           inadmissible but asymptotically efficient", arXiv:2607.18661, 2026.
    .. [5] L. Abraham, "Exact, narrow and fast: Use the Gaffke confidence
           interval for the mean of a discrete bounded variable", 2026.

    Examples
    --------
    Ratings on a scale from 1 to 5. Is the mean rating above 3?

    >>> import numpy as np
    >>> from scipy import stats
    >>> x = np.array([4, 5, 3, 4, 4, 2, 5, 3, 4, 1])
    >>> res = stats.gaffke_test(x, popmean=3, bounds=(1, 5), alternative='greater')
    >>> res.statistic
    3.5
    >>> ci = res.confidence_interval(confidence_level=0.95)

    For two-valued data the interval is the Clopper--Pearson interval:

    >>> x = np.array([1] * 4 + [0] * 6)
    >>> ci = stats.gaffke_test(x, 0.5).confidence_interval()
    >>> ci_cp = stats.binomtest(4, 10).proportion_ci(method='exact')
    >>> bool(np.allclose([ci.low, ci.high], [ci_cp.low, ci_cp.high]))
    True
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("`x` must be one-dimensional.")
    if x.size == 0:
        raise ValueError("`x` must contain at least one observation.")
    if np.isnan(x).any():
        raise ValueError("`x` must not contain NaN.")
    lb, ub = (float(b) for b in bounds)
    if not (np.isfinite(lb) and np.isfinite(ub) and lb < ub):
        raise ValueError("`bounds` must be finite with lb < ub.")
    if x.min() < lb or x.max() > ub:
        raise ValueError("All observations must be in `bounds`.")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("`alternative` must be 'two-sided', 'less' or 'greater'.")
    popmean = float(popmean)

    # p-value for 'greater': P(lb D_0 + sum D_i x_i <= popmean)
    a, w = _weights(x, lb)
    p_greater = _cdf_sf(a, w, popmean)[0]
    # p-value for 'less': P(ub D_0 + sum D_i x_i >= popmean)
    a, w = _weights(x, ub)
    cdf, sf = _cdf_sf(a, w, popmean)
    # P(Y >= popmean) = sf + P(Y = popmean), and Y is continuous unless all
    # its values are equal (then cdf/sf are 0/1 and Y = popmean is possible)
    p_less = sf if len(np.unique(w)) > 1 else float(w[0] >= popmean)
    if len(np.unique(np.append(x, lb))) == 1:
        p_greater = float(lb <= popmean)
    if alternative == "greater":
        pvalue = p_greater
    elif alternative == "less":
        pvalue = p_less
    else:
        pvalue = min(1.0, 2 * min(p_greater, p_less))
    return GaffkeTestResult(statistic=float(x.mean()), pvalue=float(pvalue),
                            _x=x, _bounds=(lb, ub), _alternative=alternative)

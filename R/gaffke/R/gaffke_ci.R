#' Gaffke confidence interval for the mean of a discrete bounded variable
#'
#' Each of \eqn{n} independent trials gives one of \eqn{k} known values
#' \code{values}, with unknown probabilities \eqn{p}; \code{counts} are the
#' numbers of trials with each value. The function returns the Gaffke
#' interval for the mean \eqn{\mu = \sum_i v_i p_i}:
#' \deqn{L = Q_{\alpha/2}(v^\top P_-),\quad U = Q_{1-\alpha/2}(v^\top P_+),}
#' with \eqn{P_- \sim \mathrm{Dir}(X + e_{\min})} and
#' \eqn{P_+ \sim \mathrm{Dir}(X + e_{\max})}.
#'
#' The coverage is at least \eqn{1 - \alpha} for every sample size and every
#' \eqn{p} (Gaffke 2005; proof by Vlassis and Thomas 2026). For two values,
#' the interval is the Clopper-Pearson interval.
#'
#' @param counts Nonnegative counts, one per value.
#' @param values Values of the categories.
#' @param alpha Level: the interval has coverage at least \code{1 - alpha}.
#' @param tol Tolerance on the endpoints, relative to
#'   \code{max(values) - min(values)}.
#' @return A named numeric vector with elements \code{lower} and \code{upper}.
#' @references
#' N. Gaffke (2005). Three test statistics for a nonparametric one-sided
#' hypothesis on the mean of a nonnegative variable. Mathematical Methods of
#' Statistics 14(4), 451-467.
#'
#' N. Vlassis and P. S. Thomas (2026). An exact distribution-free test for
#' means of nonnegative random variables. arXiv:2607.08415.
#' @examples
#' # 3 answers "1", 5 answers "2", 2 answers "3" on a scale from 1 to 3
#' gaffke_ci(c(3, 5, 2), c(1, 2, 3))
#' # two values: the Clopper-Pearson interval for 4 successes in 10 trials
#' gaffke_ci(c(6, 4), c(0, 1))
#' @export
gaffke_ci <- function(counts, values, alpha = 0.05, tol = 1e-12) {
  counts <- as.numeric(counts)
  values <- as.numeric(values)
  if (length(counts) != length(values) || length(counts) == 0L)
    stop("counts and values must have the same nonzero length")
  if (any(!is.finite(counts)) || any(counts < 0))
    stop("counts must be finite and nonnegative")
  if (any(!is.finite(values))) stop("values must be finite")
  if (!(alpha > 0 && alpha < 1)) stop("alpha must be in (0, 1)")
  a_lo <- counts
  i <- which.min(values)
  a_lo[i] <- a_lo[i] + 1
  a_hi <- counts
  j <- which.max(values)
  a_hi[j] <- a_hi[j] + 1
  c(lower = dirichlet_sum_quantile(a_lo, values, alpha / 2, tol = tol),
    upper = dirichlet_sum_quantile(a_hi, values, alpha / 2, upper = TRUE, tol = tol))
}

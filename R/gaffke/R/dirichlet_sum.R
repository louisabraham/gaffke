# Distribution of Y = w^T P with P ~ Dirichlet(a), a_i > 0.
# The method is described in the implementation note of the package
# (https://github.com/louisabraham/gaffke); this is a line-by-line port of
# the Python module gaffke.dirichlet_sum.
#
# Reduction: with independent G_i ~ Gamma(a_i, 1), P(Y <= y) = P(S <= 0)
# with S = sum_i (w_i - y) G_i, whose moment generating function is
# M(s) = prod_i (1 - s lambda_i)^(-a_i). The tail is an inversion integral
# on the line Re s = c through the saddle point, computed with a 12-point
# Gauss-Legendre rule on panels whose lengths double; the quantile comes
# from Newton's method on log(tail) as a function of log(distance to the
# end of the tail).

.gl <- local({
  # Golub-Welsch: 12-point Gauss-Legendre nodes and weights on [0, 1]
  n <- 12L
  b <- seq_len(n - 1L) / sqrt(4 * seq_len(n - 1L)^2 - 1)
  J <- matrix(0, n, n)
  J[cbind(1:(n - 1L), 2:n)] <- b
  J[cbind(2:n, 1:(n - 1L))] <- b
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  list(t = 0.5 * (e$values[o] + 1), w = e$vectors[1, o]^2)
})
.LOG_CUT <- 40
.MAX_HALF_PHASE <- 3

.reduce <- function(a, w) {
  a <- as.numeric(a)
  w <- as.numeric(w)
  keep <- a > 0
  a <- a[keep]
  w <- w[keep]
  if (length(w) == 0L) stop("all Dirichlet parameters are zero")
  wu <- sort(unique(w))
  list(a = as.numeric(tapply(a, factor(w, levels = wu), sum)), w = wu)
}

.saddle <- function(a, ls, z, bound, s0 = NULL) {
  if (z > 0) { b_lo <- bound; b_hi <- 0 } else { b_lo <- 0; b_hi <- bound }
  s <- if (is.null(s0)) -z else s0
  if (!(b_lo < s && s < b_hi)) s <- 0.5 * bound
  g_prev <- Inf
  for (it in 1:100) {
    d <- 1 / (1 - s * ls)
    al <- a * ls * d
    g <- sum(al)
    dg <- sum(al * ls * d)
    if (g * g <= 0.01 * dg) return(s)
    if (g > 0) b_hi <- s else b_lo <- s
    s_new <- s - g / dg
    if (!(b_lo < s_new && s_new < b_hi) || abs(g) > 0.5 * g_prev)
      s_new <- 0.5 * (b_lo + b_hi)
    g_prev <- abs(g)
    s <- s_new
  }
  s
}

.cut_point <- function(a, lp) {
  a2 <- lp * lp
  t <- sqrt(2 * .LOG_CUT / sum(a * a2))
  for (it in 1:60) {
    x <- t * t * a2
    g <- 0.5 * sum(a * log1p(x)) - .LOG_CUT
    if (g > -2 && g < 4) break
    dg <- sum(a * x / (1 + x))
    t <- t * exp(-max(min(g / dg, 3), -3))
  }
  t
}

.edges <- function(t_max, t_sing) {
  h0 <- min(t_max, 0.5 * t_sing, 1)
  m <- max(0, ceiling(log2(t_max / h0) - 1e-9))
  c(h0 * c(0, 2^(0:(m - 1)))[seq_len(m + 1)], t_max)
}

.nodes <- function(edges) {
  h <- diff(edges)
  t <- as.vector(outer(.gl$t, h) + rep(edges[-length(edges)], each = 12L))
  wt <- as.vector(outer(.gl$w, h))
  list(t = t, wt = wt, h = h)
}

.split <- function(edges, h, dtheta) {
  w_max <- apply(matrix(abs(dtheta), nrow = 12L), 2, max)
  parts <- ceiling(w_max * h / (2 * .MAX_HALF_PHASE))
  if (max(parts) <= 1) return(NULL)
  parts <- pmax(parts, 1)
  sub <- rep(h / parts, parts)
  start <- rep(edges[-length(edges)], parts)
  idx <- seq_along(sub) - 1 - rep(cumsum(parts) - parts, parts)
  c(start + idx * sub, edges[length(edges)])
}

.tail_pdf <- function(a, w, y, need_pdf = TRUE, s0 = NULL) {
  lam <- w - y
  sd <- sqrt(sum(a * lam * lam))
  ls <- lam / sd
  z <- sum(a * ls)
  upper <- z < 0
  bound <- 1 / (if (upper) max(ls) else min(ls))
  s <- .saddle(a, ls, z, bound, s0)
  c_min <- min(1, 0.5 * abs(bound))
  cc <- if (abs(s) >= c_min) s else sign(bound) * c_min
  p <- 1 - cc * ls
  lp <- ls / p
  k_c <- -sum(a * log1p(-cc * ls))  # not log(p): see the Python module
  t_max <- .cut_point(a, lp)
  edges <- .edges(t_max, min(abs(cc), 1 / max(abs(lp))))
  nd <- .nodes(edges)
  x <- outer(nd$t, lp)
  d <- 1 / (1 + x * x)
  new_edges <- .split(edges, nd$h, as.vector(d %*% (a * lp)))
  if (!is.null(new_edges)) {
    nd <- .nodes(new_edges)
    x <- outer(nd$t, lp)
    d <- 1 / (1 + x * x)
  }
  t <- nd$t
  theta <- as.vector(atan(x) %*% a)
  r <- exp(-0.5 * as.vector(log1p(x * x) %*% a))
  cs <- cos(theta)
  sn <- sin(theta)
  integral <- sum(nd$wt * r * (cc * cs + t * sn) / (cc * cc + t * t))
  tail <- if (upper) integral else -integral
  log_tail <- if (tail > 0) k_c + log(tail / pi) else -Inf
  log_pdf <- NULL
  if (need_pdf) {
    ap <- a / p
    s_re <- as.vector(d %*% ap)
    s_im <- as.vector((x * d) %*% ap)
    dens <- sum(nd$wt * r * (cs * s_re - sn * s_im))
    log_pdf <- if (dens > 0) k_c + log(dens / (pi * sd)) else -Inf
  }
  list(upper = upper, log_tail = log_tail, log_pdf = log_pdf, s = s)
}

.log_tails <- function(upper, log_tail) {
  other <- if (log_tail < 0) log1p(-exp(log_tail)) else -Inf
  if (upper) c(other, log_tail) else c(log_tail, other)
}

.cdf_sf <- function(a, w, y) {
  r <- .reduce(a, w)
  a <- r$a
  w <- r$w
  k <- length(w)
  if (k == 1L) return(if (y < w[1]) c(0, 1) else c(1, 0))
  if (y <= w[1]) return(c(0, 1))
  if (y >= w[k]) return(c(1, 0))
  lo <- w[1]
  R <- w[k] - w[1]
  tp <- .tail_pdf(a, (w - lo) / R, (y - lo) / R, need_pdf = FALSE)
  exp(.log_tails(tp$upper, tp$log_tail))
}

#' Law of a weighted sum of Dirichlet coordinates
#'
#' Distribution function, survival function, density and quantile function
#' of \eqn{Y = w^\top P} with \eqn{P \sim \mathrm{Dirichlet}(a)}.
#'
#' Tail probabilities are computed with a small relative error, also far in
#' the tails. The method (a Fourier inversion integral on a line through the
#' saddle point, a Gauss-Legendre rule on geometric panels, and Newton's
#' method for the quantile) is described in the implementation note of the
#' package.
#'
#' @param a Dirichlet parameters (nonnegative; zeros are dropped).
#' @param w Values, one per parameter (equal values are merged).
#' @param y Point at which to evaluate.
#' @param q Probability.
#' @param upper If \code{TRUE}, \code{dirichlet_sum_quantile} returns the
#'   \eqn{y} with \eqn{P(Y > y) = q}; use it for upper quantiles with a small
#'   \eqn{q}.
#' @param tol Tolerance on the quantile, relative to \code{max(w) - min(w)}.
#' @return A number.
#' @examples
#' dirichlet_sum_cdf(c(2, 3, 1), c(0, 0.5, 1), 0.4)
#' dirichlet_sum_quantile(c(2, 3, 1), c(0, 0.5, 1), 0.025)
#' @name dirichlet_sum
NULL

#' @rdname dirichlet_sum
#' @export
dirichlet_sum_cdf <- function(a, w, y) .cdf_sf(a, w, y)[1]

#' @rdname dirichlet_sum
#' @export
dirichlet_sum_sf <- function(a, w, y) .cdf_sf(a, w, y)[2]

#' @rdname dirichlet_sum
#' @export
dirichlet_sum_pdf <- function(a, w, y) {
  r <- .reduce(a, w)
  k <- length(r$w)
  if (k < 2L || !(r$w[1] < y && y < r$w[k])) return(0)
  lo <- r$w[1]
  R <- r$w[k] - r$w[1]
  exp(.tail_pdf(r$a, (r$w - lo) / R, (y - lo) / R)$log_pdf) / R
}

#' @rdname dirichlet_sum
#' @export
dirichlet_sum_quantile <- function(a, w, q, upper = FALSE, tol = 1e-12) {
  r <- .reduce(a, w)
  a <- r$a
  w <- r$w
  k <- length(w)
  if (q > 0.5) {
    q <- 1 - q
    upper <- !upper
  }
  lo <- w[1]
  hi <- w[k]
  if (k == 1L) return(lo)
  if (q <= 0) return(if (upper) hi else lo)
  R <- hi - lo
  w01 <- (w - lo) / R
  w01[k] <- 1
  if (k == 2L) {
    b <- stats::qbeta(q, a[2], a[1], lower.tail = !upper)
    return(lo + R * b)
  }
  A <- sum(a)
  m <- sum(a * w01) / A
  v <- sum(a * (w01 - m)^2) / (A * (A + 1))
  nu <- m * (1 - m) / v - 1
  y <- stats::qbeta(q, m * nu, (1 - m) * nu, lower.tail = !upper)
  if (!(y > 0 && y < 1)) y <- m
  d <- if (upper) 1 - y else y
  log_q <- log(q)
  d_lo <- 0
  d_hi <- 1
  s0 <- NULL
  u_prev <- NULL
  slope_prev <- NULL
  for (it in 1:200) {
    tp <- .tail_pdf(a, w01, if (upper) 1 - d else d, s0 = s0)
    s0 <- tp$s
    lt2 <- .log_tails(tp$upper, tp$log_tail)
    lt <- if (upper) lt2[2] else lt2[1]
    g <- lt - log_q
    if (g == 0) break
    if (g < 0) d_lo <- d else d_hi <- d
    u <- log(d)
    slope <- if (tp$log_pdf > -Inf) d * exp(tp$log_pdf - lt) else 0
    du <- if (slope > 0) -g / slope else NaN
    d_new <- if (is.finite(du)) d * exp(max(-50, min(50, du))) else Inf
    inside <- d_lo <= d_new && d_new <= d_hi
    astep <- abs(d_new - d)
    fast <- FALSE
    if (!is.null(u_prev) && u != u_prev && inside) {
      curv <- abs(slope - slope_prev) / abs(u - u_prev) / (2 * slope)
      fast <- 100 * d_new * curv * du * du <= tol
    }
    if (astep <= tol || fast) {
      if (inside) d <- d_new
      break
    }
    u_prev <- u
    slope_prev <- slope
    if (!inside || d_new == d_lo || d_new == d_hi) d_new <- 0.5 * (d_lo + d_hi)
    d <- d_new
    if (d_hi - d_lo <= tol) break
  }
  y <- if (upper) 1 - d else d
  lo + R * y
}

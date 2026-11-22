# Benchmark: gaffke::gaffke_ci against SBC::gaffke_ci (Monte Carlo).
#
# SBC (https://github.com/hyunjimoon/SBC, MIT license) computes the Gaffke
# interval from B Monte Carlo draws of Dirichlet(1, ..., 1) weights of length
# n + 1, for raw data scaled to [0, 1]. Its two functions are copied below
# from R/binary-calibration-tests.R (commit 1dd740d5d5, 2026-07-02), with
# MCMCpack::rdirichlet replaced by the same construction (normalized Gamma
# draws), so that MCMCpack is not needed.
#
# For data on values in [0, 1] that include 0 and 1, both functions estimate
# the same interval. The script measures, for several sample sizes and levels:
#   - the error of SBC's endpoints against gaffke's (mean and sd over R calls),
#   - how often SBC's interval is narrower than the exact one on a side,
#   - the time per interval of both.
#
# Usage: Rscript experiments/sbc_benchmark.R   (needs the gaffke R package)
# Output: experiments/results/sbc_benchmark.csv

library(gaffke)

rdirichlet_1 <- function(B, k) {  # B draws of Dirichlet(1, ..., 1) of size k
  x <- matrix(rexp(B * k), ncol = k)
  x / rowSums(x)
}

# ---- copied from SBC (MIT license), rdirichlet replaced -------------------
sbc_gaffke_m <- function(probs, B = 10000) {
  u_diff <- rdirichlet_1(B, length(probs) + 1)
  probs_sort <- sort(probs)
  z_upr <- c(probs_sort, 1)
  m_upr <- rowSums(sweep(u_diff, MARGIN = 2, STATS = z_upr, FUN = "*"))
  z_lwr <- c(rev(1 - probs_sort), 1)
  m_lwr <- rowSums(sweep(u_diff, MARGIN = 2, STATS = z_lwr, FUN = "*"))
  list(lwr = m_lwr, upr = m_upr)
}
sbc_gaffke_ci <- function(probs, B = 10000, alpha = 0.05) {
  m <- sbc_gaffke_m(probs, B)
  as.numeric(c(1 - quantile(m$lwr, probs = 1 - alpha / 2),
               quantile(m$upr, probs = 1 - alpha / 2)))
}
# ---------------------------------------------------------------------------

timeit <- function(f, min_sec = 0.5) {
  f()
  reps <- 0
  t0 <- proc.time()[["elapsed"]]
  repeat {
    f()
    reps <- reps + 1
    el <- proc.time()[["elapsed"]] - t0
    if (el >= min_sec) break
  }
  el / reps
}

set.seed(1)
values <- c(0, 0.25, 0.5, 0.75, 1)
p <- c(0.1, 0.2, 0.3, 0.25, 0.15)
R <- 200  # SBC calls per case, for the error statistics
rows <- list()
for (n in c(10, 100, 1000, 10000)) {
  counts <- as.vector(rmultinom(1, n, p))
  x <- rep(values, counts)  # raw data for SBC
  for (alpha in c(0.05, 1e-3, 1e-6)) {
    ex <- gaffke_ci(counts, values, alpha)
    Rn <- if (n >= 10000) 50 else R
    sb <- t(replicate(Rn, sbc_gaffke_ci(x, B = 10000, alpha = alpha)))
    t_g <- timeit(function() gaffke_ci(counts, values, alpha))
    t_s <- timeit(function() sbc_gaffke_ci(x, B = 10000, alpha = alpha))
    rows[[length(rows) + 1]] <- data.frame(
      n = n, alpha = alpha,
      gaffke_lower = ex[["lower"]], gaffke_upper = ex[["upper"]],
      sbc_lower_mean = mean(sb[, 1]), sbc_upper_mean = mean(sb[, 2]),
      sbc_lower_sd = sd(sb[, 1]), sbc_upper_sd = sd(sb[, 2]),
      sbc_max_abs_error = max(abs(sb[, 1] - ex[["lower"]]), abs(sb[, 2] - ex[["upper"]])),
      sbc_narrower_lower = mean(sb[, 1] > ex[["lower"]]),
      sbc_narrower_upper = mean(sb[, 2] < ex[["upper"]]),
      gaffke_ms = 1000 * t_g, sbc_ms = 1000 * t_s)
    print(tail(rows, 1)[[1]], digits = 4)
  }
}
res <- do.call(rbind, rows)
dir.create(file.path("experiments", "results"), showWarnings = FALSE)
write.csv(res, file.path("experiments", "results", "sbc_benchmark.csv"), row.names = FALSE)

num <- function(s) as.numeric(strsplit(s, " ")[[1]])

test_that("R agrees with the Python implementation", {
  ref <- utils::read.csv(test_path("ref_python.csv"), stringsAsFactors = FALSE)
  for (i in seq_len(nrow(ref))) {
    a <- num(ref$a[i]); w <- num(ref$w[i]); x <- ref$x[i]; r <- ref$ref[i]
    span <- max(w) - min(w)
    got <- switch(ref$kind[i],
      ppf = dirichlet_sum_quantile(a, w, x, upper = ref$upper[i] == 1),
      cdf = dirichlet_sum_cdf(a, w, x),
      ci_lo = gaffke_ci(a, w, x)[["lower"]],
      ci_hi = gaffke_ci(a, w, x)[["upper"]])
    if (ref$kind[i] == "cdf") {
      expect_equal(got, r, tolerance = 1e-10, info = paste("row", i))
    } else {
      expect_lt(abs(got - r), 1e-10 * span + 1e-13)
    }
  }
})

test_that("two values give the Clopper-Pearson interval", {
  for (n in c(1, 5, 20, 100)) for (x in unique(c(0, 1, n %/% 2, n - 1, n))) {
    ci <- gaffke_ci(c(n - x, x), c(0, 1), alpha = 0.05)
    lo <- if (x == 0) 0 else qbeta(0.025, x, n - x + 1)
    hi <- if (x == n) 1 else qbeta(0.975, x + 1, n - x)
    expect_equal(unname(ci), c(lo, hi), tolerance = 1e-10)
  }
})

test_that("the law agrees with simulation", {
  set.seed(1)
  a <- c(2, 3, 1, 4); w <- c(0, 1, 2.5, 3)
  G <- matrix(rgamma(4e5, shape = rep(a, each = 1e5)), ncol = 4)
  Y <- as.vector((G / rowSums(G)) %*% w)
  for (y in c(0.8, 1.5, 2.2)) {
    p <- dirichlet_sum_cdf(a, w, y)
    expect_lt(abs(mean(Y <= y) - p), 5 * sqrt(p * (1 - p) / 1e5))
  }
  q <- dirichlet_sum_quantile(a, w, 0.3)
  expect_equal(dirichlet_sum_cdf(a, w, q), 0.3, tolerance = 1e-9)
})

test_that("invariances and edge cases", {
  ci <- gaffke_ci(c(3, 5, 2), c(1, 2, 3))
  ci2 <- gaffke_ci(c(2, 3, 5), c(3, 1, 2))
  expect_equal(ci, ci2, tolerance = 1e-12)
  ci3 <- gaffke_ci(c(3, 5, 2), 10 * c(1, 2, 3) - 4)
  expect_equal(unname(ci3), unname(10 * ci - 4), tolerance = 1e-10)
  expect_equal(unname(gaffke_ci(c(0, 0, 0), c(1, 2, 3))), c(1, 3))
  expect_true(all(diff(sapply(c(0.2, 0.05, 1e-3, 1e-6), function(al)
    gaffke_ci(c(3, 5, 2), c(1, 2, 3), al)[["lower"]])) < 0))
  expect_error(gaffke_ci(c(1, 2), c(1, 2, 3)))
  expect_error(gaffke_ci(c(1, -1), c(1, 2)))
})

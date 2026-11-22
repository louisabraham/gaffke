# Changelog

## 0.1.2 (2026-11-22)

- Fix: the distribution function failed (division by zero) when the point
  was exactly the mean of the weighted Dirichlet sum. Same fix in the
  Python, R and JavaScript code.

## 0.1.1 (2026-11-22)

- Tail probabilities for large counts: the log of the moment generating
  function at the saddle point now uses log1p, which removes a relative error
  of about 1e-9 at n = 1e8 (now about 1e-12). Same change in the Python, R
  and JavaScript code.
- Python 3.10 or later.

## 0.1.0 (2026-11-22)

- First release: Python package (PyPI), R package (submitted to CRAN),
  JavaScript port and web page, implementation note.

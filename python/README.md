# gaffke (Python)

The Gaffke confidence interval for the mean of a discrete bounded variable:
**exact** (coverage at least 1 - alpha for every sample size and every
distribution), **narrow** (the shortest published interval with this
guarantee in our experiments) and **fast** (under a millisecond for any n).

```python
from gaffke import gaffke_ci

# 3 answers "1", 5 answers "2", 2 answers "3" on a 1-3 scale
gaffke_ci([3, 5, 2], [1, 2, 3], alpha=0.05)
# {'lower': 1.4276..., 'upper': 2.4186...}
```

For two values it is the Clopper-Pearson interval. The endpoints are
quantiles of a weighted sum of Dirichlet variables; `dirichlet_sum_cdf`,
`dirichlet_sum_sf`, `dirichlet_sum_pdf` and `dirichlet_sum_ppf` give this
law directly.

Method, references and experiments: see the implementation note at
https://github.com/louisabraham/gaffke.

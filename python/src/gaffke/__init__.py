"""gaffke: the Gaffke confidence interval for the mean of a discrete bounded variable.

>>> from gaffke import gaffke_ci
>>> gaffke_ci([3, 5, 2], [1, 2, 3])          # counts of the values 1, 2, 3
{'lower': 1.4276..., 'upper': 2.4186...}

The endpoints are quantiles of a weighted sum of Dirichlet variables,
computed by gaffke.dirichlet_sum (see the implementation note in the
repository).
"""

from gaffke.ci import gaffke_ci
from gaffke.dirichlet_sum import (
    dirichlet_sum_cdf,
    dirichlet_sum_pdf,
    dirichlet_sum_ppf,
    dirichlet_sum_sf,
)

__version__ = "0.1.0"

__all__ = [
    "gaffke_ci",
    "dirichlet_sum_cdf",
    "dirichlet_sum_sf",
    "dirichlet_sum_pdf",
    "dirichlet_sum_ppf",
]

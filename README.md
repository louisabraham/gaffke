# gaffke

**Exact, narrow and fast: the Gaffke confidence interval for the mean of a discrete bounded variable.**

[![arXiv](https://img.shields.io/badge/arXiv-2611.29512-b31b1b.svg)](https://doi.org/10.48550/arXiv.2611.29512)

Data such as survey ratings or test scores take a few known values. For the
mean of such data, the Gaffke interval is

- **exact**: its coverage is at least 1 - alpha for every sample size and
  every distribution (Gaffke 2005; proof by Vlassis and Thomas 2026). For two
  values it is the Clopper-Pearson interval.
- **narrow**: in our experiments it is the shortest published interval with
  this guarantee, 17-20% shorter than the betting interval of Waudby-Smith and
  Ramdas and 19-37% shorter than Anderson's.
- **fast**: this package computes it in less than a millisecond for any
  sample size, with endpoints accurate to 1e-12 of the range of the values.

**Try it in your browser:** https://louisabraham.github.io/gaffke/

## Python

```
pip install gaffke
```

```python
from gaffke import gaffke_ci

# 3 answers "1", 5 answers "2", 2 answers "3" on a scale from 1 to 3
gaffke_ci([3, 5, 2], [1, 2, 3], alpha=0.05)
# {'lower': 1.4276, 'upper': 2.4186}
```

## R

```r
install.packages("gaffke")
library(gaffke)
gaffke_ci(c(3, 5, 2), c(1, 2, 3), alpha = 0.05)
#  lower  upper
# 1.4276 2.4186 
```

## JavaScript

[`js/gaffke.js`](js/gaffke.js) is a single file with no dependencies, for the
browser and for Node. It is a line-by-line port of the Python package and
agrees with it to 1e-15 of the range of the values.

```html
<script src="https://louisabraham.github.io/gaffke/gaffke.js"></script>
<script>
  Gaffke.gaffkeCI([3, 5, 2], [1, 2, 3], 0.05);
  // {lower: 1.4276..., upper: 2.4186...}
</script>
```

```js
// Node
const Gaffke = require("./js/gaffke.js");
Gaffke.gaffkeCI([3, 5, 2], [1, 2, 3], 0.05);
```

The web version at https://louisabraham.github.io/gaffke/ uses this file.

All three implementations also give the law of `w' P` with `P ~ Dirichlet(a)`:
`dirichlet_sum_cdf`, `dirichlet_sum_sf`, `dirichlet_sum_pdf`, and
`dirichlet_sum_ppf` (Python), `dirichlet_sum_quantile` (R), or
`dirichletSumCdf`, `dirichletSumSf` and `dirichletSumQuantile` (JavaScript).

## Contents

| Folder | Content |
|---|---|
| `python/` | Python package and its tests (`pytest python/tests`) |
| `R/gaffke/` | R package and its tests |
| `paper/` | Implementation note: method, cost, accuracy, comparison |
| `js/` | JavaScript port `gaffke.js` and its test against Python (`node js/test/compare.js`) |
| `docs/` | Web page (GitHub Pages), with a copy of `js/gaffke.js` |
| `experiments/` | Coverage comparison with published intervals (`python experiments/coverage.py`) |

## Reference

L. Abraham. *Exact, narrow and fast: Use the Gaffke confidence interval for
the mean of a discrete bounded variable.* arXiv:2611.29512, 2026.
<https://doi.org/10.48550/arXiv.2611.29512>

## License

MIT

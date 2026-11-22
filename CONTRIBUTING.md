# Contributing to gaffke

Thank you for your interest. Bug reports, questions and pull requests are welcome.

## Report a problem or ask a question

Open an issue at https://github.com/louisabraham/gaffke/issues. For a wrong or
slow result, give the counts, the values, the level and the implementation
(Python, R or JavaScript) with its version.

## Change the code

The Python module `python/src/gaffke/dirichlet_sum.py` is the reference
implementation. The R package (`R/gaffke`) and the JavaScript file
(`js/gaffke.js`, copied to `docs/gaffke.js`) are line-by-line ports: a change
to the method must go to all three.

Before you open a pull request, run the three test suites:

```
pip install -e "python[test]" && pytest python/tests
R CMD build R/gaffke && R CMD check --as-cran gaffke_*.tar.gz
node js/test/compare.js
```

If you change the Python results, regenerate the reference values that the R
and JavaScript tests use (`R/gaffke/tests/testthat/ref_python.csv`).

## Support

The maintainer answers issues on a best-effort basis, usually within a few weeks.

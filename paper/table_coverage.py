"""Writes paper/table_coverage.tex from experiments/results/coverage.csv.

Run from the repository root:  python paper/table_coverage.py
"""

import csv
from pathlib import Path

ROWS = [("gaffke", "Gaffke"), ("betting", "Betting$^{*}$"), ("anderson", "Anderson"),
        ("hoeffding", "Hoeffding"), ("bernstein", "Empirical Bernstein"),
        ("student", "Student $t$"), ("wald", "Wald"), ("profile", "Profile likelihood"),
        ("jeffreys", "Bayes, Jeffreys prior")]
NS = [5, 10, 30]

d = {(r["method"], int(r["n"])): r for r in csv.DictReader(open("experiments/results/coverage.csv"))
     if r["design"] == "k3"}
lines = []
for key, label in ROWS:
    cells = [f"{float(d[(key, n)]['min_cov']):.3f}" for n in NS] + \
            [f"{float(d[(key, n)]['mean_width']):.3f}" for n in NS]
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    if key == "bernstein":
        lines.append(r"\midrule")
tex = r"""\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{3}{c}{minimum coverage} & \multicolumn{3}{c}{mean width} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Interval & $n = 5$ & $10$ & $30$ & $5$ & $10$ & $30$ \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
Path("paper/table_coverage.tex").write_text(tex)
print(tex)

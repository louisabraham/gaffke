// node docs/test/compare.js : compares gaffke.js with the Python reference values
const fs = require("fs"), path = require("path");
const G = require("../gaffke.js");
const csv = fs.readFileSync(path.join(__dirname, "../../R/gaffke/tests/testthat/ref_python.csv"), "utf8").trim().split(/\r?\n/);
const head = csv[0].split(",");
let worst = 0, worstRow = null, n = 0;
for (const line of csv.slice(1)) {
  const r = Object.fromEntries(line.split(",").map((v, i) => [head[i], v]));
  const a = r.a.split(" ").map(Number), w = r.w.split(" ").map(Number), x = Number(r.x), ref = Number(r.ref);
  const span = Math.max(...w) - Math.min(...w);
  let got, err;
  if (r.kind === "ppf") { got = G.dirichletSumQuantile(a, w, x, r.upper === "1"); err = Math.abs(got - ref) / span; }
  else if (r.kind === "cdf") { got = G.dirichletSumCdf(a, w, x); err = Math.abs(got - ref) / Math.max(Math.min(ref, 1 - ref), 1e-300) * 1e-2; }
  else { const ci = G.gaffkeCI(a, w, x); got = r.kind === "ci_lo" ? ci.lower : ci.upper; err = Math.abs(got - ref) / span; }
  n++;
  if (!isFinite(got)) { console.log('NOT FINITE', r); }
  if (err > worst) { worst = err; worstRow = { ...r, got }; }
}
console.log(JSON.stringify({ cases: n, worst_relative_error: worst, worstRow }));
const beta = JSON.parse(fs.readFileSync(path.join(__dirname, "beta_ref.json"), "utf8"));
let bw = 0;
let ba = 0;
for (const [q, a, b, up, ref] of beta) { const got = G._betaQuantile(q, a, b, up); if (!isFinite(got)) console.log('beta NOT FINITE', q, a, b); bw = Math.max(bw, Math.abs(got - ref) / Math.max(Math.abs(ref), 1e-300)); ba = Math.max(ba, Math.abs(got - ref)); }
console.log(JSON.stringify({ beta_cases: beta.length, beta_worst_relative_error: bw, beta_worst_absolute_error: ba }));
// the copy served by GitHub Pages must be the same file
const same = fs.readFileSync(path.join(__dirname, "../gaffke.js"), "utf8") === fs.readFileSync(path.join(__dirname, "../../docs/gaffke.js"), "utf8");
console.log(JSON.stringify({ docs_copy_identical: same }));
if (!same) process.exit(1);

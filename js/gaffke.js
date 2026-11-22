// gaffke.js: the Gaffke confidence interval for the mean of a discrete
// bounded variable, in plain JavaScript. A line-by-line port of the Python
// package gaffke (https://github.com/louisabraham/gaffke); the method is
// described in the implementation note of that repository.
//
// API:
//   Gaffke.gaffkeCI(counts, values, alpha = 0.05, tol = 1e-12) -> {lower, upper}
//   Gaffke.dirichletSumCdf(a, w, y), Gaffke.dirichletSumSf(a, w, y)
//   Gaffke.dirichletSumQuantile(a, w, q, upper = false, tol = 1e-12)
(function (root) {
  "use strict";

  // ── special functions ─────────────────────────────────────────────────

  // log Gamma(x) for x > 0: Lanczos approximation (g = 7, 9 terms), with
  // the reflection-free form; relative error about 1e-15.
  const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    let s = LG[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) s += LG[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
  }
  // Stirling correction: lgamma(x) - ((x - 0.5) log x - x + log sqrt(2 pi)),
  // for x >= 10 (error below 1e-16).
  function lgammacor(x) {
    const x2 = 1 / (x * x);
    return (1 / 12 + x2 * (-1 / 360 + x2 * (1 / 1260 + x2 * (-1 / 1680 + x2 * (1 / 1188
      + x2 * (-691 / 360360 + x2 * (1 / 156))))))) / x;
  }
  const LN_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
  // log Beta(a, b), without the cancellation of lgamma(a) + lgamma(b) -
  // lgamma(a + b) for large arguments (the method of R's lbeta).
  function lbeta(a, b) {
    const p = Math.min(a, b), q = Math.max(a, b);
    if (p >= 10) {
      const corr = lgammacor(p) + lgammacor(q) - lgammacor(p + q);
      return -0.5 * Math.log(q) + LN_SQRT_2PI + corr + (p - 0.5) * Math.log(p / (p + q)) + q * Math.log1p(-p / (p + q));
    }
    if (q >= 10) {
      const corr = lgammacor(q) - lgammacor(p + q);
      return lgamma(p) + corr + p - p * Math.log(p + q) + (q - 0.5) * Math.log1p(-p / (p + q));
    }
    return lgamma(p) + lgamma(q) - lgamma(p + q);
  }

  // Continued fraction of the incomplete Beta function (modified Lentz).
  function betacf(a, b, x) {
    const TINY = 1e-300;
    let c = 1, d = 1 - (a + b) * x / (a + 1);
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 1000000; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-16) break;
    }
    return h;
  }

  // [I_x(a, b), 1 - I_x(a, b)], each tail computed directly on the side
  // where it is small, so a small tail keeps its relative accuracy.
  function betaTails(x, a, b) {
    if (x <= 0) return [0, 1];
    if (x >= 1) return [1, 0];
    const lf = a * Math.log(x) + b * Math.log1p(-x) - lbeta(a, b);
    if (x < (a + 1) / (a + b + 2)) {
      const lo = Math.exp(lf) * betacf(a, b, x) / a;
      return [lo, 1 - lo];
    }
    const up = Math.exp(lf) * betacf(b, a, 1 - x) / b;
    return [1 - up, up];
  }

  // x with I_x(a, b) = q (upper = false) or 1 - I_x(a, b) = q (upper = true).
  // Newton's method on log(tail), inside a bracket, with bisection.
  function betaQuantile(q, a, b, upper) {
    if (q <= 0) return upper ? 1 : 0;
    if (q >= 1) return upper ? 0 : 1;
    let lo = 0, hi = 1;
    // start: normal approximation of the Beta law
    const m = a / (a + b), sd = Math.sqrt(a * b / ((a + b) * (a + b) * (a + b + 1)));
    let x = m + (upper ? 1 : -1) * sd * normQuantileUpper(q);
    if (!(x > 0 && x < 1)) x = m;
    const lq = Math.log(q), lb = lbeta(a, b);
    for (let it = 0; it < 300; it++) {
      const t = betaTails(x, a, b);
      const tail = upper ? t[1] : t[0];
      const g = Math.log(tail) - lq;  // increasing in x if lower tail
      if ((upper ? -g : g) > 0) hi = x; else lo = x;
      const lpdf = (a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x) - lb;
      const slope = (upper ? -1 : 1) * Math.exp(lpdf) / tail;  // d log(tail) / dx
      let xn = x - g / slope;
      if (!(xn > lo && xn < hi) || !isFinite(xn)) xn = 0.5 * (lo + hi);
      if (Math.abs(xn - x) <= 1e-15 * Math.max(x, 1e-300) || hi - lo <= 1e-16 * hi) { x = xn; break; }
      x = xn;
    }
    return x;
  }

  // upper-tail quantile of the standard normal (Acklam's approximation; only
  // a start point)
  function normQuantileUpper(q) {
    const p = Math.min(Math.max(1 - q, 1e-300), 1 - 1e-16);
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425;
    let r;
    if (p < pl) { const s = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1); }
    if (p > 1 - pl) { const s = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1); }
    const s = p - 0.5; r = s * s;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * s / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // ── Gauss-Legendre rule, 12 points on [0, 1] ──────────────────────────
  const GL = (function () {
    const n = 12, t = [], w = [];
    for (let i = 1; i <= n; i++) {
      let x = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), dp = 0;
      for (let it = 0; it < 100; it++) {
        let p0 = 1, p1 = x;
        for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
        dp = n * (x * p1 - p0) / (x * x - 1);
        const dx = p1 / dp;
        x -= dx;
        if (Math.abs(dx) < 1e-16) break;
      }
      t.push(0.5 * (1 - x));
      w.push(1 / ((1 - x * x) * dp * dp));  // = half the weight on [-1, 1]
    }
    return { t, w };
  })();
  const LOG_CUT = 40, MAX_HALF_PHASE = 3;

  // ── law of Y = w^T P, P ~ Dirichlet(a) ────────────────────────────────

  function reduce(a, w) {
    const map = new Map();
    for (let i = 0; i < a.length; i++) {
      if (!(a[i] > 0)) continue;
      map.set(w[i], (map.get(w[i]) || 0) + a[i]);
    }
    if (map.size === 0) throw new Error("all Dirichlet parameters are zero");
    const wu = Array.from(map.keys()).sort((x, y) => x - y);
    return { a: wu.map((v) => map.get(v)), w: wu };
  }
  const dot = (x, y) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; };

  function saddle(a, ls, z, bound, s0) {
    let bLo, bHi;
    if (z > 0) { bLo = bound; bHi = 0; } else { bLo = 0; bHi = bound; }
    let s = s0 === undefined || s0 === null ? -z : s0;
    if (!(bLo < s && s < bHi)) s = 0.5 * bound;
    let gPrev = Infinity;
    for (let it = 0; it < 100; it++) {
      let g = 0, dg = 0;
      for (let i = 0; i < a.length; i++) { const d = 1 / (1 - s * ls[i]); const al = a[i] * ls[i] * d; g += al; dg += al * ls[i] * d; }
      if (g * g <= 0.01 * dg) return s;
      if (g > 0) bHi = s; else bLo = s;
      let sNew = s - g / dg;
      if (!(bLo < sNew && sNew < bHi) || Math.abs(g) > 0.5 * gPrev) sNew = 0.5 * (bLo + bHi);
      gPrev = Math.abs(g);
      s = sNew;
    }
    return s;
  }

  function cutPoint(a, lp) {
    const a2 = lp.map((v) => v * v);
    let t = Math.sqrt(2 * LOG_CUT / dot(a, a2));
    for (let it = 0; it < 60; it++) {
      let g = -LOG_CUT, dg = 0;
      for (let i = 0; i < a.length; i++) { const x = t * t * a2[i]; g += 0.5 * a[i] * Math.log1p(x); dg += a[i] * x / (1 + x); }
      if (g > -2 && g < 4) break;
      t *= Math.exp(-Math.max(Math.min(g / dg, 3), -3));
    }
    return t;
  }

  function panelEdges(tMax, tSing) {
    const h0 = Math.min(tMax, 0.5 * tSing, 1);
    const m = Math.max(0, Math.ceil(Math.log2(tMax / h0) - 1e-9));
    const e = [0];
    for (let j = 0; j < m; j++) e.push(h0 * Math.pow(2, j));
    e.push(tMax);
    return e;
  }

  function nodes(edges) {
    const t = [], wt = [], h = [];
    for (let p = 0; p + 1 < edges.length; p++) {
      const hp = edges[p + 1] - edges[p];
      h.push(hp);
      for (let j = 0; j < 12; j++) { t.push(edges[p] + hp * GL.t[j]); wt.push(hp * GL.w[j]); }
    }
    return { t, wt, h };
  }

  function splitPanels(edges, h, dtheta) {
    const parts = [];
    let need = false;
    for (let p = 0; p < h.length; p++) {
      let wMax = 0;
      for (let j = 0; j < 12; j++) wMax = Math.max(wMax, Math.abs(dtheta[p * 12 + j]));
      const k = Math.max(1, Math.ceil(wMax * h[p] / (2 * MAX_HALF_PHASE)));
      if (k > 1) need = true;
      parts.push(k);
    }
    if (!need) return null;
    const out = [];
    for (let p = 0; p < h.length; p++) for (let j = 0; j < parts[p]; j++) out.push(edges[p] + j * h[p] / parts[p]);
    out.push(edges[edges.length - 1]);
    return out;
  }

  function tailPdf(a, w, y, needPdf, s0) {
    const k = a.length;
    const lam = w.map((v) => v - y);
    let sd = 0; for (let i = 0; i < k; i++) sd += a[i] * lam[i] * lam[i];
    sd = Math.sqrt(sd);
    const ls = lam.map((v) => v / sd);
    const z = dot(a, ls);
    const upper = z < 0;
    const bound = 1 / (upper ? Math.max(...ls) : Math.min(...ls));
    const s = saddle(a, ls, z, bound, s0);
    const cMin = Math.min(1, 0.5 * Math.abs(bound));
    const c = Math.abs(s) >= cMin ? s : Math.sign(bound) * cMin;
    const p = ls.map((v) => 1 - c * v);
    const lp = ls.map((v, i) => v / p[i]);
    let kc = 0; for (let i = 0; i < k; i++) kc -= a[i] * Math.log(p[i]);
    const tMax = cutPoint(a, lp);
    const edges = panelEdges(tMax, Math.min(Math.abs(c), 1 / Math.max(...lp.map(Math.abs))));
    let nd = nodes(edges);
    const alp = a.map((v, i) => v * lp[i]);
    const dth = nd.t.map((t) => { let s2 = 0; for (let i = 0; i < k; i++) { const x = t * lp[i]; s2 += alp[i] / (1 + x * x); } return s2; });
    const ne = splitPanels(edges, nd.h, dth);
    if (ne) nd = nodes(ne);
    let integral = 0, dens = 0;
    const ap = a.map((v, i) => v / p[i]);
    for (let q = 0; q < nd.t.length; q++) {
      const t = nd.t[q];
      let theta = 0, lr = 0, sRe = 0, sIm = 0;
      for (let i = 0; i < k; i++) {
        const x = t * lp[i], x2 = x * x, d = 1 / (1 + x2);
        theta += a[i] * Math.atan(x);
        lr += a[i] * Math.log1p(x2);
        sRe += d * ap[i];
        sIm += x * d * ap[i];
      }
      const r = Math.exp(-0.5 * lr), cs = Math.cos(theta), sn = Math.sin(theta);
      integral += nd.wt[q] * r * (c * cs + t * sn) / (c * c + t * t);
      if (needPdf) dens += nd.wt[q] * r * (cs * sRe - sn * sIm);
    }
    const tail = upper ? integral : -integral;
    const logTail = tail > 0 ? kc + Math.log(tail / Math.PI) : -Infinity;
    const logPdf = needPdf ? (dens > 0 ? kc + Math.log(dens / (Math.PI * sd)) : -Infinity) : null;
    return { upper, logTail, logPdf, s };
  }

  function logTails(upper, logTail) {
    const other = logTail < 0 ? Math.log1p(-Math.exp(logTail)) : -Infinity;
    return upper ? [other, logTail] : [logTail, other];
  }

  function cdfSf(a0, w0, y) {
    const { a, w } = reduce(a0, w0);
    const k = w.length;
    if (k === 1) return y < w[0] ? [0, 1] : [1, 0];
    if (y <= w[0]) return [0, 1];
    if (y >= w[k - 1]) return [1, 0];
    const lo = w[0], R = w[k - 1] - w[0];
    const tp = tailPdf(a, w.map((v) => (v - lo) / R), (y - lo) / R, false, null);
    return logTails(tp.upper, tp.logTail).map(Math.exp);
  }

  function dirichletSumQuantile(a0, w0, q, upper = false, tol = 1e-12) {
    const { a, w } = reduce(a0, w0);
    const k = w.length;
    if (q > 0.5) { q = 1 - q; upper = !upper; }
    const lo = w[0], hi = w[k - 1];
    if (k === 1) return lo;
    if (q <= 0) return upper ? hi : lo;
    const R = hi - lo;
    const w01 = w.map((v) => (v - lo) / R);
    w01[k - 1] = 1;
    if (k === 2) return lo + R * betaQuantile(q, a[1], a[0], upper);
    const A = a.reduce((s, v) => s + v, 0);
    const m = dot(a, w01) / A;
    let v = 0; for (let i = 0; i < k; i++) v += a[i] * (w01[i] - m) * (w01[i] - m);
    v /= A * (A + 1);
    const nu = m * (1 - m) / v - 1;
    let y = betaQuantile(q, m * nu, (1 - m) * nu, upper);
    if (!(y > 0 && y < 1)) y = m;
    let d = upper ? 1 - y : y;
    const logQ = Math.log(q);
    let dLo = 0, dHi = 1, s0 = null, uPrev = null, slopePrev = null;
    for (let it = 0; it < 200; it++) {
      const tp = tailPdf(a, w01, upper ? 1 - d : d, true, s0);
      s0 = tp.s;
      const lt2 = logTails(tp.upper, tp.logTail);
      const lt = upper ? lt2[1] : lt2[0];
      const g = lt - logQ;
      if (g === 0) break;
      if (g < 0) dLo = d; else dHi = d;
      const u = Math.log(d);
      const slope = tp.logPdf > -Infinity ? d * Math.exp(tp.logPdf - lt) : 0;
      const du = slope > 0 ? -g / slope : NaN;
      let dNew = isFinite(du) ? d * Math.exp(Math.max(-50, Math.min(50, du))) : Infinity;
      const inside = dLo <= dNew && dNew <= dHi;
      const astep = Math.abs(dNew - d);
      let fast = false;
      if (uPrev !== null && u !== uPrev && inside) {
        const curv = Math.abs(slope - slopePrev) / Math.abs(u - uPrev) / (2 * slope);
        fast = 100 * dNew * curv * du * du <= tol;
      }
      if (astep <= tol || fast) { if (inside) d = dNew; break; }
      uPrev = u; slopePrev = slope;
      if (!inside || dNew === dLo || dNew === dHi) dNew = 0.5 * (dLo + dHi);
      d = dNew;
      if (dHi - dLo <= tol) break;
    }
    return lo + R * (upper ? 1 - d : d);
  }

  function gaffkeCI(counts, values, alpha = 0.05, tol = 1e-12) {
    if (counts.length !== values.length || counts.length === 0) throw new Error("counts and values must have the same nonzero length");
    if (counts.some((c) => !(c >= 0) || !isFinite(c))) throw new Error("counts must be finite and nonnegative");
    if (values.some((v) => !isFinite(v))) throw new Error("values must be finite");
    if (!(alpha > 0 && alpha < 1)) throw new Error("alpha must be in (0, 1)");
    const iMin = values.indexOf(Math.min(...values)), iMax = values.indexOf(Math.max(...values));
    const aLo = counts.slice(); aLo[iMin] += 1;
    const aHi = counts.slice(); aHi[iMax] += 1;
    return {
      lower: dirichletSumQuantile(aLo, values, alpha / 2, false, tol),
      upper: dirichletSumQuantile(aHi, values, alpha / 2, true, tol),
    };
  }

  const api = {
    gaffkeCI, dirichletSumQuantile,
    dirichletSumCdf: (a, w, y) => cdfSf(a, w, y)[0],
    dirichletSumSf: (a, w, y) => cdfSf(a, w, y)[1],
    _betaQuantile: betaQuantile, _betaTails: betaTails, _lgamma: lgamma,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Gaffke = api;
})(typeof window !== "undefined" ? window : globalThis);

// GTA VI trophy rarity forecaster.
//
// Reads every case file (*.in) from MAKEFUN_INPUT_DIR (default /input) and
// writes one forecast file per case to MAKEFUN_OUTPUT_DIR (default /output).
// Each forecast gives, for every trophy, five quantiles (5/25/50/75/95%)
// of the percent of tracked players who will hold the trophy at the
// case's target time. Standard library only, no network, deterministic.
//
// Model, in outline:
//  * Work in logit space, with rates clamped to 0.01..99.99 percent.
//  * Collapse the history to distinct reads (Exophase refreshes in steps
//    and is flat between refreshes), keeping the time of each distinct read.
//  * Per trophy, fit a recency-weighted least-squares line of logit rate
//    against x = ln(1 + hours since 2026-11-19T00:00:00Z).
//  * Separately fit log players-tracked and log earners (rate * players)
//    against the same x, and turn those two forecasts into an implied
//    rate at the target. Blend the direct rate trend and the implied rate
//    in logit space, weighting each by how well it fitted the history.
//    The split matters because most rate movement comes from the
//    denominator: when Exophase starts tracking many new players, rates
//    fall even though the absolute number of earners keeps growing.
//  * Shrink each trophy's slope toward its tier's median slope and the
//    global median slope, more so when the trophy has few distinct reads
//    or a noisy fit, and clamp slopes so no forecast explodes.
//  * Spread comes from the fit residuals, from how far the target sits
//    beyond the history, and from the size of past refresh steps in logit
//    units. A flat history gets a wide spread: it gives no evidence
//    against a big refresh move before the target.

import * as fs from "node:fs";
import * as path from "node:path";

const ORIGIN_MS = Date.parse("2026-11-19T00:00:00Z");
const LO = 0.01;
const HI = 99.99;
const DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

// ---------- small numeric helpers ----------

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function logit(pct: number): number {
  const p = clamp(pct, LO, HI) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  if (l >= 0) {
    const e = Math.exp(-l);
    return 100 / (1 + e);
  }
  const e = Math.exp(l);
  return (100 * e) / (1 + e);
}

function round4(v: number): number {
  const r = Math.round(v * 10000) / 10000;
  return Object.is(r, -0) ? 0 : r;
}

// Inverse standard normal CDF (Acklam's rational approximation).
// Accurate to ~1e-9, deterministic, no tables needed.
function normInv(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285984469534e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
    4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// x coordinate: ln(1 + hours since the origin). Hours before the origin are
// clamped at -0.9 so the logarithm stays defined for launch-night reads.
function xOf(timeMs: number): number {
  const hours = (timeMs - ORIGIN_MS) / 3600000;
  return Math.log(1 + Math.max(hours, -0.9));
}

// ---------- weighted least squares ----------

interface Point { x: number; y: number; w: number }

interface Fit {
  a: number;          // intercept at x = 0
  b: number;          // slope (clamped)
  rawB: number;       // slope before clamping
  residStd: number;   // residual standard deviation
  xBar: number;       // weighted mean of x
  sxx: number;        // weighted sum of (x - xBar)^2
  n: number;
}

function wls(points: Point[], slopeLo: number, slopeHi: number): Fit | null {
  if (points.length < 2) return null;
  let sw = 0, sx = 0, sy = 0;
  for (const p of points) { sw += p.w; sx += p.w * p.x; sy += p.w * p.y; }
  if (sw <= 0) return null;
  const xBar = sx / sw;
  const yBar = sy / sw;
  let sxx = 0, sxy = 0;
  for (const p of points) {
    const dx = p.x - xBar;
    sxx += p.w * dx * dx;
    sxy += p.w * dx * (p.y - yBar);
  }
  if (sxx < 1e-12) return null;
  const rawB = sxy / sxx;
  const b = clamp(rawB, slopeLo, slopeHi);
  const a = yBar - b * xBar;
  let ss = 0;
  for (const p of points) {
    const r = p.y - (a + b * p.x);
    ss += p.w * r * r;
  }
  const dof = Math.max(points.length - 2, 1);
  const residVar = ss / (sw * (dof / points.length));
  const residStd = Math.sqrt(Math.max(residVar, 0));
  return { a, b, rawB, residStd, xBar, sxx, n: points.length };
}

// Refit the intercept for a chosen slope (used after shrinkage).
function interceptFor(points: Point[], b: number): number {
  let sw = 0, s = 0;
  for (const p of points) { sw += p.w; s += p.w * (p.y - b * p.x); }
  return sw > 0 ? s / sw : 0;
}

// ---------- case processing ----------

interface Trophy { id: number; tier: string }
interface DistinctRead { timeMs: number; players: number; rates: (number | null)[] }

function forecastCase(kase: any): { trophies: { id: number; q: number[] }[] } {
  const trophies: Trophy[] = Array.isArray(kase.trophies) ? kase.trophies : [];
  const levels: number[] = Array.isArray(kase.levels) && kase.levels.length > 0
    ? kase.levels : DEFAULT_LEVELS;
  const zVals = levels.map((p) => normInv(clamp(p, 0.001, 0.999)));
  const targetMs = Date.parse(kase.target ?? kase.at ?? "2026-11-26T00:00:00Z");
  const xTarget = xOf(Number.isFinite(targetMs) ? targetMs : ORIGIN_MS);
  const horizonHours = Math.max(
    ((Number.isFinite(targetMs) ? targetMs : ORIGIN_MS) -
      Date.parse(kase.at ?? "2026-11-19T00:00:00Z")) / 3600000, 0.25);

  const history: any[] = Array.isArray(kase.history) ? kase.history : [];

  // Deduplicate to distinct reads: a new read counts when its rates or its
  // players-tracked count differ from the previous distinct read. A read
  // that repeats the current value does not add a point, but it does extend
  // that value's validity: the distinct read's time is the last moment the
  // value was confirmed, so a flat history is anchored at the checkpoint
  // rather than at the first time the value was seen.
  const distinct: DistinctRead[] = [];
  let prevKey = "";
  for (const read of history) {
    if (!read || !Array.isArray(read.rates)) continue;
    const players = typeof read.players_tracked === "number" && read.players_tracked > 0
      ? read.players_tracked : null;
    if (players === null) continue;
    const timeMs = Date.parse(read.slot ?? read.at ?? "");
    if (!Number.isFinite(timeMs)) continue;
    const key = players + "|" + read.rates.map((r: any) => (r === null || r === undefined ? "?" : String(r))).join(",");
    if (key === prevKey && distinct.length > 0) {
      distinct[distinct.length - 1].timeMs = timeMs;
      continue;
    }
    prevKey = key;
    distinct.push({ timeMs, players, rates: read.rates.slice(0, trophies.length) });
  }
  distinct.sort((a, b) => a.timeMs - b.timeMs);

  const lastTime = distinct.length > 0 ? distinct[distinct.length - 1].timeMs : ORIGIN_MS;
  const lastRates: (number | null)[] = distinct.length > 0
    ? distinct[distinct.length - 1].rates : trophies.map(() => null);

  // Recency weights: recent reads count most, older reads still inform.
  function weight(timeMs: number): number {
    const ageHours = Math.max((lastTime - timeMs) / 3600000, 0);
    return Math.exp(-ageHours / 36) + 0.15;
  }

  // --- shared players-tracked fit (log players vs x) ---
  const playerPoints: Point[] = distinct.map((d) => ({
    x: xOf(d.timeMs), y: Math.log(d.players), w: weight(d.timeMs),
  }));
  const playerFit = wls(playerPoints, 0, 1.5);
  const logPlayersTarget = playerFit
    ? playerFit.a + playerFit.b * xTarget
    : (distinct.length > 0 ? Math.log(distinct[distinct.length - 1].players) : Math.log(638));

  // --- first pass: per-trophy direct fits, for tier/global slope priors ---
  interface TrophyWork {
    id: number; tier: string;
    points: Point[]; earnPoints: Point[];
    lastLogit: number | null; lastRate: number | null;
    direct: Fit | null; earnFit: Fit | null;
    steps: number[];
  }
  const work: TrophyWork[] = trophies.map((t, i) => {
    const points: Point[] = [];
    const earnPoints: Point[] = [];
    const steps: number[] = [];
    let prevLogit: number | null = null;
    for (const d of distinct) {
      const rate = d.rates[i];
      if (rate === null || rate === undefined || typeof rate !== "number" || !Number.isFinite(rate)) continue;
      const l = logit(rate);
      points.push({ x: xOf(d.timeMs), y: l, w: weight(d.timeMs) });
      // Earners: absolute count holding the trophy. Floor at half a player
      // so a 0% rate still has a finite log.
      const earners = Math.max((rate / 100) * d.players, 0.5);
      earnPoints.push({ x: xOf(d.timeMs), y: Math.log(earners), w: weight(d.timeMs) });
      if (prevLogit !== null) steps.push(l - prevLogit);
      prevLogit = l;
    }
    const lastRate = lastRates[i];
    return {
      id: t.id, tier: t.tier ?? "bronze",
      points, earnPoints,
      lastLogit: lastRate === null || lastRate === undefined ? null : logit(lastRate),
      lastRate: lastRate === null || lastRate === undefined ? null : lastRate,
      direct: wls(points, -1.5, 1.5),
      earnFit: wls(earnPoints, 0, 1.5),
      steps,
    };
  });

  // Slope priors: median slope per tier (when the tier has enough trophies)
  // and the global median slope, for both the direct and earners fits.
  function medianSlope(list: Fit[] | (Fit | null)[], tier?: string): number {
    const vals: number[] = [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (!f) continue;
      if (tier !== undefined && work[i].tier !== tier) continue;
      vals.push(f.b);
    }
    return vals.length > 0 ? median(vals) : 0;
  }
  const globalDirectSlope = medianSlope(work.map((w) => w.direct));
  const globalEarnSlope = medianSlope(work.map((w) => w.earnFit));

  const rows: { id: number; q: number[] }[] = [];

  for (const w of work) {
    const d = w.points.length; // distinct reads carrying this trophy
    const xLast = xOf(lastTime);

    // ----- direct rate-trend forecast (shrunk slope) -----
    let directLogit: number | null = null;
    let directSd = 0.5;
    let directPredVar = 0.36;
    if (w.direct && d >= 3) {
      const tierSlope = medianSlope(work.map((x) => x.direct), w.tier);
      const tierCount = work.filter((x) => x.tier === w.tier && x.direct).length;
      const prior = tierCount >= 3 ? 0.6 * tierSlope + 0.4 * globalDirectSlope : globalDirectSlope;
      let r = d / (d + 4);
      if (w.direct.residStd > 0.25) r *= 0.7;
      if (w.direct.residStd > 0.5) r *= 0.5;
      r = clamp(r, 0.05, 0.95);
      const b = clamp(r * w.direct.b + (1 - r) * prior, -1.5, 1.5);
      const a = interceptFor(w.points, b);
      directLogit = a + b * xTarget;
      directSd = clamp(w.direct.residStd, 0.08, 1.0);
      const dx = xTarget - w.direct.xBar;
      directPredVar = Math.max(w.direct.residStd * w.direct.residStd, 0.0064) *
        (1 + (dx * dx) / Math.max(w.direct.sxx, 1e-9));
    } else if (w.lastLogit !== null) {
      directLogit = w.lastLogit;
      directSd = 0.5;
      directPredVar = 0.36;
    }

    // ----- implied rate from earners / players -----
    let impliedLogit: number | null = null;
    let impliedSd = 0.5;
    if (playerFit && w.earnFit && d >= 3) {
      let r = d / (d + 4);
      if (w.earnFit.residStd > 0.25) r *= 0.7;
      r = clamp(r, 0.05, 0.95);
      const b = clamp(r * w.earnFit.b + (1 - r) * globalEarnSlope, 0, 1.5);
      const a = interceptFor(w.earnPoints, b);
      const logEarnersTarget = a + b * xTarget;
      const impliedRate = clamp(100 * Math.exp(logEarnersTarget - logPlayersTarget), LO, HI);
      impliedLogit = logit(impliedRate);
      const sdE = clamp(w.earnFit.residStd, 0.08, 1.0);
      const sdN = clamp(playerFit.residStd, 0.05, 0.6);
      impliedSd = Math.sqrt(sdE * sdE + sdN * sdN);
    } else if (w.lastLogit !== null) {
      impliedLogit = w.lastLogit;
      impliedSd = 0.5;
    }

    // ----- blend the two forecasts in logit space -----
    let medianLogit: number;
    let fitSd: number;
    if (directLogit !== null && impliedLogit !== null) {
      const wD = 1 / (directSd * directSd + 0.05);
      const wI = 1 / (impliedSd * impliedSd + 0.05);
      medianLogit = (wD * directLogit + wI * impliedLogit) / (wD + wI);
      fitSd = Math.sqrt(1 / (wD + wI));
    } else if (directLogit !== null) {
      medianLogit = directLogit;
      fitSd = directSd;
    } else if (impliedLogit !== null) {
      medianLogit = impliedLogit;
      fitSd = impliedSd;
    } else {
      // No history at all for this trophy: neutral 5% with a wide spread.
      medianLogit = logit(5);
      fitSd = 1.0;
    }

    // ----- spread -----
    // Refresh steps observed in the history, in logit units.
    let jump = 0;
    if (w.steps.length > 0) {
      const sq = w.steps.reduce((s, v) => s + v * v, 0) / w.steps.length;
      jump = Math.sqrt(sq);
    }
    const extrap = Math.max(xTarget - xLast, 0);
    let spreadSq: number;
    if (d >= 3 && w.direct) {
      const jumpTerm = jump * Math.sqrt(1 + extrap / 2);
      spreadSq = directPredVar + jumpTerm * jumpTerm + 0.09 + fitSd * fitSd * 0.25;
    } else if (d === 2) {
      const jumpTerm = Math.max(jump, 0.9) * Math.sqrt(1 + extrap / 2);
      spreadSq = 0.16 + jumpTerm * jumpTerm + 0.09;
    } else if (d === 1) {
      // Flat history: no evidence against a big refresh move.
      const jumpTerm = 1.2 * Math.sqrt(1 + extrap / 2);
      spreadSq = 0.36 + jumpTerm * jumpTerm + 0.09;
    } else {
      spreadSq = 2.25; // no reads at all
    }
    // Longer horizons are more uncertain.
    spreadSq *= 1 + 0.15 * Math.log1p(horizonHours / 24);
    let spread = Math.sqrt(spreadSq);
    // Stable, widely-held trophies move less.
    const medianPct = invLogit(medianLogit);
    if (medianPct >= 90) spread *= 0.7;
    else if (medianPct >= 50) spread *= 0.85;
    spread = clamp(spread, 0.3, 2.5);
    // A trophy already at 100% stays near 100%.
    if (w.lastRate !== null && w.lastRate >= 99.99 && d >= 1) {
      medianLogit = logit(99.99);
      spread = 0.15;
    }

    const q = zVals.map((z) => {
      const pct = clamp(invLogit(medianLogit + z * spread), 0, 100);
      return round4(pct);
    });
    // Rounding can only keep the order, but enforce it explicitly.
    for (let i = 1; i < q.length; i++) {
      if (q[i] < q[i - 1]) q[i] = q[i - 1];
    }
    rows.push({ id: w.id, q });
  }

  return { trophies: rows };
}

// ---------- driver ----------

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  let names: string[] = [];
  try {
    names = fs.readdirSync(inputDir).filter((n) => n.endsWith(".in")).sort();
  } catch {
    names = [];
  }
  for (const name of names) {
    const caseId = name.slice(0, -3);
    let kase: any = null;
    try {
      kase = JSON.parse(fs.readFileSync(path.join(inputDir, name), "utf8"));
    } catch (e) {
      kase = null;
    }
    let out: { trophies: { id: number; q: number[] }[] };
    if (kase && typeof kase === "object") {
      try {
        out = forecastCase(kase);
      } catch {
        out = { trophies: [] };
      }
    } else {
      out = { trophies: [] };
    }
    // If anything went wrong, still emit a well-formed fallback so the
    // case is not left without an output file.
    if (!out.trophies || out.trophies.length === 0) {
      const ids: number[] = kase && Array.isArray(kase.trophies)
        ? kase.trophies.map((t: any) => t.id) : [];
      const fallback = ids.length > 0 ? ids : Array.from({ length: 55 }, (_, i) => i + 1);
      out = {
        trophies: fallback.map((id) => ({
          id,
          q: [0.5, 2, 5, 12, 30].map((v) => round4(v)),
        })),
      };
    }
    const outPath = path.join(outputDir, `${kase && kase.id ? String(kase.id) : caseId}.out`);
    fs.writeFileSync(outPath, JSON.stringify(out) + "\n", "utf8");
  }
}

main();

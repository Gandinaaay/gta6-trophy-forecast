# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Gandinaaay/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Gandinaaay/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players will earn each trophy in week one: for every one of the game's 55 trophies, it predicts the percent of Exophase-tracked players holding the trophy at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%). Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

`main.ts` (Node 22, standard library only, no network) reads every case file (`*.in`) from the input directory and writes one forecast file per case (`<id>.out`) to the output directory.

The model works in logit space, with rates clamped to 0.01–99.99%:

- **Deduplicate.** Exophase refreshes in steps and is flat between refreshes, so the history is collapsed to distinct reads, each kept until the next change (a repeated read extends the current value's validity to the checkpoint).
- **Direct trend.** Per trophy, a recency-weighted least-squares line of logit rate against `ln(1 + hours since 2026-11-19T00:00:00Z)`. Slopes are shrunk toward the trophy's tier median slope and the global median slope — more so with few distinct reads or a noisy fit — and clamped so no forecast explodes.
- **Denominator / numerator split.** Log players-tracked and log earners (rate × players) are fitted separately against the same time axis, giving an implied rate at the target. This matters because most rate movement comes from the denominator: when Exophase starts tracking many new players, rates fall even while the absolute number of earners keeps growing. The direct trend and the implied rate are blended in logit space, weighted by fit quality.
- **Spread.** Per-trophy spread comes from the fit residuals, from how far the target sits beyond the history, and from the size of past refresh steps in logit units. A flat history gets a wide spread — it gives no evidence against a big refresh move before the target — while stable, widely-held trophies get a narrower one. Quantiles are the median logit plus the standard normal quantiles (−1.645, −0.674, 0, 0.674, 1.645) times the spread, converted back to percent.

The program is deterministic: two runs on the same input write byte-identical output, and it runs in well under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Each input case `input/<id>.in` produces `out/<id>.out`, a JSON object `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}` with one row per trophy and five non-decreasing percentages per row.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fywkmv2t&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

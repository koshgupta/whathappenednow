# whathappenednow

A data and modelling pipeline that produces a daily pre-market advisory signal for
the S&P 500 from a **9:00 AM ET snapshot** of ES continuous futures, macro data, and
pre-market news sentiment.

The prediction is made at 9:00 AM ET — after the 8:30 AM macro releases settle, but
before the 9:30 AM cash open — so every feature is provably knowable before the
session it forecasts. **No trades are placed; the signal is read directly by the user
as pre-market context.**

This repo covers **data collection, feature engineering, and model training only**.
API/serving is out of scope.

---

## The signal

**Realized range** (`app/predict_vol.py`) — predicts today's RTH realized range,
`(high − low) / open` over the 9:30 AM–4:00 PM ET session. This is the project's
reliable, shipping signal.

| Metric (232-day test set) | Value |
|---|---|
| R² | +0.165 |
| RMSE | 0.00434 |
| MAE | 0.00302 (median abs error ≈ 22 bps) |
| Within 30 bps | 64% of days |
| Within 50 bps | 84% of days |
| vs. persistence baseline | +22% RMSE |

In plain terms: the model tells you how *wide* a day to expect (Calm / Normal /
Active), calibrated against the trailing-252-day distribution of realized range. It
slightly over-predicts on quiet days and under-predicts true tail-vol days.

> **Background — direction signal.** An earlier classifier predicted whether the
> session closed above its open. It peaked at AUC ~0.55 with calibration collapsing
> to base-rate: directional prediction from a daily snapshot sits at the
> efficient-market ceiling. That model was retired; realized range is the signal that
> carries usable information.

---

## How it works

`app/daily_fetch.py` is the single production entrypoint, designed to run from cron at
**9:05 AM ET** (5 minutes after the 9:00 bar prints, before the cash open). It
hard-fails on any error so cron emails surface drift immediately.

Pipeline order:

1. Connect to IB Gateway (live, `127.0.0.1:4001` by default).
2. Backfill incremental data: ES + VX + ZN 1-minute bars via IB; news via Mediastack;
   sentiment via FinBERT + a local LLM + Qwen3 embeddings.
3. Settle yesterday's prediction by computing its actual RTH realized range.
4. Rebuild every feature parquet.
5. Retrain the vol model — refit with cached params daily; full Optuna search if no
   cache exists or the last search was > 7 days ago (`OPTUNA_REFRESH_DAYS`).
6. Build today's 9 AM feature row and predict the realized range.
7. Append the prediction to `data/predictions.parquet`.
8. Broadcast a rich Discord embed to every configured forecast channel; push a
   traceback to the error channel on failure.

### Idempotent same-day reruns

Every IB fetch pins its upper bound to **today 09:01 ET**, so the cache mechanically
never contains today's bars past 09:00 ET regardless of when the script runs. A 09:05
cron, an 11:00 AM manual rerun, and an 18:00 catch-up all produce identical caches,
training sets, and forecasts (today is always excluded from training).

### Safe-window guard

The script refuses to run outside `[09:01, 23:59:59] ET`. Before 09:01 the 09:00 bar
may not be finalised in IB; after midnight ET the calendar date rolls over before the
new "today" reaches 09:01. Use `--skip-time-guard` only for manual backfills.

---

## Architecture

### Data sources

- **ES continuous futures (`ES.c.0`)** — primary price source. Futures trade overnight,
  giving a smooth view into the 9 AM snapshot without cash-session opening-gap
  alignment headaches.
  - **Databento** (`GLBX.MDP3`, `ohlcv-1m`) — 5-year historical cold-start cache.
    Prices are fixed-point `int64` (÷1e9); bars are **left-labeled** (`ts_event` = open).
  - **Interactive Brokers (IB Gateway)** — daily incremental bars on top of the
    Databento cache. ES → `ContFuture('ES','CME')`, VX → `ContFuture('VX','CFE')`,
    ZN → `ContFuture('ZN','CBOT')`. `app/ib_client.py` normalises exchange-tz bars to
    UTC to match the Databento schema exactly. Powered by `ib-async`.
- **VIX cash + 10Y Treasury** — macro context (Databento intraday, IB incremental).
- **Mediastack** (paid plan) — news across 7 countries × 4 categories.

### Pipeline stages

| Module | Role |
|---|---|
| `app/market_data.py` | ES futures → clean 9 AM snapshot feature matrix (technicals, pre-market gap/trend, 8:30 macro window). All price indicators in stationary `(indicator/price) − 1` form. |
| `app/macro_data.py` | VIX + 10Y relative features, as-of joined at 09:00 ET. |
| `app/news_data.py` | Mediastack fetch (monthly chunks) + time-boxing of articles to the 4:00 PM T-1 → 8:59 AM T pre-market window (drops intraday news to prevent lookahead; shifts weekend articles to Monday). |
| `app/sentiment-local.py` | Two-tier NLP — **FinBERT** (`ProsusAI/finbert`) filters neutral articles; a local LLM scores high-impact ones; Qwen3 embeddings → PCA dims 1–5. |
| `app/calendar_features.py` | FOMC / OPEX / day-of-week calendar features. |
| `app/merge_features.py` | Inner-merges all sources on Date → `data/model_data.parquet` (1 row/day). |
| `app/predict_vol.py` | XGBoost regression for realized range. |
| `app/notifications.py` | Discord embed broadcast + error-channel traceback. |

---

## Anti-lookahead protocol

Every feature represents only what was knowable at 9:00 AM ET. Three datasets are
time-aligned independently *before* merging on Date:

1. **Daily technicals** — compute indicators, then `shift(1)` (yesterday's values on
   today's index).
2. **Intraday snapshot** — filter to exactly the 9:00 AM row.
3. **News sentiment** — aggregate over the custom 4 PM T-1 → 9 AM T window.

Inner-merge on Date. Tail flags use a 252-day warmup + `shift(1)` so trailing
percentile cutoffs never see the current row.

---

## Model details (`predict_vol.py`)

- **Target:** same-day RTH realized range `(high − low) / open`. Features locked at
  9 AM, session is the outcome window — anti-lookahead-clean.
- **Objective:** `reg:absoluteerror` (MAE). Squared-error collapses to a constant
  predictor on fat-tailed vol days; MAE is robust to those outliers. Run with
  `--mode mae` (do **not** use `--mode aggressive` — it overfits).
- **Validation:** single chronological holdout (last 20% → validation). Optuna TPE,
  50 trials, with collapse rejection (`TrialPruned` when `pred.std() < 1e-3 × y.std()`).
  Final refit on full train+val with fixed `best_iter × 1.1` — no early-stopping leak.
- **Extra features (no API):** `rv_lag_5`, `rv_lag_22` (trailing realized-range means,
  shifted), `vix_level` (VIX cash at 09:00).
- **News block** contributes ~40% of total feature importance. Contrarian tail flags
  `is_top_2pct_weighted_impact` / `is_bot_2pct_weighted_impact` rank in the top 10.
- **Ceiling:** R² ≈ 0.15–0.20 is the empirical limit for daily realized range from 9 AM
  features. Further gains require *different data* (intraday vol, options IV surface,
  dealer gamma), not more engineering on this set.

### Feature set (all locked at 9:00 AM)

| Category | Features |
|---|---|
| Pre-market intraday | `overnight_gap`, `premarket_trend`, `macro_vol_830`, `macro_dir_830` |
| Daily technicals (shifted +1) | `dist_sma_50/200`, `rsi_14`, `pct_atr`, `bb_pct_b`, `relative_volume`, `adx_14`, `dmp_14`, `dmn_14`, `stoch_k/d`, `pct_macd`/`_hist`/`_signal`, `prev_rth_momentum`, `prev_day_range` |
| News sentiment | Aggregated FinBERT + LLM scores + Qwen3 PCA dims for the pre-market window |
| News extreme flags | `is_top_2pct_weighted_impact`, `is_bot_2pct_weighted_impact` |

---

## Setup

Requires **Python 3.12** and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

Create `app/.env`:

| Variable | Source |
|---|---|
| `DATABENTO_API_KEY` | Databento (ES futures 1-min bars) |
| `NEWS_API_KEY` | Mediastack (**paid plan required**) |
| `HF_TOKEN` | HuggingFace (FinBERT inference) |

Runtime prerequisites for a live daily run:

- IB Gateway running with the live API enabled (`127.0.0.1:4001`; override with
  `IB_HOST` / `IB_PORT`).
- A local LLM server (LM Studio) serving the Tier-2 sentiment model.
- All base parquets cold-started (see below).

---

## Usage

```bash
uv run python app/daily_fetch.py                   # end-to-end daily run (cron @ 9:05 AM ET)
uv run python app/daily_fetch.py --dry-run         # print planned steps, no fetch/train
uv run python app/daily_fetch.py --force-search    # force a full Optuna search this run
uv run python app/daily_fetch.py --skip-time-guard # bypass the safe-window check (backfills)
uv run python app/daily_fetch.py --no-notify       # suppress Discord notifications (backfills)

uv run python app/predict_vol.py --mode mae        # standalone vol-model retrain
uv run python app/sentiment-local.py               # standalone sentiment scoring (costs API $$)

uv add <package>                                    # add a dependency
```

### Cold-starting a fresh machine

The historical cold-start scripts no longer expose script-level entrypoints — the
daily orchestrator hard-fails if any base cache is missing rather than silently
triggering a multi-year refetch. Bootstrap the underlying functions manually, e.g.:

```bash
uv run python -c "from app.news_data import fetch_sp500_news_history; \
  import os; fetch_sp500_news_history(os.environ['NEWS_API_KEY'])"
```

---

## Discord notifications

The orchestrator broadcasts a rich embed to one or more forecast webhooks on success
and pushes a traceback to an error webhook on failure (the cron email fires either way
— Discord is a courtesy layer).

Webhook URLs live in `app/discord_webhooks.json` (gitignored — see
`app/discord_webhooks.json.example` for the schema; override the path with
`DISCORD_WEBHOOK_CONFIG`):

```json
{
  "forecast_channels": [
    {"label": "personal", "url": "https://discord.com/api/webhooks/..."}
  ],
  "error_channel": {"label": "ops", "url": "https://discord.com/api/webhooks/..."}
}
```

The embed reports the predicted range with confidence band, a Calm/Normal/Active day
band, the most-recent settled prediction with 7-day rolling MAE, calendar/macro context
(DOW, days to FOMC/OPEX, VIX@09:00), a news block (tone, FinBERT mix, momentum, tail
flags), baselines (persistence + 5-day MA with model edge), and model provenance.

Forecast webhooks retry with exponential backoff on 5xx/429 (honouring `Retry-After`)
and hard-fail the step if any channel ultimately fails; the error webhook soft-fails
(the cron email remains source of truth).

---

## Project layout

```
app/
  daily_fetch.py            # production orchestrator (cron entrypoint)
  ib_client.py              # IB Gateway client, exchange-tz → UTC normalisation
  market_data.py            # ES futures → 9 AM snapshot features
  macro_data.py             # VIX + 10Y relative features
  news_data.py              # Mediastack fetch + pre-market time-boxing
  sentiment-local.py        # FinBERT + local LLM + Qwen3 embeddings
  calendar_features.py      # FOMC / OPEX / DOW features
  merge_features.py         # inner-merge all sources → model_data.parquet
  predict_vol.py            # XGBoost realized-range model
  notifications.py          # Discord broadcast
  discord_webhooks.json.example
  old/                      # retired/experimental scripts (gitignored)
data/                       # parquet caches, models, predictions (gitignored)
```

### Data files (`data/`, gitignored)

```
es_intraday_raw.parquet            # ES 1-min bars (cached)
macro_vix_intraday_raw.parquet     # VIX cash 1-min bars
macro_tenY_intraday_raw.parquet    # 10Y Treasury 1-min bars
sp500_news.parquet                 # raw Mediastack articles
aligned_premarket_news.parquet     # time-boxed, tz-adjusted news
article_scores_checkpoint.parquet  # per-article sentiment scores
features_9am_snapshot.parquet      # market-only feature matrix
macro_features.parquet             # VIX + 10Y features
calendar_features.parquet          # calendar features
model_data.parquet                 # merged feature matrix (1 row/day)
vol_model_mae.joblib / .json       # vol model + metrics (the one to use)
predictions.parquet                # predicted/actual range log
```

---

## Status & next steps

The realized-range model is production-ready and the recommended signal. Remaining
research directions:

- **Tail-vol improvement** — the MAE model under-predicts extreme days; a secondary
  "is today a tail day" classifier could gate a separate high-vol estimate.
- **Beyond-ceiling features** — intraday range/tick volume post-9:30, options IV
  surface (term structure, skew), dealer gamma positioning.
- **Richer news encoding** — semantic context from Qwen3 embeddings beyond the current
  PCA dims.
- **Regime detection** — a trend-vs-mean-reversion clustering feature fed to XGBoost.
```

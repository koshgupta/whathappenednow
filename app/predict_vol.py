"""
XGBoost regression for next-day S&P 500 RTH realized range — vol pivot, v2.

What's different from v1
  v1 used TimeSeriesSplit(n_splits=5) for Optuna. With 926 train_val rows, fold
  1 trained on 151 rows and produced wildly mis-specified val predictions
  (R² ≈ −1.4). Optuna averaged the noisy fold RMSEs and converged on the only
  thing that minimizes the average — a constant predictor (gamma=2.07,
  min_child_weight=13, reg_lambda=8.17 → zero splits, std(pred)=0 on test).

  v2 fixes that by:
    1. Single chronological train/val holdout inside train_val (last
       VAL_FRACTION → val). Optuna optimizes RMSE on this one window with
       early stopping. No 150-row fold disasters.
    2. Hyperparameter caps that forbid the search from picking parameters
       that produce a constant-predictor model.

Anti-collapse caps
  max_depth         : [3, 6]    (≥3 forces at least some branching)
  learning_rate     : [0.01, 0.10] (log)
  subsample         : [0.6, 1.0]
  colsample_bytree  : [0.6, 1.0]
  min_child_weight  : [1, 5]    (was [1, 20])
  reg_lambda        : [0.0, 2.0] (was [0.0, 10.0])
  reg_alpha         : [0.0, 1.0] (was [0.0, 5.0])
  gamma             : [0.0, 0.5] (was [0.0, 5.0])

Target / features
  Target: realized_range[T] = (RTH_high[T] − RTH_low[T]) / RTH_open[T]
          — today's RTH range, predicted from features locked at 9:00 AM
          before the 9:30 cash open.
  Features: all columns from data/model_data.parquet (minus y),
            plus vix_level, rv_lag_5, rv_lag_22 derived from cached parquets.

Anti-lookahead
  - realized_range is built from RTH bars (09:30–15:59 ET) on day T. It is
    *the target*, not a feature — no shift applied. The features are all
    anti-lookahead-locked at 9:00 AM (before RTH opens), so using today's
    9 AM snapshot to predict today's RTH range introduces no leakage.
  - HAR-RV aggregates use shift(1) so today's row holds the trailing mean
    of past 5/22 days through yesterday's close.
  - vix_level uses as-of rule (bars with ts ≤ 09:00 ET).

Pipeline
  1. Chronological split: last TEST_FRACTION rows → test.
  2. Inside train_val: last VAL_FRACTION rows → val (the rest → train).
  3. Optuna TPE over capped param space, scoring val RMSE with early stopping
     on val. Capture the trial's best_iteration alongside its params.
  4. Final refit on the full train_val (train + val merged) with fixed boost
     budget = best_iter × 1.1. No early stopping leak into test.
  5. Evaluate on test: R², MAE, RMSE, plus persistence baseline
     (prev_day_range) and a 5-day-MA baseline (rv_lag_5).
  6. Save bundle + metrics.

Pure local. Reads:
  data/es_intraday_raw.parquet
  data/macro_vix_intraday_raw.parquet
  data/model_data.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, time, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths (filled in per mode at runtime)
# ---------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parent.parent
ES_RAW_PATH = _PROJ_ROOT / "data" / "es_intraday_raw.parquet"
VIX_RAW_PATH = _PROJ_ROOT / "data" / "macro_vix_intraday_raw.parquet"
FEATURES_PATH = _PROJ_ROOT / "data" / "model_data.parquet"

# ---------------------------------------------------------------------------
# Modes
#   "mae"        : reg:absoluteerror objective. Robust to outliers (Aug 2024
#                  carry-unwind, tariff days). Keep the v2 cap ranges.
#   "aggressive" : reg:squarederror objective + tighter caps that force the
#                  search away from the constant-predictor basin (max_depth
#                  floor raised, gamma fixed at 0, regs sharply capped).
# Always on (both modes): trials whose val pred_std < threshold are pruned —
# Optuna can NEVER pick a collapsed trial as best.
# ---------------------------------------------------------------------------
COLLAPSE_REJECT_REL_THRESHOLD = 1e-3  # prune if pred.std() < this × y_va.std()

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

TEST_FRACTION = 0.20   # last 20 % of all rows → held-out test
VAL_FRACTION = 0.20    # last 20 % of train_val → single Optuna val window

# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
N_TRIALS = 50
EARLY_STOP_ROUNDS = 50
MAX_BOOST_ROUNDS = 5000
OPTUNA_TIMEOUT_SEC = None

# HAR-RV aggregation windows
HAR_SHORT_WINDOW = 5
HAR_LONG_WINDOW = 22

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    return df


def _asof_last_per_date(df_et: pd.DataFrame, cutoff: time) -> pd.DataFrame:
    """Latest bar per ET date with timestamp ≤ cutoff (inclusive)."""
    window = df_et.between_time(time(0, 0), cutoff)
    return window.groupby(window.index.normalize(), sort=False).tail(1)


# ---------------------------------------------------------------------------
# Target + HAR-RV features from cached ES bars
# ---------------------------------------------------------------------------

def build_realized_range_table() -> pd.DataFrame:
    log.info("Loading ES intraday bars: %s", ES_RAW_PATH)
    df = pd.read_parquet(ES_RAW_PATH)
    df_et = _to_et(df)
    rth = df_et.between_time("09:30", "15:59")
    daily = rth.groupby(rth.index.date).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily = daily[(daily.index.dayofweek < 5) & (daily["close"] > 0)]

    realized = (daily["high"] - daily["low"]) / daily["open"]
    realized.name = "realized_range"

    out = pd.DataFrame({"realized_range": realized})
    out["rv_lag_5"] = realized.rolling(HAR_SHORT_WINDOW).mean().shift(1)
    out["rv_lag_22"] = realized.rolling(HAR_LONG_WINDOW).mean().shift(1)
    # Target: TODAY's realized range. Features are locked at 9:00 AM (before
    # the 9:30 cash open), so predicting today's RTH session from today's
    # snapshot is anti-lookahead-clean.
    out["target"] = realized
    log.info(
        "  built %d daily realized-range rows (%s → %s)",
        len(out), out.index.min().date(), out.index.max().date(),
    )
    log.info(
        "  realized_range: mean=%.5f std=%.5f min=%.5f max=%.5f",
        realized.mean(), realized.std(), realized.min(), realized.max(),
    )
    return out


# ---------------------------------------------------------------------------
# VIX absolute level at 09:00 ET
# ---------------------------------------------------------------------------

def build_vix_level() -> pd.Series:
    log.info("Loading VIX intraday bars: %s", VIX_RAW_PATH)
    df = pd.read_parquet(VIX_RAW_PATH)
    df_et = _to_et(df)
    morning = _asof_last_per_date(df_et, time(9, 0))
    is_exact_9 = morning.index.time == time(9, 0)
    values = np.where(is_exact_9, morning["open"].values, morning["close"].values)
    s = pd.Series(
        values,
        index=pd.DatetimeIndex(morning.index.date, name="date"),
        name="vix_level",
    )
    s = s.sort_index()
    log.info(
        "  vix_level rows=%d (%s → %s) mean=%.2f min=%.2f max=%.2f",
        len(s), s.index.min().date(), s.index.max().date(),
        s.mean(), s.min(), s.max(),
    )
    return s


# ---------------------------------------------------------------------------
# Assemble training matrix
# ---------------------------------------------------------------------------

def assemble() -> tuple[pd.DataFrame, pd.Series]:
    rr = build_realized_range_table()
    vix = build_vix_level()

    log.info("Loading existing feature matrix: %s", FEATURES_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    feats = feats.drop(columns=["y"], errors="ignore")
    feats.index.name = "date"

    df = feats.join(rr[["rv_lag_5", "rv_lag_22", "target"]], how="inner")
    df = df.join(vix, how="left")
    df = df.dropna(subset=["target"]).copy()

    log.info(
        "Assembled %d rows × %d cols (target plus %d features) | %s → %s",
        len(df), df.shape[1], df.shape[1] - 1,
        df.index.min().date(), df.index.max().date(),
    )
    n_vix_nan = int(df["vix_level"].isna().sum())
    if n_vix_nan:
        log.info("  vix_level NaN on %d rows (kept; XGB tolerates NaN)", n_vix_nan)

    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------

def chronological_split(X: pd.DataFrame, y: pd.Series, test_fraction: float = TEST_FRACTION):
    n = len(X)
    cutoff = int(n * (1 - test_fraction))
    X_tv, X_te = X.iloc[:cutoff], X.iloc[cutoff:]
    y_tv, y_te = y.iloc[:cutoff], y.iloc[cutoff:]
    log.info(
        "Train+val: %d rows (%s → %s) | target mean=%.5f std=%.5f",
        len(X_tv), X_tv.index.min().date(), X_tv.index.max().date(),
        y_tv.mean(), y_tv.std(),
    )
    log.info(
        "Test     : %d rows (%s → %s) | target mean=%.5f std=%.5f",
        len(X_te), X_te.index.min().date(), X_te.index.max().date(),
        y_te.mean(), y_te.std(),
    )
    return X_tv, y_tv, X_te, y_te


def train_val_split(X_tv: pd.DataFrame, y_tv: pd.Series, val_fraction: float = VAL_FRACTION):
    n = len(X_tv)
    cutoff = int(n * (1 - val_fraction))
    X_tr, X_va = X_tv.iloc[:cutoff], X_tv.iloc[cutoff:]
    y_tr, y_va = y_tv.iloc[:cutoff], y_tv.iloc[cutoff:]
    log.info(
        "  train: %d rows (%s → %s) | mean=%.5f std=%.5f",
        len(X_tr), X_tr.index.min().date(), X_tr.index.max().date(),
        y_tr.mean(), y_tr.std(),
    )
    log.info(
        "  val  : %d rows (%s → %s) | mean=%.5f std=%.5f",
        len(X_va), X_va.index.min().date(), X_va.index.max().date(),
        y_va.mean(), y_va.std(),
    )
    return X_tr, y_tr, X_va, y_va


# ---------------------------------------------------------------------------
# Optuna search — single val window, capped params, anti-collapse
# ---------------------------------------------------------------------------

def _base_params(mode: str, seed: int = SEED) -> dict:
    if mode == "mae":
        obj, metric = "reg:absoluteerror", "mae"
    else:  # aggressive
        obj, metric = "reg:squarederror", "rmse"
    return {
        "objective": obj,
        "eval_metric": metric,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": seed,
        "n_jobs": -1,
    }


def _suggest_params(trial: optuna.Trial, mode: str) -> dict:
    """Mode-specific cap ranges. Aggressive forces depth ≥ 5 and ~zero reg."""
    base = _base_params(mode)
    if mode == "mae":
        # Original v2 caps — robust loss handles outliers without further caps.
        return {
            **base,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "n_estimators": MAX_BOOST_ROUNDS,
            "early_stopping_rounds": EARLY_STOP_ROUNDS,
        }
    # aggressive
    return {
        **base,
        "max_depth": trial.suggest_int("max_depth", 5, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.2),
        "gamma": 0.0,
        "n_estimators": MAX_BOOST_ROUNDS,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
    }


def _score(mode: str, y_true: pd.Series, pred: np.ndarray) -> float:
    if mode == "mae":
        return float(mean_absolute_error(y_true, pred))
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def _objective_factory(mode: str, X_tr: pd.DataFrame, y_tr: pd.Series,
                       X_va: pd.DataFrame, y_va: pd.Series):
    """Capture per-trial best_iter. Prune collapsed trials so they cannot be best."""
    best_iters: dict[int, int] = {}
    threshold = COLLAPSE_REJECT_REL_THRESHOLD * float(y_va.std())

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, mode)
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        if float(pred.std()) < threshold:
            # Hard reject: Optuna cannot return this trial as best.
            raise optuna.TrialPruned()
        best_iters[trial.number] = int(model.best_iteration)
        return _score(mode, y_va, pred)

    return objective, best_iters


def run_optuna(mode: str, X_tr: pd.DataFrame, y_tr: pd.Series,
               X_va: pd.DataFrame, y_va: pd.Series) -> tuple[dict, int]:
    metric_name = "MAE" if mode == "mae" else "RMSE"
    log.info(
        "Optuna TPE: %d trials. mode=%s. Single train→val holdout. "
        "Collapse rejected via TrialPruned.",
        N_TRIALS, mode,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective, best_iters = _objective_factory(mode, X_tr, y_tr, X_va, y_va)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT_SEC,
                   show_progress_bar=False)

    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    log.info("Search done: %d complete / %d pruned (collapsed).", n_complete, n_pruned)
    if n_complete == 0:
        raise RuntimeError(
            f"All {N_TRIALS} trials collapsed even with mode={mode}. "
            "The hyperparameter cap space cannot produce a non-constant predictor "
            "on this val window — strong evidence the features don't carry signal "
            "for this target."
        )
    best_trial = study.best_trial
    best_iter = best_iters.get(best_trial.number, EARLY_STOP_ROUNDS)
    log.info("Best val %s: %.6f", metric_name, study.best_value)
    log.info("Best params: %s", best_trial.params)
    log.info("Best trial early-stop iteration: %d", best_iter)
    return best_trial.params, best_iter


# ---------------------------------------------------------------------------
# Final fit on full train_val
# ---------------------------------------------------------------------------

def fit_final(mode: str, X_tv: pd.DataFrame, y_tv: pd.Series, search_params: dict,
              best_iter: int) -> XGBRegressor:
    n_estimators = max(int(best_iter * 1.1), 10)
    log.info(
        "Final fit on %d rows with n_estimators=%d (no early stopping).",
        len(X_tv), n_estimators,
    )
    final_params = {**_base_params(mode), **search_params, "n_estimators": n_estimators}
    final_params.pop("early_stopping_rounds", None)
    model = XGBRegressor(**final_params)
    model.fit(X_tv, y_tv)
    return model


# ---------------------------------------------------------------------------
# Evaluation (model vs persistence vs rv_lag_5)
# ---------------------------------------------------------------------------

def _metrics(y_true: pd.Series, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "pred_std": float(np.std(pred)),
    }


def evaluate(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    model_pred = model.predict(X)
    persist_pred = X["prev_day_range"].values if "prev_day_range" in X.columns else None
    rv5_pred = X["rv_lag_5"].values if "rv_lag_5" in X.columns else None

    out = {"model": _metrics(y, model_pred)}
    if persist_pred is not None and not np.isnan(persist_pred).all():
        m = ~np.isnan(persist_pred)
        out["persist"] = _metrics(y.iloc[m], persist_pred[m])
    if rv5_pred is not None and not np.isnan(rv5_pred).all():
        m = ~np.isnan(rv5_pred)
        out["rv_lag_5"] = _metrics(y.iloc[m], rv5_pred[m])

    log.info("=== %s (n=%d, y mean=%.5f std=%.5f) ===",
             label, len(y), float(y.mean()), float(y.std()))
    for name, m in out.items():
        log.info("  %-10s  rmse=%.6f  mae=%.6f  R²=%+.4f  pred_std=%.6f",
                 name, m["rmse"], m["mae"], m["r2"], m["pred_std"])
    if "persist" in out:
        d = out["model"]["rmse"] - out["persist"]["rmse"]
        log.info("  Δ rmse (model − persist) = %+.6f  (negative = model wins)", d)
    if "rv_lag_5" in out:
        d = out["model"]["rmse"] - out["rv_lag_5"]["rmse"]
        log.info("  Δ rmse (model − rv_lag_5) = %+.6f  (negative = model wins)", d)
    return out


def report_feature_importance(model: XGBRegressor, names: list[str]) -> dict:
    imp = np.asarray(model.feature_importances_, dtype=float)
    paired = sorted(zip(names, imp), key=lambda kv: kv[1], reverse=True)
    log.info("Feature importance (gain, normalised):")
    for name, score in paired:
        bar = "#" * int(round(score * 60))
        log.info("  %-22s %.4f  %s", name, score, bar)
    return {name: float(score) for name, score in paired}


# ---------------------------------------------------------------------------
# Daily retrain / predict API (used by daily_fetch.py)
# ---------------------------------------------------------------------------

def _model_paths(mode: str) -> tuple[Path, Path]:
    return (
        _PROJ_ROOT / "data" / f"vol_model_{mode}.joblib",
        _PROJ_ROOT / "data" / f"vol_model_{mode}.json",
    )


def train_or_refit(mode: str = "mae", force_search: bool = False) -> dict:
    """
    Fit the vol model for daily production use.

    Modes of operation
    ------------------
    force_search=True  : run full Optuna search (chronological train/val/test
                         split, 50 trials, pick best_params + best_iter),
                         then refit on the FULL dataset for production.
                         Updates last_optuna_search timestamp in meta.
    force_search=False : load cached best_params + best_iter from the model
                         meta JSON and refit on the full dataset. ~30 s.
                         If meta is missing, falls back to force_search=True.

    Returns the saved metadata dict.
    """
    model_path, meta_path = _model_paths(mode)
    X, y = assemble()
    feature_names = list(X.columns)
    now_iso = datetime.now(timezone.utc).isoformat()

    have_cache = meta_path.exists()
    do_search = force_search or not have_cache

    if do_search:
        log.info("=" * 70)
        log.info("Vol training — FULL OPTUNA SEARCH (mode=%s)", mode)
        log.info("=" * 70)
        X_tv, y_tv, X_te, y_te = chronological_split(X, y)
        X_tr, y_tr, X_va, y_va = train_val_split(X_tv, y_tv)

        best_params, best_iter = run_optuna(mode, X_tr, y_tr, X_va, y_va)

        # Eval on test BEFORE the final refit on full data, so metrics stay
        # comparable to the original holdout-based numbers.
        held_out_model = fit_final(mode, X_tv, y_tv, best_params, best_iter)
        val_refit_params = {**_base_params(mode), **best_params,
                            "n_estimators": max(int(best_iter * 1.1), 10)}
        val_refit_params.pop("early_stopping_rounds", None)
        val_model = XGBRegressor(**val_refit_params).fit(X_tr, y_tr)
        val_metrics = evaluate(val_model, X_va, y_va, "VAL (pre-final-refit)")
        test_metrics = evaluate(held_out_model, X_te, y_te, "TEST (held-out)")
        last_optuna_search = now_iso
    else:
        log.info("Vol training — REFIT with cached params (mode=%s)", mode)
        meta = json.loads(meta_path.read_text())
        best_params = meta["best_params"]
        best_iter = int(meta["best_iter"])
        val_metrics = meta.get("val_metrics")
        test_metrics = meta.get("test_metrics")
        last_optuna_search = meta.get("last_optuna_search", meta.get("trained_at"))
        log.info("  loaded best_params=%s best_iter=%d (last search: %s)",
                 best_params, best_iter, last_optuna_search)

    # Production refit: full X, y — every available labelled day, including
    # yesterday which was just settled. This is the model that scores today.
    final_params = {
        **_base_params(mode),
        **best_params,
        "n_estimators": max(int(best_iter * 1.1), 10),
    }
    final_params.pop("early_stopping_rounds", None)
    log.info("Production refit on %d rows (n_estimators=%d).",
             len(X), final_params["n_estimators"])
    prod_model = XGBRegressor(**final_params).fit(X, y)
    importance = report_feature_importance(prod_model, feature_names)

    bundle = {
        "model": prod_model,
        "feature_names": feature_names,
        "best_params": best_params,
        "best_iter": best_iter,
        "mode": mode,
        "target": "same_day_realized_range",
        "trained_at": now_iso,
        "train_rows": int(len(X)),
        "train_date_range": [str(X.index.min().date()), str(X.index.max().date())],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    meta = {
        "mode": mode,
        "target": "same_day_realized_range",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "best_params": best_params,
        "best_iter": best_iter,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance_gain": importance,
        "seed": SEED,
        "trained_at": now_iso,
        "last_optuna_search": last_optuna_search,
        "train_rows": int(len(X)),
        "train_date_range": [str(X.index.min().date()), str(X.index.max().date())],
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    log.info("Saved model bundle → %s", model_path)
    log.info("Saved metrics meta → %s", meta_path)
    return meta


def load_model_bundle(mode: str = "mae") -> dict:
    model_path, _ = _model_paths(mode)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Vol model not found at {model_path}. Run train_or_refit() first."
        )
    return joblib.load(model_path)


def predict_one(features: pd.DataFrame, mode: str = "mae", bundle: dict | None = None) -> pd.Series:
    """Predict realized range for a feature DataFrame (one or more rows).

    The bundle's feature_names define the column order — extra columns are
    ignored, missing columns raise KeyError. Returns a Series indexed by
    features.index.
    """
    bundle = bundle or load_model_bundle(mode)
    cols = bundle["feature_names"]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"Features missing required columns: {missing}")
    X = features[cols]
    pred = bundle["model"].predict(X)
    return pd.Series(pred, index=features.index, name="predicted_range")


def build_features_for_date(target_date: pd.Timestamp) -> pd.DataFrame:
    """Build the feature row for one specific date (typically today).

    Used by daily_fetch.py to score a date that has not yet completed its RTH
    session — and therefore has no realized_range row of its own. All three
    feature pieces are fully knowable at 9:00 AM:

      * model_data.parquet — target_date must be present (built with
        keep_unsettled=True so the unsettled row survives the merge).
      * rv_lag_5, rv_lag_22 — computed here as the trailing 5- / 22-day mean
        of the *prior* labelled realized_range series. The HAR-RV builder's
        own rv_lag columns are not used for today's row because today is not
        in its output index (it groups RTH bars, and today has no RTH bars
        yet at 9 AM).
      * vix_level — as-of-09:00 ET lookup from the refreshed VX cache.

    Raises if any required input is missing for target_date.
    """
    target_date = pd.Timestamp(target_date).normalize()
    if target_date.tzinfo is not None:
        target_date = target_date.tz_localize(None)

    feats = pd.read_parquet(FEATURES_PATH)
    feats = feats.drop(columns=["y"], errors="ignore")
    feats.index.name = "date"
    if target_date not in feats.index:
        raise KeyError(
            f"{target_date.date()} not in {FEATURES_PATH}. "
            "Rebuild with keep_unsettled=True."
        )

    rr = build_realized_range_table()
    realized = rr["realized_range"].dropna()
    prior = realized[realized.index < target_date]
    if len(prior) < HAR_LONG_WINDOW:
        raise RuntimeError(
            f"Only {len(prior)} prior labelled realized_range days available; "
            f"need ≥{HAR_LONG_WINDOW} to compute rv_lag_22 for {target_date.date()}."
        )
    lag_cols = pd.DataFrame(
        {
            "rv_lag_5": [float(prior.tail(HAR_SHORT_WINDOW).mean())],
            "rv_lag_22": [float(prior.tail(HAR_LONG_WINDOW).mean())],
        },
        index=pd.DatetimeIndex([target_date], name="date"),
    )

    vix = build_vix_level()
    if target_date not in vix.index:
        raise KeyError(
            f"{target_date.date()} not in vix_level series — verify 09:00 ET "
            "VX bar exists (cron may have fired before the 9:00 print)."
        )
    vix_row = pd.DataFrame(
        {"vix_level": [float(vix.loc[target_date])]},
        index=pd.DatetimeIndex([target_date], name="date"),
    )

    row = feats.loc[[target_date]].join(lag_cols, how="left").join(vix_row, how="left")
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vol regression — same-day realized range.")
    parser.add_argument("--mode", choices=("mae", "aggressive"), default="aggressive",
                        help="mae: reg:absoluteerror, original caps. "
                             "aggressive: reg:squarederror, tighter caps (forces splits).")
    args = parser.parse_args()
    mode = args.mode

    model_path = _PROJ_ROOT / "data" / f"vol_model_{mode}.joblib"
    meta_path = _PROJ_ROOT / "data" / f"vol_model_{mode}.json"

    log.info("=" * 70)
    log.info("Vol prediction — MODE=%s", mode)
    log.info("=" * 70)

    X, y = assemble()
    X_tv, y_tv, X_te, y_te = chronological_split(X, y)

    log.info("Train+val split:")
    X_tr, y_tr, X_va, y_va = train_val_split(X_tv, y_tv)

    feature_names = list(X_tv.columns)

    best_params, best_iter = run_optuna(mode, X_tr, y_tr, X_va, y_va)
    val_refit_params = {**_base_params(mode), **best_params,
                        "n_estimators": max(int(best_iter * 1.1), 10)}
    val_refit_params.pop("early_stopping_rounds", None)
    val_model = XGBRegressor(**val_refit_params).fit(X_tr, y_tr)
    val_metrics = evaluate(val_model, X_va, y_va, "VAL (pre-final-refit)")

    model = fit_final(mode, X_tv, y_tv, best_params, best_iter)
    test_metrics = evaluate(model, X_te, y_te, "TEST")
    importance = report_feature_importance(model, feature_names)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "best_params": best_params,
            "best_iter": best_iter,
            "mode": mode,
            "target": "same_day_realized_range",
        },
        model_path,
    )
    meta_path.write_text(json.dumps(
        {
            "mode": mode,
            "target": "same_day_realized_range",
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "best_params": best_params,
            "best_iter": best_iter,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance_gain": importance,
            "seed": SEED,
        },
        indent=2,
        default=str,
    ))
    log.info("Saved model bundle → %s", model_path)
    log.info("Saved metrics meta → %s", meta_path)


if __name__ == "__main__":
    main()

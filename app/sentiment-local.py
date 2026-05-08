"""
Three-tier sentiment pipeline for S&P 500 direction prediction.

Tier 1 : FinBERT (ProsusAI/finbert) — HF Serverless API
          Scores ALL articles. P(Neutral) > NEUTRAL_THRESHOLD → tagged as noise,
          excluded from Tier 2, but FinBERT probs still averaged into daily base features.
Tier 2 : Qwen3.5-27B — HF Dedicated Endpoint (auto-created and torn down per run)
          Scores non-neutral articles for sector relevance, novelty, and subjectivity.
Tier 3 : Qwen3-Embedding — LM Studio instance (Mac) via /v1/embeddings
          Embeds ALL articles; per-day mean vectors reduced via PCA to 5 context features.

Checkpoint : data/article_scores_checkpoint.parquet — flushed after every batch so the
             run can be interrupted and resumed without re-calling the API.

PCA model  : data/qwen3_pca.joblib — fitted once on the full historical run, then persisted.
             Daily / incremental runs must LOAD and transform, never re-fit. Re-fitting
             would shift the basis vectors and break joinability with prior training rows.

Input  : data/aligned_premarket_news.parquet  (Date, published_at, news_text)
Output : data/sentiment_features.parquet      (one row per trading day, join on Date)
"""

import hashlib
import json
import logging
import os
import re
import requests
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

_PROJ_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"
TIER2_MODEL = "qwen2.5-7b-instruct@q4_k_m"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
NEUTRAL_THRESHOLD = 0.90  # P(Neutral) above this → skip Tier 2
BATCH_SIZE = 256          # unique texts per delta checkpoint file / progress log
TIER2_WORKERS = 4         # concurrent Tier 2 calls per batch. Real ceiling is the
                          # dedicated endpoint's TGI max_batch_size; raising past it
                          # just queues server-side without further speedup.
PCA_COMPONENTS = 5
BUZZ_WINDOW = 10          # trading days for buzz_factor rolling denominator
EWMA_SPAN = 5             # span for sentiment_ewma_5d

# ---------------------------------------------------------------------------
# LM Studio Local Instance (for Tier 2) — adjust these if using a different hosting strategy or model
# ---------------------------------------------------------------------------

LM_STUDIO_BASE_URL="http://100.93.64.114:1234/v1"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ALIGNED_NEWS_PATH = _PROJ_ROOT / "data" / "aligned_premarket_news.parquet"
CHECKPOINT_PATH = _PROJ_ROOT / "data" / "article_scores_checkpoint.parquet"
CHECKPOINT_DIR = _PROJ_ROOT / "data" / "checkpoint_batches"
PCA_PATH = _PROJ_ROOT / "data" / "qwen3_pca.joblib"
OUTPUT_PATH = _PROJ_ROOT / "data" / "sentiment_features.parquet"

# ---------------------------------------------------------------------------
# Tier 2 prompt
# ---------------------------------------------------------------------------
_TIER2_SYSTEM = (
    "You are a financial news analyst. Given a financial news article, assess its potential "
    "impact on broad equity markets and return ONLY a valid JSON object with exactly these "
    "three keys:\n"
    '  "relevance": float 0.1-1.0  '
    "(0.1 = single-stock ticker, 0.5 = sector-wide, 1.0 = global macro/index-wide)\n"
    '  "novelty": float 0.1-1.0  '
    "(0.1 = routine update/old news, 1.0 = breaking event/black swan)\n"
    '  "subjectivity": float 0.1-1.0  '
    "(0.1 = opinion/rumor, 1.0 = hard fact/government data/SEC filing)\n"
    "Return ONLY the JSON object. No preamble, no explanation, no markdown."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{[^{}]+\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# 4xx statuses that indicate permanent client errors — retrying cannot fix them
# and burns the retry budget. 429 is excluded because it is transient by nature.
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 422})


def _retry(fn, max_retries: int = 5, base_delay: float = 2.0):
    """
    Status-aware retry with exponential backoff.

    Strategy by HTTP status:
      • 429 (rate limit)     — honor Retry-After header; else 60s × 2**attempt.
      • 503 (model loading)  — honor Retry-After header; else 10s × 2**attempt.
                               HF returns 503 while a serverless model cold-loads.
      • 4xx in _PERMANENT_STATUSES — raise immediately (auth, malformed, missing model).
      • Other errors / network — base_delay × 2**attempt.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except (HfHubHTTPError, requests.exceptions.HTTPError) as exc:
            status = exc.response.status_code if exc.response is not None else None

            if status in _PERMANENT_STATUSES:
                log.error("Permanent HTTP %s, not retrying: %s", status, exc)
                raise

            if attempt == max_retries - 1:
                raise

            retry_after = (
                exc.response.headers.get("Retry-After") if exc.response is not None else None
            )
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    # Retry-After can also be an HTTP-date — fall back to default
                    delay = base_delay * (2 ** attempt)
            elif status == 429:
                delay = 60.0 * (2 ** attempt)  # 60, 120, 240, 480, 960 s
            elif status == 503:
                delay = 10.0 * (2 ** attempt)  # 10, 20, 40, 80, 160 s
            else:
                delay = base_delay * (2 ** attempt)

            log.warning(
                "HTTP %s (attempt %d/%d). Backing off %.1fs.",
                status, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Tier 1: FinBERT
# ---------------------------------------------------------------------------

_FINBERT_SUB_BATCH = 32  # HF Inference API silently truncates large batches


def _score_finbert_batch(texts: list, token: str) -> list:
    """
    Returns [{"prob_pos", "prob_neg", "prob_neu"}, ...] in the same order as `texts`.

    Splits into sub-batches of _FINBERT_SUB_BATCH to avoid silent truncation by the
    HF Inference API, which has an undocumented per-request input limit.
    """
    def _call_sub(sub: list):
        resp = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{FINBERT_MODEL}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inputs": sub,
                "parameters": {"top_k": None},
                "options": {"wait_for_model": True},
            },
            timeout=120,
        )
        resp.raise_for_status()
        parsed = resp.json()
        log.debug("FinBERT raw response (first 2 items): %s", parsed[:2])
        log.info("FinBERT response: len=%d, first_item_type=%s", len(parsed), type(parsed[0]).__name__ if parsed else "empty")
        # HF router sometimes wraps batch results in an extra outer list
        if (
            len(parsed) == 1
            and isinstance(parsed[0], list)
            and len(parsed[0]) == len(sub)
        ):
            parsed = parsed[0]
        if len(parsed) != len(sub):
            raise ValueError(
                f"FinBERT returned {len(parsed)} results for {len(sub)} inputs"
            )
        return parsed

    results = []
    for start in range(0, len(texts), _FINBERT_SUB_BATCH):
        sub = texts[start : start + _FINBERT_SUB_BATCH]
        parsed = _retry(lambda s=sub: _call_sub(s))
        for item in parsed:
            row = {entry["label"].lower(): entry["score"] for entry in item}
            results.append({
                "prob_pos": row.get("positive", 0.0),
                "prob_neg": row.get("negative", 0.0),
                "prob_neu": row.get("neutral", 0.0),
            })
    return results


# ---------------------------------------------------------------------------
# Tier 2: Qwen3.5-9B
# ---------------------------------------------------------------------------

def _parse_tier2_response(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_RE.search(raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _score_tier2(text: str, client: str) -> dict | None:
    """
    Returns {"relevance", "novelty", "subjectivity"} or None on parse/API failure.
    None causes this article to be excluded from Tier 2 daily aggregation.
    Its FinBERT scores are still included in the base daily means.
    """
    def _call():
        resp = requests.post(
            url="http://100.93.64.114:1234/v1/chat/completions",
            json={
                "model": "qwen2.5-7b-instruct-mlx@8bit",
                "messages": [
                    {"role": "system", "content": _TIER2_SYSTEM},
                    {"role": "user", "content": f"Article: {text}"},
                ],
                "max_tokens": 128,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw = _retry(_call)
    except Exception as exc:
        log.warning("Tier 2 exhausted retries: %s", exc)
        return None

    parsed = _parse_tier2_response(raw)
    if parsed is None:
        log.warning("Tier 2 JSON parse failed. Raw: %.200s", raw)
        return None

    def _get(d: dict, exact: str, prefix: str):
        if exact in d:
            return float(d[exact])
        fallback = next((v for k, v in d.items() if k.startswith(prefix)), None)
        if fallback is not None:
            return float(fallback)
        raise KeyError(exact)

    try:
        return {
            "relevance": _get(parsed, "relevance", "rel"),
            "novelty": _get(parsed, "novelty", "nov"),
            "subjectivity": _get(parsed, "subjectivity", "sub"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Tier 2 field extraction failed (%s). Parsed: %s", exc, parsed)
        return None


# ---------------------------------------------------------------------------
# Tier 3: Qwen3-Embedding (LM Studio on Mac)
# ---------------------------------------------------------------------------

def _load_embedding_model(model_id: str = EMBEDDING_MODEL) -> SentenceTransformer:
    log.info("Loading embedding model %s on CPU (first run downloads ~1.2 GB)...", model_id)
    model = SentenceTransformer(model_id, device="cpu")
    log.info("Embedding model loaded.")
    return model


def _embed_batch(texts: list, model: SentenceTransformer) -> np.ndarray:
    """
    Returns float32 array of shape (n, embedding_dim) in the same order as `texts`.
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    log.info("Embedding: %d texts → shape %s, dim=%d", len(texts), embeddings.shape, embeddings.shape[1])
    return embeddings

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: Path, checkpoint_dir: Path) -> pd.DataFrame:
    """
    Reads the consolidated checkpoint and any pending delta files, returning a single
    deduplicated DataFrame.  Delta files override consolidated rows for the same
    text_hash (keep='last'), so Phase 1 Tier 2 retries written to the consolidated
    file always win over stale delta entries.
    """
    frames = []
    if checkpoint_path.exists():
        frames.append(pd.read_parquet(checkpoint_path))
    if checkpoint_dir.exists():
        for f in sorted(checkpoint_dir.glob("batch_*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("text_hash", keep="last").reset_index(drop=True)


def _consolidate_checkpoint(
    df_all: pd.DataFrame, checkpoint_path: Path, checkpoint_dir: Path
) -> None:
    """
    Merges all delta files into the single consolidated checkpoint parquet and removes
    the delta directory.  Called once at the end of a completed scoring run so that the
    next resume starts from a single clean file rather than scanning many small deltas.
    """
    df_all.to_parquet(checkpoint_path, index=False)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    log.info("Checkpoint consolidated: %d articles at %s", len(df_all), checkpoint_path)


# ---------------------------------------------------------------------------
# Article scoring with checkpoint / resume
# ---------------------------------------------------------------------------

def score_articles(
    df: pd.DataFrame,
    hf_token: str,
    tier2_client: str,
    embed_model: SentenceTransformer,
    checkpoint_path: Path,
    checkpoint_dir: Path,
) -> pd.DataFrame:
    """
    Runs all three tiers over df["news_text"] and returns a per-article scored DataFrame.

    Checkpoint strategy: articles are keyed by a hash of their news_text. On resume:
    - tier2_scored=True  → skip entirely (neutral articles, or non-neutral with valid scores).
    - tier2_scored=False → Tier 2 previously failed; retry Tier 2 only — FinBERT scores
                           and embeddings are preserved from the checkpoint row.
    - Not in checkpoint  → run all three tiers.

    Each batch writes only its own rows to an append-only delta file in checkpoint_dir
    rather than rewriting the entire checkpoint. This keeps per-batch I/O O(1) instead
    of O(n). At the end of the run _consolidate_checkpoint merges all deltas into the
    single checkpoint_path and removes checkpoint_dir.
    """
    df = df.copy()
    df["text_hash"] = df["news_text"].apply(_text_hash)
    text_by_hash = df.set_index("text_hash")["news_text"].to_dict()

    df_ckpt = _load_checkpoint(checkpoint_path, checkpoint_dir)

    if len(df_ckpt) > 0:
        # Backwards compatibility: checkpoints written before tier2_scored was added
        if "tier2_scored" not in df_ckpt.columns:
            df_ckpt["tier2_scored"] = df_ckpt["is_neutral"] | df_ckpt["relevance"].notna()
        fully_done_hashes = set(df_ckpt.loc[df_ckpt["tier2_scored"], "text_hash"])
        tier2_retry_hashes = set(df_ckpt.loc[~df_ckpt["tier2_scored"], "text_hash"])
        log.info(
            "Checkpoint: %d fully scored, %d awaiting Tier 2 retry.",
            len(fully_done_hashes), len(tier2_retry_hashes),
        )
    else:
        fully_done_hashes = set()
        tier2_retry_hashes = set()

    # --- Phase 1: Retry Tier 2 for articles that previously failed ---
    if tier2_retry_hashes:
        log.info("Retrying Tier 2 for %d articles...", len(tier2_retry_hashes))
        recovered = 0
        for idx in df_ckpt.index[df_ckpt["text_hash"].isin(tier2_retry_hashes)]:
            text = text_by_hash.get(df_ckpt.at[idx, "text_hash"])
            if text is None:
                continue
            tier2 = _score_tier2(text, tier2_client)
            if tier2:
                df_ckpt.at[idx, "relevance"] = tier2["relevance"]
                df_ckpt.at[idx, "novelty"] = tier2["novelty"]
                df_ckpt.at[idx, "subjectivity"] = tier2["subjectivity"]
                df_ckpt.at[idx, "tier2_scored"] = True
                recovered += 1
        if recovered:
            # Write the updated consolidated checkpoint so Phase 1 corrections are
            # durable before Phase 2 delta files are created
            df_ckpt.to_parquet(checkpoint_path, index=False)
        log.info("Tier 2 retry: %d/%d articles recovered.", recovered, len(tier2_retry_hashes))

    # --- Phase 2: Score articles not yet in the checkpoint ---
    # Dedup at the API layer: identical news_text → one set of API calls. Scores are
    # broadcast back to every original row sharing that hash, so frequency-based signals
    # (article_count, buzz_factor, weighted means) are unchanged from the non-deduped path.
    all_seen_hashes = fully_done_hashes | tier2_retry_hashes
    pending = df[~df["text_hash"].isin(all_seen_hashes)].reset_index(drop=True)
    pending_unique = pending.drop_duplicates("text_hash").reset_index(drop=True)
    total_rows = len(pending)
    total_unique = len(pending_unique)
    log.info(
        "%d new article rows (%d unique texts) to score.", total_rows, total_unique,
    )

    if total_unique > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_start = time.time()

        for batch_start in range(0, total_unique, BATCH_SIZE):
            batch_num = batch_start // BATCH_SIZE
            batch = pending_unique.iloc[batch_start : batch_start + BATCH_SIZE]
            texts = batch["news_text"].tolist()

            # Tier 1: FinBERT (must complete before Tier 2 — its is_neutral flag gates
            # which articles get sent to the dedicated endpoint).
            fb_scores = _score_finbert_batch(texts, hf_token)

            # Tier 2 + Tier 3 run concurrently — Tier 2 is network-bound (LM Studio),
            # embed is CPU-bound (local PyTorch, releases GIL during forward pass), so
            # they overlap without contending. TIER2_WORKERS threads issue parallel Tier 2
            # requests; +1 worker handles the embedding call.
            with ThreadPoolExecutor(max_workers=TIER2_WORKERS + 1) as executor:
                embed_future = executor.submit(_embed_batch, texts, embed_model)
                tier2_futures = {
                    i: executor.submit(_score_tier2, texts[i], tier2_client)
                    for i, fb in enumerate(fb_scores)
                    if fb["prob_neu"] <= NEUTRAL_THRESHOLD
                }
                embeddings = embed_future.result()
                tier2_results = {i: f.result() for i, f in tier2_futures.items()}

            # Cache scores keyed by text_hash for broadcast to all duplicate rows
            scores_by_hash: dict = {}
            for i, urow in enumerate(batch.itertuples(index=False)):
                fb = fb_scores[i]
                is_neutral = fb["prob_neu"] > NEUTRAL_THRESHOLD
                scores_by_hash[urow.text_hash] = {
                    "fb": fb,
                    "is_neutral": is_neutral,
                    "tier2": tier2_results.get(i),
                    "embedding": embeddings[i],
                }

            # Broadcast scores onto every original (potentially duplicated) row whose
            # text_hash was scored in this batch. One write per batch — O(1) per flush.
            batch_hashes = set(batch["text_hash"])
            broadcast_rows = pending[pending["text_hash"].isin(batch_hashes)]

            batch_rows = []
            for row in broadcast_rows.itertuples(index=False):
                s = scores_by_hash[row.text_hash]
                tier2 = s["tier2"]
                fb = s["fb"]
                batch_rows.append({
                    "text_hash": row.text_hash,
                    "Date": row.Date,
                    "published_at": row.published_at,
                    "prob_pos": fb["prob_pos"],
                    "prob_neg": fb["prob_neg"],
                    "prob_neu": fb["prob_neu"],
                    "is_neutral": s["is_neutral"],
                    "relevance": tier2["relevance"] if tier2 else np.nan,
                    "novelty": tier2["novelty"] if tier2 else np.nan,
                    "subjectivity": tier2["subjectivity"] if tier2 else np.nan,
                    "tier2_scored": s["is_neutral"] or tier2 is not None,
                    "embedding": s["embedding"].tolist(),
                })

            pd.DataFrame(batch_rows).to_parquet(
                checkpoint_dir / f"batch_{batch_num:06d}.parquet", index=False
            )

            done_unique = min(batch_start + BATCH_SIZE, total_unique)
            elapsed = time.time() - run_start
            rate = done_unique / elapsed if elapsed > 0 else 0.0
            eta_min = (total_unique - done_unique) / rate / 60 if rate > 0 else float("inf")
            log.info(
                "Scored %d/%d unique texts (%.1f tx/s, ETA ~%.0f min)",
                done_unique, total_unique, rate, eta_min,
            )

    # Merge consolidated checkpoint + all deltas into a single file, remove delta dir
    df_all = _load_checkpoint(checkpoint_path, checkpoint_dir)
    _consolidate_checkpoint(df_all, checkpoint_path, checkpoint_dir)
    return df_all


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _aggregate_daily(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses per-article scores into one row per trading day.

    FinBERT means      — averaged over ALL articles (neutral + non-neutral).
    Tier 2 aggregates  — averaged over NON-NEUTRAL articles with valid Tier 2 scores.
    weighted_impact    — mean of (p_pos − p_neg) × relevance × subjectivity
                         over non-neutral articles.
    sentiment_entropy  — std of raw_impact scores (NaN for days with ≤1 article).
    article_count      — non-neutral articles with valid Tier 2 scores; used as the
                         buzz_factor denominator.
    mean_embedding     — per-day mean of ALL article embeddings (fed into PCA).
                         Neutral articles are included because embeddings capture topic,
                         not sentiment polarity.
    """
    # FinBERT means over all articles
    finbert_agg = df_scored.groupby("Date").agg(
        prob_pos_mean=("prob_pos", "mean"),
        prob_neg_mean=("prob_neg", "mean"),
        prob_neu_mean=("prob_neu", "mean"),
    )

    # Tier 2 metrics over non-neutral articles only
    non_neutral = df_scored[~df_scored["is_neutral"]].copy()
    non_neutral["polarity"] = non_neutral["prob_pos"] - non_neutral["prob_neg"]
    non_neutral["raw_impact"] = (
        non_neutral["polarity"] * non_neutral["relevance"] * non_neutral["subjectivity"]
    )
    tier2_agg = non_neutral.groupby("Date").agg(
        relevance_mean=("relevance", "mean"),
        novelty_mean=("novelty", "mean"),
        subjectivity_mean=("subjectivity", "mean"),
        weighted_impact=("raw_impact", "mean"),
        # std of raw polarity (p_pos − p_neg) — measures disagreement in sentiment signal
        sentiment_entropy=("polarity", "std"),
        # polarity is always non-NaN for non-neutral articles regardless of Tier 2 outcome,
        # so this counts all articles that passed the FinBERT filter, not just those with
        # successful Tier 2 responses
        article_count=("polarity", "count"),
    )

    # Per-day mean embedding (all articles)
    df_scored = df_scored.copy()
    df_scored["_emb"] = df_scored["embedding"].apply(np.asarray)
    emb_agg = (
        df_scored.groupby("Date")["_emb"]
        .apply(lambda arrs: np.stack(arrs.values).mean(axis=0))
        .rename("mean_embedding")
    )

    df_daily = finbert_agg.join(tier2_agg, how="left").join(emb_agg, how="left")
    df_daily = df_daily.reset_index()
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    return df_daily


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def _compute_pca(df_daily: pd.DataFrame, pca_path: Path) -> pd.DataFrame:
    """
    Reduces daily mean embeddings to PCA_COMPONENTS dimensions.

    On the first (historical bulk) run the model is fitted and saved to pca_path.
    All subsequent runs — including daily incremental production runs — must load
    the saved model and call transform() only so the basis vectors never shift.
    """
    emb_matrix = np.stack(df_daily["mean_embedding"].values).astype(np.float32)

    if pca_path.exists():
        log.info("Loading PCA model from %s", pca_path)
        pca = joblib.load(pca_path)
    else:
        log.info(
            "Fitting PCA (n_components=%d) on %d daily embeddings...",
            PCA_COMPONENTS, len(emb_matrix),
        )
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        pca.fit(emb_matrix)
        joblib.dump(pca, pca_path)
        pct = np.round(pca.explained_variance_ratio_ * 100, 1).tolist()
        log.info("PCA saved to %s. Variance explained per component: %s%%", pca_path, pct)

    coords = pca.transform(emb_matrix)
    for i in range(PCA_COMPONENTS):
        df_daily[f"context_pca_{i + 1}"] = coords[:, i]

    return df_daily.drop(columns=["mean_embedding"])


# ---------------------------------------------------------------------------
# Time-series features
# ---------------------------------------------------------------------------

def _build_time_series_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling and lag features. All operations are strictly backward-looking.

    buzz_factor      : article_count / BUZZ_WINDOW-day SMA of article_count.
                       Days with article_count NaN (all-neutral) are treated as 0
                       before the rolling mean to avoid shrinking the window silently.
    sentiment_momentum: weighted_impact delta vs the previous trading day.
    sentiment_ewma_5d : exponential WMA of weighted_impact (span=EWMA_SPAN days).
    impact_lag_1..5  : explicit lags of weighted_impact for 1–5 trading days back.
    """
    df = df_daily.sort_values("Date").reset_index(drop=True).copy()

    count = df["article_count"].fillna(0)
    # Denominator uses only past days so today's volume is compared against yesterday's baseline
    df["buzz_factor"] = count / count.shift(1).rolling(BUZZ_WINDOW, min_periods=1).mean()

    df["sentiment_momentum"] = df["weighted_impact"] - df["weighted_impact"].shift(1)
    df["sentiment_ewma_5d"] = df["weighted_impact"].ewm(span=EWMA_SPAN, adjust=False).mean()

    for lag in range(1, 6):
        df[f"impact_lag_{lag}"] = df["weighted_impact"].shift(lag)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_sentiment_pipeline(
    aligned_path: Path = ALIGNED_NEWS_PATH,
    output_path: Path = OUTPUT_PATH,
    pca_path: Path = PCA_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> pd.DataFrame:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set. Add it to app/.env.")

    log.info("Loading aligned news from %s", aligned_path)
    df_news = pd.read_parquet(aligned_path)
    log.info(
        "%d articles across %d trading days.", len(df_news), df_news["Date"].nunique()
    )

    embed_model = _load_embedding_model()

    df_scored = score_articles(
        df=df_news,
        hf_token=HF_TOKEN,
        tier2_client=LM_STUDIO_BASE_URL,
        embed_model=embed_model,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
    )

    log.info("Aggregating %d scored articles into daily rows...", len(df_scored))
    df_daily = _aggregate_daily(df_scored)
    df_daily = _compute_pca(df_daily, pca_path)
    df_daily = _build_time_series_features(df_daily)

    cols = [
        "Date",
        "prob_pos_mean", "prob_neg_mean", "prob_neu_mean",
        "relevance_mean", "novelty_mean", "subjectivity_mean",
        "article_count",
        "weighted_impact", "sentiment_entropy",
        "buzz_factor", "sentiment_momentum",
        "sentiment_ewma_5d",
        "impact_lag_1", "impact_lag_2", "impact_lag_3", "impact_lag_4", "impact_lag_5",
        "context_pca_1", "context_pca_2", "context_pca_3", "context_pca_4", "context_pca_5",
    ]
    df_daily = df_daily[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_parquet(output_path, index=False)
    log.info("Saved %d daily sentiment rows to %s", len(df_daily), output_path)
    return df_daily


if __name__ == "__main__":
    run_sentiment_pipeline()

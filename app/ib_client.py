"""
Shared IB Gateway/TWS client for incremental 1 min bar fetching.

Used by the daily_fetch.py orchestrator. Replaces Databento for the rolling
incremental window only — the historical 5 yr cold-start cache stays on
Databento (see market_data.fetch_es_futures / macro_data.fetch_series).

Connection
----------
Live IB Gateway on 127.0.0.1:4001 by default. Override with IB_HOST / IB_PORT
env vars. CLIENT_ID defaults to 1; if you run two pipelines concurrently,
override IB_CLIENT_ID to keep them from clobbering each other's connection.

Output schema (matches Databento parquet cache exactly)
-------------------------------------------------------
fetch_continuous_bars returns a DataFrame with:
  index    : UTC-aware DatetimeIndex, tz=UTC, sorted ascending, no duplicates.
  columns  : open, high, low, close (float64), volume (float64).

This shape is identical to what _normalize_databento_df produces, so the two
sources can be pd.concat'd into the same parquet file without any further
massaging.

Supported continuous futures
----------------------------
  ES → CME      (S&P 500 e-mini)
  VX → CFE      (VIX)
  ZN → CBOT     (10Y Treasury Note)

IB ContFuture roll
------------------
ib_async resolves ContFuture(symbol, exchange) to the front-month at request
time. As contracts roll, successive daily runs will fetch from the new front
month — the underlying price series remains continuous because the gap fetch
on roll day spans the rollover and IB stitches the front-month bars at the
exchange level. This matches Databento's `c.0` semantic, with the same minor
caveat that prices around the roll moment may exhibit a small discontinuity.

ContFuture endDateTime + per-symbol pacing
------------------------------------------
IB Error 10339 fires on *unresolved* ContFuture requests with an explicit
endDateTime. Once we call qualifyContractsAsync, the contract returned is a
concrete Future (conId, localSymbol, lastTradeDateOrContractMonth filled
in) and IB accepts explicit endDateTime against it — that's what enables
chunked fetches.

IB HMDS pacing for 1-min bars is contract-specific. ES (the highest-
volume contract on CME) tolerates ~30 D in a single request, but lower-
volume contracts cancel server-side with Error 162 well below that:
empirically VX (CFE) tops out around 5 D and ZN (CBOT) around 7 D.
PER_SYMBOL_CHUNK_DAYS encodes those caps. When a requested gap exceeds
the per-symbol cap, fetch_continuous_bars walks endDateTime backwards
in chunk-sized windows and concatenates the results.

Idempotency: we still trim to the half-open window [start, end) on the
client side, so same-day reruns with a 09:01 ET upper bound remain
deterministic.

Backfills larger than MAX_DURATION_DAYS — or any backfill spanning a
contract roll for monthly-roll products (VX) — should be cold-started
from Databento rather than chunk-fetched. The resolved front-month
contract only holds data for its own life, so older chunks past the roll
return zero bars and the chunking loop stops.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone  # noqa: F401  (re-exported)

import pandas as pd
from ib_async import IB, ContFuture, Future, util

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection config
# ---------------------------------------------------------------------------

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "4001"))         # live IB Gateway
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
IB_CONNECT_TIMEOUT_SEC = 10.0

# Canonical fetch upper-bound for the daily pipeline. See today_9am_et_as_utc().
_FETCH_END_HOUR_ET = 9
_FETCH_END_MINUTE_ET = 1

# IB HMDS pacing for 1-min bars is contract-specific. ES (highest-volume
# contract on its exchange) tolerates large windows, but lower-volume
# contracts like VX (CFE) and ZN (CBOT) silently cancel with Error 162
# beyond ~5–7 days. When the gap exceeds the per-symbol cap we chunk into
# sequential single-contract requests walked back via endDateTime.
MAX_DURATION_DAYS = 30  # hard ceiling for backfill before cold-start required
PER_SYMBOL_CHUNK_DAYS: dict[str, int] = {
    "ES": 30,
    "VX": 5,
    "ZN": 7,
}
INTER_CHUNK_SLEEP_SEC = 2.0  # IB HMDS pacing: ~6 identical reqs / 2 sec
HIST_REQUEST_TIMEOUT_SEC = 120.0


@dataclass(frozen=True)
class ContractSpec:
    """Canonical IB ContFuture spec for one symbol used by the pipeline."""
    symbol: str
    exchange: str
    currency: str = "USD"

    def to_contract(self) -> ContFuture:
        return ContFuture(symbol=self.symbol, exchange=self.exchange, currency=self.currency)


# Canonical specs used across market_data.py and macro_data.py.
SPECS: dict[str, ContractSpec] = {
    "ES": ContractSpec("ES", "CME"),
    "VX": ContractSpec("VIX", "CFE"),
    "ZN": ContractSpec("ZN", "CBOT"),
}


# ---------------------------------------------------------------------------
# Fetch upper-bound for the daily pipeline
# ---------------------------------------------------------------------------

def today_9am_et_as_utc() -> pd.Timestamp:
    """Return the canonical fetch upper-bound: today 09:01 ET as a UTC timestamp.

    Why pin to a fixed time-of-day rather than utcnow()?
    -----------------------------------------------------
    The IB fetch trims results with `combined.index < end` (exclusive). If we
    use utcnow() and the script runs at 11 AM ET, IB returns bars through 11 AM
    today — including 09:30-10:59 RTH bars. Those bars then enter
    build_realized_range_table's groupby, produce a partial-day realized_range
    for today, and pollute the next retrain with a wrong label.

    Pinning end to today 09:01 ET (exclusive) means IB only returns bars
    timestamped strictly before 09:01 ET — i.e. the 09:00 left-labelled bar
    (which covers 09:00:00-09:00:59) plus everything before. No 09:30+ RTH
    bars enter the cache no matter when the script is invoked.

    Effect: the entire daily pipeline becomes idempotent across reruns within
    a single ET day. Running at 09:05, 11:00, or 18:00 all produce the same
    cache state and the same forecast.

    Raises
    ------
    RuntimeError
        If current ET time < 09:01 — today's 09:00 bar may not be finalised
        in IB yet, so IB would return nothing or a partial bar. The caller's
        time guard normally catches this before we reach here.
    """
    now_et = pd.Timestamp.now(tz="America/New_York")
    end_et = now_et.normalize().replace(
        hour=_FETCH_END_HOUR_ET, minute=_FETCH_END_MINUTE_ET, second=0, microsecond=0
    )
    if now_et < end_et:
        raise RuntimeError(
            f"Cannot compute fetch-end before 09:01 ET — today's 09:00 bar may "
            f"not be finalised in IB yet (current ET: {now_et.strftime('%H:%M:%S')})."
        )
    return end_et.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def ib_session(
    host: str = IB_HOST,
    port: int = IB_PORT,
    client_id: int = IB_CLIENT_ID,
):
    """Connect to IB Gateway/TWS for the duration of the with-block.

    Yields a connected IB instance. Always disconnects on exit, including on
    exception — critical because IB enforces a one-client-id-at-a-time rule
    and a dangling connection blocks the next run.
    """
    ib = IB()
    log.info("Connecting to IB at %s:%d (clientId=%d)...", host, port, client_id)
    try:
        await asyncio.wait_for(
            ib.connectAsync(host, port, clientId=client_id),
            timeout=IB_CONNECT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        raise ConnectionError(
            f"IB connect to {host}:{port} timed out after {IB_CONNECT_TIMEOUT_SEC}s. "
            "Is IB Gateway running and API access enabled?"
        ) from exc
    log.info("IB connected. Accounts: %s", ib.managedAccounts())
    try:
        yield ib
    finally:
        if ib.isConnected():
            ib.disconnect()
            log.info("IB disconnected.")


# ---------------------------------------------------------------------------
# Bar fetch
# ---------------------------------------------------------------------------

def _bars_to_df(bars) -> pd.DataFrame:
    """Convert ib_async BarData list into a UTC-indexed OHLCV DataFrame.

    `util.df` already does the heavy lifting (datetime → column, prices as
    float). We then normalise the timezone to UTC and select the canonical
    column set so the result is drop-in compatible with the Databento
    parquet cache.
    """
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = util.df(bars)
    # ib_async sometimes returns the bar timestamp as the index, sometimes as
    # a 'date' column. Handle both.
    if "date" in df.columns:
        df = df.set_index("date")
    df.index.name = "ts_event"

    # Bars for 1 min size come back tz-aware in the exchange timezone (e.g.
    # US/Central for CME). Convert to UTC; localize as UTC if (rare) tz-naive.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].astype("float64").sort_index()

    # Drop duplicate timestamps if any (defensive — should not happen with
    # single-symbol continuous requests, but cheap insurance).
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    return df


async def fetch_continuous_bars(
    ib: IB,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    use_rth: bool = False,
) -> pd.DataFrame:
    """Fetch 1 min bars for a continuous future in one request, trim client-side.

    IB does not allow `endDateTime` on ContFuture historical requests
    (Error 10339), so we cannot pin the upper bound at the API layer. We
    instead request bars ending "now" with a durationStr that comfortably
    covers (now - start), then trim the result to the requested half-open
    window [start, end). Idempotency of the cache is preserved because
    `end` is pinned by the caller to today 09:01 ET — any bars IB returned
    beyond that point are dropped before saving.

    Parameters
    ----------
    ib          : connected IB instance (use the ib_session context manager).
    symbol      : one of SPECS keys ("ES", "VX", "ZN").
    start, end  : tz-aware (UTC) timestamps. Inclusive of start, exclusive of
                  end. start must be < end.
    bar_size    : IB bar size string ("1 min" default).
    what_to_show: "TRADES" by default — matches Databento ohlcv-1m semantics.
    use_rth     : False (default) → full ETH session, matching our existing
                  cache that needs the overnight + pre-market windows.

    Returns
    -------
    UTC-indexed OHLCV DataFrame in the same schema as the Databento cache.
    """
    if symbol not in SPECS:
        raise ValueError(f"Unknown symbol {symbol!r}. Known: {sorted(SPECS)}")
    if start >= end:
        raise ValueError(f"start {start} must be strictly before end {end}")

    spec = SPECS[symbol]
    contract = spec.to_contract()
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified or qualified[0] is None:
        raise RuntimeError(
            f"IB failed to qualify ContFuture(symbol={spec.symbol!r}, "
            f"exchange={spec.exchange!r}). Verify market-data subscription "
            "and that the symbol/exchange combination is correct for IB."
        )
    contract = qualified[0]
    log.info(
        "[%s] qualified: localSymbol=%s expiry=%s",
        symbol, contract.localSymbol, contract.lastTradeDateOrContractMonth,
    )

    now_utc = pd.Timestamp.now(tz="UTC")
    gap = now_utc - start
    total_days = max(1, gap.days + (1 if gap.seconds > 0 else 0) + 1)
    if total_days > MAX_DURATION_DAYS:
        raise RuntimeError(
            f"[{symbol}] gap {start} → now ({total_days} D) exceeds backfill "
            f"ceiling ({MAX_DURATION_DAYS} D). Cold-start refresh the cache "
            f"via the per-source Databento path instead of incremental backfill."
        )

    chunk_days = PER_SYMBOL_CHUNK_DAYS.get(symbol, MAX_DURATION_DAYS)
    if total_days <= chunk_days:
        df = await _fetch_single(
            ib, contract, symbol,
            end_dt="", duration_str=f"{total_days} D",
            bar_size=bar_size, what_to_show=what_to_show, use_rth=use_rth,
        )
    else:
        # Chunking requires an explicit endDateTime, which IB rejects on
        # ContFuture (Error 10339). Rebuild as a concrete Future from the
        # qualified front-month's conId — IB accepts endDateTime on that.
        concrete = Future(
            conId=contract.conId,
            exchange=contract.exchange,
            currency=contract.currency,
            localSymbol=contract.localSymbol,
            lastTradeDateOrContractMonth=contract.lastTradeDateOrContractMonth,
        )
        log.info("[%s] gap=%d D > chunk_days=%d → chunked fetch on %s",
                 symbol, total_days, chunk_days, contract.localSymbol)
        df = await _fetch_chunked(
            ib, concrete, symbol,
            start=start, total_days=total_days, chunk_days=chunk_days,
            bar_size=bar_size, what_to_show=what_to_show, use_rth=use_rth,
        )

    if df.empty:
        log.info("[%s] IB returned 0 bars", symbol)
        return df

    # Trim to the requested half-open window [start, end). The end trim is
    # what protects the cache from absorbing post-09:01 ET bars.
    trimmed = df.loc[(df.index >= start) & (df.index < end)]
    log.info("[%s] fetched %d bars, trimmed to %d in [%s, %s)",
             symbol, len(df), len(trimmed), start, end)
    return trimmed


async def _fetch_single(
    ib: IB,
    contract,
    symbol: str,
    *,
    end_dt: str,
    duration_str: str,
    bar_size: str,
    what_to_show: str,
    use_rth: bool,
) -> pd.DataFrame:
    """One reqHistoricalDataAsync call, ContFuture-compatible.

    end_dt="" → "now" (the only form ContFuture accepts pre-qualification).
    Once `contract` is a qualified concrete Future (conId/localSymbol set by
    qualifyContractsAsync), IB accepts an explicit endDateTime — that's what
    enables chunking. Error 10339 only fires on unresolved ContFutures.
    """
    log.info("[%s] req: duration=%s end=%r", symbol, duration_str, end_dt or "now")
    bars = await asyncio.wait_for(
        ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_dt,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
        ),
        timeout=HIST_REQUEST_TIMEOUT_SEC,
    )
    return _bars_to_df(bars)


async def _fetch_chunked(
    ib: IB,
    contract,
    symbol: str,
    *,
    start: pd.Timestamp,
    total_days: int,
    chunk_days: int,
    bar_size: str,
    what_to_show: str,
    use_rth: bool,
) -> pd.DataFrame:
    """Walk endDateTime backwards in chunk_days-sized windows, concat results.

    The first chunk uses end_dt="" (now); subsequent chunks step `chunk_days`
    further back. We stop once the covered span reaches `total_days`.

    Caveat: for monthly-roll contracts (VX), chunks that pre-date the current
    front-month's listing may return zero bars — the resolved contract only
    holds data for its own life. Backfills spanning a roll need a cold-start
    refresh from the Databento path.
    """
    frames: list[pd.DataFrame] = []
    # IB endDateTime UTC format: "YYYYMMDD-HH:MM:SS"
    fmt = "%Y%m%d-%H:%M:%S"
    end_anchor = pd.Timestamp.now(tz="UTC")
    covered = 0
    chunk_idx = 0
    while covered < total_days:
        remaining = total_days - covered
        this_chunk = min(chunk_days, remaining)
        # First request uses end_dt="" so IB's notion of "now" aligns
        # exactly with the server clock (no client-skew issues).
        # IB accepts "yyyymmdd-hh:mm:ss" as UTC (the dash form implies UTC;
        # appending " UTC" is rejected as Error 10314).
        end_dt = "" if chunk_idx == 0 else (
            (end_anchor - pd.Timedelta(days=covered)).strftime(fmt)
        )
        df_chunk = await _fetch_single(
            ib, contract, symbol,
            end_dt=end_dt, duration_str=f"{this_chunk} D",
            bar_size=bar_size, what_to_show=what_to_show, use_rth=use_rth,
        )
        log.info("[%s] chunk %d: end=%s duration=%d D → %d bars",
                 symbol, chunk_idx, end_dt or "now", this_chunk, len(df_chunk))
        if not df_chunk.empty:
            frames.append(df_chunk)
        elif chunk_idx > 0:
            # Empty older chunk usually means we've walked past the contract's
            # earliest available data (esp. for monthly-roll VX after a roll).
            log.info("[%s] empty older chunk — likely past contract listing; "
                     "stop chunking.", symbol)
            break
        covered += this_chunk
        chunk_idx += 1
        if covered < total_days:
            await asyncio.sleep(INTER_CHUNK_SLEEP_SEC)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# ---------------------------------------------------------------------------
# Convenience sync wrapper for callers that don't manage their own event loop
# ---------------------------------------------------------------------------

def fetch_continuous_bars_sync(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    **kwargs,
) -> pd.DataFrame:
    """Synchronous one-shot wrapper: open connection, fetch, close.

    Convenient for the legacy market_data.update_es_cache_incremental() and
    macro_data update path. The orchestrator (daily_fetch.py) uses the async
    API directly to share one connection across multiple symbols.
    """
    async def _run():
        async with ib_session() as ib:
            return await fetch_continuous_bars(ib, symbol, start, end, **kwargs)

    return asyncio.run(_run())

import asyncio
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)

# NED allows 200 requests per five minutes. Share pacing across stores and pages.
_request_lock = asyncio.Lock()
_last_request_at = 0.0


class NedDataStore(DataStore):
    """Caches Dutch load and renewable generation data from the NED API."""

    API_URL = "https://api.ned.nl/v1/utilizations"
    # (NED type, activity, model feature column)
    SERIES = (
        (59, 2, "ned_load_mw"),
        (2, 1, "ned_solar_mw"),
        (1, 1, "ned_wind_onshore_mw"),
        (17, 1, "ned_wind_offshore_mw"),
    )
    FEATURE_COLUMNS = tuple(series[2] for series in SERIES)

    def __init__(self, region: PriceRegion, storage_dir: str | None = None):
        # v1 contains hourly values repeated four times; never mix them with v2.
        super().__init__(region, storage_dir, "nedforecast_v2")
        self.api_key = os.getenv("EPEXPREDICTOR_NED_API_KEY") or None
        self.update_lock = asyncio.Lock()
        self._next_fetch_attempt = datetime.min.replace(tzinfo=timezone.utc)
        if self.api_key is None:
            log.warning("EPEXPREDICTOR_NED_API_KEY is not configured; skipping NED features")

    def _update_data(self, df: pd.DataFrame) -> bool:
        # Other data stores drop incomplete rows, but each NED series can have gaps.
        old = self.data
        self.data = df.combine_first(self.data).sort_index()
        changed = not old.round(10).equals(self.data.round(10))
        if changed:
            self.last_updated = datetime.now(timezone.utc)
            self.known_source_horizon = None
            self.source_horizon_revalitation_ts = None
        return changed

    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        if self.api_key is None or self.region.country_code != "NL":
            return False
        async with self.update_lock:
            now = datetime.now(timezone.utc)
            if now < self._next_fetch_attempt:
                return False
            # Retry absent/partial records hourly, including records the source
            # has not published yet. Training and prediction share this store.
            self._next_fetch_attempt = now + timedelta(hours=1)
            changed = False
            # Re-fetch a rolling window because NED can update recent records.
            refresh_start = max(start, datetime.now(timezone.utc) - timedelta(days=2))
            missing_end = min(end, refresh_start - timedelta(minutes=15))
            missing = self.gen_missing_date_ranges(start, missing_end) if missing_end >= start else []
            for left, right in missing:
                changed = await self._refresh_range(left.to_pydatetime(), right.to_pydatetime()) or changed
            changed = await self._refresh_range(refresh_start, end) or changed
            return changed

    def gen_missing_date_ranges(self, start: datetime, end: datetime):
        # A missing offshore value must be retried even when load/solar exist.
        needed = pd.date_range(pd.Timestamp(start).floor("15min"), pd.Timestamp(end).ceil("15min"), freq="15min")
        missing = self.data.reindex(index=needed, columns=self.FEATURE_COLUMNS).isna().any(axis=1)
        indices = missing.index[missing]
        if indices.empty:
            return []
        groups = indices.to_series().diff().ne(pd.Timedelta(minutes=15)).cumsum()
        return list(indices.to_series().groupby(groups).agg(["min", "max"]).itertuples(index=False, name=None))

    async def refresh_range(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            return await self._refresh_range(start, end)

    async def _refresh_range(self, start: datetime, end: datetime) -> bool:
        if self.api_key is None or self.region.country_code != "NL" or end < start:
            return False
        start = pd.Timestamp(start).tz_convert("UTC").floor("15min").to_pydatetime()
        end = pd.Timestamp(end).tz_convert("UTC").floor("15min").to_pydatetime()
        # The current quarter belongs to the forecast request. Splitting at the
        # wall-clock second can otherwise omit that quarter from both responses.
        now = pd.Timestamp.now(tz="UTC").floor("15min").to_pydatetime()
        timeout = aiohttp.ClientTimeout(total=45)
        headers = {"X-AUTH-TOKEN": self.api_key, "Accept": "application/ld+json"}
        changed = False
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            # Persist each week separately, so an interrupted first backfill can
            # resume without repeating six months of requests.
            chunk_start = start
            while chunk_start <= end:
                chunk_end = min(end, chunk_start + timedelta(days=7) - timedelta(minutes=15))
                feature_series: dict[str, pd.Series] = {}
                for type_id, activity_id, feature in self.SERIES:
                    # Classification 2 is actual/current; classification 1 is the forecast.
                    # Use actuals through now and NED forecasts after now.
                    for left, right, classification in (
                        (chunk_start, min(chunk_end, now - timedelta(minutes=15)), 2),
                        (max(chunk_start, now), chunk_end, 1),
                    ):
                        if right < left:
                            continue
                        params = {
                            "point": 0,
                            "type": type_id,
                            "granularity": 4,  # genuine 15-minute observations/forecasts
                            "granularitytimezone": 0,  # UTC
                            "classification": classification,
                            "activity": activity_id,
                            "itemsPerPage": 200,
                            "validfrom[after]": left.date().isoformat(),
                            "validfrom[strictly_before]": (right + timedelta(days=1)).date().isoformat(),
                        }
                        try:
                            records = await self._fetch_records(session, params)
                        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                            log.error("%s: NED %s request failed: %s", self.region.bidding_zone_entsoe, feature, exc)
                            # Preserve successful series/chunks; gaps use the
                            # model without NED and are retried next hour.
                            continue
                        rows: dict[pd.Timestamp, float] = {}
                        revisions: dict[pd.Timestamp, pd.Timestamp] = {}
                        for record in records:
                            try:
                                ts = pd.to_datetime(record["validfrom"], utc=True)
                                validto = pd.to_datetime(record["validto"], utc=True)
                                value = float(record["capacity"]) / 1000.0
                                revision = pd.to_datetime(record.get("lastupdate", "1970-01-01"), utc=True)
                                if pd.isna(revision):
                                    revision = pd.Timestamp(0, tz="UTC")
                                if (left <= ts.to_pydatetime() <= right
                                        and ts == ts.floor("15min")
                                        and validto - ts == pd.Timedelta(minutes=15)
                                        and math.isfinite(value)
                                        and revision >= revisions.get(ts, pd.Timestamp(0, tz="UTC"))):
                                    rows[ts] = value
                                    revisions[ts] = revision
                            except (KeyError, TypeError, ValueError):
                                continue
                        if rows:
                            new_series = pd.Series(rows, name=feature, dtype="float64")
                            # Actuals and forecasts are fetched in separate
                            # requests, but share one model feature column.
                            # Prefer the later response where timestamps overlap.
                            previous = feature_series.get(feature)
                            feature_series[feature] = new_series.combine_first(previous) if previous is not None else new_series
                if feature_series:
                    df = pd.concat(feature_series.values(), axis=1, sort=True)
                    # Leave missing quarters missing. Never forward-fill NED.
                    if self._update_data(df):
                        changed = True
                        await self.serialize()
                chunk_start = chunk_end + timedelta(minutes=15)
        if changed:
            log.info("%s: updated NED forecast features", self.region.bidding_zone_entsoe)
        return changed

    async def load(self):
        # Preserve gaps between NED series. DataStore.load() drops any row that
        # contains a NaN, which is appropriate for scalar stores but not here.
        fn = self.get_storage_file()
        if fn is not None and os.path.exists(fn):
            log.info("%s: loading persisted NED data", self.region.bidding_zone_entsoe)
            self.data = await asyncio.to_thread(pd.read_json, fn, compression="gzip")
            if pd.api.types.is_integer_dtype(self.data.index.dtype):
                self.data.index = pd.to_datetime(self.data.index, unit="ms", utc=True)
            elif isinstance(self.data.index, pd.DatetimeIndex) and self.data.index.tz is None:
                self.data.index = self.data.index.tz_localize("UTC")
            elif not isinstance(self.data.index, pd.DatetimeIndex):
                self.data.index = pd.to_datetime(self.data.index, utc=True)
            self.data.index.set_names("time", inplace=True)
            self.last_updated = datetime.fromtimestamp(os.path.getmtime(fn), tz=timezone.utc)
        return self

    async def _fetch_records(self, session: aiohttp.ClientSession, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = self.API_URL
        records: list[dict[str, Any]] = []
        first = True
        while url:
            await self._pace_request()
            async with session.get(url, params=params if first else None, allow_redirects=False) as response:
                if response.status == 429:
                    retry_after = min(int(response.headers.get("Retry-After", "10")), 60)
                    await asyncio.sleep(retry_after)
                    await self._pace_request()
                    async with session.get(url, params=params if first else None, allow_redirects=False) as retry:
                        retry.raise_for_status()
                        payload = await retry.json()
                else:
                    response.raise_for_status()
                    payload = await response.json()
            members = payload.get("hydra:member", []) if isinstance(payload, dict) else []
            records.extend(item for item in members if isinstance(item, dict))
            next_link = payload.get("hydra:view", {}).get("hydra:next") if isinstance(payload, dict) else None
            if not next_link:
                break
            url = urljoin(self.API_URL, next_link)
            if urlparse(url).netloc != urlparse(self.API_URL).netloc or urlparse(url).scheme != "https":
                raise ValueError("Unexpected NED pagination origin")
            first = False
        return records

    @staticmethod
    async def _pace_request():
        global _last_request_at
        async with _request_lock:
            delay = 1.6 - (time.monotonic() - _last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            _last_request_at = time.monotonic()

    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=1)

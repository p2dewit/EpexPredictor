"""Weekly out-of-fold base predictions for training the second model stage.

This isolates PRICE labels chronologically. Cached source features may have
been revised later; it is not a reconstruction of historic forecast vintages.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .pricepredictor import PricePredictor

log = logging.getLogger(__name__)
_fit_slots = asyncio.Semaphore(2)
_cache_locks: dict[str, asyncio.Lock] = {}


class ChronologicalPredictionCache:
    MIN_HISTORY_DAYS = 60
    BLOCK_DAYS = 7
    MAX_HISTORY_DAYS = 180
    CACHE_VERSION = "stacking_oof_v1"
    COLUMNS = ("price", "train_end_ms")

    def __init__(self, base: PricePredictor):
        self.base = base
        self.lock = asyncio.Lock()
        self.signature = None
        self.data = self._empty()

    @classmethod
    def _empty(cls):
        return pd.DataFrame(columns=cls.COLUMNS, dtype=float,
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))

    def _cache_identity(self, frame):
        region = self.base.region
        settings = {
            "version": self.CACHE_VERSION,
            "features": list(frame.drop(columns=["price"]).columns),
            "region": region.bidding_zone_entsoe,
            "latitudes": region.latitudes,
            "longitudes": region.longitudes,
            "ned": region.use_ned_forecasts,
            "entsoe": region.use_entsoe_load_forecast,
            "gas": region.use_de_nat_gas_price,
            "min_history": self.MIN_HISTORY_DAYS,
            "block_days": self.BLOCK_DAYS,
            "max_history": self.MAX_HISTORY_DAYS,
        }
        signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.base.weatherstore.storage_dir
        path = None if directory is None else Path(directory) / f"{self.CACHE_VERSION}_{region.bidding_zone_entsoe}_{signature}.json.gz"
        return signature, path

    async def _load(self, signature, path):
        if self.signature == signature and path is None:
            return
        self.signature = signature
        self.data = self._empty()
        if path is None or not path.exists():
            return
        try:
            data = await asyncio.to_thread(pd.read_json, path, compression="gzip")
            if pd.api.types.is_integer_dtype(data.index.dtype):
                data.index = pd.to_datetime(data.index, unit="ms", utc=True)
            else:
                data.index = pd.to_datetime(data.index, utc=True)
            data = data.reindex(columns=self.COLUMNS).apply(pd.to_numeric, errors="coerce")
            prediction_ms = pd.Series([ts.timestamp() * 1000 for ts in data.index], index=data.index)
            # Never accept cache entries whose model could have seen the target.
            valid = np.isfinite(data["price"]) & np.isfinite(data["train_end_ms"]) & (data["train_end_ms"] < prediction_ms)
            data = data.loc[valid]
            self.data = data.loc[~data.index.duplicated(keep="first")].sort_index()
            self.data.index.name = "time"
        except (OSError, EOFError, ValueError, TypeError) as exc:
            log.warning("%s: ignoring unreadable chronological cache: %s", self.base.region.bidding_zone_entsoe, exc)

    async def _save(self, path):
        if path is None:
            return
        # Write an entire fold atomically; interruption leaves the old cache valid.
        snapshot = self.data.copy()

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            os.close(fd)
            try:
                snapshot.to_json(temporary, compression="gzip", double_precision=10)
                os.chmod(temporary, 0o644)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

        await asyncio.to_thread(write)

    async def get_data(self, start, end):
        if self.base.cross_predictors:
            raise ValueError("Chronological predictions require a first-stage base model")
        async with self.lock:
            frame = await self.base.prepare_dataframe(start, end)
            if frame is None or frame.empty:
                return self._empty()[["price"]]
            signature, path = self._cache_identity(frame)
            # Live and evaluation predictors may share a disk cache. Serialize
            # their updates, and reload under this lock to avoid losing folds.
            key = str(path) if path is not None else f"memory:{id(self)}"
            async with _cache_locks.setdefault(key, asyncio.Lock()):
                return await self._build_data(frame, start, end, signature, path)

    async def _build_data(self, frame, start, end, signature, path):
        await self._load(signature, path)
        labels = frame["price"].dropna()
        if labels.empty:
            return self.data.loc[start:end, ["price"]].copy()

        # Fixed Monday 00:00 UTC boundaries make folds reusable across updates.
        first = pd.Timestamp(start).tz_convert("UTC").normalize()
        block_start = first - pd.Timedelta(days=first.weekday())
        target_end = min(pd.Timestamp(end).tz_convert("UTC"), labels.index.max())
        while block_start <= target_end:
            block_stop = block_start + pd.Timedelta(days=self.BLOCK_DAYS)
            targets = labels.index[(labels.index >= block_start) & (labels.index < block_stop)]
            needed = targets.difference(self.data.index)
            if not needed.empty:
                learn_start = max(frame.index.min(), block_start - pd.Timedelta(days=self.MAX_HISTORY_DAYS))
                # Exclusive boundary: no label in the predicted week is used.
                learn = frame.loc[(frame.index >= learn_start) & (frame.index < block_start)]
                usable = learn
                if self.base.region.use_ned_forecasts:
                    usable = usable.drop(columns=list(self.base.NED_MODEL_COLUMNS))
                usable = usable.dropna()
                enough_history = (
                    len(usable) >= 7 * 96
                    and block_start - usable.index.min() >= pd.Timedelta(days=self.MIN_HISTORY_DAYS)
                ) if not usable.empty else False
                if enough_history:
                    # Fresh boosters use frame slices, never the live booster.
                    fold_model = type(self.base)(self.base.region)
                    async with _fit_slots:
                        await fold_model._train_dataframe(learn, num_threads=1)
                        predicted = await asyncio.to_thread(
                            fold_model._predict_dataframe, frame.loc[needed].drop(columns=["price"]))
                    if not np.isfinite(predicted["price"]).all():
                        raise ValueError("Non-finite chronological base predictions")
                    predicted["train_end_ms"] = (block_start - pd.Timedelta(minutes=15)).timestamp() * 1000
                    self.data = self.data.combine_first(predicted).sort_index()
                    await self._save(path)
                    log.info("%s: chronological block %s to %s: %d quarters; labels strictly before %s",
                             self.base.region.bidding_zone_entsoe, block_start.isoformat(), block_stop.isoformat(),
                             len(predicted), block_start.isoformat())
            block_start = block_stop

        # The initial 60 days have no out-of-fold features. They remain NaN
        # when joined to stage-two training, so they cannot leak via a fallback.
        return self.data.loc[start:end, ["price"]].copy()

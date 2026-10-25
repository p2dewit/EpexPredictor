#!/usr/bin/python3

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, cast

import numpy as np
import pandas as pd
import lightgbm as lgb

from .auxdatastore import AuxDataStore
from .priceregion import PriceRegion
from .pricestore import PriceStore
from .weatherstore import WeatherStore
from .entsoedatastore import EntsoeDataStore
from .gaspricestore import GasPriceStore
from .neddatastore import NedDataStore
from .chronological import ChronologicalPredictionCache

log = logging.getLogger(__name__)


class PricePredictor:
    NED_MODEL_COLUMNS = (*NedDataStore.FEATURE_COLUMNS, "ned_residual_load_mw")
    region: PriceRegion
    weatherstore: WeatherStore
    pricestore: PriceStore
    entsoestore: EntsoeDataStore
    auxstore: AuxDataStore
    gasstore: GasPriceStore
    nedstore: NedDataStore

    traindata: pd.DataFrame | None = None

    predictor: lgb.Booster | None = None

    # 2-stage forecasting: stage-1 predictors whose price forecasts are added as extra features
    cross_predictors: list["PricePredictor"]

    def __init__(self, region: PriceRegion, storage_dir: str | None = None, cross_predictors: list["PricePredictor"] | None = None):
        self.region = region
        self.weatherstore = WeatherStore(region, storage_dir)
        self.pricestore = PriceStore(region, storage_dir)
        self.auxstore = AuxDataStore(region, storage_dir)
        self.entsoestore = EntsoeDataStore(region, storage_dir)
        self.gasstore = GasPriceStore(region, storage_dir)
        self.nedstore = NedDataStore(region, storage_dir)
        self.predictor = None
        self.fallback_predictor: lgb.Booster | None = None
        self.chronological_predictions = ChronologicalPredictionCache(self)

        # When set, the price forecasts of these (stage-1) predictors are added as extra
        # input features. Each cross predictor is responsible for delivering its own
        # forecast (and for any horizon handling of its data).
        self.cross_predictors = cross_predictors or []

    async def load_from_persistence(self):
        await asyncio.gather(
            self.weatherstore.load(),
            self.pricestore.load(),
            self.auxstore.load(),
            self.entsoestore.load(),
            self.gasstore.load(),
            self.nedstore.load()
        )
        return self
    
    def last_data_update(self) -> datetime:
        return max(self.weatherstore.last_updated, self.pricestore.last_updated, self.entsoestore.last_updated, self.gasstore.last_updated, self.nedstore.last_updated)

    def use_datastores_from(self, other: "PricePredictor"):
        assert self.region.bidding_zone_entsoe == other.region.bidding_zone_entsoe
        self.weatherstore = other.weatherstore
        self.pricestore = other.pricestore
        self.auxstore = other.auxstore
        self.entsoestore = other.entsoestore
        self.gasstore = other.gasstore
        self.nedstore = other.nedstore

    def is_trained(self) -> bool:
        return self.predictor is not None or self.fallback_predictor is not None


    async def train(self, start: datetime, end: datetime):
        # Stage two must learn from predictions made without its target prices.
        frame = await self.prepare_dataframe(start, end, for_training=True)
        if frame is None:
            return
        await self._train_dataframe(frame)

    async def _train_dataframe(self, frame: pd.DataFrame, num_threads: int | None = None):
        """Fit only the supplied rows; used by isolated chronological base models."""
        params_lgb = {
            "force_col_wise": True,
            "verbosity": -1,
        }
        if num_threads is not None:
            params_lgb["num_threads"] = num_threads
        fallback_model = None
        if self.region.use_ned_forecasts:
            # Train on every usable weather/calendar/price row, including dates
            # for which NED is absent. Do not let NED gaps remove fallback data.
            fallback_data = frame.drop(columns=list(self.NED_MODEL_COLUMNS)).dropna()
            if len(fallback_data) >= 2:
                fallback_model = await asyncio.to_thread(
                    lgb.train, params=params_lgb,
                    train_set=lgb.Dataset(fallback_data.drop(columns=["price"]), label=fallback_data["price"]),
                )

        training_data = frame.dropna()
        model = None
        if len(training_data) >= 2:
            model = await asyncio.to_thread(
                lgb.train, params=params_lgb,
                train_set=lgb.Dataset(training_data.drop(columns=["price"]), label=training_data["price"]),
            )
        if model is None and fallback_model is None:
            raise ValueError(f"{self.region.bidding_zone_entsoe}: no usable price training rows")
        # Publish both models together; never expose a half-retrained pair.
        self.traindata = training_data
        self.predictor = model
        self.fallback_predictor = fallback_model



    async def predict(self, start: datetime, end: datetime, fill_known=True) -> pd.DataFrame:
        assert self.is_trained()

        df = await self.prepare_dataframe(start, end)
        assert df is not None

        prices_known = df["price"]

        params = df.drop(columns=["price"])
        resultdf = self._predict_dataframe(params)

        if fill_known:
            resultdf.update(prices_known)

        return resultdf

    def _predict_dataframe(self, params: pd.DataFrame) -> pd.DataFrame:
        """Predict from supplied features without fetching data or using prices."""
        assert self.is_trained()
        resultdf = pd.DataFrame(index=params.index)
        resultdf["price"] = np.nan
        if params.empty:
            return resultdf
        if self.region.use_ned_forecasts:
            ned = params.reindex(columns=self.NED_MODEL_COLUMNS)
            # All four NED inputs must be present. Gaps anywhere in the week,
            # including different per-series horizons, use the model without NED.
            use_ned = pd.Series(np.isfinite(ned.to_numpy(dtype=float)).all(axis=1), index=params.index)
            if self.predictor is None:
                use_ned[:] = False
            fallback_rows = ~use_ned
            if fallback_rows.any():
                if self.fallback_predictor is None:
                    raise ValueError(f"{self.region.bidding_zone_entsoe}: missing NED inputs and no fallback model")
                fallback_params = params.loc[fallback_rows].reindex(columns=self.fallback_predictor.feature_name())
                resultdf.loc[fallback_rows, "price"] = np.asarray(self.fallback_predictor.predict(fallback_params))
                log.info("%s: using model without NED for %d of %d quarters",
                         self.region.bidding_zone_entsoe, fallback_rows.sum(), len(params))
            if use_ned.any():
                resultdf.loc[use_ned, "price"] = np.asarray(self.predictor.predict(
                    params.loc[use_ned].reindex(columns=self.predictor.feature_name())))
        else:
            resultdf["price"] = np.asarray(self.predictor.predict(params.reindex(columns=self.predictor.feature_name())))

        return resultdf

    def to_price_dict(self, df : pd.DataFrame) -> Dict[datetime, float]:
        result = {}
        for time, row in df.iterrows():
            ts = cast(pd.Timestamp, time).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result



    async def get_cross_features(self, start: datetime, end: datetime, for_training: bool = False) -> pd.DataFrame | None:
        """
        Build the extra input features from the stage-1 (cross) predictors.

        For each cross predictor we produce a single column named after its region:
        the predicted price for each timestamp. Each cross predictor is responsible for
        delivering its own forecast; any horizon handling of its data is its concern.
        Training uses weekly chronological predictions, serving uses the live bases.
        """
        if len(self.cross_predictors) == 0:
            return None

        async def predict_cross(cp):
            col = f"cross_price_{cp.region.bidding_zone_entsoe}"
            if for_training:
                pred = await cp.chronological_predictions.get_data(start, end)
            else:
                pred = await cp.predict(start, end, fill_known=False)
            return pred.rename(columns={"price": col})

        frames = await asyncio.gather(*(predict_cross(cp) for cp in self.cross_predictors))

        return pd.concat(frames, axis=1, sort=True)

    async def prepare_dataframe(self, actual_start: datetime, end: datetime, for_training: bool = False) -> pd.DataFrame | None:
        # gas prices are usually not available for today or the last few days. If forecast range is in the future, we might have nothing to ffill. Ensure we do
        start = min(datetime.now(timezone.utc) - timedelta(days=14), actual_start)

        weather, prices, auxdata = await asyncio.gather(
            self.weatherstore.get_data(start, end),
            self.pricestore.get_data(start, end),
            self.auxstore.get_data(start, end)
        )

        df = pd.concat([weather, auxdata], axis=1, sort=True)

        
        if self.region.use_entsoe_load_forecast:
            entsoedata = await self.entsoestore.get_data(start, end)
            if len(entsoedata) > 0:
                df = pd.concat([df, entsoedata], axis=1, sort=True)

        if self.region.use_ned_forecasts:
            neddata = await self.nedstore.get_data(start, end)
            # Keep a stable schema even during an outage or the first v2 backfill.
            # The fallback model supplies predictions when these fields are NaN.
            neddata = neddata.reindex(columns=NedDataStore.FEATURE_COLUMNS).copy()
            # Missing inputs propagate to NaN, selecting the existing model
            # without NED. Negative residual demand is valid and stays negative.
            neddata["ned_residual_load_mw"] = (
                neddata["ned_load_mw"] - neddata["ned_solar_mw"]
                - neddata["ned_wind_onshore_mw"] - neddata["ned_wind_offshore_mw"]
            )
            df = pd.concat([df, neddata], axis=1, sort=True)

        if self.region.use_de_nat_gas_price:
            gasprices = await self.gasstore.get_data(start, end)
            gasprices = gasprices.reindex(weather.index).ffill()
            df = pd.concat([df, gasprices], axis=1, sort=True)

        if len(self.cross_predictors) > 0:
            cross_features = await self.get_cross_features(actual_start, end, for_training=for_training)
            if cross_features is not None:
                df = pd.concat([df, cross_features], axis=1, sort=True)

        df = pd.concat([df, prices], axis=1, sort=True)
        df = df.loc[actual_start:end]
        return df

    async def refresh_forecasts(self, start : datetime, end: datetime):
        """
            Will re-fetch everything starting from yesterday during next training
            Not sure when past data becomes "stable", so better be sure and fetch a bit more
        """
        await self.weatherstore.refresh_range(start, end)
        await self.entsoestore.refresh_range(start, end)
        await self.nedstore.refresh_range(start, end)


    def cleanup(self):
        """
        Delete data older than 5 years
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=1825)
        self.weatherstore.drop_before(cutoff)
        self.pricestore.drop_before(cutoff)
        self.auxstore.drop_before(cutoff)
        self.entsoestore.drop_before(cutoff)
        self.nedstore.drop_before(cutoff)
        self.gasstore.drop_before(cutoff)





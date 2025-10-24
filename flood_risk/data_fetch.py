"""Utilities for loading hydrological datasets.

This module provides helpers to download and cache time series data from
USGS and NOAA APIs for use in flood risk modeling workflows.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class USGSConfig:
    """Configuration for USGS streamflow API requests."""

    site: str
    start_date: dt.date
    end_date: dt.date
    parameter_code: str = "00060"  # discharge


@dataclass
class NOAAConfig:
    """Configuration for NOAA precipitation data requests.

    Parameters
    ----------
    station:
        Station identifier (e.g., ``GHCND:USW00014734``) as used by the Climate Data Online API.
    start_date / end_date:
        Inclusive time range for the query.
    dataset:
        NOAA dataset identifier. If left as ``PRECIP_HLY`` but a GHCND station is supplied,
        the fetcher will automatically switch to the ``GHCND`` dataset for convenience.
    datatype:
        Optional NOAA datatype identifier (e.g., ``PRCP``). When omitted, a sensible default is
        chosen based on the dataset.
    limit:
        Page size for API pagination. Increase when querying long time ranges.
    units:
        Desired measurement units as accepted by the API (``metric`` or ``standard``).
    """

    station: str
    start_date: dt.date
    end_date: dt.date
    dataset: str = "PRECIP_HLY"
    datatype: Optional[str] = None
    limit: int = 1000
    units: str = "metric"


def fetch_usgs_streamflow(config: USGSConfig) -> pd.DataFrame:
    """Download streamflow data from USGS Water Services.

    Parameters
    ----------
    config:
        Settings controlling the API query.

    Returns
    -------
    DataFrame with datetime index and discharge observations in cubic feet per second.
    """

    base_url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": config.site,
        "startDT": config.start_date.isoformat(),
        "endDT": config.end_date.isoformat(),
        "parameterCd": config.parameter_code,
    }
    logger.debug("Requesting USGS data: %s", params)
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    series = payload["value"]["timeSeries"]
    if not series:
        raise ValueError("No USGS time series found for given configuration")

    points = series[0]["values"][0]["value"]
    records = [
        {
            "datetime": pd.to_datetime(point["dateTime"]),
            "discharge_cfs": float(point["value"]),
        }
        for point in points
    ]
    df = pd.DataFrame.from_records(records).set_index("datetime").sort_index()
    if df.index.tz is not None:
        # Normalize to UTC and drop timezone for downstream joins
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    logger.info("Fetched %d USGS observations", len(df))
    return df


def fetch_noaa_precipitation(config: NOAAConfig, token: Optional[str] = None) -> pd.DataFrame:
    """Download daily precipitation data from NOAA's Climate Data Online API."""

    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    station_prefix = config.station.split(":", 1)[0].upper() if ":" in config.station else ""
    dataset = config.dataset.upper()

    if dataset == "PRECIP_HLY" and station_prefix == "GHCND":
        logger.info(
            "Detected GHCND station id '%s' – switching dataset to GHCND for compatibility",
            config.station,
        )
        dataset = "GHCND"

    params = {
        "datasetid": dataset,
        "stationid": config.station,
        "startdate": config.start_date.isoformat(),
        "enddate": config.end_date.isoformat(),
        "limit": config.limit,
        "units": config.units,
    }

    datatype = config.datatype
    if datatype is None:
        if dataset == "GHCND":
            datatype = "PRCP"
        elif dataset == "PRECIP_HLY":
            datatype = "HPCP"
    if datatype:
        params["datatypeid"] = datatype

    headers = {"token": token} if token else {}

    logger.debug("Requesting NOAA data: %s", params)

    results: list[dict] = []
    offset = 1
    while True:
        params["offset"] = offset
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        chunk = payload.get("results", [])
        if not chunk:
            break
        results.extend(chunk)

        metadata = payload.get("metadata", {}).get("resultset", {})
        count = metadata.get("count")
        if count is None or offset + config.limit > count:
            break
        offset += config.limit

    if not results:
        raise ValueError("No NOAA precipitation data returned for the requested configuration")

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df.dropna(subset=["value"])
        .pivot_table(index="date", values="value", aggfunc="sum")
        .rename(columns={"value": "precip_mm"})
        .sort_index()
    )

    if dataset in {"GHCND", "PRECIP_HLY"}:
        df["precip_mm"] = df["precip_mm"] / 10.0

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    logger.info("Fetched %d NOAA precipitation observations", len(df))
    return df



def load_local_csv(path: str, datetime_col: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load time series data from a CSV file."""

    df = pd.read_csv(path, **read_csv_kwargs)
    if datetime_col not in df.columns:
        raise ValueError(f"datetime column '{datetime_col}' not in CSV")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    return df


def merge_hydro_meteorological(
    streamflow: pd.DataFrame, precip: pd.DataFrame, freq: str = "D"
) -> pd.DataFrame:
    """Merge streamflow and precipitation data at the desired temporal resolution."""

    discharge = streamflow.resample(freq).mean().rename(columns={"discharge_cfs": "discharge_cfs"})
    rainfall = precip.resample(freq).sum().rename(columns={"precip_mm": "precip_mm"})
    merged = discharge.join(rainfall, how="inner").dropna()
    return merged


def create_supervised_sequences(
    data: pd.DataFrame, target_col: str, lookback: int, horizon: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Transform a time series DataFrame into supervised sequences for LSTM models."""

    sequences = []
    targets = []
    for idx in range(lookback, len(data) - horizon + 1):
        window = data.iloc[idx - lookback : idx]
        sequences.append(window.values)
        targets.append(data.iloc[idx : idx + horizon][target_col].values)

    x = pd.Series(sequences, name="inputs")
    y = pd.Series(targets, name="targets")
    return x, y

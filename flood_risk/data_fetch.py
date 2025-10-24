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
    """Configuration for NOAA precipitation data requests."""

    station: str
    start_date: dt.date
    end_date: dt.date
    dataset: str = "PRECIP_HLY"


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
    logger.info("Fetched %d USGS observations", len(df))
    return df


def fetch_noaa_precipitation(config: NOAAConfig, token: Optional[str] = None) -> pd.DataFrame:
    """Download hourly precipitation data from NOAA's Climate Data Online API.

    Parameters
    ----------
    config:
        Settings controlling the API query.
    token:
        NOAA token string for authenticated access.

    Returns
    -------
    DataFrame with datetime index and precipitation in millimeters.
    """

    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token} if token else {}
    params = {
        "datasetid": config.dataset,
        "stationid": config.station,
        "startdate": config.start_date.isoformat(),
        "enddate": config.end_date.isoformat(),
        "limit": 1000,
        "units": "metric",
    }

    logger.debug("Requesting NOAA data: %s", params)
    response = requests.get(base_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        raise ValueError("No NOAA precipitation data returned")

    df = (
        pd.DataFrame(results)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .pivot_table(index="date", values="value", aggfunc="sum")
        .rename(columns={"value": "precip_mm"})
        .sort_index()
    )
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

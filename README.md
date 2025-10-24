# Flood Risk Estimation with LSTM Models

This project provides an end-to-end workflow for forecasting river discharge and estimating flood risk using LSTM models. It combines USGS streamflow and NOAA precipitation data, visualizes the training process, and includes an AWS Lambda alerting function that publishes notifications when risk exceeds a configurable threshold.

## Project Structure

```
climate_risk/
├── flood_risk/
│   ├── __init__.py
│   ├── alerting.py
│   ├── data_fetch.py
│   ├── model.py
│   ├── preprocess.py
│   └── training.py
├── aws_lambda/
│   └── handler.py
├── notebooks/
│   └── flood_risk_lstm.ipynb
├── requirements.txt
└── README.md
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch Jupyter**

   ```bash
   jupyter notebook notebooks/flood_risk_lstm.ipynb
   ```

3. **Configure data access**
   * Request a NOAA API token and set it as the `NOAA_TOKEN` environment variable when running the notebook.
   * Identify the USGS site and NOAA station IDs for the region of interest.
   * GHCND station identifiers are detected automatically; the loader switches to the `GHCND/PRCP` daily dataset unless you override `NOAAConfig`.

## Notebook Workflow

The `flood_risk_lstm.ipynb` notebook demonstrates the complete workflow:

1. Fetch and merge USGS streamflow with NOAA precipitation data.
2. Engineer supervised learning sequences with configurable lookback and forecast horizon.
3. Train an LSTM model in PyTorch and visualize loss curves and predictions.
4. Calculate exceedance-based flood risk scores and render daily risk plots.
5. Publish alerts to AWS when forecasted discharge surpasses the risk threshold.

The notebook uses the reusable modules under `flood_risk/` for data loading, preprocessing, modeling, and training. Synthetic data generation utilities are included to allow offline experimentation when live APIs are unavailable.

## AWS Lambda Alerting

Deploy the AWS Lambda handler found in `aws_lambda/handler.py` to enable automated alerts. The function expects events with the following structure:

```json
{
  "predictions": [123.4, 150.2, 210.8],
  "threshold": 180.0,
  "region": "Lower Mississippi",
  "topic_arn": "arn:aws:sns:us-east-1:123456789012:flood-alerts"
}
```

When the peak predicted discharge exceeds the threshold, the Lambda publishes a message to the supplied SNS topic.

## Extending the Project

* Integrate additional meteorological covariates (soil moisture, snowpack) to improve forecasts.
* Use AWS Step Functions to orchestrate daily model runs and Lambda alerts.
* Configure automated retraining pipelines leveraging Amazon SageMaker or AWS Batch.

## License

This project is provided as-is for demonstration purposes.

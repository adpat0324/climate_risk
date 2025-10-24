"""AWS Lambda handler for flood risk alerts."""
from __future__ import annotations

import logging
from typing import Any, Dict

from flood_risk.alerting import AlertEvent, SNSAlertSink, evaluate_risk

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:  # pragma: no cover
    """Entry point for AWS Lambda deployments."""

    logger.info("Event received: %s", event)
    predictions = event.get("predictions", [])
    threshold = float(event.get("threshold", 1.0))
    region = event.get("region", "unknown")
    topic_arn = event["topic_arn"]

    risk_score = evaluate_risk(predictions, threshold)
    alert = AlertEvent(region=region, risk_score=risk_score, threshold=1.0)

    if risk_score >= 1.0:
        sink = SNSAlertSink(topic_arn=topic_arn)
        sink.send(subject=f"Flood risk alert for {region}", message=alert.to_message())
        return {"status": "ALERT_TRIGGERED", "risk_score": risk_score}

    logger.info("Risk below threshold; no alert issued.")
    return {"status": "NO_ALERT", "risk_score": risk_score}

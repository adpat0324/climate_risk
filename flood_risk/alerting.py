"""Alerting utilities including AWS Lambda handler."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, Protocol

import boto3

logger = logging.getLogger(__name__)


class AlertSink(Protocol):
    """Protocol for alert notification sinks."""

    def send(self, subject: str, message: str) -> None:
        """Send an alert notification."""


@dataclass
class SNSAlertSink:
    topic_arn: str

    def __post_init__(self) -> None:
        self.client = boto3.client("sns")

    def send(self, subject: str, message: str) -> None:
        logger.info("Publishing alert to %s", self.topic_arn)
        self.client.publish(TopicArn=self.topic_arn, Subject=subject, Message=message)


@dataclass
class AlertEvent:
    region: str
    risk_score: float
    threshold: float

    def to_message(self) -> str:
        status = "CRITICAL" if self.risk_score >= self.threshold else "WARNING"
        return (
            f"Flood risk status for {self.region}: {status}.\n"
            f"Risk score: {self.risk_score:.2f}, Threshold: {self.threshold:.2f}"
        )


def evaluate_risk(predictions: Iterable[float], threshold: float) -> float:
    """Aggregate predictions to a risk score."""

    scores = list(predictions)
    if not scores:
        raise ValueError("No predictions to evaluate")
    peak = max(scores)
    risk_score = peak / threshold
    logger.debug("Peak predicted discharge %.2f vs threshold %.2f -> risk %.2f", peak, threshold, risk_score)
    return risk_score


def lambda_handler(event, context):  # pragma: no cover - AWS entry point
    """AWS Lambda entry point triggered by EventBridge or Step Functions."""

    logger.info("Received event: %s", json.dumps(event))
    threshold = float(event["threshold"])
    predictions = event["predictions"]
    region = event.get("region", "unknown")
    topic_arn = event["topic_arn"]

    sink = SNSAlertSink(topic_arn=topic_arn)
    risk_score = evaluate_risk(predictions, threshold)
    alert_event = AlertEvent(region=region, risk_score=risk_score, threshold=1.0)

    if risk_score >= 1.0:
        sink.send(subject=f"Flood risk alert for {region}", message=alert_event.to_message())
        return {"status": "ALERT_TRIGGERED", "risk_score": risk_score}

    logger.info("Risk below threshold; no alert.")
    return {"status": "NO_ALERT", "risk_score": risk_score}

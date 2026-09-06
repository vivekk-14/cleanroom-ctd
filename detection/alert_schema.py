"""
Standard alert schema.

Every alert emitted by the system has this shape, whatever produced it. The
dashboard reads only these keys, so Student B can build against the schema
before the model exists.

Schema (v1)
-----------
    {
      "schema_version": 1,
      "timestamp":       "2026-09-05T18:42:31+00:00",   observation time (UTC)
      "flow_id":         "F000123",                     replay sequence id
      "source_flow_id":  "172.16.0.1-192.168.10.50-...", dataset's own Flow ID
      "src_ip":          "172.16.0.1",
      "dst_ip":          "192.168.10.50",
      "src_port":        59116,
      "dst_port":        80,
      "protocol":        "TCP",
      "threat_class":    "DDoS",
      "confidence":      0.96,
      "severity":        "CRITICAL",
      "base_severity":   "HIGH",
      "escalated":       true,
      "severity_reason": "...",
      "is_alertable":    true,
      "evidence":        {"Fwd Packet Length Mean": 7.0, ...},
      "evidence_detail": [ {feature, value, display, why, comparison,
                            importance}, ... ],
      "evidence_method": "...",
      "aggregate_context": "..." | null,
      "class_probabilities": {"BENIGN": 0.01, "DDoS": 0.96, ...},
      "capture_timestamp": "7/7/2017 3:30",
      "ground_truth":    "DDoS" | null,
      "correct":         true | false | null,
      "model_ready":     true,
      "observation_mode": "PASSIVE_REPLAY"
    }

Design notes
------------
timestamp vs capture_timestamp
    `timestamp` is when this system observed the flow, generated at replay
    time in UTC with an explicit offset. `capture_timestamp` is the original
    string from the dataset, kept verbatim for provenance and never parsed:
    CIC-IDS2017 records '7/7/2017 3:30' in 12-hour form with no AM/PM marker,
    so parsing it would silently invent a time.

ground_truth / correct
    Present only because this is a replay of a LABELLED dataset, which lets the
    dashboard show live accuracy. A real deployment observing production
    traffic would have no ground truth, so both fields would be null. They are
    marked clearly as demo-only rather than presented as a detection result.

JSON Lines transport
    Alerts are appended one JSON object per line to runtime/alerts.jsonl. This
    is deliberately the simplest thing that decouples the two students'
    processes: the replay engine appends, the dashboard tails. No socket, no
    broker, no shared memory, and a partially written final line is simply
    skipped by the reader.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_config import protocol_name  # noqa: E402

SCHEMA_VERSION = 2

# Valid values for the `reliability` field, from detection/ood.py. Duplicated as
# a literal tuple rather than imported so that alert_schema stays importable
# without the OOD module, and so a reader can see the full set here.
RELIABILITY_VALUES: tuple[str, ...] = (
    "RELIABLE", "DEGRADED", "UNRELIABLE", "UNKNOWN",
)

# The threat class reported when input lies outside the training distribution.
OOD_CLASS = "UNKNOWN_OOD"

# Placeholder used when a dataset variant carries no identity columns. The
# MachineLearningCVE variant of CIC-IDS2017 (79 columns) has no IP fields; the
# TrafficLabelling variant used here (85 columns) does, so real addresses are
# normally present. These constants exist so an alert is never silently blank,
# and so a placeholder is obviously a placeholder.
UNKNOWN_IP = "0.0.0.0"
UNKNOWN_PORT = 0
UNKNOWN_STR = "UNKNOWN"


@dataclass
class Alert:
    """One alert. Use `Alert.build()` rather than constructing directly."""

    # --- identity ---
    timestamp: str
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    # --- classification ---
    # `threat_class` is the REPORTED verdict. It equals `model_prediction` when
    # the flow is in distribution, and becomes 'UNKNOWN_OOD' when it is not.
    threat_class: str
    confidence: float
    severity: str

    # --- severity derivation ---
    base_severity: str
    escalated: bool
    severity_reason: str
    is_alertable: bool

    # --- evidence ---
    evidence: dict[str, float] = field(default_factory=dict)
    evidence_detail: list[dict] = field(default_factory=list)
    evidence_method: str = ""
    aggregate_context: str | None = None

    # --- classifier output, deliberately separate from the verdict ---------
    # These two fields exist because conflating "the model is confident" with
    # "the answer is reliable" is the failure this schema version was written to
    # prevent. On the synthetic OOD fixture the classifier said BENIGN for all
    # 20 rows at 0.913 mean confidence; `confidence` records that certainty,
    # while `in_distribution=False` records that it means nothing here.
    model_prediction: str | None = None
    model_confidence: float = 0.0
    confidence_means: str = ""

    # --- distribution assessment ------------------------------------------
    # in_distribution is None when no reference was available, NOT False:
    # "unchecked" and "checked and failed" are different states.
    in_distribution: bool | None = None
    reliability: str = "UNKNOWN"
    ood_score: float = 0.0
    ood_features: list[str] = field(default_factory=list)
    ood_violations: list[dict] = field(default_factory=list)
    ood_reason: str = ""
    novelty_score: float | None = None
    novelty_flagged: bool | None = None

    # --- triage -----------------------------------------------------------
    # A second queue, independent of is_alertable. A benign-looking flow with
    # one out-of-range feature is not an actionable alert, but suppressing it
    # entirely is how the fixture's PortScan and Botnet rows would vanish.
    needs_review: bool = False

    # --- model output ---
    class_probabilities: dict[str, float] = field(default_factory=dict)
    model_ready: bool = True
    contract_satisfied: bool = True
    profile: str | None = None

    # --- provenance ---
    schema_version: int = SCHEMA_VERSION
    source_flow_id: str | None = None
    capture_timestamp: str | None = None
    observation_mode: str = "PASSIVE_REPLAY"

    # --- demo-only, because the replayed dataset is labelled ---
    ground_truth: str | None = None
    correct: bool | None = None

    @staticmethod
    def build(
        prediction: dict,
        flow_id: str,
        identity: dict | None = None,
        ground_truth: str | None = None,
        observation_mode: str = "PASSIVE_REPLAY",
    ) -> "Alert":
        """Assemble an Alert from a predict_threat() result plus flow identity.

        Args:
            prediction: the dict returned by model.predict.predict_threat().
            flow_id: replay-assigned identifier, e.g. 'F000123'.
            identity: optional raw identity fields from the flow record
                (Source IP, Destination IP, Source Port, Destination Port,
                Protocol, Flow ID, capture_timestamp).
            ground_truth: true label, when replaying labelled data.
            observation_mode: how the flow was observed.

        Every field is defensively coerced. A malformed row must not stop a
        live replay, so bad values become placeholders rather than exceptions.
        """
        identity = identity or {}
        threat = str(prediction.get("threat", UNKNOWN_STR))
        model_prediction = prediction.get("model_prediction")

        # Correctness is judged against what the CLASSIFIER said, not against
        # the reported verdict. An OOD flow reports UNKNOWN_OOD, which would
        # never equal a dataset label, so scoring the verdict would make every
        # OOD row count as a miss and understate the classifier's accuracy.
        judged_against = (str(model_prediction) if model_prediction is not None
                          else threat)

        return Alert(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            flow_id=flow_id,
            src_ip=_as_ip(identity.get("Source IP")),
            dst_ip=_as_ip(identity.get("Destination IP")),
            src_port=_as_port(identity.get("Source Port")),
            dst_port=_as_port(identity.get("Destination Port")),
            protocol=protocol_name(identity.get("Protocol")),
            threat_class=threat,
            confidence=_as_confidence(prediction.get("confidence")),
            severity=str(prediction.get("severity", UNKNOWN_STR)),
            base_severity=str(prediction.get("base_severity",
                                             prediction.get("severity",
                                                            UNKNOWN_STR))),
            escalated=bool(prediction.get("escalated", False)),
            severity_reason=str(prediction.get("severity_reason", "")),
            is_alertable=bool(prediction.get("is_alertable", False)),
            evidence=dict(prediction.get("evidence", {})),
            evidence_detail=list(prediction.get("evidence_detail", [])),
            evidence_method=str(prediction.get("evidence_method", "")),
            aggregate_context=prediction.get("aggregate_context"),
            model_prediction=(str(model_prediction)
                              if model_prediction is not None else None),
            model_confidence=_as_confidence(
                prediction.get("model_confidence",
                               prediction.get("confidence"))),
            confidence_means=str(prediction.get("confidence_means", "")),
            in_distribution=prediction.get("in_distribution"),
            reliability=str(prediction.get("reliability", "UNKNOWN")),
            ood_score=_as_confidence(prediction.get("ood_score", 0.0)),
            ood_features=list(prediction.get("ood_features", [])),
            ood_violations=list(prediction.get("ood_violations", [])),
            ood_reason=str(prediction.get("ood_reason", "")),
            novelty_score=prediction.get("novelty_score"),
            novelty_flagged=prediction.get("novelty_flagged"),
            needs_review=bool(prediction.get("needs_review", False)),
            class_probabilities=dict(prediction.get("class_probabilities", {})),
            model_ready=bool(prediction.get("model_ready", False)),
            contract_satisfied=bool(prediction.get("contract_satisfied", True)),
            profile=prediction.get("profile"),
            source_flow_id=_as_optional_str(identity.get("Flow ID")),
            capture_timestamp=_as_optional_str(identity.get("capture_timestamp")),
            observation_mode=observation_mode,
            ground_truth=ground_truth,
            correct=(None if ground_truth is None
                     else judged_against == ground_truth),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        """Serialise to a single line of JSON, newline-terminated.

        `default=str` is a safety net: a stray numpy scalar that escaped
        coercion is stringified rather than raising mid-replay.
        """
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False) + "\n"

    def summary(self) -> str:
        """One-line console rendering, used by the replay engine's log."""
        arrow = f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port}"
        verdict = (f"{self.threat_class:<11} {self.confidence:5.1%} "
                   f"{self.severity:<8}")

        # An OOD row must be visually distinct in the log: the whole point is
        # that its confidence figure does not support its verdict.
        flag = ""
        if self.reliability == "UNRELIABLE":
            said = self.model_prediction or "?"
            flag = f"  [OOD: said {said}, {len(self.ood_features)} feats out]"
        elif self.reliability == "DEGRADED":
            flag = "  [degraded]"

        mark = ""
        if self.correct is False:
            mark = f"  [MISS: actually {self.ground_truth}]"
        return (f"{self.flow_id}  {arrow:<44} {self.protocol:<5} "
                f"{verdict}{flag}{mark}")


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _as_ip(value: object) -> str:
    """Return a printable IP string, or the placeholder if unusable."""
    if value is None:
        return UNKNOWN_IP
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return UNKNOWN_IP
    return text


def _as_port(value: object) -> int:
    """Return a port as int, or 0. Tolerates floats such as 80.0 from pandas."""
    try:
        port = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return UNKNOWN_PORT
    return port if 0 <= port <= 65535 else UNKNOWN_PORT


def _as_confidence(value: object) -> float:
    """Clamp confidence into [0, 1]; NaN and junk become 0.0."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return min(max(number, 0.0), 1.0)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return None
    return text


def parse_alert_line(line: str) -> dict | None:
    """Parse one line of alerts.jsonl, returning None if it is unusable.

    The dashboard may read the file while the replay engine is mid-write, so
    the final line can be truncated. Returning None instead of raising lets the
    reader skip it and pick it up on the next refresh.
    """
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


# Keys the dashboard relies on. Used by tests to catch accidental removals.
REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version", "timestamp", "flow_id",
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "threat_class", "confidence", "severity", "is_alertable",
    "evidence", "model_ready",
    # v2: the verdict and the classifier's raw answer are separate fields, and
    # the reliability of the answer is reported alongside it.
    "model_prediction", "model_confidence", "in_distribution", "reliability",
    "ood_score", "ood_features", "needs_review",
)


def validate(record: dict) -> list[str]:
    """Return a list of schema problems; empty means valid."""
    problems = [f"missing key: {key}" for key in REQUIRED_KEYS
                if key not in record]

    for key in ("confidence", "model_confidence", "ood_score"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not 0.0 <= value <= 1.0:
            problems.append(f"{key} out of range: {value}")

    if not isinstance(record.get("evidence", {}), dict):
        problems.append("evidence is not a dict")

    if not isinstance(record.get("ood_features", []), list):
        problems.append("ood_features is not a list")

    # in_distribution must stay tri-state. Collapsing None to False would
    # conflate "not checked" with "checked and outside the distribution".
    #
    # The identity check is deliberate: `0 in (True, False, None)` is True in
    # Python because 0 == False and 1 == True, so a membership test would let
    # integers through. A record carrying in_distribution=0 from some other
    # producer must be rejected, not silently read as False.
    in_distribution = record.get("in_distribution", None)
    if not (in_distribution is None or in_distribution is True
            or in_distribution is False):
        problems.append(f"in_distribution must be True/False/None, "
                        f"got {in_distribution!r} "
                        f"({type(in_distribution).__name__})")

    reliability = record.get("reliability")
    if reliability is not None and reliability not in RELIABILITY_VALUES:
        problems.append(f"unknown reliability {reliability!r}")

    # An UNRELIABLE verdict must carry the OOD threat class, and vice versa.
    # Catching a mismatch here prevents a partially-applied change to the OOD
    # layer from silently reporting a trusted-looking class.
    if reliability == "UNRELIABLE" and record.get("threat_class") != OOD_CLASS:
        problems.append(
            f"reliability=UNRELIABLE but threat_class="
            f"{record.get('threat_class')!r}, expected {OOD_CLASS!r}")
    if record.get("threat_class") == OOD_CLASS and reliability != "UNRELIABLE":
        problems.append(
            f"threat_class={OOD_CLASS!r} but reliability={reliability!r}")

    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        problems.append(f"schema_version {version}, expected {SCHEMA_VERSION}")

    return problems


if __name__ == "__main__":
    print("=" * 76)
    print(f"ALERT SCHEMA v{SCHEMA_VERSION}")
    print("=" * 76)

    demo_prediction = {
        "threat": "DDoS",
        "confidence": 0.9612,
        "severity": "CRITICAL",
        "base_severity": "HIGH",
        "escalated": True,
        "severity_reason": ("Confidence 96% maps to CRITICAL; DDoS escalation "
                            "cannot raise it further."),
        "is_alertable": True,
        "evidence": {
            "Fwd Packet Length Mean": 7.0,
            "Flow Duration": 1876595.0,
            "Init_Win_bytes_forward": 256.0,
        },
        "evidence_detail": [
            {"feature": "Fwd Packet Length Mean", "value": 7.0,
             "display": "7 bytes", "why": "Small, uniform forward payload.",
             "comparison": "5.6x below the benign median of 39 bytes",
             "importance": 0.1148},
        ],
        "evidence_method": "Feature-based evidence.",
        "aggregate_context": "The flood exists in the aggregate.",
        "class_probabilities": {"BENIGN": 0.0288, "Botnet": 0.0,
                                "DDoS": 0.9612, "PortScan": 0.01},
        "model_ready": True,
    }
    demo_identity = {
        "Flow ID": "172.16.0.1-192.168.10.50-53687-80-6",
        "Source IP": "172.16.0.1", "Destination IP": "192.168.10.50",
        "Source Port": 53687.0, "Destination Port": 80.0, "Protocol": 6,
        "capture_timestamp": "7/7/2017 3:57",
    }

    alert = Alert.build(demo_prediction, "F000123", demo_identity,
                        ground_truth="DDoS")

    print("\nExample alert:")
    print(json.dumps(alert.to_dict(), indent=2)[:1600] + "\n  ...")

    print(f"\nConsole summary line:\n  {alert.summary()}")

    problems = validate(alert.to_dict())
    print(f"\nValidation: {'PASS' if not problems else problems}")

    print("\nCoercion of malformed input:")
    messy = Alert.build(
        {"threat": "PortScan", "confidence": 1.7, "severity": "CRITICAL"},
        "F000999",
        {"Source IP": float("nan"), "Destination Port": "not-a-port",
         "Protocol": None, "Source Port": 70000},
    )
    print(f"  confidence 1.7   -> {messy.confidence}")
    print(f"  src_ip NaN       -> {messy.src_ip!r}")
    print(f"  dst_port 'x'     -> {messy.dst_port}")
    print(f"  src_port 70000   -> {messy.src_port}  (out of range)")
    print(f"  protocol None    -> {messy.protocol!r}")
    print(f"  validation       : {validate(messy.to_dict()) or 'PASS'}")

    print("\nTruncated-line handling:")
    good = alert.to_json_line()
    print(f"  full line     -> {'parsed' if parse_alert_line(good) else 'None'}")
    print(f"  truncated     -> "
          f"{'parsed' if parse_alert_line(good[:60]) else 'None (skipped)'}")
    print(f"  empty         -> "
          f"{'parsed' if parse_alert_line('   ') else 'None (skipped)'}")
    print("=" * 76)

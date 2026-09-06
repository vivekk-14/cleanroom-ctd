"""
Explainable evidence for alerts.

For each classified flow this module selects the handful of feature values that
best justify the classification, and states in plain language why each one is
suspicious.

Scope claim
-----------
This is FEATURE-BASED evidence, not SHAP, not LIME, not counterfactual
explanation. Two honest sources are combined:

  1. The model's global feature importances (read from model/metrics.json),
     which say which features mattered across the whole training set.
  2. Per-class reference statistics MEASURED from the processed dataset, which
     let a flow's value be compared against the typical value for its class and
     for benign traffic.

What it does NOT do: attribute this individual prediction to individual
features. Random-forest global importance is not a per-prediction attribution.
The wording in every returned note reflects that limit, and the dashboard
labels the panel "feature-based evidence" rather than "explainable AI".

Why the reference statistics are hardcoded
------------------------------------------
The baselines below were computed from the processed dataset produced by
preprocessing/clean_data.py (151,947 rows: 50,000 each of BENIGN/DDoS/PortScan,
1,947 Botnet). They are stored as constants so evidence generation stays a
pure, fast, dependency-free function during replay -- it must not re-read a
35 MB CSV per flow. Regenerate them with:

    python -m detection.evidence --recompute

IMPORTANT FINDING: the problem statement's suggested evidence rule for DDoS
("very high packet rate, high SYN count") does not hold on this dataset.
Measured medians for Fwd Packets/s:

    PortScan  20,000.0     <- the high-rate class
    Botnet        57.7
    BENIGN        39.6
    DDoS           1.7     <- LOWER than benign

CIC-IDS2017's DDoS is a distributed HTTP flood: each individual flow is slow
and ordinary-looking, and the attack exists in the AGGREGATE -- 49,998 flows
from a single source IP to a single destination. A per-flow "high packet rate"
rule would therefore have flagged the wrong class. The evidence rules below use
what actually separates the classes: forward payload size, flow duration, TCP
initial window size, and inter-arrival regularity.

Similarly, 'SYN Flag Count' cannot support the suggested rule: it is a binary
0/1 indicator in this dataset (measured), not a count, and its model importance
is 0.0014. It is not used as evidence.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_PROFILE, PATHS  # noqa: E402

# ---------------------------------------------------------------------------
# Per-class reference medians, MEASURED from data/processed/flows_clean.csv
# ---------------------------------------------------------------------------
# Rows: BENIGN 50,000 | DDoS 50,000 | PortScan 50,000 | Botnet 1,947
REFERENCE_MEDIANS: dict[str, dict[str, float]] = {
    "Fwd Packets/s":               {"BENIGN": 39.6, "Botnet": 57.7, "DDoS": 1.7, "PortScan": 20000.0},
    "Flow Packets/s":              {"BENIGN": 76.1, "Botnet": 100.9, "DDoS": 2.7, "PortScan": 40000.0},
    "Fwd Packet Length Mean":      {"BENIGN": 39.0, "Botnet": 6.0, "DDoS": 7.0, "PortScan": 0.0},
    "Fwd Packet Length Max":       {"BENIGN": 42.0, "Botnet": 6.0, "DDoS": 20.0, "PortScan": 0.0},
    "Total Fwd Packets":           {"BENIGN": 2.0, "Botnet": 3.0, "DDoS": 4.0, "PortScan": 1.0},
    "Total Length of Fwd Packets": {"BENIGN": 70.0, "Botnet": 6.0, "DDoS": 26.0, "PortScan": 0.0},
    "Flow Duration":               {"BENIGN": 48699.0, "Botnet": 71053.0, "DDoS": 1876595.0, "PortScan": 50.0},
    "Fwd IAT Mean":                {"BENIGN": 4.0, "Botnet": 23062.7, "DDoS": 482246.8, "PortScan": 0.0},
    "Fwd IAT Std":                 {"BENIGN": 0.0, "Botnet": 1653.2, "DDoS": 908248.3, "PortScan": 0.0},
    "Init_Win_bytes_forward":      {"BENIGN": 122.0, "Botnet": 8192.0, "DDoS": 256.0, "PortScan": 29200.0},
    "act_data_pkt_fwd":            {"BENIGN": 1.0, "Botnet": 0.0, "DDoS": 3.0, "PortScan": 0.0},
    "min_seg_size_forward":        {"BENIGN": 20.0, "Botnet": 20.0, "DDoS": 20.0, "PortScan": 32.0},
    "Fwd Header Length":           {"BENIGN": 64.0, "Botnet": 92.0, "DDoS": 80.0, "PortScan": 40.0},
    "Destination Port":            {"BENIGN": 80.0, "Botnet": 8080.0, "DDoS": 80.0, "PortScan": 3527.0},
}

# Human-readable units, so the dashboard does not show a bare microsecond count.
FEATURE_UNITS: dict[str, str] = {
    "Flow Duration": "microseconds",
    "Fwd IAT Mean": "microseconds",
    "Fwd IAT Std": "microseconds",
    "Fwd IAT Max": "microseconds",
    "Fwd IAT Min": "microseconds",
    "Fwd Packets/s": "packets/sec",
    "Flow Packets/s": "packets/sec",
    "Flow Bytes/s": "bytes/sec",
    "Fwd Packet Length Mean": "bytes",
    "Fwd Packet Length Max": "bytes",
    "Fwd Packet Length Min": "bytes",
    "Fwd Packet Length Std": "bytes",
    "Total Length of Fwd Packets": "bytes",
    "Fwd Header Length": "bytes",
    "Init_Win_bytes_forward": "bytes",
    "min_seg_size_forward": "bytes",
    "Average Packet Size": "bytes",
}

# ---------------------------------------------------------------------------
# Per-class evidence rules
# ---------------------------------------------------------------------------
# Each rule names a feature, a direction, and the plain-language reason that
# value is characteristic of the class. Every threshold traces to the measured
# distributions printed by `--recompute`.
#
# 'direction' controls how the value is compared to the benign baseline:
#   'high'  -> notable when well above the benign median
#   'low'   -> notable when well below
#   'near'  -> notable when close to this class's own median (a signature value,
#              e.g. a specific TCP initial window size)
#   'nonstandard' -> for port numbers: notable when the port maps to no
#              well-known service, which is what makes a scan target list
#              look unlike ordinary client traffic


@dataclass(frozen=True)
class Rule:
    feature: str
    direction: str
    why: str


CLASS_RULES: dict[str, tuple[Rule, ...]] = {
    "PortScan": (
        Rule("Fwd Packet Length Mean", "low",
             "Probe packets carry little or no payload: the sender is testing "
             "whether a port answers, not exchanging data."),
        Rule("Flow Duration", "low",
             "The flow is abandoned almost immediately, as soon as the port's "
             "state is known, instead of carrying a session."),
        Rule("Fwd Packets/s", "high",
             "Forward packet rate far above normal for this network, consistent "
             "with automated sweeping rather than human-driven traffic."),
        Rule("Total Fwd Packets", "low",
             "Very few forward packets: a probe, not a conversation."),
        Rule("Init_Win_bytes_forward", "near",
             "TCP initial window size matches the scanner signature seen across "
             "port-scan traffic in training, rather than a negotiated "
             "application connection."),
        Rule("Destination Port", "nonstandard",
             "Destination port corresponds to no standard service. Scans walk "
             "port ranges rather than connecting to services in use: 999 "
             "distinct destination ports were observed against a single host "
             "in the processed dataset."),
    ),
    "DDoS": (
        Rule("Fwd Packet Length Mean", "near",
             "Forward payload is small and closely matches the size seen across "
             "this flood in training (median 7 bytes, std 1.2 over 50,000 "
             "flows): machine-generated identical requests, not varied human "
             "traffic."),
        Rule("Flow Duration", "high",
             "The flow is held open far longer than normal traffic on this "
             "network, consuming server-side connection resources."),
        Rule("Fwd IAT Std", "high",
             "High variance in forward inter-arrival time, characteristic of a "
             "connection deliberately kept alive with sporadic traffic."),
        Rule("Init_Win_bytes_forward", "near",
             "TCP initial window size matches the flood client's signature and "
             "is distinctly smaller than benign traffic on this network."),
        Rule("Total Length of Fwd Packets", "low",
             "Total forward bytes far below benign despite the long duration: "
             "high connection cost, negligible useful payload."),
    ),
    "Botnet": (
        Rule("Init_Win_bytes_forward", "near",
             "TCP initial window size matches the implant's client-stack "
             "fingerprint, repeated consistently across botnet flows."),
        Rule("Destination Port", "near",
             "Connects to the fixed high port used by this botnet's controller, "
             "consistent with hard-coded command-and-control rather than varied "
             "user browsing."),
        Rule("Fwd IAT Std", "low",
             "Low variability in forward inter-arrival time relative to the "
             "mean: regular, periodic beaconing rather than bursty human "
             "activity."),
        Rule("act_data_pkt_fwd", "low",
             "Almost no forward packets carry payload: a check-in beacon, not a "
             "data transfer."),
        Rule("Fwd Packet Length Mean", "low",
             "Very small forward payload, consistent with short status messages "
             "to a controller."),
    ),
    "BENIGN": (
        Rule("Fwd Packet Length Mean", "near",
             "Forward payload size is within the normal range for this network."),
        Rule("Flow Duration", "near",
             "Flow duration is unremarkable for observed traffic."),
        Rule("Fwd Packets/s", "near",
             "Packet rate is consistent with ordinary application traffic."),
    ),
}

# Behavioural notes that require correlation across many flows and therefore
# CANNOT be asserted from a single flow record. Surfaced in the dashboard as
# context, explicitly marked as not evaluated per-flow, so the prototype never
# implies it detected something it did not.
AGGREGATE_CONTEXT: dict[str, str] = {
    "DDoS": (
        "Per-flow features alone understate this threat. In the source capture "
        "the DDoS consists of 49,998 flows from a single source IP to a single "
        "destination -- the flood exists in the aggregate. This prototype "
        "classifies each flow independently and does not perform cross-flow "
        "correlation."
    ),
    "PortScan": (
        "Scan breadth (many destination ports per source, many hosts) is an "
        "aggregate property. Measured in the source capture: 999 distinct "
        "destination ports against one host. This prototype evaluates each flow "
        "independently."
    ),
    "Botnet": (
        "Beacon periodicity is best confirmed across a sequence of flows to the "
        "same destination. Measured in the source capture: 1,242 of 1,947 botnet "
        "flows target a single external IP. This prototype evaluates each flow "
        "independently."
    ),
}

# Ratio above/below the benign median at which a value is called notable.
NOTABLE_HIGH_RATIO = 3.0
NOTABLE_LOW_RATIO = 0.34
# Tolerance for 'near' matches against a class's own median.
NEAR_TOLERANCE = 0.35

# Keyed by profile: importances are fitted per feature set.
_IMPORTANCE_CACHE: dict[str, dict[str, float]] = {}


def _load_importances(profile: str | None = None) -> dict[str, float]:
    """Read global feature importances from a profile's metrics.json.

    Cached per profile: importances differ between profiles because they are
    fitted on different feature sets, so a single shared cache would attribute
    one profile's importances to another's features.

    Returns an empty dict if that profile is not trained yet, so evidence
    generation degrades gracefully instead of failing.
    """
    name = (profile or FEATURE_PROFILE).strip().upper()
    if name in _IMPORTANCE_CACHE:
        return _IMPORTANCE_CACHE[name]
    try:
        path = PATHS.metrics_file(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        _IMPORTANCE_CACHE[name] = dict(data.get("feature_importance", {}))
    except (OSError, json.JSONDecodeError, KeyError):
        _IMPORTANCE_CACHE[name] = {}
    return _IMPORTANCE_CACHE[name]


# Features whose values are IDENTIFIERS, not magnitudes. Ratio comparisons
# against them are nonsense ("port 3527 is 44x the benign median of 80" says
# nothing), so they get categorical comparisons instead.
CATEGORICAL_FEATURES: frozenset[str] = frozenset({"Destination Port"})

# Well-known service ports, for describing a destination port in words.
WELL_KNOWN_PORTS: dict[int, str] = {
    20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 67: "DHCP", 80: "HTTP", 110: "POP3", 123: "NTP",
    135: "MS-RPC", 139: "NetBIOS", 143: "IMAP", 161: "SNMP", 389: "LDAP",
    443: "HTTPS", 445: "SMB", 502: "Modbus", 993: "IMAPS", 995: "POP3S",
    1433: "MSSQL", 1521: "Oracle", 2404: "IEC-104", 3306: "MySQL",
    3389: "RDP", 5432: "PostgreSQL", 5900: "VNC", 6379: "Redis",
    8080: "HTTP-alt", 8443: "HTTPS-alt", 20000: "DNP3",
}


def _describe_port(value: float) -> str:
    """Describe a destination port categorically rather than numerically."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        return "unrecognised port value"

    known = WELL_KNOWN_PORTS.get(port)
    if known:
        return f"port {port} ({known})"
    if port < 1024:
        return f"port {port} (system/privileged range)"
    if port < 49152:
        return f"port {port} (registered range, no standard service)"
    return f"port {port} (ephemeral range)"


def _format_value(feature: str, value: float) -> str:
    """Render a feature value for display, with units where meaningful."""
    unit = FEATURE_UNITS.get(feature, "")

    if feature == "Flow Duration" and value >= 1000:
        seconds = value / 1_000_000
        return f"{value:,.0f} us ({seconds:.3f} s)"
    if feature == "Destination Port":
        return f"{int(value)}"
    if abs(value) >= 1000:
        text = f"{value:,.0f}"
    elif abs(value) >= 1:
        text = f"{value:,.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return f"{text} {unit}".strip()


def _comparison(feature: str, value: float, threat: str) -> str | None:
    """Compare a value against the benign baseline, if one is known.

    Categorical features get a descriptive comparison; only genuine magnitudes
    get a ratio.
    """
    refs = REFERENCE_MEDIANS.get(feature)
    if not refs or "BENIGN" not in refs:
        return None
    benign = refs["BENIGN"]

    if feature in CATEGORICAL_FEATURES:
        description = _describe_port(value)
        typical = refs.get(threat)
        if typical is not None and abs(value - typical) < 1:
            return (f"{description}; this is the port most often seen for "
                    f"{threat} in the training data")
        return f"{description}; benign traffic here is mostly {_describe_port(benign)}"

    if benign == 0:
        if value > 0:
            return f"benign median is 0 {FEATURE_UNITS.get(feature, '')}".strip()
        return None

    ratio = value / benign
    if ratio >= NOTABLE_HIGH_RATIO:
        return f"{ratio:,.1f}x the benign median of {_format_value(feature, benign)}"
    if ratio <= NOTABLE_LOW_RATIO:
        if value == 0:
            return f"zero, against a benign median of {_format_value(feature, benign)}"
        return (f"{1 / max(ratio, 1e-9):,.1f}x below the benign median of "
                f"{_format_value(feature, benign)}")
    return f"benign median is {_format_value(feature, benign)}"


def _is_notable(feature: str, value: float, threat: str, direction: str) -> bool:
    """Decide whether a value actually supports the classification.

    Prevents evidence that contradicts itself: without this check a DDoS flow
    with a perfectly ordinary duration would still be presented with the text
    "held open 38x longer than benign".
    """
    refs = REFERENCE_MEDIANS.get(feature)

    if direction == "nonstandard":
        # A port is notable evidence of scanning only if no standard service
        # lives there. A scan of port 80 is indistinguishable, on this feature
        # alone, from ordinary web traffic.
        try:
            port = int(value)
        except (TypeError, ValueError):
            return False
        return port not in WELL_KNOWN_PORTS

    if not refs:
        return True

    if direction == "near":
        target = refs.get(threat)
        if target is None:
            return True
        if target == 0:
            return abs(value) <= 1.0
        return abs(value - target) / abs(target) <= NEAR_TOLERANCE

    benign = refs.get("BENIGN")
    if benign is None:
        return True
    if direction == "high":
        return value >= benign * NOTABLE_HIGH_RATIO if benign > 0 else value > 0
    if direction == "low":
        return value <= benign * NOTABLE_LOW_RATIO if benign > 0 else value == 0
    return True


@dataclass
class Evidence:
    """Evidence supporting one classification."""

    threat: str
    items: list[dict] = field(default_factory=list)
    aggregate_note: str | None = None
    method: str = (
        "Feature-based evidence: measured per-class reference statistics "
        "combined with the model's global feature importances. This is not a "
        "per-prediction attribution method such as SHAP."
    )

    def as_simple_dict(self) -> dict[str, float]:
        """Flat {feature: value} mapping, the shape the problem statement asks for."""
        return {item["feature"]: item["value"] for item in self.items}

    def to_dict(self) -> dict:
        return {
            "evidence": self.as_simple_dict(),
            "evidence_detail": self.items,
            "evidence_method": self.method,
            "aggregate_context": self.aggregate_note,
        }


def build_evidence(threat: str, features: dict[str, float],
                   max_items: int = 4, profile: str | None = None) -> Evidence:
    """Select the features that best justify `threat` for this flow.

    Args:
        threat: predicted class name.
        features: {canonical_feature_name: numeric_value} for the flow. May
            contain more features than the model uses.
        max_items: maximum number of evidence items to return.

    Returns:
        An Evidence object. Never raises: an unknown class or an empty feature
        dict yields evidence with no items rather than an exception, because
        this runs inside the replay loop.
    """
    rules = CLASS_RULES.get(threat, ())
    importances = _load_importances(profile)

    scored: list[tuple[float, dict]] = []
    fallback: list[tuple[float, dict]] = []

    for rule in rules:
        if rule.feature not in features:
            continue
        try:
            value = float(features[rule.feature])
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue

        item = {
            "feature": rule.feature,
            "value": value,
            "display": _format_value(rule.feature, value),
            "why": rule.why,
            "comparison": _comparison(rule.feature, value, threat),
            "importance": round(importances.get(rule.feature, 0.0), 4),
        }
        # Rank by model importance so the most influential features lead.
        score = importances.get(rule.feature, 0.0)
        if _is_notable(rule.feature, value, threat, rule.direction):
            scored.append((score, item))
        else:
            # Value does not support the rule. Kept only as a fallback, with the
            # claim removed so nothing misleading is displayed.
            weak = dict(item)
            weak["why"] = (
                f"Recorded for context: this value is not unusual for "
                f"{rule.feature} in this traffic."
            )
            fallback.append((score, weak))

    scored.sort(key=lambda pair: -pair[0])
    fallback.sort(key=lambda pair: -pair[0])

    items = [item for _, item in scored[:max_items]]
    if len(items) < 2:
        # Always show at least a couple of values so an alert is never empty.
        items += [item for _, item in fallback[: 2 - len(items)]]

    return Evidence(
        threat=threat,
        items=items,
        aggregate_note=AGGREGATE_CONTEXT.get(threat),
    )


# ---------------------------------------------------------------------------
def _recompute() -> int:
    """Recompute REFERENCE_MEDIANS from the processed dataset and print them.

    Use after changing the dataset, the per-class cap, or the feature profile,
    then paste the output over the constants above.
    """
    import pandas as pd

    if not PATHS.processed_csv.exists():
        print(f"ERROR: {PATHS.processed_csv} not found.")
        print("Run:  python -m preprocessing.clean_data")
        return 1

    frame = pd.read_csv(PATHS.processed_csv, low_memory=False)
    features = [f for f in REFERENCE_MEDIANS if f in frame.columns]
    grouped = frame.groupby("threat_class")[features].median()

    print("REFERENCE_MEDIANS: dict[str, dict[str, float]] = {")
    for feature in features:
        entries = ", ".join(
            f'"{cls}": {grouped.loc[cls, feature]:.6g}'
            for cls in grouped.index
        )
        print(f'    "{feature}": {{{entries}}},')
    print("}")
    print()
    print(f"# Computed from {PATHS.processed_csv.name}: "
          f"{len(frame):,} rows")
    for cls, n in frame["threat_class"].value_counts().items():
        print(f"#   {cls}: {n:,}")
    return 0


if __name__ == "__main__":
    if "--recompute" in sys.argv:
        raise SystemExit(_recompute())

    print("=" * 76)
    print("EVIDENCE GENERATION - worked examples")
    print("=" * 76)
    imps = _load_importances()
    print(f"  feature importances loaded: "
          f"{len(imps)} (from {PATHS.metrics_file(FEATURE_PROFILE).name})"
          if imps else "  no trained model found; importances default to 0")

    # Representative flows built from the measured per-class medians.
    samples = {
        "PortScan": {
            "Fwd Packet Length Mean": 0.0, "Flow Duration": 50.0,
            "Fwd Packets/s": 20000.0, "Total Fwd Packets": 1.0,
            "Init_Win_bytes_forward": 29200.0, "Destination Port": 3527.0,
        },
        "DDoS": {
            "Fwd Packet Length Mean": 7.0, "Flow Duration": 1876595.0,
            "Fwd IAT Std": 908248.3, "Init_Win_bytes_forward": 256.0,
            "Total Length of Fwd Packets": 26.0, "Fwd Packets/s": 1.7,
        },
        "Botnet": {
            "Init_Win_bytes_forward": 8192.0, "Destination Port": 8080.0,
            "Fwd IAT Std": 1653.2, "act_data_pkt_fwd": 0.0,
            "Fwd Packet Length Mean": 6.0,
        },
        "BENIGN": {
            "Fwd Packet Length Mean": 39.0, "Flow Duration": 48699.0,
            "Fwd Packets/s": 39.6,
        },
    }

    for threat, flow in samples.items():
        ev = build_evidence(threat, flow)
        print(f"\n{'-' * 76}\n{threat}\n{'-' * 76}")
        for item in ev.items:
            print(f"  {item['feature']} = {item['display']}"
                  f"   [importance {item['importance']:.4f}]")
            if item["comparison"]:
                print(f"      {item['comparison']}")
            print(f"      {item['why']}")
        if ev.aggregate_note:
            print(f"\n  Aggregate context (NOT evaluated per-flow):")
            print(f"      {ev.aggregate_note}")

    print(f"\n{'-' * 76}")
    print("Contradiction guard: a 'DDoS' flow with benign-looking values")
    print(f"{'-' * 76}")
    odd = build_evidence("DDoS", {
        "Fwd Packet Length Mean": 39.0,   # benign-typical, not DDoS-typical
        "Flow Duration": 48699.0,          # benign-typical
        "Init_Win_bytes_forward": 256.0,   # genuinely DDoS-typical
    })
    for item in odd.items:
        print(f"  {item['feature']} = {item['display']}")
        print(f"      {item['why']}")
    print("\n  Unsupported claims are replaced with neutral wording rather than")
    print("  asserting a pattern the values do not show.")
    print("=" * 76)

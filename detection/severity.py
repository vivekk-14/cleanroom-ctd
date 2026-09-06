"""
Severity assignment.

Confidence answers "how sure is the model?". Severity answers "how much should
an analyst care?". They are not the same question, so severity is derived from
confidence and then adjusted for the threat class.

Rules, in order
---------------
1. BENIGN is always INFO, whatever the confidence.
   A high-confidence BENIGN classification is not an alert. Ranking it
   alongside real detections would flood the analyst's queue with rows that
   say "nothing is wrong", which is how real SOC dashboards become useless.

2. Otherwise, map confidence to a base band (config.SEVERITY.bands):

       confidence in [0.00, 0.50)  ->  LOW
       confidence in [0.50, 0.75)  ->  MEDIUM
       confidence in [0.75, 0.90)  ->  HIGH
       confidence in [0.90, 1.00]  ->  CRITICAL

   Bands are half-open so every value in [0, 1] maps to exactly one band, with
   no gaps and no overlaps.

3. Escalate one band if the threat class is in config.SEVERITY.escalate
   (currently DDoS and Botnet). Applied at most once; cannot exceed CRITICAL.

   Why those two:
     DDoS   - direct availability impact on the production network. In a
              critical-infrastructure context, loss of availability is the
              damage, so a merely probable DDoS still warrants attention.
     Botnet - implies a host inside the monitored network is already
              compromised and beaconing to external command-and-control. The
              breach has happened; confidence only concerns whether this
              particular flow is part of it.

   PortScan is deliberately NOT escalated. It is reconnaissance: it precedes
   impact rather than causing it, and scans are frequent enough that
   escalating them would dilute the higher bands.

Every returned decision records its own derivation (base band, whether
escalation fired, and why) so the dashboard can show the reasoning instead of
an unexplained label. Nothing about the mapping is hidden from the analyst.

All thresholds live in config.SEVERITY. Editing them changes behaviour here
with no code change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SEVERITY  # noqa: E402


@dataclass(frozen=True)
class SeverityDecision:
    """A severity assignment together with its full derivation.

    Attributes:
        severity:       final label, e.g. 'CRITICAL'
        base_severity:  band from confidence alone, before escalation
        escalated:      True if the threat class bumped it up one band
        reason:         one-line human-readable explanation
        confidence:     the confidence the decision was based on
        is_alertable:   True if this belongs in the actionable alert queue
        needs_review:   True if a human should look at this flow even though it
                        is not an actionable alert. Set for DEGRADED flows,
                        including benign ones that `is_alertable` suppresses.
        reliability:    the OOD verdict that fed into the decision, or None when
                        no distribution check was performed
    """

    severity: str
    base_severity: str
    escalated: bool
    reason: str
    confidence: float
    is_alertable: bool
    needs_review: bool = False
    reliability: str | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "base_severity": self.base_severity,
            "escalated": self.escalated,
            "severity_reason": self.reason,
            "is_alertable": self.is_alertable,
            "needs_review": self.needs_review,
        }


def base_band(confidence: float) -> str:
    """Return the severity band for a confidence value, ignoring threat class.

    Bands are half-open [lower, next_lower). Confidence is clamped to [0, 1]
    first: a NaN or out-of-range probability should still yield a valid band
    rather than raising during a live replay.

    >>> base_band(0.42)
    'LOW'
    >>> base_band(0.50)
    'MEDIUM'
    >>> base_band(0.9)
    'CRITICAL'
    """
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.0
    if value != value:  # NaN
        value = 0.0
    value = min(max(value, 0.0), 1.0)

    label = SEVERITY.bands[0][1]
    for lower_bound, name in SEVERITY.bands:
        if value >= lower_bound:
            label = name
        else:
            break
    return label


def escalate_once(severity: str) -> str:
    """Return the next severity up the ladder, stopping at the top.

    >>> escalate_once('MEDIUM')
    'HIGH'
    >>> escalate_once('CRITICAL')
    'CRITICAL'
    """
    ladder = SEVERITY.ladder
    if severity not in ladder:
        return severity
    index = ladder.index(severity)
    return ladder[min(index + 1, len(ladder) - 1)]


def assess(threat: str, confidence: float,
           reliability: str | None = None,
           model_prediction: str | None = None) -> SeverityDecision:
    """Assign severity for a classified flow.

    Args:
        threat: the reported class name, e.g. 'DDoS' or 'UNKNOWN_OOD'.
        confidence: the CLASSIFIER's probability for its own prediction, in
            [0, 1]. Note this is not a probability that `threat` is correct when
            the flow is out of distribution.
        reliability: verdict from detection.ood ('RELIABLE', 'DEGRADED',
            'UNRELIABLE', 'UNKNOWN'). None skips the reliability rule entirely,
            which is what callers without a distribution check get.
        model_prediction: what the classifier actually said, used in the reason
            text when `threat` has been replaced by UNKNOWN_OOD.

    Returns:
        A SeverityDecision carrying the final severity and its derivation.
    """
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.0
    if value != value:
        value = 0.0
    value = min(max(value, 0.0), 1.0)

    # Rule 0: an unreliable prediction is triaged on reliability, not on class.
    #
    # This rule runs FIRST, ahead of the BENIGN suppression below, and that
    # ordering is the entire point. The dangerous failure mode is a confident
    # BENIGN on input the model has never seen anything like: on the synthetic
    # fixture the classifier said BENIGN at 0.913 mean confidence for all 20
    # rows. Under Rule 1 alone every one of those would become INFO and vanish
    # from the analyst's queue. Checking reliability first surfaces them.
    #
    # Severity is HIGH rather than CRITICAL: an unreliable answer warrants human
    # review, but it is not evidence of an attack, and ranking it above a
    # confirmed high-confidence DDoS would invert the analyst's priorities.
    if reliability == "UNRELIABLE":
        said = model_prediction or threat
        return SeverityDecision(
            severity=SEVERITY.ood_severity,
            base_severity=SEVERITY.ood_severity,
            escalated=False,
            reason=(
                f"Input lies outside the distribution the model was trained on, "
                f"so its answer is not trustworthy. The classifier said {said} "
                f"at {value:.0%} confidence, but that confidence describes its "
                f"internal certainty, not the correctness of the answer on "
                f"unfamiliar input. Flagged {SEVERITY.ood_severity} for human "
                f"review rather than suppressed."
            ),
            confidence=value,
            is_alertable=SEVERITY.ood_severity in SEVERITY.alert_severities,
            needs_review=True,
            reliability=reliability,
        )

    # Rule 1: BENIGN never produces a ranked alert.
    #
    # A DEGRADED benign flow is still suppressed from the ACTIONABLE queue -- it
    # is not evidence of an attack -- but `needs_review` is set so it remains
    # visible in a separate low-priority queue. Without that, the fixture's
    # PortScan and Botnet rows (one violation each, all classified BENIGN)
    # would disappear entirely.
    #
    # MEASURED, which is why the threshold stays at 2 rather than dropping to 1:
    # counting a single violation as UNRELIABLE would catch 19 of 20 fixture
    # rows instead of 10, but flag 1.759% of Monday's real benign traffic
    # (9,316 flows) instead of 0.887% (4,694). Routing DEGRADED flows to a
    # review queue surfaces all 19 without putting any in the alert queue.
    if SEVERITY.benign_is_informational and threat == SEVERITY.benign_label:
        suffix = ""
        if reliability == "DEGRADED":
            suffix = (" One feature lies outside the training range, so this "
                      "classification is less certain than the confidence "
                      "suggests. Routed to the review queue, not the alert "
                      "queue.")
        return SeverityDecision(
            severity=SEVERITY.informational_severity,
            base_severity=SEVERITY.informational_severity,
            escalated=False,
            reason=(
                f"Classified {threat} at {value:.0%} confidence. Benign traffic "
                f"is recorded for context only and is never ranked as an alert."
                + suffix
            ),
            confidence=value,
            is_alertable=False,
            needs_review=(reliability == "DEGRADED"),
            reliability=reliability,
        )

    # Rule 2: confidence -> base band.
    base = base_band(value)

    # Rule 3: threat-class escalation.
    should_escalate = threat in SEVERITY.escalate
    final = escalate_once(base) if should_escalate else base

    if should_escalate and final != base:
        reason = (
            f"Confidence {value:.0%} maps to {base}; escalated to {final} "
            f"because {threat} has direct operational impact "
            f"(availability loss or an already-compromised internal host)."
        )
    elif should_escalate:
        reason = (
            f"Confidence {value:.0%} maps to {base}, already the highest band; "
            f"{threat} escalation cannot raise it further."
        )
    else:
        reason = (
            f"Confidence {value:.0%} maps to {base}. {threat} is not escalated: "
            f"it indicates reconnaissance rather than realised impact."
        )

    if reliability == "DEGRADED":
        reason += (
            " One feature lies outside the training range, so treat this "
            "classification with caution."
        )
    elif reliability == "UNKNOWN":
        reason += (
            " No distribution reference was available, so reliability could "
            "not be checked."
        )

    return SeverityDecision(
        severity=final,
        base_severity=base,
        escalated=should_escalate and final != base,
        reason=reason,
        confidence=value,
        is_alertable=final in SEVERITY.alert_severities,
        needs_review=(reliability == "DEGRADED"),
        reliability=reliability,
    )


def severity_rank(severity: str) -> int:
    """Numeric rank for sorting, higher = more severe. Unknown labels rank -1."""
    try:
        return SEVERITY.ladder.index(severity)
    except ValueError:
        return -1


def describe_policy() -> str:
    """Render the active policy as text, for the dashboard and the README.

    Generated from config rather than written by hand, so the documented policy
    cannot drift away from the enforced one.
    """
    lines = ["Rule 0 - reliability overrides everything:"]
    lines.append(
        f"  input outside the training distribution -> "
        f"{SEVERITY.ood_severity}, threat_class becomes UNKNOWN_OOD"
    )
    lines.append(
        "  This is checked BEFORE the benign rule below, so a confident BENIGN"
    )
    lines.append(
        "  on unfamiliar input is surfaced rather than suppressed to INFO."
    )
    lines.append("")
    lines.append("Confidence -> severity bands:")
    bands = list(SEVERITY.bands)
    for i, (lower, name) in enumerate(bands):
        upper = bands[i + 1][0] if i + 1 < len(bands) else None
        span = f"{lower:.2f} <= confidence < {upper:.2f}" if upper else \
               f"confidence >= {lower:.2f}"
        lines.append(f"  {span:<32} {name}")

    lines.append("")
    lines.append(f"Escalated one band: {', '.join(SEVERITY.escalate)}")
    lines.append("  DDoS   - availability impact on the production network")
    lines.append("  Botnet - internal host already compromised, beaconing to C2")
    lines.append("")
    lines.append(
        f"{SEVERITY.benign_label} is always "
        f"{SEVERITY.informational_severity} and never alertable, UNLESS the "
        f"flow is out of distribution."
    )
    lines.append(f"Counted as actionable: {', '.join(SEVERITY.alert_severities)}")
    lines.append("")
    lines.append("Two separate queues:")
    lines.append("  is_alertable  - actionable alert queue")
    lines.append("  needs_review  - lower-priority queue for flows whose")
    lines.append("                  classification is less trustworthy than its")
    lines.append("                  confidence suggests (DEGRADED reliability),")
    lines.append("                  including benign ones the alert queue drops")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("SEVERITY POLICY")
    print("=" * 70)
    print(describe_policy())

    print("\n" + "=" * 70)
    print("WORKED EXAMPLES")
    print("=" * 70)
    print(f"  {'threat':<10} {'conf':>6}  {'base':<9} {'final':<9} {'alert?':<7} esc")
    print(f"  {'-' * 10} {'-' * 6}  {'-' * 9} {'-' * 9} {'-' * 7} ---")
    cases = [
        ("BENIGN", 0.99), ("BENIGN", 0.55),
        ("PortScan", 0.42), ("PortScan", 0.60),
        ("PortScan", 0.82), ("PortScan", 0.97),
        ("DDoS", 0.42), ("DDoS", 0.60),
        ("DDoS", 0.82), ("DDoS", 0.96),
        ("Botnet", 0.55), ("Botnet", 0.91),
    ]
    for threat, conf in cases:
        d = assess(threat, conf)
        print(f"  {threat:<10} {conf:>6.2f}  {d.base_severity:<9} "
              f"{d.severity:<9} {str(d.is_alertable):<7} "
              f"{'yes' if d.escalated else '-'}")

    print("\n  Boundary and malformed-input checks:")
    for threat, conf in [("DDoS", 0.0), ("DDoS", 0.4999), ("DDoS", 0.50),
                         ("DDoS", 0.90), ("DDoS", 1.0), ("PortScan", 1.5),
                         ("PortScan", -0.2), ("PortScan", float("nan"))]:
        d = assess(threat, conf)
        shown = "nan" if conf != conf else f"{conf:.4f}"
        print(f"    {threat:<9} conf={shown:<8} -> {d.base_severity:<8} "
              f"-> {d.severity}")

    print("\n" + "=" * 70)
    print("RELIABILITY INTERACTION - the case the fixture exposed")
    print("=" * 70)
    print(f"  {'reported':<13} {'said':<9} {'conf':>6} {'reliability':<11} "
          f"{'sev':<9} {'alert':>6} {'review':>7}")
    print(f"  {'-' * 13} {'-' * 9} {'-' * 6} {'-' * 11} {'-' * 9} "
          f"{'-' * 6} {'-' * 7}")
    reliability_cases = [
        # The dangerous case: a confident BENIGN on unfamiliar input. Without
        # Rule 0 this is INFO and invisible.
        ("UNKNOWN_OOD", "BENIGN", 0.98, "UNRELIABLE"),
        ("UNKNOWN_OOD", "DDoS", 0.95, "UNRELIABLE"),
        ("BENIGN", "BENIGN", 0.98, "DEGRADED"),
        ("BENIGN", "BENIGN", 0.98, "RELIABLE"),
        ("DDoS", "DDoS", 0.82, "DEGRADED"),
        ("DDoS", "DDoS", 0.82, "RELIABLE"),
        ("PortScan", "PortScan", 0.60, "UNKNOWN"),
    ]
    for reported, said, conf, reliability in reliability_cases:
        d = assess(reported, conf, reliability=reliability,
                   model_prediction=said)
        print(f"  {reported:<13} {said:<9} {conf:>6.2f} {reliability:<11} "
              f"{d.severity:<9} {str(d.is_alertable):>6} "
              f"{str(d.needs_review):>7}")

    print("\n  The first two rows are why Rule 0 runs before the BENIGN rule.")
    print("  On the synthetic OOD fixture the classifier said BENIGN for all 20")
    print("  rows at 0.913 mean confidence. Under the BENIGN rule alone, every")
    print("  one would be INFO and absent from the analyst's queue.")

    print("\n  Example reason strings:")
    print(f"    normal : {assess('DDoS', 0.82).reason}")
    print(f"    OOD    : {assess('UNKNOWN_OOD', 0.98, reliability='UNRELIABLE', model_prediction='BENIGN').reason}")
    print("=" * 70)

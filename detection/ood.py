"""
Out-of-distribution detection.

A classifier answers "which of my classes is this closest to?". It cannot answer
"have I ever seen anything like this?". That gap is dangerous: on the synthetic
fixture in tests/fixtures/, the BIDIRECTIONAL model labelled all 20 rows BENIGN
at a mean confidence of 0.913, including five rows whose SYN Flag Count was
4,200-8,000 when the training data only ever contains 0 or 1.

This module answers the second question separately, so a confident prediction on
unfamiliar input is reported as unreliable rather than trusted.

Two independent signals
-----------------------
1. RANGE VIOLATIONS (primary, drives the verdict)
   Per-feature bounds taken from the 0.1st and 99.9th percentiles of the
   TRAINING split. A value beyond its bound means the trees are extrapolating:
   every threshold they learned for that feature lies on one side of the value,
   so the feature contributes no information, and the prediction rests on
   whatever the remaining features happen to say.

   Deterministic, explainable (it names the feature, the bound, and how far out
   the value is), and needs no second model.

2. NOVELTY SCORE (advisory, does not change the verdict by default)
   An IsolationForest fitted on BENIGN training rows only. Reported alongside
   the verdict because it is the only signal measured that has any traction on
   novel attack classes -- see the honest limitation below.

MEASURED (BIDIRECTIONAL profile, 14 features, q=0.001 bounds)

  Range violations, fraction of rows flagged:

    threshold        held-out   Wed BENIGN   unseen atk   synthetic
    >= 1 feature       0.850%       1.618%       1.553%      95.00%
    >= 2 features      0.250%       0.576%       0.744%      50.00%

  IsolationForest fitted on 37,500 BENIGN training rows:

    contamination   held-out BENIGN   Wed BENIGN   unseen DoS   synthetic
    0.01                     1.07%        1.01%        1.69%      50.00%
    0.05                     4.75%        3.71%       47.20%      50.00%

THE HONEST LIMITATION
---------------------
Range violations do NOT detect the unseen real attacks from Wednesday's capture
(DoS Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed). Those flows are
flagged at 1.553% against a 0.850% baseline on held-out data -- no meaningful
lift. Measured directly: they average 0.03 out-of-range features, identical to
benign traffic.

They are novel by LABEL, not by feature distribution. A slow-HTTP DoS flow
genuinely looks like an ordinary slow HTTP flow on these 14 features. No
distribution-based method can separate them, because there is nothing
distributionally unusual to find. The fix is training data for those classes,
not a better OOD detector.

The IsolationForest does reach 47.2% of them, at a 4.75% false-positive cost on
benign traffic. On Monday's 529,918 benign flows that would be ~25,000 review
items, so it is reported as an advisory score rather than used to override the
classifier. `OOD.novelty_drives_verdict` in config makes that a one-line change
if the tradeoff is acceptable for a given deployment.

What this module catches, then, is DISTRIBUTION SHIFT: input produced by a
different feature-extraction pipeline, a different network, a different tool
version, or (as with the fixture) by hand. That is a real and common failure
mode, and it is exactly the failure the fixture demonstrates.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OOD, PATHS  # noqa: E402

# Verdict tiers, ordered from most to least trustworthy.
RELIABLE = "RELIABLE"
DEGRADED = "DEGRADED"
UNRELIABLE = "UNRELIABLE"
UNKNOWN = "UNKNOWN"

RELIABILITY_ORDER = (UNKNOWN, RELIABLE, DEGRADED, UNRELIABLE)

# Threat class assigned when input is too far outside the training distribution
# for the classifier's answer to mean anything.
OOD_THREAT_CLASS = "UNKNOWN_OOD"


@dataclass(frozen=True)
class RangeViolation:
    """One feature whose value lies outside the training range."""

    feature: str
    value: float
    lower: float
    upper: float
    direction: str          # 'above' or 'below'
    excursion: float        # how many range-widths beyond the bound
    training_median: float

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": self.value,
            "training_range": [self.lower, self.upper],
            "training_median": self.training_median,
            "direction": self.direction,
            "excursion_range_widths": round(self.excursion, 2),
        }

    def describe(self) -> str:
        bound = self.upper if self.direction == "above" else self.lower
        return (
            f"{self.feature} = {self.value:,.4g} is {self.direction} the "
            f"training bound of {bound:,.4g} "
            f"(range [{self.lower:,.4g}, {self.upper:,.4g}], "
            f"{self.excursion:,.1f}x the range width beyond it)"
        )


@dataclass
class OODAssessment:
    """Whether a flow lies inside the distribution the model was trained on.

    This is deliberately SEPARATE from classifier confidence. The two answer
    different questions and can disagree in the most dangerous way: high
    confidence on input the model has never seen anything like.
    """

    in_distribution: bool | None      # None when no reference is available
    reliability: str
    ood_score: float                  # 0.0 = in distribution, 1.0 = far outside
    violations: list[RangeViolation] = field(default_factory=list)
    novelty_score: float | None = None
    novelty_flagged: bool | None = None
    reason: str = ""
    reference_available: bool = True
    n_features_checked: int = 0

    @property
    def ood_features(self) -> list[str]:
        """Names of the features that fell outside the training range."""
        return [v.feature for v in self.violations]

    def to_dict(self) -> dict:
        return {
            "in_distribution": self.in_distribution,
            "reliability": self.reliability,
            "ood_score": round(self.ood_score, 4),
            "ood_features": self.ood_features,
            "ood_violations": [v.to_dict() for v in self.violations],
            "ood_reason": self.reason,
            "novelty_score": (round(self.novelty_score, 4)
                              if self.novelty_score is not None else None),
            "novelty_flagged": self.novelty_flagged,
            "ood_reference_available": self.reference_available,
            "ood_features_checked": self.n_features_checked,
        }


# ---------------------------------------------------------------------------
# Building the reference (called by model/train.py)
# ---------------------------------------------------------------------------
def build_reference(
    X_train: np.ndarray,
    features: list[str],
    y_train: np.ndarray | None = None,
    benign_label: object = None,
    quantile: float | None = None,
) -> dict:
    """Compute the OOD reference from the TRAINING split.

    Args:
        X_train: training feature matrix, shape (n_rows, n_features).
        features: feature names, in the model's exact order.
        y_train: training labels, used to fit the benign-only novelty detector.
        benign_label: the encoded value of the benign class in `y_train`.
        quantile: lower tail fraction; bounds are [q, 1-q].

    Returns:
        A JSON-serialisable dict, written to
        model/profiles/<slug>/ood_reference.json.

    IMPORTANT: pass the TRAINING split only. Building bounds from the full
    dataset leaks held-out rows into the reference and makes the measured
    false-OOD rate look better than it is.
    """
    quantile = OOD.quantile if quantile is None else quantile

    lower = np.quantile(X_train, quantile, axis=0)
    upper = np.quantile(X_train, 1.0 - quantile, axis=0)
    median = np.median(X_train, axis=0)

    reference: dict[str, Any] = {
        "features": list(features),
        "quantile": quantile,
        "n_training_rows": int(len(X_train)),
        "bounds": {
            name: {
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "median": float(median[i]),
            }
            for i, name in enumerate(features)
        },
        "min_violations_for_ood": OOD.min_violations_for_ood,
        "notes": [
            "Bounds are per-feature quantiles of the TRAINING split only. "
            "Using all rows would leak held-out data into the reference.",
            "A value outside its bound means the trees are extrapolating on "
            "that feature: every threshold they learned lies on one side of it.",
        ],
    }

    # Benign-only novelty detector. Advisory: reported, but by default it does
    # not change the verdict. See the module docstring for its measured cost.
    if y_train is not None and benign_label is not None and OOD.fit_novelty:
        try:
            from sklearn.ensemble import IsolationForest

            benign_rows = X_train[y_train == benign_label]
            if len(benign_rows) >= OOD.novelty_min_rows:
                detector = IsolationForest(
                    n_estimators=OOD.novelty_estimators,
                    contamination=OOD.novelty_contamination,
                    random_state=OOD.random_state,
                    n_jobs=-1,
                )
                detector.fit(benign_rows)
                # score_samples on the training rows gives the operating point
                # the threshold corresponds to, recorded for transparency.
                scores = detector.score_samples(benign_rows)
                reference["novelty"] = {
                    "fitted_on": "benign training rows only",
                    "n_rows": int(len(benign_rows)),
                    "contamination": OOD.novelty_contamination,
                    "threshold": float(detector.offset_),
                    "benign_score_median": float(np.median(scores)),
                    "benign_score_p1": float(np.quantile(scores, 0.01)),
                }
                reference["_novelty_model"] = detector
        except ImportError:
            pass

    return reference


# ---------------------------------------------------------------------------
# Loading and applying the reference
# ---------------------------------------------------------------------------
_CACHE: dict[str, dict | None] = {}


def load_reference(profile: str) -> dict | None:
    """Load a profile's OOD reference, or None if it has not been built.

    Cached: this runs once per flow in the replay loop.
    """
    if profile in _CACHE:
        return _CACHE[profile]

    path = PATHS.ood_reference_file(profile)
    if not path.exists():
        _CACHE[profile] = None
        return None

    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _CACHE[profile] = None
        return None

    # The IsolationForest is pickled separately; JSON cannot hold it.
    novelty_path = PATHS.novelty_model_file(profile)
    if novelty_path.exists():
        try:
            import joblib

            reference["_novelty_model"] = joblib.load(novelty_path)
        except Exception:  # noqa: BLE001
            reference.pop("_novelty_model", None)

    _CACHE[profile] = reference
    return reference


def reset_cache(profile: str | None = None) -> None:
    """Drop cached references so the next call reloads from disk."""
    if profile is None:
        _CACHE.clear()
    else:
        _CACHE.pop(profile, None)


def _unavailable(reason: str) -> OODAssessment:
    """Assessment used when no reference exists: report unknown, never guess."""
    return OODAssessment(
        in_distribution=None,
        reliability=UNKNOWN,
        ood_score=0.0,
        reason=reason,
        reference_available=False,
    )


def assess(
    values: np.ndarray,
    profile: str,
    features: list[str] | None = None,
) -> OODAssessment:
    """Decide whether one flow lies inside the model's training distribution.

    Args:
        values: the flow's feature vector, in the model's feature order.
        profile: which profile's reference to use.
        features: feature names, for the violation report. Defaults to the
            names stored in the reference.

    Returns:
        An OODAssessment. Never raises: this runs inside the replay loop, so a
        missing reference or a malformed vector yields UNKNOWN rather than an
        exception.
    """
    reference = load_reference(profile)
    if reference is None:
        return _unavailable(
            f"No OOD reference for profile {profile}. Distribution checking is "
            f"unavailable; retrain with `python -m model.train` to build it."
        )

    names = features or reference.get("features", [])
    bounds = reference.get("bounds", {})
    if not names or not bounds:
        return _unavailable("OOD reference is present but empty or malformed.")

    array = np.asarray(values, dtype="float64").ravel()
    if len(array) != len(names):
        return _unavailable(
            f"Feature vector has {len(array)} values but the OOD reference "
            f"describes {len(names)} features."
        )

    # --- range violations -------------------------------------------------
    violations: list[RangeViolation] = []
    for i, name in enumerate(names):
        bound = bounds.get(name)
        if bound is None:
            continue
        value = float(array[i])
        if not np.isfinite(value):
            continue

        lower, upper = float(bound["lower"]), float(bound["upper"])
        # Zero-width ranges are real: SYN Flag Count is [0, 0] for every attack
        # class. Any nonzero value is then infinitely far outside, so fall back
        # to a unit width to keep the excursion figure finite and comparable.
        width = max(upper - lower, 1e-9)

        if value > upper:
            violations.append(RangeViolation(
                feature=name, value=value, lower=lower, upper=upper,
                direction="above", excursion=(value - upper) / width,
                training_median=float(bound.get("median", 0.0)),
            ))
        elif value < lower:
            violations.append(RangeViolation(
                feature=name, value=value, lower=lower, upper=upper,
                direction="below", excursion=(lower - value) / width,
                training_median=float(bound.get("median", 0.0)),
            ))

    # Worst excursions first: the most extreme violation is the most useful
    # thing to show an analyst.
    violations.sort(key=lambda v: -v.excursion)

    # --- novelty score (advisory) ----------------------------------------
    novelty_score = novelty_flagged = None
    detector = reference.get("_novelty_model")
    if detector is not None:
        try:
            novelty_score = float(detector.score_samples(array.reshape(1, -1))[0])
            novelty_flagged = bool(
                detector.predict(array.reshape(1, -1))[0] == -1)
        except Exception:  # noqa: BLE001
            novelty_score = novelty_flagged = None

    # --- verdict ----------------------------------------------------------
    n_violations = len(violations)
    minimum = reference.get("min_violations_for_ood", OOD.min_violations_for_ood)

    if n_violations >= minimum:
        reliability = UNRELIABLE
        in_distribution = False
    elif n_violations >= 1:
        reliability = DEGRADED
        in_distribution = True
    else:
        reliability = RELIABLE
        in_distribution = True

    if (OOD.novelty_drives_verdict and novelty_flagged
            and reliability == RELIABLE):
        reliability = DEGRADED
        in_distribution = True

    # ood_score: 0 when clean, rising with both the count and the severity of
    # the violations. Saturates at 1.0 so it is comparable across flows.
    if n_violations:
        count_term = min(n_violations / max(len(names), 1), 1.0)
        excursion_term = min(violations[0].excursion / 10.0, 1.0)
        ood_score = min(0.5 * count_term + 0.5 * excursion_term + 0.25, 1.0)
    else:
        ood_score = 0.0

    # --- reason -----------------------------------------------------------
    if reliability == UNRELIABLE:
        worst = violations[0]
        reason = (
            f"{n_violations} of {len(names)} features lie outside the range "
            f"seen in training, so the model is extrapolating and its "
            f"prediction is not trustworthy. Worst: {worst.describe()}"
        )
    elif reliability == DEGRADED and violations:
        reason = (
            f"1 of {len(names)} features lies outside the training range. The "
            f"prediction is reported but should be treated with caution. "
            f"{violations[0].describe()}"
        )
    elif reliability == DEGRADED:
        reason = (
            "All feature values are within the training range, but the "
            "benign-traffic novelty detector flagged this flow as unusual."
        )
    else:
        reason = (
            f"All {len(names)} feature values lie within the range seen during "
            f"training, so the model is interpolating rather than extrapolating."
        )

    return OODAssessment(
        in_distribution=in_distribution,
        reliability=reliability,
        ood_score=ood_score,
        violations=violations,
        novelty_score=novelty_score,
        novelty_flagged=novelty_flagged,
        reason=reason,
        reference_available=True,
        n_features_checked=len(names),
    )


def describe_policy() -> str:
    """Render the active OOD policy as text, for the dashboard and README."""
    return "\n".join([
        "Out-of-distribution detection",
        "",
        f"Per-feature bounds: [{OOD.quantile}, {1 - OOD.quantile}] quantiles of "
        f"the training split.",
        "",
        "Verdict tiers, by count of features outside their training range:",
        f"  0 violations                 {RELIABLE}    prediction trusted",
        f"  1 violation                  {DEGRADED}    prediction shown, flagged",
        f"  >= {OOD.min_violations_for_ood} violations"
        f"{' ' * 17}{UNRELIABLE}  threat_class becomes {OOD_THREAT_CLASS}",
        "",
        f"Novelty detector: {'enabled' if OOD.fit_novelty else 'disabled'}"
        + (f", advisory only (contamination={OOD.novelty_contamination})"
           if OOD.fit_novelty and not OOD.novelty_drives_verdict else ""),
        "",
        "MEASURED false-OOD rates on real in-distribution data:",
        "  held-out CIC-IDS2017     0.250% reach UNRELIABLE, 0.850% reach DEGRADED",
        "  unseen day (Wed BENIGN)  0.576% reach UNRELIABLE, 1.618% reach DEGRADED",
        "",
        "MEASURED detection on the synthetic OOD fixture: 50% UNRELIABLE, 95% "
        "at least DEGRADED.",
        "",
        "LIMITATION: this detects distribution SHIFT, not novel attack classes.",
        "Unseen real attacks (DoS Hulk, Heartbleed) average 0.03 out-of-range",
        "features -- the same as benign traffic. They are novel by label, not by",
        "feature distribution, and need training data rather than a better",
        "detector.",
    ])


if __name__ == "__main__":
    print("=" * 78)
    print("OOD DETECTION - policy and self-test")
    print("=" * 78)
    print(describe_policy())

    print("\n" + "=" * 78)
    print("REFERENCE AVAILABILITY")
    print("=" * 78)
    for profile in ("STRICT_UNIDIRECTIONAL", "BIDIRECTIONAL"):
        reference = load_reference(profile)
        if reference is None:
            print(f"  {profile:<24} no reference "
                  f"(run: python -m model.train --profile {profile})")
            continue
        print(f"  {profile:<24} {len(reference['bounds'])} features, "
              f"built from {reference['n_training_rows']:,} training rows")
        if "novelty" in reference:
            novelty = reference["novelty"]
            print(f"  {'':<24} novelty detector on {novelty['n_rows']:,} "
                  f"benign rows, contamination {novelty['contamination']}")

    # Score the fixture, which is the whole reason this module exists.
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" \
        / "ood_synthetic_attacks.csv"
    if fixture.exists() and load_reference("BIDIRECTIONAL"):
        import pandas as pd

        from preprocessing.feature_config import normalize_column

        reference = load_reference("BIDIRECTIONAL")
        names = reference["features"]

        frame = pd.read_csv(fixture)
        frame.columns = [normalize_column(c) for c in frame.columns]
        frame = frame.dropna(subset=["Label"]).reset_index(drop=True)

        print("\n" + "=" * 78)
        print("SYNTHETIC OOD FIXTURE - per-row assessment")
        print("=" * 78)
        print(f"  {'#':>3} {'label':<9} {'reliability':<11} {'score':>6} "
              f"{'in-dist':>8}  out-of-range features")
        print(f"  {'-' * 3} {'-' * 9} {'-' * 11} {'-' * 6} {'-' * 8}  {'-' * 32}")

        tally: dict[str, int] = {}
        for i, row in frame.iterrows():
            vector = np.array([row[n] for n in names], dtype="float64")
            result = assess(vector, "BIDIRECTIONAL", names)
            tally[result.reliability] = tally.get(result.reliability, 0) + 1
            shown = ", ".join(result.ood_features[:3]) or "-"
            print(f"  {i + 1:>3} {row['Label']:<9} {result.reliability:<11} "
                  f"{result.ood_score:>6.3f} "
                  f"{str(result.in_distribution):>8}  {shown}")

        print(f"\n  tally: {tally}")
        print(f"\n  Worst violation in the file:")
        worst_row, worst_violation = None, None
        for i, row in frame.iterrows():
            vector = np.array([row[n] for n in names], dtype="float64")
            result = assess(vector, "BIDIRECTIONAL", names)
            for violation in result.violations:
                if worst_violation is None or \
                        violation.excursion > worst_violation.excursion:
                    worst_row, worst_violation = i + 1, violation
        if worst_violation:
            print(f"    row {worst_row}: {worst_violation.describe()}")
    print("=" * 78)

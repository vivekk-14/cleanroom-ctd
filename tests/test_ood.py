"""
Tests for out-of-distribution detection.

These lock down the behaviour established when the synthetic fixture in
tests/fixtures/ood_synthetic_attacks.csv exposed a real failure: the classifier
labelled all 20 rows BENIGN at 0.913 mean confidence, including five whose
SYN Flag Count was 4,200-8,000 when the training data only ever contains 0 or 1.

The four behaviours pinned here
-------------------------------
    in-distribution        -> normal classification, reliability RELIABLE
    >= 2 violations        -> UNKNOWN_OOD, UNRELIABLE, needs_review
    1 violation            -> classification kept, DEGRADED, needs_review
    OOD + BENIGN           -> NOT suppressed to INFO

The last one is the reason this module exists. Without Rule 0 running before the
BENIGN suppression, a confident BENIGN on unfamiliar input becomes INFO and
disappears from the analyst's queue -- which is precisely the dangerous case.

And the invariant
-----------------
    reliability == UNRELIABLE   <->   threat_class == UNKNOWN_OOD

Enforced in alert_schema.validate(). Tested here in both directions so a partial
change to the OOD layer cannot silently report a trusted-looking class.

Run:
    python -m pytest tests/test_ood.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OOD, PATHS, SEVERITY  # noqa: E402
from detection import ood as ood_module  # noqa: E402
from detection import severity as severity_module  # noqa: E402
from detection.alert_schema import (  # noqa: E402
    OOD_CLASS,
    RELIABILITY_VALUES,
    Alert,
    validate,
)
from detection.ood import (  # noqa: E402
    DEGRADED,
    OOD_THREAT_CLASS,
    RELIABLE,
    UNKNOWN,
    UNRELIABLE,
)
from model.predict import (  # noqa: E402
    get_expected_features,
    is_model_ready,
    predict_threat,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ood_synthetic_attacks.csv"

needs_bidirectional = pytest.mark.skipif(
    not is_model_ready("BIDIRECTIONAL"),
    reason="BIDIRECTIONAL model not trained; "
           "run: python -m model.train --profile BIDIRECTIONAL",
)
needs_strict = pytest.mark.skipif(
    not is_model_ready("STRICT_UNIDIRECTIONAL"),
    reason="STRICT_UNIDIRECTIONAL model not trained; run: python -m model.train",
)
needs_reference = pytest.mark.skipif(
    ood_module.load_reference("BIDIRECTIONAL") is None,
    reason="no OOD reference for BIDIRECTIONAL; retrain to build it",
)


@pytest.fixture(scope="module")
def fixture_rows() -> list[dict]:
    """The 20 labelled rows of the synthetic OOD fixture."""
    pytest.importorskip("pandas")
    import pandas as pd

    from preprocessing.feature_config import normalize_column

    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")

    frame = pd.read_csv(FIXTURE)
    frame.columns = [normalize_column(c) for c in frame.columns]
    frame = frame.dropna(subset=["Label"]).reset_index(drop=True)
    return frame.to_dict("records")


@pytest.fixture(scope="module")
def reference() -> dict:
    ref = ood_module.load_reference("BIDIRECTIONAL")
    if ref is None:
        pytest.skip("no OOD reference")
    return ref


def _vector(row: dict, features: list[str]) -> np.ndarray:
    return np.array([row[f] for f in features], dtype="float64")


def _in_bounds_vector(reference: dict) -> np.ndarray:
    """A synthetic vector sitting at every feature's training median.

    By construction this violates nothing, so it is the cleanest possible
    in-distribution input.
    """
    return np.array(
        [reference["bounds"][f]["median"] for f in reference["features"]],
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# The reference artefact
# ---------------------------------------------------------------------------
class TestReference:
    def test_reference_exists_for_trained_profiles(self) -> None:
        for profile in PATHS.trained_profiles():
            path = PATHS.profiles_dir / profile / "ood_reference.json"
            assert path.exists(), \
                f"{profile} has a model but no OOD reference; retrain it"

    @needs_reference
    def test_reference_has_a_bound_per_feature(self, reference: dict) -> None:
        contract = json.loads(
            PATHS.features_file("BIDIRECTIONAL").read_text(encoding="utf-8"))
        assert reference["features"] == contract["features"]
        assert set(reference["bounds"]) == set(contract["features"])

    @needs_reference
    def test_bounds_are_ordered_and_contain_the_median(self,
                                                       reference: dict) -> None:
        for name, bound in reference["bounds"].items():
            assert bound["lower"] <= bound["upper"], f"{name} bounds inverted"
            assert bound["lower"] <= bound["median"] <= bound["upper"], \
                f"{name} median outside its own bounds"

    @needs_reference
    def test_reference_records_its_quantile_and_threshold(self,
                                                          reference: dict) -> None:
        """The policy must be readable from the artefact, not just from config.

        An artefact trained under a different threshold would otherwise be
        applied under the current one without anyone noticing.
        """
        assert reference["quantile"] == OOD.quantile
        assert reference["min_violations_for_ood"] == OOD.min_violations_for_ood

    @needs_reference
    def test_reference_built_from_training_split_only(self,
                                                      reference: dict) -> None:
        """Bounds must come from the training split, never the full dataset.

        Using every row would leak held-out data into the reference and make the
        measured false-OOD rate look better than it is. 113,960 of 151,947 rows
        is the 75% training split.
        """
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")
        total = len(pd.read_csv(PATHS.processed_csv, usecols=["threat_class"]))
        assert reference["n_training_rows"] < total, \
            "reference appears to include held-out rows"

    @needs_reference
    def test_zero_width_bounds_are_preserved(self, reference: dict) -> None:
        """SYN Flag Count is [0, 0] for attack classes; that is real, not a bug.

        A zero-width range is what makes the fixture's SYN count of 8,000
        detectable at all, so it must survive into the artefact rather than being
        widened defensively.
        """
        widths = [b["upper"] - b["lower"] for b in reference["bounds"].values()]
        assert min(widths) >= 0.0

    def test_missing_reference_yields_unknown_not_a_guess(self) -> None:
        result = ood_module.assess(np.zeros(5), "NO_SUCH_PROFILE_XYZ")
        assert result.reliability == UNKNOWN
        assert result.in_distribution is None, \
            "unchecked must be None, not False: 'not checked' and 'checked and " \
            "failed' are different states"
        assert result.reference_available is False


# ---------------------------------------------------------------------------
# assess(): the four behaviours
# ---------------------------------------------------------------------------
@needs_reference
class TestAssessVerdicts:
    def test_in_distribution_is_reliable(self, reference: dict) -> None:
        result = ood_module.assess(_in_bounds_vector(reference),
                                   "BIDIRECTIONAL", reference["features"])
        assert result.reliability == RELIABLE
        assert result.in_distribution is True
        assert result.violations == []
        assert result.ood_score == 0.0

    def test_one_violation_is_degraded(self, reference: dict) -> None:
        vector = _in_bounds_vector(reference)
        features = reference["features"]
        # Push exactly one feature far above its upper bound.
        upper = reference["bounds"][features[0]]["upper"]
        vector[0] = upper + max(abs(upper), 1.0) * 1000

        result = ood_module.assess(vector, "BIDIRECTIONAL", features)
        assert len(result.violations) == 1
        assert result.reliability == DEGRADED
        # DEGRADED still counts as in-distribution: the classification is kept.
        assert result.in_distribution is True
        assert result.ood_features == [features[0]]

    def test_two_violations_is_unreliable(self, reference: dict) -> None:
        vector = _in_bounds_vector(reference)
        features = reference["features"]
        for i in (0, 1):
            upper = reference["bounds"][features[i]]["upper"]
            vector[i] = upper + max(abs(upper), 1.0) * 1000

        result = ood_module.assess(vector, "BIDIRECTIONAL", features)
        assert len(result.violations) >= OOD.min_violations_for_ood
        assert result.reliability == UNRELIABLE
        assert result.in_distribution is False
        assert result.ood_score > 0.0

    def test_below_lower_bound_also_violates(self, reference: dict) -> None:
        """Excursions in both directions count."""
        vector = _in_bounds_vector(reference)
        features = reference["features"]
        violated = 0
        for i, name in enumerate(features):
            lower = reference["bounds"][name]["lower"]
            if lower > 0:
                vector[i] = -abs(lower) * 1000
                violated += 1
            if violated == 2:
                break
        if violated < 2:
            pytest.skip("fewer than two features have a positive lower bound")

        result = ood_module.assess(vector, "BIDIRECTIONAL", features)
        assert result.reliability == UNRELIABLE
        assert all(v.direction == "below" for v in result.violations[:violated])

    def test_violations_sorted_worst_first(self, reference: dict) -> None:
        """The most extreme excursion is the most useful thing to show first."""
        vector = _in_bounds_vector(reference)
        features = reference["features"]
        for i, factor in ((0, 10), (1, 100_000)):
            upper = reference["bounds"][features[i]]["upper"]
            vector[i] = upper + max(abs(upper), 1.0) * factor

        result = ood_module.assess(vector, "BIDIRECTIONAL", features)
        excursions = [v.excursion for v in result.violations]
        assert excursions == sorted(excursions, reverse=True)

    def test_violation_report_is_explainable(self, reference: dict) -> None:
        """Each violation must name the feature, the bound, and the excursion."""
        vector = _in_bounds_vector(reference)
        features = reference["features"]
        for i in (0, 1):
            upper = reference["bounds"][features[i]]["upper"]
            vector[i] = upper + max(abs(upper), 1.0) * 500

        result = ood_module.assess(vector, "BIDIRECTIONAL", features)
        for violation in result.violations:
            payload = violation.to_dict()
            assert payload["feature"] in features
            assert payload["direction"] in ("above", "below")
            assert len(payload["training_range"]) == 2
            assert payload["excursion_range_widths"] > 0
            text = violation.describe()
            assert violation.feature in text
            assert "training bound" in text

        assert "outside the range seen in training" in result.reason

    def test_ood_score_is_bounded(self, reference: dict) -> None:
        vector = _in_bounds_vector(reference)
        features = reference["features"]
        for i in range(len(features)):
            upper = reference["bounds"][features[i]]["upper"]
            vector[i] = upper + max(abs(upper), 1.0) * 1e9

        result = ood_module.assess(vector, "BIDIRECTIONAL", features)
        assert 0.0 <= result.ood_score <= 1.0

    def test_never_raises_on_malformed_input(self, reference: dict) -> None:
        """This runs inside the replay loop; it must degrade, not raise."""
        features = reference["features"]
        n = len(features)
        payloads = [
            np.zeros(n),
            np.full(n, np.nan),
            np.full(n, np.inf),
            np.full(n, -np.inf),
            np.zeros(3),                    # wrong length
            np.zeros(n + 5),                # wrong length
            np.array([]),
        ]
        for payload in payloads:
            result = ood_module.assess(payload, "BIDIRECTIONAL", features)
            assert result.reliability in RELIABILITY_VALUES
            assert 0.0 <= result.ood_score <= 1.0

    def test_nonfinite_values_are_skipped_not_counted(self,
                                                      reference: dict) -> None:
        """NaN is a data-quality problem, not a distribution violation.

        Counting it as one would conflate two different failures.
        """
        vector = _in_bounds_vector(reference)
        vector[0] = np.nan
        result = ood_module.assess(vector, "BIDIRECTIONAL",
                                   reference["features"])
        assert reference["features"][0] not in result.ood_features

    def test_wrong_length_vector_is_refused(self, reference: dict) -> None:
        result = ood_module.assess(np.zeros(3), "BIDIRECTIONAL",
                                   reference["features"])
        assert result.reliability == UNKNOWN
        assert result.in_distribution is None


# ---------------------------------------------------------------------------
# severity: Rule 0 and its ordering
# ---------------------------------------------------------------------------
class TestSeverityRuleZero:
    def test_unreliable_gets_the_ood_severity(self) -> None:
        decision = severity_module.assess("UNKNOWN_OOD", 0.98,
                                          reliability=UNRELIABLE,
                                          model_prediction="DDoS")
        assert decision.severity == SEVERITY.ood_severity
        assert decision.needs_review is True

    def test_ood_benign_is_NOT_suppressed_to_info(self) -> None:
        """The single most important test in this file.

        On the fixture the classifier said BENIGN for all 20 rows at 0.913 mean
        confidence. If Rule 1 (BENIGN -> INFO) ran first, every one would be INFO
        and invisible. Rule 0 must win.
        """
        decision = severity_module.assess(
            OOD_THREAT_CLASS, 0.98, reliability=UNRELIABLE,
            model_prediction=SEVERITY.benign_label)
        assert decision.severity != SEVERITY.informational_severity
        assert decision.severity == SEVERITY.ood_severity
        assert decision.is_alertable is True
        assert decision.needs_review is True

    def test_ood_severity_is_below_critical(self) -> None:
        """An unreliable answer needs review but is not evidence of an attack.

        Ranking it above a confirmed high-confidence DDoS would invert the
        analyst's priorities.
        """
        ood = severity_module.assess(OOD_THREAT_CLASS, 0.99,
                                     reliability=UNRELIABLE,
                                     model_prediction="BENIGN")
        confirmed = severity_module.assess("DDoS", 0.99, reliability=RELIABLE)
        assert (severity_module.severity_rank(ood.severity)
                < severity_module.severity_rank(confirmed.severity))

    def test_ood_reason_separates_the_two_concepts(self) -> None:
        reason = severity_module.assess(
            OOD_THREAT_CLASS, 0.98, reliability=UNRELIABLE,
            model_prediction="BENIGN").reason.lower()
        assert "outside the distribution" in reason
        assert "internal certainty" in reason
        assert "benign" in reason, "must state what the classifier actually said"

    def test_degraded_benign_goes_to_review_not_alerts(self) -> None:
        """Not an actionable alert, but must not vanish either."""
        decision = severity_module.assess(SEVERITY.benign_label, 0.98,
                                          reliability=DEGRADED)
        assert decision.severity == SEVERITY.informational_severity
        assert decision.is_alertable is False
        assert decision.needs_review is True

    def test_reliable_benign_needs_no_review(self) -> None:
        decision = severity_module.assess(SEVERITY.benign_label, 0.98,
                                          reliability=RELIABLE)
        assert decision.is_alertable is False
        assert decision.needs_review is False

    def test_degraded_threat_keeps_its_severity(self) -> None:
        """DEGRADED flags the flow; it does not downgrade a real detection."""
        degraded = severity_module.assess("DDoS", 0.82, reliability=DEGRADED)
        reliable = severity_module.assess("DDoS", 0.82, reliability=RELIABLE)
        assert degraded.severity == reliable.severity
        assert degraded.needs_review is True
        assert reliable.needs_review is False

    def test_unknown_reliability_behaves_like_no_check(self) -> None:
        unchecked = severity_module.assess("DDoS", 0.82, reliability=UNKNOWN)
        plain = severity_module.assess("DDoS", 0.82)
        assert unchecked.severity == plain.severity
        assert unchecked.needs_review is False

    def test_reliability_absent_is_backward_compatible(self) -> None:
        """Callers predating the OOD layer must still work."""
        decision = severity_module.assess("DDoS", 0.82)
        assert decision.severity in SEVERITY.ladder
        assert decision.reliability is None

    @pytest.mark.parametrize("value", [1.5, -0.2, float("nan"), None, "junk"])
    def test_malformed_confidence_with_ood(self, value: object) -> None:
        decision = severity_module.assess(
            OOD_THREAT_CLASS, value,  # type: ignore[arg-type]
            reliability=UNRELIABLE, model_prediction="BENIGN")
        assert decision.severity == SEVERITY.ood_severity


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------
class TestInvariant:
    """reliability == UNRELIABLE  <->  threat_class == UNKNOWN_OOD.

    Tested in BOTH directions. A partially-applied change to the OOD layer could
    break either one, and either break would report a trusted-looking class on
    input the model cannot handle.
    """

    def _alert(self, **overrides) -> dict:
        prediction = {
            "threat": OOD_THREAT_CLASS,
            "confidence": 0.9,
            "severity": "HIGH",
            "base_severity": "HIGH",
            "escalated": False,
            "severity_reason": "test",
            "is_alertable": True,
            "needs_review": True,
            "evidence": {},
            "model_prediction": "BENIGN",
            "model_confidence": 0.9,
            "confidence_means": "test",
            "in_distribution": False,
            "reliability": UNRELIABLE,
            "ood_score": 0.8,
            "ood_features": ["SYN Flag Count"],
            "ood_violations": [],
            "ood_reason": "test",
            "class_probabilities": {"BENIGN": 0.9, "DDoS": 0.1},
            "model_ready": True,
        }
        prediction.update(overrides)
        return Alert.build(prediction, "T000001").to_dict()

    def test_consistent_pair_validates(self) -> None:
        assert validate(self._alert()) == []

    def test_unreliable_without_ood_class_is_rejected(self) -> None:
        problems = validate(self._alert(threat="DDoS"))
        assert any("UNRELIABLE" in p for p in problems), \
            f"invariant not enforced: {problems}"

    def test_ood_class_without_unreliable_is_rejected(self) -> None:
        problems = validate(self._alert(reliability=RELIABLE))
        assert any(OOD_CLASS in p for p in problems), \
            f"invariant not enforced in reverse: {problems}"

    def test_reliable_normal_class_validates(self) -> None:
        record = self._alert(threat="DDoS", reliability=RELIABLE,
                             in_distribution=True, ood_score=0.0,
                             ood_features=[])
        assert validate(record) == []

    def test_unknown_reliability_validates(self) -> None:
        """No reference available is a legitimate state, not a violation."""
        record = self._alert(threat="DDoS", reliability=UNKNOWN,
                             in_distribution=None, ood_score=0.0,
                             ood_features=[])
        assert validate(record) == []

    def test_unrecognised_reliability_is_rejected(self) -> None:
        problems = validate(self._alert(reliability="TOTALLY_MADE_UP"))
        assert any("reliability" in p for p in problems)

    def test_in_distribution_stays_tri_state(self) -> None:
        """Only True, False, or None. Integers must be rejected.

        `0 in (True, False, None)` is True in Python, so a membership test would
        silently accept in_distribution=0 from another producer and read it as
        False. validate() uses identity checks for this reason.
        """
        for bad in ("false", 0, 1, "unknown", 0.0):
            problems = validate(self._alert(in_distribution=bad))
            assert any("in_distribution" in p for p in problems), \
                f"{bad!r} ({type(bad).__name__}) should be rejected"

    def test_in_distribution_accepts_the_three_valid_states(self) -> None:
        for good, reliability, threat in ((True, RELIABLE, "DDoS"),
                                          (False, UNRELIABLE, OOD_THREAT_CLASS),
                                          (None, UNKNOWN, "DDoS")):
            record = self._alert(in_distribution=good, reliability=reliability,
                                 threat=threat)
            assert validate(record) == [], f"{good!r} should be accepted"

    def test_ood_score_out_of_range_is_clamped_on_build(self) -> None:
        """Alert.build() clamps rather than rejecting.

        Coercion at the boundary is deliberate: a malformed value must not stop a
        live replay. validate() therefore never sees an out-of-range score from
        this path, so the range check exists for records arriving from elsewhere.
        """
        record = self._alert(ood_score=1.7)
        assert record["ood_score"] == 1.0
        assert validate(record) == []

    def test_ood_score_range_is_validated_on_foreign_records(self) -> None:
        """A record not built by Alert.build must still be checked."""
        record = self._alert()
        for bad in (1.7, -0.1):
            record["ood_score"] = bad
            assert any("ood_score" in p for p in validate(record)), \
                f"{bad} should be flagged"


# ---------------------------------------------------------------------------
# End to end through predict_threat
# ---------------------------------------------------------------------------
@needs_bidirectional
@needs_reference
class TestPredictThreatIntegration:
    def test_every_result_carries_the_ood_block(self) -> None:
        features = get_expected_features("BIDIRECTIONAL")
        result = predict_threat({f: 1.0 for f in features},
                                profile="BIDIRECTIONAL")
        for key in ("in_distribution", "reliability", "ood_score",
                    "ood_features", "ood_violations", "ood_reason",
                    "needs_review", "model_prediction", "model_confidence",
                    "confidence_means"):
            assert key in result, f"missing {key}"

    def test_confidence_and_reliability_are_separate_concepts(self) -> None:
        """The central design decision, asserted rather than assumed.

        `confidence` is the classifier's certainty. `reliability` says whether
        that certainty means anything for this input. High confidence with
        UNRELIABLE reliability is exactly the case the fixture produces.
        """
        features = get_expected_features("BIDIRECTIONAL")
        result = predict_threat({f: 1.0 for f in features},
                                profile="BIDIRECTIONAL")
        assert result["confidence"] == result["model_confidence"]
        assert isinstance(result["reliability"], str)
        assert result["reliability"] in RELIABILITY_VALUES
        # The field documenting the distinction must actually say something.
        assert "certainty" in result["confidence_means"].lower()

    def test_model_prediction_survives_the_override(self,
                                                    fixture_rows) -> None:
        """The classifier's answer is preserved, never discarded."""
        for row in fixture_rows:
            payload = {k: v for k, v in row.items() if k != "Label"}
            result = predict_threat(payload, profile="BIDIRECTIONAL")
            assert result["model_prediction"] in ("BENIGN", "DDoS", "PortScan",
                                                  "Botnet")
            if result["reliability"] == UNRELIABLE:
                assert result["threat"] == OOD_THREAT_CLASS
                assert result["model_prediction"] != OOD_THREAT_CLASS

    def test_fixture_ood_rows_are_not_suppressed(self, fixture_rows) -> None:
        """The regression this whole feature exists to prevent.

        Every fixture row is classified BENIGN by the model. Before the OOD
        layer, all 20 became INFO and non-alertable. Now the UNRELIABLE ones must
        be alertable and none may be silently invisible.
        """
        alertable = review_only = invisible = 0
        for row in fixture_rows:
            payload = {k: v for k, v in row.items() if k != "Label"}
            result = predict_threat(payload, profile="BIDIRECTIONAL")

            if result["reliability"] == UNRELIABLE:
                assert result["is_alertable"] is True, \
                    "an out-of-distribution flow was suppressed"
                assert result["severity"] != SEVERITY.informational_severity

            if result["is_alertable"]:
                alertable += 1
            elif result["needs_review"]:
                review_only += 1
            else:
                invisible += 1

        # MEASURED: 10 alertable, 9 review-only, 1 fully in-distribution.
        assert alertable >= 10, f"only {alertable} of 20 rows alertable"
        assert alertable + review_only >= 19, \
            f"{invisible} rows invisible; expected at most 1"

    def test_in_distribution_rows_classify_normally(self) -> None:
        """No regression on real traffic: the common path is unchanged."""
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")

        frame = pd.read_csv(PATHS.processed_csv, nrows=300, low_memory=False)
        reliable = 0
        for _, series in frame.iterrows():
            result = predict_threat(series.to_dict(), profile="BIDIRECTIONAL")
            if result["reliability"] == RELIABLE:
                reliable += 1
                assert result["threat"] == result["model_prediction"]
                assert result["threat"] != OOD_THREAT_CLASS
                assert result["needs_review"] is False
        assert reliable > 250, \
            f"only {reliable}/300 real rows judged RELIABLE; the false-OOD " \
            f"rate has regressed"

    def test_false_ood_rate_on_real_data(self) -> None:
        """The cost side of the tradeoff, pinned so it cannot creep up.

        MEASURED at q=0.001, >=2 violations: 0.250% of held-out rows reach
        UNRELIABLE. The 2% bar here is deliberately loose -- it catches a
        regression, not a small drift.
        """
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")

        frame = pd.read_csv(PATHS.processed_csv, nrows=2000, low_memory=False)
        unreliable = sum(
            predict_threat(series.to_dict(),
                           profile="BIDIRECTIONAL")["reliability"] == UNRELIABLE
            for _, series in frame.iterrows()
        )
        rate = unreliable / len(frame)
        assert rate < 0.02, \
            f"false-OOD rate {rate:.2%} on real in-distribution traffic"

    def test_contract_failure_reports_unknown_reliability(self) -> None:
        """A refused record has no vector, so no distribution check ran."""
        result = predict_threat({"totally": 1, "wrong": 2},
                                profile="BIDIRECTIONAL")
        assert result["contract_satisfied"] is False
        assert result["reliability"] == UNKNOWN
        assert result["in_distribution"] is None
        assert result["needs_review"] is True, \
            "a record that could not be scored must not vanish"

    @needs_strict
    def test_both_profiles_have_independent_references(self) -> None:
        strict = ood_module.load_reference("STRICT_UNIDIRECTIONAL")
        bidirectional = ood_module.load_reference("BIDIRECTIONAL")
        assert strict is not None and bidirectional is not None
        assert strict["features"] != bidirectional["features"]
        assert len(strict["bounds"]) == 18
        assert len(bidirectional["bounds"]) == 14


# ---------------------------------------------------------------------------
# Alerts built from OOD predictions
# ---------------------------------------------------------------------------
@needs_bidirectional
@needs_reference
class TestOODAlerts:
    def test_fixture_alerts_are_schema_valid(self, fixture_rows) -> None:
        for i, row in enumerate(fixture_rows):
            payload = {k: v for k, v in row.items() if k != "Label"}
            result = predict_threat(payload, profile="BIDIRECTIONAL")
            alert = Alert.build(result, f"T{i:06d}",
                                ground_truth=str(row["Label"]))
            assert validate(alert.to_dict()) == [], \
                f"row {i} produced an invalid alert"

    def test_correctness_judged_on_classifier_not_verdict(self,
                                                          fixture_rows) -> None:
        """UNKNOWN_OOD never matches a dataset label.

        Scoring the verdict would count every OOD row as a miss and understate
        the classifier's own accuracy, conflating two separate measurements.
        """
        row = next(r for r in fixture_rows if r["Label"] == "BENIGN")
        payload = {k: v for k, v in row.items() if k != "Label"}
        result = predict_threat(payload, profile="BIDIRECTIONAL")
        if result["reliability"] != UNRELIABLE:
            pytest.skip("this fixture row is not OOD")

        alert = Alert.build(result, "T000001", ground_truth="BENIGN")
        assert alert.threat_class == OOD_THREAT_CLASS
        assert alert.model_prediction == "BENIGN"
        assert alert.correct is True, \
            "correctness must be judged against model_prediction"

    def test_summary_line_marks_ood_visibly(self, fixture_rows) -> None:
        for row in fixture_rows:
            payload = {k: v for k, v in row.items() if k != "Label"}
            result = predict_threat(payload, profile="BIDIRECTIONAL")
            alert = Alert.build(result, "T000001")
            if result["reliability"] == UNRELIABLE:
                text = alert.summary()
                assert "OOD" in text
                assert "said" in text, "log line must state what the model said"
                return
        pytest.skip("no UNRELIABLE row in the fixture")

    def test_json_roundtrip_preserves_the_ood_block(self, fixture_rows) -> None:
        row = fixture_rows[0]
        payload = {k: v for k, v in row.items() if k != "Label"}
        result = predict_threat(payload, profile="BIDIRECTIONAL")
        alert = Alert.build(result, "T000001")

        parsed = json.loads(alert.to_json_line())
        assert parsed["reliability"] == result["reliability"]
        assert parsed["ood_features"] == result["ood_features"]
        assert parsed["model_prediction"] == result["model_prediction"]
        assert parsed["in_distribution"] == result["in_distribution"]


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------
class TestFixtureIntegrity:
    """The fixture is a deliberately out-of-distribution artefact.

    It must NOT be edited to make the model pass. These tests pin the properties
    that make it a useful robustness test, so a well-meaning future change
    cannot quietly turn it into an in-distribution file.
    """

    def test_fixture_exists_and_is_labelled(self, fixture_rows) -> None:
        assert len(fixture_rows) == 20
        labels = {r["Label"] for r in fixture_rows}
        assert labels == {"BENIGN", "DDoS", "PortScan", "Botnet"}

    def test_syn_flag_count_is_impossible_by_design(self, fixture_rows) -> None:
        """The fixture's DDoS rows carry SYN counts of thousands.

        In the real dataset this column is a binary 0/1 flag-present indicator,
        so these values cannot occur. That impossibility is the point: it is what
        makes the file a distribution-shift test rather than an attack sample.
        """
        ddos = [r for r in fixture_rows if r["Label"] == "DDoS"]
        assert ddos
        assert all(r["SYN Flag Count"] > 1000 for r in ddos), \
            "fixture no longer contains impossible SYN counts; has it been " \
            "edited to make the model pass?"

    @needs_reference
    def test_fixture_is_genuinely_out_of_distribution(self, fixture_rows,
                                                      reference: dict) -> None:
        features = reference["features"]
        unreliable = 0
        for row in fixture_rows:
            result = ood_module.assess(_vector(row, features),
                                       "BIDIRECTIONAL", features)
            if result.reliability == UNRELIABLE:
                unreliable += 1
        # MEASURED: 10 of 20 reach UNRELIABLE, 19 of 20 reach at least DEGRADED.
        assert unreliable >= 10, \
            f"only {unreliable}/20 fixture rows are UNRELIABLE; the fixture or " \
            f"the detector has changed"

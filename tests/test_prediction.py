"""
Tests for the prediction contract.

Run:
    python -m pytest tests/ -v

These tests cover the interface Student B depends on. If any of them fail, the
dashboard will misbehave, so they are the ones to run before a demo.

Tests that need a trained model are skipped, not failed, when model.pkl is
absent -- a fresh clone has no artefacts and `pytest` should still be green.
Run `python -m model.train` first to exercise the full suite.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS, SEVERITY  # noqa: E402
from detection import severity as severity_module  # noqa: E402
from detection.alert_schema import (  # noqa: E402
    REQUIRED_KEYS,
    SCHEMA_VERSION,
    Alert,
    parse_alert_line,
    validate,
)
from detection.evidence import build_evidence  # noqa: E402
from model.predict import (  # noqa: E402
    get_expected_features,
    is_model_ready,
    model_status,
    predict_threat,
    reset_cache,
)
from preprocessing.feature_config import (  # noqa: E402
    FEATURE_PROFILES,
    TARGET_CLASSES,
    get_features,
    map_label,
    normalize_column,
    protocol_name,
)

# Keys every predict_threat() result must carry, trained or not.
CONTRACT_KEYS = {
    "threat", "confidence", "severity", "evidence",
    "base_severity", "escalated", "severity_reason", "is_alertable",
    "evidence_detail", "evidence_method", "aggregate_context",
    "class_probabilities", "model_ready", "missing_features", "profile",
}

needs_model = pytest.mark.skipif(
    not is_model_ready(),
    reason="no trained model; run: python -m model.train",
)


# ---------------------------------------------------------------------------
# Column normalisation -- the CIC-IDS2017 whitespace problem
# ---------------------------------------------------------------------------
class TestNormalization:
    """The dataset's headers are inconsistently spaced; verified on disk."""

    @pytest.mark.parametrize("raw,expected", [
        (" Destination Port", "Destination Port"),
        (" Flow Duration", "Flow Duration"),
        ("Total Length of Fwd Packets", "Total Length of Fwd Packets"),
        ("Flow Bytes/s", "Flow Bytes/s"),
        (" Flow Packets/s", "Flow Packets/s"),
        ("  Fwd IAT Mean  ", "Fwd IAT Mean"),
        ("\ufeff Flow Duration", "Flow Duration"),
        ("Flow\xa0Duration", "Flow Duration"),
        ("Flow  Duration", "Flow Duration"),
    ])
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_column(raw) == expected

    def test_idempotent(self) -> None:
        for name in (" Flow Duration", "Flow Bytes/s", " ACK Flag Count"):
            once = normalize_column(name)
            assert normalize_column(once) == once

    def test_duplicate_suffix_preserved(self) -> None:
        # 'Fwd Header Length' appears twice in the real files; pandas renames
        # the second to '.1'. Normalisation must keep the suffix so the
        # duplicate can be dropped by name.
        assert normalize_column(" Fwd Header Length.1") == "Fwd Header Length.1"


class TestLabelMapping:
    """Dataset labels differ from the project's class names."""

    @pytest.mark.parametrize("raw,expected", [
        ("BENIGN", "BENIGN"),
        ("DDoS", "DDoS"),
        ("PortScan", "PortScan"),
        ("Bot", "Botnet"),          # the rename that matters
        (" BENIGN ", "BENIGN"),     # stray whitespace
    ])
    def test_mapped(self, raw: str, expected: str) -> None:
        assert map_label(raw) == expected

    @pytest.mark.parametrize("raw", [
        "DoS Hulk", "FTP-Patator", "Heartbleed", "Infiltration",
        "Web Attack \x96 XSS",      # latin-1 en-dash, as stored on disk
    ])
    def test_out_of_scope_dropped(self, raw: str) -> None:
        assert map_label(raw) is None

    def test_unknown_label_returns_none_not_raises(self) -> None:
        # A new CSV with an unexpected label must shrink the dataset, not crash
        # the pipeline.
        assert map_label("SomeFutureAttack2030") is None
        assert map_label(None) is None

    def test_all_target_classes_reachable(self) -> None:
        mapped = {map_label(k) for k in
                  ("BENIGN", "DDoS", "PortScan", "Bot")}
        assert mapped == set(TARGET_CLASSES)


class TestProtocolNames:
    @pytest.mark.parametrize("value,expected", [
        (6, "TCP"), (17, "UDP"), (0, "OTHER"), (1, "ICMP"),
        (6.0, "TCP"), ("6", "TCP"),
        (None, "UNKNOWN"), ("garbage", "UNKNOWN"), (float("nan"), "UNKNOWN"),
    ])
    def test_protocol_name(self, value: object, expected: str) -> None:
        assert protocol_name(value) == expected


# ---------------------------------------------------------------------------
# Feature profiles
# ---------------------------------------------------------------------------
class TestFeatureProfiles:
    def test_both_profiles_defined(self) -> None:
        assert "STRICT_UNIDIRECTIONAL" in FEATURE_PROFILES
        assert "BIDIRECTIONAL" in FEATURE_PROFILES

    def test_strict_profile_excludes_backward_features(self) -> None:
        """The point of the strict profile: no reverse-path features.

        A unidirectional tap may not expose them at all, so a model trained on
        them could not run in the deployment the problem statement describes.
        """
        for feature in get_features("STRICT_UNIDIRECTIONAL"):
            lowered = feature.lower()
            assert "bwd" not in lowered, f"{feature} is a backward feature"
            assert "backward" not in lowered, f"{feature} is a backward feature"
            assert "down/up" not in lowered, f"{feature} needs both directions"

    def test_bidirectional_profile_does_contain_backward(self) -> None:
        features = get_features("BIDIRECTIONAL")
        assert any("Bwd" in f or "Backward" in f for f in features)

    def test_no_duplicate_features(self) -> None:
        for name in FEATURE_PROFILES:
            features = get_features(name)
            assert len(features) == len(set(features))

    def test_features_are_normalized(self) -> None:
        for name in FEATURE_PROFILES:
            for feature in get_features(name):
                assert feature == normalize_column(feature)

    def test_unknown_profile_raises_with_valid_options(self) -> None:
        with pytest.raises(ValueError, match="Unknown feature profile"):
            get_features("NOT_A_PROFILE")


# ---------------------------------------------------------------------------
# Severity policy
# ---------------------------------------------------------------------------
class TestSeverity:
    @pytest.mark.parametrize("confidence,expected", [
        (0.00, "LOW"), (0.49, "LOW"),
        (0.50, "MEDIUM"), (0.74, "MEDIUM"),
        (0.75, "HIGH"), (0.89, "HIGH"),
        (0.90, "CRITICAL"), (1.00, "CRITICAL"),
    ])
    def test_bands(self, confidence: float, expected: str) -> None:
        assert severity_module.base_band(confidence) == expected

    def test_bands_have_no_gaps(self) -> None:
        """Every confidence in [0, 1] must fall in exactly one band."""
        for step in range(0, 1001):
            band = severity_module.base_band(step / 1000)
            assert band in SEVERITY.ladder

    @pytest.mark.parametrize("value", [1.5, -0.2, float("nan"), None, "junk"])
    def test_malformed_confidence_never_raises(self, value: object) -> None:
        decision = severity_module.assess("DDoS", value)  # type: ignore[arg-type]
        assert decision.severity in SEVERITY.ladder

    def test_benign_is_never_alertable(self) -> None:
        """A high-confidence BENIGN must not appear in the alert queue."""
        for confidence in (0.1, 0.5, 0.9, 1.0):
            decision = severity_module.assess("BENIGN", confidence)
            assert decision.severity == "INFO"
            assert decision.is_alertable is False

    def test_ddos_escalates_one_band(self) -> None:
        decision = severity_module.assess("DDoS", 0.80)
        assert decision.base_severity == "HIGH"
        assert decision.severity == "CRITICAL"
        assert decision.escalated is True

    def test_botnet_escalates_one_band(self) -> None:
        decision = severity_module.assess("Botnet", 0.55)
        assert decision.base_severity == "MEDIUM"
        assert decision.severity == "HIGH"

    def test_portscan_does_not_escalate(self) -> None:
        """Reconnaissance is not escalated; only realised impact is."""
        decision = severity_module.assess("PortScan", 0.80)
        assert decision.severity == decision.base_severity == "HIGH"
        assert decision.escalated is False

    def test_escalation_cannot_exceed_critical(self) -> None:
        decision = severity_module.assess("DDoS", 0.99)
        assert decision.severity == "CRITICAL"
        assert decision.escalated is False  # already at the top

    def test_escalation_applied_at_most_once(self) -> None:
        for confidence in (0.1, 0.4, 0.6, 0.8):
            decision = severity_module.assess("DDoS", confidence)
            base_index = SEVERITY.ladder.index(decision.base_severity)
            final_index = SEVERITY.ladder.index(decision.severity)
            assert final_index - base_index <= 1

    def test_reason_is_populated(self) -> None:
        decision = severity_module.assess("DDoS", 0.82)
        assert len(decision.reason) > 30
        assert "82%" in decision.reason

    def test_rank_orders_ladder(self) -> None:
        ranks = [severity_module.severity_rank(s) for s in SEVERITY.ladder]
        assert ranks == sorted(ranks)
        assert severity_module.severity_rank("NOT_A_SEVERITY") == -1


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
class TestEvidence:
    def test_every_class_produces_evidence(self) -> None:
        flows = {
            "PortScan": {"Fwd Packet Length Mean": 0.0, "Flow Duration": 50.0,
                         "Fwd Packets/s": 20000.0, "Total Fwd Packets": 1.0,
                         "Init_Win_bytes_forward": 29200.0,
                         "Destination Port": 3527.0},
            "DDoS": {"Fwd Packet Length Mean": 7.0, "Flow Duration": 1876595.0,
                     "Fwd IAT Std": 908248.0,
                     "Init_Win_bytes_forward": 256.0},
            "Botnet": {"Init_Win_bytes_forward": 8192.0,
                       "Destination Port": 8080.0, "Fwd IAT Std": 1653.0,
                       "act_data_pkt_fwd": 0.0,
                       "Fwd Packet Length Mean": 6.0},
            "BENIGN": {"Fwd Packet Length Mean": 39.0,
                       "Flow Duration": 48699.0, "Fwd Packets/s": 39.6},
        }
        for threat, flow in flows.items():
            evidence = build_evidence(threat, flow)
            assert evidence.items, f"no evidence for {threat}"
            for item in evidence.items:
                assert item["feature"] in flow
                assert item["display"]
                assert item["why"]

    def test_evidence_never_raises(self) -> None:
        for threat in ("DDoS", "UNKNOWN", "", "NotAClass"):
            for flow in ({}, {"junk": "x"},
                         {"Flow Duration": float("nan")},
                         {"Flow Duration": None}):
                evidence = build_evidence(threat, flow)  # type: ignore[arg-type]
                assert isinstance(evidence.items, list)

    def test_method_does_not_overclaim(self) -> None:
        """The project must not claim SHAP-style attribution it does not do."""
        evidence = build_evidence("DDoS", {"Fwd Packet Length Mean": 7.0})
        method = evidence.method.lower()
        assert "not" in method and "shap" in method

    def test_aggregate_context_present_for_flow_level_gaps(self) -> None:
        """DDoS and scans are aggregate phenomena; the alert must say so."""
        for threat in ("DDoS", "PortScan", "Botnet"):
            evidence = build_evidence(threat, {"Flow Duration": 1000.0})
            assert evidence.aggregate_note
            assert "independently" in evidence.aggregate_note

    def test_contradictory_values_are_not_asserted(self) -> None:
        """A benign-looking value must not be described as attack-like."""
        # A benign-typical duration presented under a DDoS classification.
        evidence = build_evidence("DDoS", {"Flow Duration": 48699.0,
                                           "Init_Win_bytes_forward": 256.0})
        for item in evidence.items:
            if item["feature"] == "Flow Duration":
                assert "held open" not in item["why"].lower()
                assert "context" in item["why"].lower()

    def test_port_comparison_is_categorical_not_ratio(self) -> None:
        """'Port 3527 is 44x the median of 80' is meaningless; must not appear."""
        evidence = build_evidence("PortScan", {"Destination Port": 3527.0})
        for item in evidence.items:
            if item["feature"] == "Destination Port":
                assert "x the benign median" not in (item["comparison"] or "")
                assert "port" in (item["comparison"] or "").lower()


# ---------------------------------------------------------------------------
# predict_threat -- the contract Student B calls
# ---------------------------------------------------------------------------
class TestPredictContract:
    """These tests must pass whether or not a model is trained."""

    def test_returns_all_contract_keys(self) -> None:
        result = predict_threat({"Flow Duration": 1000.0})
        assert CONTRACT_KEYS.issubset(result.keys())

    def test_never_raises_on_malformed_input(self) -> None:
        payloads = [
            {}, None, [], "string", 42, 3.14,
            {"junk": "value"},
            {" Flow Duration": 100},            # un-normalised header
            {"Flow Duration": None},
            {"Flow Duration": "not a number"},
            {"Flow Duration": float("nan")},
            {"Flow Duration": float("inf")},
            {"Flow Duration": float("-inf")},
            {"Flow Duration": -999},
            {"Flow Duration": 1e300},
            {"Flow Duration": [1, 2, 3]},
            {"Flow Duration": {"nested": "dict"}},
        ]
        for payload in payloads:
            result = predict_threat(payload)  # type: ignore[arg-type]
            assert isinstance(result, dict)
            assert CONTRACT_KEYS.issubset(result.keys())

    def test_confidence_always_in_unit_range(self) -> None:
        for payload in ({}, {"Flow Duration": 1}, {"junk": 1}):
            confidence = predict_threat(payload)["confidence"]
            assert 0.0 <= confidence <= 1.0

    def test_severity_always_valid(self) -> None:
        for payload in ({}, {"Flow Duration": 1}, {"junk": 1}):
            assert predict_threat(payload)["severity"] in SEVERITY.ladder

    def test_evidence_is_always_a_dict(self) -> None:
        for payload in ({}, {"Flow Duration": 1}, {"junk": 1}):
            assert isinstance(predict_threat(payload)["evidence"], dict)

    def test_types_are_json_serialisable(self) -> None:
        """The replay engine writes results as JSON; numpy types would break it."""
        result = predict_threat({"Flow Duration": 1000.0})
        json.dumps(result)  # must not raise
        assert isinstance(result["confidence"], float)
        assert isinstance(result["threat"], str)
        assert isinstance(result["escalated"], bool)


@needs_model
class TestPredictWithModel:
    """Tests requiring model.pkl, features.json and label_encoder.pkl."""

    def test_status_reports_ready(self) -> None:
        status = model_status()
        assert status["model_ready"] is True
        assert status["n_features"] == len(status["features"])
        assert status["classes"]

    def test_feature_order_matches_features_json(self) -> None:
        """predict.py must use the exact order train.py fitted on."""
        from config import FEATURE_PROFILE

        contract = json.loads(
            PATHS.features_file(FEATURE_PROFILE).read_text(encoding="utf-8"))
        assert get_expected_features() == contract["features"]
        assert contract["profile"] == FEATURE_PROFILE
        assert contract["feature_count"] == len(contract["features"])

    def test_feature_order_is_respected(self) -> None:
        """Two dicts with the same values in different key order must agree.

        This is the test that catches a feature-ordering bug, which would
        otherwise produce plausible but wrong predictions.
        """
        features = get_expected_features()
        values = {name: float(i + 1) for i, name in enumerate(features)}
        forward = predict_threat(values)
        reversed_order = predict_threat(dict(reversed(list(values.items()))))
        assert forward["threat"] == reversed_order["threat"]
        assert forward["confidence"] == reversed_order["confidence"]

    def test_probabilities_sum_to_one(self) -> None:
        features = get_expected_features()
        result = predict_threat({name: 1.0 for name in features})
        total = sum(result["class_probabilities"].values())
        assert math.isclose(total, 1.0, abs_tol=0.01)

    def test_confidence_is_the_max_probability(self) -> None:
        features = get_expected_features()
        result = predict_threat({name: 1.0 for name in features})
        assert math.isclose(result["confidence"],
                            max(result["class_probabilities"].values()),
                            abs_tol=0.001)

    def test_model_prediction_is_the_argmax(self) -> None:
        """`model_prediction` always holds the classifier's argmax class.

        `threat` does NOT: it is the REPORTED verdict, and the OOD layer replaces
        it with UNKNOWN_OOD when the input lies outside the training
        distribution. Separating the two is deliberate -- a synthetic row with
        SYN Flag Count = 8,000 (impossible in the real data, max is 1) still has
        an argmax, but reporting it as the verdict would present an
        extrapolation as a finding. This test pins the classifier's raw output;
        `test_threat_is_ood_when_out_of_distribution` pins the verdict.
        """
        features = get_expected_features()
        result = predict_threat({name: 5.0 for name in features})
        best = max(result["class_probabilities"].items(), key=lambda kv: kv[1])
        assert result["model_prediction"] == best[0]
        assert math.isclose(result["model_confidence"], best[1], abs_tol=0.001)

    def test_threat_equals_model_prediction_when_in_distribution(self) -> None:
        """With no distribution violation, the verdict is the classifier's answer."""
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")

        frame = pd.read_csv(PATHS.processed_csv, nrows=200, low_memory=False)
        checked = 0
        for _, series in frame.iterrows():
            result = predict_threat(series.to_dict())
            if result["reliability"] != "UNRELIABLE":
                assert result["threat"] == result["model_prediction"]
                checked += 1
        assert checked > 100, "too few in-distribution rows to be meaningful"

    def test_missing_features_are_reported(self) -> None:
        """Filling a gap with 0.0 must never be silent."""
        features = get_expected_features()
        partial = {name: 1.0 for name in features[:3]}
        result = predict_threat(partial)
        assert set(result["missing_features"]) == set(features[3:])

    def test_all_features_missing_yields_unknown(self) -> None:
        """An all-zero vector must not produce a confident answer."""
        result = predict_threat({"totally": 1, "wrong": 2, "keys": 3})
        assert result["threat"] == "UNKNOWN"
        assert result["confidence"] == 0.0
        assert result["model_ready"] is True  # the MODEL is fine; the input is not

    def test_extra_keys_are_ignored(self) -> None:
        features = get_expected_features()
        base = {name: 1.0 for name in features}
        with_extra = {**base, "Source IP": "10.0.0.1", "unused": 99}
        assert predict_threat(base)["threat"] == predict_threat(with_extra)["threat"]

    def test_identity_fields_do_not_change_the_prediction(self) -> None:
        """The model must not be influenced by IP addresses or ports.

        Training on identity would memorise the lab topology instead of learning
        traffic behaviour.
        """
        features = get_expected_features()
        base = {name: 1.0 for name in features}
        for identity in ({"Source IP": "1.2.3.4"},
                         {"Destination IP": "255.255.255.255"},
                         {"Flow ID": "x-y-z"},
                         {"Source Port": 65535}):
            assert (predict_threat({**base, **identity})["confidence"]
                    == predict_threat(base)["confidence"])

    def test_real_flows_are_classified_correctly(self) -> None:
        """End-to-end sanity check on real rows from the processed dataset."""
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset; run preprocessing.clean_data")

        frame = pd.read_csv(PATHS.processed_csv, low_memory=False)
        sample = frame.sample(min(200, len(frame)), random_state=11)
        correct = sum(
            predict_threat(record.to_dict())["threat"] == record["threat_class"]
            for _, record in sample.iterrows()
        )
        # Held-out accuracy is ~0.9987; this sample overlaps the training set,
        # so the bar is set well below that to catch breakage, not to measure
        # generalisation.
        assert correct / len(sample) > 0.95, \
            f"only {correct}/{len(sample)} correct -- the pipeline is broken"

    def test_reset_cache_reloads(self) -> None:
        first = predict_threat({f: 1.0 for f in get_expected_features()})
        reset_cache()
        second = predict_threat({f: 1.0 for f in get_expected_features()})
        assert first["threat"] == second["threat"]


# ---------------------------------------------------------------------------
# Alert schema
# ---------------------------------------------------------------------------
class TestAlertSchema:
    def _prediction(self) -> dict:
        return {
            "threat": "DDoS", "confidence": 0.96, "severity": "CRITICAL",
            "base_severity": "HIGH", "escalated": True,
            "severity_reason": "test", "is_alertable": True,
            "evidence": {"Flow Duration": 1000.0}, "evidence_detail": [],
            "evidence_method": "test", "aggregate_context": None,
            "class_probabilities": {"DDoS": 0.96, "BENIGN": 0.04},
            "model_ready": True,
        }

    def test_build_produces_valid_alert(self) -> None:
        alert = Alert.build(self._prediction(), "F000001",
                            {"Source IP": "10.0.0.1",
                             "Destination IP": "10.0.0.2",
                             "Source Port": 1234, "Destination Port": 80,
                             "Protocol": 6})
        assert validate(alert.to_dict()) == []

    def test_required_keys_present(self) -> None:
        alert = Alert.build(self._prediction(), "F000001")
        record = alert.to_dict()
        for key in REQUIRED_KEYS:
            assert key in record

    def test_json_roundtrip(self) -> None:
        alert = Alert.build(self._prediction(), "F000001")
        line = alert.to_json_line()
        assert line.endswith("\n")
        parsed = parse_alert_line(line)
        assert parsed is not None
        assert parsed["flow_id"] == "F000001"
        assert validate(parsed) == []

    def test_truncated_line_is_skipped_not_raised(self) -> None:
        """The dashboard may read while the replay engine is mid-write."""
        line = Alert.build(self._prediction(), "F000001").to_json_line()
        assert parse_alert_line(line[:40]) is None
        assert parse_alert_line("") is None
        assert parse_alert_line("   \n") is None
        assert parse_alert_line("{not json}") is None

    @pytest.mark.parametrize("identity,field,expected", [
        ({"Source IP": float("nan")}, "src_ip", "0.0.0.0"),
        ({"Source IP": None}, "src_ip", "0.0.0.0"),
        ({"Destination Port": "bad"}, "dst_port", 0),
        ({"Destination Port": 70000}, "dst_port", 0),
        ({"Destination Port": 80.0}, "dst_port", 80),
        ({"Protocol": None}, "protocol", "UNKNOWN"),
        ({"Protocol": 6}, "protocol", "TCP"),
    ])
    def test_identity_coercion(self, identity: dict, field: str,
                               expected: object) -> None:
        alert = Alert.build(self._prediction(), "F1", identity)
        assert getattr(alert, field) == expected

    def test_confidence_is_clamped(self) -> None:
        for raw, expected in ((1.7, 1.0), (-0.5, 0.0), ("junk", 0.0),
                              (float("nan"), 0.0)):
            alert = Alert.build({**self._prediction(), "confidence": raw}, "F1")
            assert alert.confidence == expected

    def test_ground_truth_marks_correctness(self) -> None:
        hit = Alert.build(self._prediction(), "F1", ground_truth="DDoS")
        miss = Alert.build(self._prediction(), "F2", ground_truth="BENIGN")
        blind = Alert.build(self._prediction(), "F3")
        assert hit.correct is True
        assert miss.correct is False
        assert blind.correct is None  # production has no ground truth

    def test_schema_version_is_recorded(self) -> None:
        alert = Alert.build(self._prediction(), "F1")
        assert alert.to_dict()["schema_version"] == SCHEMA_VERSION

    def test_validate_catches_problems(self) -> None:
        assert validate({}) != []
        assert any("confidence" in p for p in
                   validate({**Alert.build(self._prediction(), "F1").to_dict(),
                             "confidence": 5.0}))


# ---------------------------------------------------------------------------
# Integration: predict -> alert
# ---------------------------------------------------------------------------
@needs_model
class TestIntegration:
    def test_prediction_flows_into_a_valid_alert(self) -> None:
        """The exact path the replay engine takes."""
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")

        frame = pd.read_csv(PATHS.processed_csv, nrows=25, low_memory=False)
        for _, series in frame.iterrows():
            record = series.to_dict()
            prediction = predict_threat(record)
            alert = Alert.build(prediction, "F000001", record,
                                ground_truth=record.get("threat_class"))
            assert validate(alert.to_dict()) == []
            json.dumps(alert.to_dict())  # must be serialisable

    def test_benign_alerts_are_not_alertable(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")

        frame = pd.read_csv(PATHS.processed_csv, low_memory=False)
        benign = frame[frame["threat_class"] == "BENIGN"].head(30)
        for _, series in benign.iterrows():
            result = predict_threat(series.to_dict())
            if result["threat"] == "BENIGN":
                assert result["is_alertable"] is False
                assert result["severity"] == "INFO"

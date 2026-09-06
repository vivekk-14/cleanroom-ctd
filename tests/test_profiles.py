"""
Tests for multi-profile model artefacts and feature-contract enforcement.

Each feature profile is a SEPARATE model with its own feature contract, stored
under model/profiles/<slug>/. These tests exist because of a real failure found
during testing: a 14-feature BIDIRECTIONAL record was passed to the 18-feature
STRICT_UNIDIRECTIONAL model, 13 absent features were silently filled with zeros,
and the model returned confident predictions computed mostly from invented data.

Run:
    python -m pytest tests/test_profiles.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_PROFILE, PATHS  # noqa: E402
from model.predict import (  # noqa: E402
    available_profiles,
    get_expected_features,
    is_model_ready,
    model_status,
    predict_threat,
    reset_cache,
)
from preprocessing.feature_config import (  # noqa: E402
    BIDIRECTIONAL,
    FEATURE_PROFILES,
    STRICT_UNIDIRECTIONAL,
    FeatureContractError,
    describe_contract_mismatch,
    get_features,
    profiles_satisfied_by,
    validate_features,
)

TRAINED = available_profiles()
needs_both = pytest.mark.skipif(
    len(TRAINED) < 2,
    reason="needs both profiles trained; run: python -m model.train --all",
)
needs_any = pytest.mark.skipif(
    not TRAINED, reason="no trained model; run: python -m model.train --all",
)


# ---------------------------------------------------------------------------
# Artefact layout
# ---------------------------------------------------------------------------
class TestArtefactLayout:
    def test_profile_slugs_are_directory_safe(self) -> None:
        for profile in FEATURE_PROFILES:
            slug = PATHS.profile_slug(profile)
            assert slug == slug.lower()
            assert " " not in slug
            assert slug.isidentifier(), f"{slug!r} is not a safe directory name"

    def test_each_profile_has_a_distinct_directory(self) -> None:
        directories = {PATHS.profile_dir(p) for p in FEATURE_PROFILES}
        assert len(directories) == len(FEATURE_PROFILES)

    def test_artefact_paths_are_profile_specific(self) -> None:
        """No two profiles may share a model.pkl.

        A single shared artefact is what allowed one profile's data to reach the
        other's model.
        """
        model_paths = {PATHS.model_file(p) for p in FEATURE_PROFILES}
        assert len(model_paths) == len(FEATURE_PROFILES)

        feature_paths = {PATHS.features_file(p) for p in FEATURE_PROFILES}
        assert len(feature_paths) == len(FEATURE_PROFILES)

    def test_artefact_roles_are_complete(self) -> None:
        artefacts = PATHS.profile_artefacts(FEATURE_PROFILE)
        for role in ("model", "features", "label_encoder", "metrics",
                     "ood_reference"):
            assert role in artefacts
            assert artefacts[role].parent == PATHS.profile_dir(FEATURE_PROFILE)

    @needs_any
    def test_trained_profiles_are_discoverable(self) -> None:
        assert PATHS.trained_profiles()
        for slug in PATHS.trained_profiles():
            assert (PATHS.profiles_dir / slug / "model.pkl").exists()


# ---------------------------------------------------------------------------
# Feature contracts on disk
# ---------------------------------------------------------------------------
@needs_any
class TestFeatureContracts:
    @pytest.mark.parametrize("profile", sorted(FEATURE_PROFILES))
    def test_contract_declares_its_own_profile(self, profile: str) -> None:
        """A features.json must name the profile whose directory it sits in.

        Catches artefacts copied or renamed by hand, after which nothing
        downstream could be trusted.
        """
        path = PATHS.features_file(profile)
        if not path.exists():
            pytest.skip(f"{profile} not trained")
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["profile"] == profile

    @pytest.mark.parametrize("profile", sorted(FEATURE_PROFILES))
    def test_contract_count_matches_list(self, profile: str) -> None:
        path = PATHS.features_file(profile)
        if not path.exists():
            pytest.skip(f"{profile} not trained")
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["feature_count"] == len(contract["features"])

    @pytest.mark.parametrize("profile", sorted(FEATURE_PROFILES))
    def test_contract_matches_feature_config(self, profile: str) -> None:
        """The saved contract must equal the profile definition, in order."""
        path = PATHS.features_file(profile)
        if not path.exists():
            pytest.skip(f"{profile} not trained")
        contract = json.loads(path.read_text(encoding="utf-8"))
        assert contract["features"] == get_features(profile)

    @needs_both
    def test_the_two_contracts_differ(self) -> None:
        strict = json.loads(
            PATHS.features_file("STRICT_UNIDIRECTIONAL").read_text(
                encoding="utf-8"))
        bidir = json.loads(
            PATHS.features_file("BIDIRECTIONAL").read_text(encoding="utf-8"))
        assert strict["feature_count"] == 18
        assert bidir["feature_count"] == 14
        assert strict["features"] != bidir["features"]

    @needs_both
    def test_models_have_different_feature_counts(self) -> None:
        import joblib

        strict = joblib.load(PATHS.model_file("STRICT_UNIDIRECTIONAL"))
        bidir = joblib.load(PATHS.model_file("BIDIRECTIONAL"))
        assert strict.n_features_in_ == 18
        assert bidir.n_features_in_ == 14


# ---------------------------------------------------------------------------
# validate_features
# ---------------------------------------------------------------------------
class TestValidateFeatures:
    def test_returns_contract_order_not_input_order(self) -> None:
        result = validate_features({"c": 3, "a": 1, "b": 2}, ["a", "b", "c"])
        assert list(result) == ["a", "b", "c"]

    def test_missing_features_raise(self) -> None:
        with pytest.raises(FeatureContractError) as info:
            validate_features({"a": 1}, ["a", "b", "c"])
        assert info.value.missing == ["b", "c"]

    def test_extra_keys_allowed_by_default_and_excluded(self) -> None:
        """The replay engine passes whole CSV rows, identity columns included."""
        result = validate_features(
            {"a": 1, "Source IP": "10.0.0.1", "Flow ID": "x"}, ["a"])
        assert result == {"a": 1}
        assert "Source IP" not in result

    def test_extra_keys_rejected_when_requested(self) -> None:
        with pytest.raises(FeatureContractError):
            validate_features({"a": 1, "b": 2}, ["a"], allow_extra=False)

    def test_non_dict_input_raises_contract_error(self) -> None:
        for bad in (None, [], "string", 42):
            with pytest.raises(FeatureContractError):
                validate_features(bad, ["a"])  # type: ignore[arg-type]

    def test_the_exact_bug_is_caught(self) -> None:
        """A BIDIRECTIONAL record must not satisfy the STRICT contract."""
        record = {f: 1.0 for f in BIDIRECTIONAL}
        with pytest.raises(FeatureContractError) as info:
            validate_features(record, list(STRICT_UNIDIRECTIONAL))
        assert len(info.value.missing) == 13

    def test_profiles_satisfied_by_identifies_the_right_one(self) -> None:
        assert profiles_satisfied_by({f: 1.0 for f in BIDIRECTIONAL}) == \
            ["BIDIRECTIONAL"]
        strict_record = {f: 1.0 for f in STRICT_UNIDIRECTIONAL}
        assert "STRICT_UNIDIRECTIONAL" in profiles_satisfied_by(strict_record)

    def test_a_full_record_satisfies_both_profiles(self) -> None:
        both = {f: 1.0 for f in
                set(STRICT_UNIDIRECTIONAL) | set(BIDIRECTIONAL)}
        assert set(profiles_satisfied_by(both)) == set(FEATURE_PROFILES)

    def test_mismatch_description_suggests_a_profile(self) -> None:
        message = describe_contract_mismatch(
            {f: 1.0 for f in BIDIRECTIONAL}, "STRICT_UNIDIRECTIONAL")
        assert "BIDIRECTIONAL" in message
        assert "13 of 18" in message

    def test_mismatch_description_when_nothing_fits(self) -> None:
        message = describe_contract_mismatch({"junk": 1}, FEATURE_PROFILE)
        assert "no configured profile" in message


# ---------------------------------------------------------------------------
# predict_threat profile routing
# ---------------------------------------------------------------------------
@needs_any
class TestPredictProfileRouting:
    def test_default_profile_is_the_configured_one(self) -> None:
        record = {f: 1.0 for f in get_features(FEATURE_PROFILE)}
        assert predict_threat(record)["profile"] == FEATURE_PROFILE

    @needs_both
    def test_profile_argument_selects_the_model(self) -> None:
        bidir = {f: 1.0 for f in BIDIRECTIONAL}
        result = predict_threat(bidir, profile="BIDIRECTIONAL")
        assert result["profile"] == "BIDIRECTIONAL"
        assert result["threat"] != "UNKNOWN"
        assert result["contract_satisfied"] is True

    @needs_both
    def test_wrong_profile_is_refused_not_zero_filled(self) -> None:
        """The central regression test for this change.

        Before per-profile contracts, this returned a confident classification
        computed from 13 zero-filled features.
        """
        bidir = {f: 1.0 for f in BIDIRECTIONAL}
        result = predict_threat(bidir, profile="STRICT_UNIDIRECTIONAL")
        assert result["threat"] == "UNKNOWN"
        assert result["confidence"] == 0.0
        assert result["contract_satisfied"] is False
        assert result["is_alertable"] is False
        # The model exists; the input is what is wrong.
        assert result["model_ready"] is True
        assert len(result["missing_features"]) == 13

    @needs_both
    def test_refusal_names_the_right_profile(self) -> None:
        bidir = {f: 1.0 for f in BIDIRECTIONAL}
        result = predict_threat(bidir, profile="STRICT_UNIDIRECTIONAL")
        assert "BIDIRECTIONAL" in result.get("error", "")

    @needs_any
    def test_strict_false_permits_zero_fill(self) -> None:
        """The escape hatch still works, and still reports what it filled."""
        features = get_features(FEATURE_PROFILE)
        partial = {f: 1.0 for f in features[:5]}
        result = predict_threat(partial, strict=False)
        assert result["contract_satisfied"] is True  # not contract-checked
        assert len(result["missing_features"]) == len(features) - 5

    def test_unknown_profile_returns_unknown_not_raises(self) -> None:
        result = predict_threat({"a": 1}, profile="NOT_A_PROFILE")
        assert result["threat"] == "UNKNOWN"
        assert result["contract_satisfied"] is False

    @needs_both
    def test_both_profiles_usable_in_one_process(self) -> None:
        """Per-profile caching: loading one must not evict the other."""
        strict_record = {f: 1.0 for f in STRICT_UNIDIRECTIONAL}
        bidir_record = {f: 1.0 for f in BIDIRECTIONAL}

        first = predict_threat(strict_record, profile="STRICT_UNIDIRECTIONAL")
        second = predict_threat(bidir_record, profile="BIDIRECTIONAL")
        third = predict_threat(strict_record, profile="STRICT_UNIDIRECTIONAL")

        assert first["profile"] == "STRICT_UNIDIRECTIONAL"
        assert second["profile"] == "BIDIRECTIONAL"
        assert third["confidence"] == first["confidence"]

    @needs_both
    def test_get_expected_features_is_profile_specific(self) -> None:
        assert len(get_expected_features("STRICT_UNIDIRECTIONAL")) == 18
        assert len(get_expected_features("BIDIRECTIONAL")) == 14

    def test_get_expected_features_falls_back_to_config(self) -> None:
        """Discoverable before training, so a caller can prepare input."""
        reset_cache()
        for profile in FEATURE_PROFILES:
            assert get_expected_features(profile) == get_features(profile)

    def test_get_expected_features_unknown_profile_is_empty(self) -> None:
        assert get_expected_features("NOT_A_PROFILE") == []

    @needs_any
    def test_model_status_reports_all_profiles(self) -> None:
        status = model_status()
        assert "profiles" in status
        for profile in FEATURE_PROFILES:
            assert profile in status["profiles"]
            assert "trained" in status["profiles"][profile]
            assert status["profiles"][profile]["n_features"] == \
                len(get_features(profile))

    @needs_any
    def test_is_model_ready_accepts_a_profile(self) -> None:
        assert is_model_ready(FEATURE_PROFILE) is True
        assert is_model_ready("NOT_A_PROFILE") is False

    @needs_both
    def test_profiles_disagree_on_the_same_flow(self) -> None:
        """Two different models must be able to reach different conclusions.

        If they always agreed, the profile argument would be doing nothing.
        """
        pytest.importorskip("pandas")
        import pandas as pd

        if not PATHS.processed_csv.exists():
            pytest.skip("no processed dataset")

        frame = pd.read_csv(PATHS.processed_csv, nrows=400, low_memory=False)
        differences = 0
        for _, row in frame.iterrows():
            record = row.to_dict()
            a = predict_threat(record, profile="STRICT_UNIDIRECTIONAL")
            b = predict_threat(record, profile="BIDIRECTIONAL")
            assert a["profile"] != b["profile"]
            if a["confidence"] != b["confidence"]:
                differences += 1
        assert differences > 0, \
            "the two profiles produced identical confidences on 400 flows"


# ---------------------------------------------------------------------------
# The OOD fixture, scored through the correct profile
# ---------------------------------------------------------------------------
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ood_synthetic_attacks.csv"


@pytest.mark.skipif(not FIXTURE.exists(), reason="OOD fixture absent")
class TestOODFixtureProfileRouting:
    """The fixture has exactly the 14 BIDIRECTIONAL columns plus Label.

    It must be scored with BIDIRECTIONAL. Testing it against the 18-feature
    model is an invalid comparison, and these tests pin that down.
    """

    def _fixture_rows(self) -> list[dict]:
        pytest.importorskip("pandas")
        import pandas as pd

        from preprocessing.feature_config import normalize_column

        frame = pd.read_csv(FIXTURE)
        frame.columns = [normalize_column(c) for c in frame.columns]
        frame = frame.dropna(subset=["Label"])
        return frame.to_dict("records")

    def test_fixture_satisfies_exactly_bidirectional(self) -> None:
        rows = self._fixture_rows()
        assert rows, "fixture is empty"
        satisfied = profiles_satisfied_by(rows[0])
        assert satisfied == ["BIDIRECTIONAL"], \
            f"fixture should satisfy only BIDIRECTIONAL, got {satisfied}"

    def test_fixture_has_twenty_labelled_rows(self) -> None:
        rows = self._fixture_rows()
        assert len(rows) == 20
        labels = {r["Label"] for r in rows}
        assert labels == {"BENIGN", "DDoS", "PortScan", "Botnet"}

    @needs_both
    def test_fixture_refused_by_strict_profile(self) -> None:
        for row in self._fixture_rows():
            result = predict_threat(row, profile="STRICT_UNIDIRECTIONAL")
            assert result["contract_satisfied"] is False
            assert result["threat"] == "UNKNOWN"

    @needs_both
    def test_fixture_scored_by_bidirectional_profile(self) -> None:
        """The fixture satisfies the BIDIRECTIONAL contract and is scored.

        The reported `threat` may be UNKNOWN_OOD: the contract is about whether
        the right FEATURES are present, which is a separate question from whether
        their VALUES resemble anything in training. This fixture passes the first
        check and deliberately fails the second. `model_prediction` always holds
        a real class either way.
        """
        for row in self._fixture_rows():
            result = predict_threat(row, profile="BIDIRECTIONAL")
            assert result["contract_satisfied"] is True
            assert result["model_prediction"] in ("BENIGN", "DDoS", "PortScan",
                                                  "Botnet")
            assert result["threat"] in ("BENIGN", "DDoS", "PortScan", "Botnet",
                                        "UNKNOWN_OOD")

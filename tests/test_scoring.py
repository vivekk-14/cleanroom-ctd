"""
Tests for scoring external datasets.

Covers the two capabilities added so the model can be pointed at data it has
never seen:

  1. preprocessing/clean_data.py accepts CSVs with NO label column, so genuinely
     unlabelled traffic can be prepared for scoring.
  2. model/score_dataset.py scores any CSV, in labelled or unlabelled mode.

Run:
    python -m pytest tests/test_scoring.py -v

Tests needing a trained model are skipped, not failed, when model.pkl is absent.
Fixtures build small CSVs in a temp directory from the real processed dataset, so
nothing here depends on network access or on the 260 MB raw files.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS  # noqa: E402
from model.predict import is_model_ready  # noqa: E402
from preprocessing.clean_data import (  # noqa: E402
    TARGET_COLUMN,
    _clean_one,
    _wanted_columns,
)
from preprocessing.feature_config import (  # noqa: E402
    ALL_PROFILE_FEATURES,
    EVIDENCE_EXTRA_COLUMNS,
    LABEL_COLUMN,
    normalize_column,
)

needs_model = pytest.mark.skipif(
    not is_model_ready(),
    reason="no trained model; run: python -m model.train",
)
needs_processed = pytest.mark.skipif(
    not PATHS.processed_csv.exists(),
    reason="no processed dataset; run: python -m preprocessing.clean_data",
)

FEATURE_COLUMNS = sorted(set(ALL_PROFILE_FEATURES) | set(EVIDENCE_EXTRA_COLUMNS))


@pytest.fixture(scope="module")
def labelled_frame():
    """A small real slice of the processed dataset."""
    pytest.importorskip("pandas")
    import pandas as pd

    if not PATHS.processed_csv.exists():
        pytest.skip("no processed dataset")
    return pd.read_csv(PATHS.processed_csv, nrows=600, low_memory=False)


@pytest.fixture(scope="module")
def labelled_csv(tmp_path_factory, labelled_frame) -> Path:
    """A CSV that keeps its threat_class column."""
    path = tmp_path_factory.mktemp("scoring") / "labelled.csv"
    labelled_frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def unlabelled_csv(tmp_path_factory, labelled_frame) -> Path:
    """The same rows with the label column removed entirely."""
    path = tmp_path_factory.mktemp("scoring") / "unlabelled.csv"
    labelled_frame.drop(columns=[TARGET_COLUMN]).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Cleaning without labels
# ---------------------------------------------------------------------------
class TestCleaningUnlabelledData:
    """The label column must be optional.

    Before this, _clean_one raised KeyError: 'Label' on any CSV without labels,
    which meant the pipeline could only process data whose answers were already
    known. That is useless for scoring traffic handed to us at a hackathon.
    """

    def _raw_slice(self, with_label: bool):
        pytest.importorskip("pandas")
        import pandas as pd

        source = PATHS.raw_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        if not source.exists():
            pytest.skip("raw CSV not present in data/raw/")

        wanted = _wanted_columns()
        frame = pd.read_csv(
            source, nrows=300,
            usecols=lambda c: normalize_column(c) in wanted,
            low_memory=False,
        )
        frame.columns = [normalize_column(c) for c in frame.columns]
        if not with_label and LABEL_COLUMN in frame.columns:
            frame = frame.drop(columns=[LABEL_COLUMN])
        return frame

    def test_labelled_input_produces_target_column(self) -> None:
        frame = self._raw_slice(with_label=True)
        stats: Counter = Counter()
        cleaned = _clean_one(frame, FEATURE_COLUMNS, stats, verbose=False)
        assert TARGET_COLUMN in cleaned.columns
        assert LABEL_COLUMN not in cleaned.columns
        assert cleaned[TARGET_COLUMN].notna().all()

    def test_unlabelled_input_does_not_raise(self) -> None:
        frame = self._raw_slice(with_label=False)
        stats: Counter = Counter()
        cleaned = _clean_one(frame, FEATURE_COLUMNS, stats, verbose=False)
        assert len(cleaned) > 0

    def test_unlabelled_input_has_no_target_column(self) -> None:
        """No invented labels. Absent ground truth must stay absent."""
        frame = self._raw_slice(with_label=False)
        stats: Counter = Counter()
        cleaned = _clean_one(frame, FEATURE_COLUMNS, stats, verbose=False)
        assert TARGET_COLUMN not in cleaned.columns

    def test_unlabelled_input_is_counted(self) -> None:
        frame = self._raw_slice(with_label=False)
        stats: Counter = Counter()
        _clean_one(frame, FEATURE_COLUMNS, stats, verbose=False)
        assert stats["unlabelled_files"] == 1

    def test_features_still_cleaned_without_labels(self) -> None:
        """Numeric coercion and inf handling must run regardless of labels."""
        pytest.importorskip("pandas")
        import numpy as np

        frame = self._raw_slice(with_label=False)
        stats: Counter = Counter()
        cleaned = _clean_one(frame, FEATURE_COLUMNS, stats, verbose=False)
        present = [c for c in FEATURE_COLUMNS if c in cleaned.columns]
        values = cleaned[present].to_numpy(dtype="float64", na_value=np.nan)
        assert not np.isinf(values).any(), "infinity survived cleaning"

    def test_end_to_end_unlabelled_file(self, tmp_path) -> None:
        """clean_dataset on a real raw CSV stripped of its label column."""
        pytest.importorskip("pandas")
        import pandas as pd

        from preprocessing.clean_data import clean_dataset

        source = PATHS.raw_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        if not source.exists():
            pytest.skip("raw CSV not present in data/raw/")

        # Write a label-free copy into data/raw/, because clean_dataset resolves
        # filenames against that directory.
        stripped = PATHS.raw_dir / "_pytest_unlabelled.csv"
        try:
            frame = pd.read_csv(source, nrows=400, low_memory=False)
            label = [c for c in frame.columns if normalize_column(c) == LABEL_COLUMN]
            frame.drop(columns=label).to_csv(stripped, index=False)

            out = tmp_path / "cleaned.csv"
            result = clean_dataset(files=[stripped.name], max_per_class=None,
                                   out_path=out, verbose=False)

            assert out.exists()
            assert TARGET_COLUMN not in result.columns
            assert len(result) > 0

            meta = json.loads(out.with_suffix(".meta.json").read_text(
                encoding="utf-8"))
            assert meta["labelled"] is False
            assert meta["class_counts"] is None
            assert meta["target_column"] is None
            assert any("NO LABEL COLUMN" in note for note in meta["notes"])
        finally:
            stripped.unlink(missing_ok=True)

    def test_cap_is_skipped_without_labels(self, tmp_path) -> None:
        """A per-class cap cannot apply without classes.

        Applying a global sample instead would silently discard most of the
        traffic the user asked to have scored.
        """
        pytest.importorskip("pandas")
        import pandas as pd

        from preprocessing.clean_data import clean_dataset

        source = PATHS.raw_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        if not source.exists():
            pytest.skip("raw CSV not present in data/raw/")

        stripped = PATHS.raw_dir / "_pytest_unlabelled_cap.csv"
        try:
            frame = pd.read_csv(source, nrows=500, low_memory=False)
            label = [c for c in frame.columns if normalize_column(c) == LABEL_COLUMN]
            frame.drop(columns=label).to_csv(stripped, index=False)

            out = tmp_path / "capped.csv"
            # A cap of 10 would decimate the output if wrongly applied.
            result = clean_dataset(files=[stripped.name], max_per_class=10,
                                   out_path=out, verbose=False)
            assert len(result) > 100, "cap was applied despite absent labels"
        finally:
            stripped.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The scoring tool
# ---------------------------------------------------------------------------
@needs_model
@needs_processed
class TestScoreDataset:
    def test_unlabelled_mode(self, unlabelled_csv, tmp_path) -> None:
        from model.score_dataset import score

        summary = score(unlabelled_csv, show_examples=0, output_dir=tmp_path)
        assert summary["labelled"] is False
        assert summary["rows_scored"] > 0
        # No ground truth means no accuracy may be claimed.
        assert "accuracy_known_classes" not in summary
        assert "unseen_classes" not in summary

    def test_labelled_mode_reports_accuracy(self, labelled_csv, tmp_path) -> None:
        from model.score_dataset import score

        summary = score(labelled_csv, show_examples=0, output_dir=tmp_path)
        assert summary["labelled"] is True
        assert 0.0 <= summary["accuracy_known_classes"] <= 1.0
        assert summary["per_class"]
        # This slice comes from the training data, so agreement should be high.
        # The bar catches breakage rather than measuring generalisation.
        assert summary["accuracy_known_classes"] > 0.90

    def test_predictions_file_is_written(self, unlabelled_csv, tmp_path) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        from model.score_dataset import score

        summary = score(unlabelled_csv, show_examples=0, output_dir=tmp_path)
        path = Path(summary["predictions_file"])
        assert path.exists()

        output = pd.read_csv(path)
        assert len(output) == summary["rows_scored"]
        for column in ("reported_threat", "model_prediction", "confidence",
                       "reliability", "ood_score", "severity"):
            assert column in output.columns
        assert output["confidence"].between(0.0, 1.0).all()

    def test_summary_json_is_written_and_valid(self, unlabelled_csv, tmp_path) -> None:
        from model.score_dataset import score

        summary = score(unlabelled_csv, show_examples=0, output_dir=tmp_path)
        # Output names carry the profile slug, so scoring one file under two
        # profiles cannot overwrite one summary with the other.
        written = sorted(tmp_path.glob("*.summary.json"))
        assert len(written) == 1, f"expected one summary, found {written}"
        path = written[0]
        assert unlabelled_csv.stem in path.name
        assert PATHS.profile_slug(summary["profile"]) in path.name
        json.loads(path.read_text(encoding="utf-8"))  # must parse

    def test_severity_counts_match_row_count(self, unlabelled_csv, tmp_path) -> None:
        from model.score_dataset import score

        summary = score(unlabelled_csv, show_examples=0, output_dir=tmp_path)
        assert sum(summary["severity_counts"].values()) == summary["rows_scored"]
        assert sum(summary["predicted_counts"].values()) == summary["rows_scored"]

    def test_max_flows_limits_work(self, unlabelled_csv, tmp_path) -> None:
        from model.score_dataset import score

        summary = score(unlabelled_csv, max_flows=50, show_examples=0, output_dir=tmp_path)
        assert summary["rows_read"] <= 50

    def test_save_alerts_produces_valid_schema(self, labelled_csv, tmp_path) -> None:
        from detection.alert_schema import validate
        from model.score_dataset import score

        summary = score(labelled_csv, save_alerts=True, show_examples=0, output_dir=tmp_path)
        path = Path(summary["alerts_file"])
        assert path.exists()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines, "no alerts written"
        for line in lines[:50]:
            record = json.loads(line)
            assert validate(record) == []
            assert record["observation_mode"] == "BATCH_SCORING"

    def test_written_records_belong_to_a_queue(self, labelled_csv,
                                               tmp_path) -> None:
        """Every written record is either actionable or flagged for review.

        The stream carries BOTH queues, not just alerts. Writing only alertable
        flows would drop every DEGRADED-but-benign row -- on the OOD fixture that
        is 9 of 20, precisely the flows the review queue exists to preserve. The
        dashboard separates them on load via `is_alertable` / `needs_review`.
        """
        from config import SEVERITY
        from model.score_dataset import score

        summary = score(labelled_csv, save_alerts=True, show_examples=0,
                        output_dir=tmp_path)
        for line in Path(summary["alerts_file"]).read_text(
                encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["is_alertable"] or record["needs_review"], \
                f"{record['flow_id']} belongs to neither queue"
            if record["is_alertable"]:
                assert record["severity"] in SEVERITY.alert_severities

    def test_review_only_records_are_not_lost(self, tmp_path) -> None:
        """The OOD fixture must produce review records, not only alerts."""
        from model.score_dataset import score

        fixture = (Path(__file__).resolve().parent / "fixtures"
                   / "ood_synthetic_attacks.csv")
        if not fixture.exists():
            pytest.skip("OOD fixture missing")
        if not is_model_ready("BIDIRECTIONAL"):
            pytest.skip("BIDIRECTIONAL model not trained")

        summary = score(fixture, profile="BIDIRECTIONAL", save_alerts=True,
                        show_examples=0, output_dir=tmp_path)
        assert summary["review_written"] > 0, \
            "review-queue records were dropped from the alert stream"
        assert summary["alerts_written"] > summary["review_written"] or \
            summary["alerts_written"] >= 10

    def test_missing_features_raise_a_clear_error(self, tmp_path) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        from model.score_dataset import score

        path = tmp_path / "wrong_format.csv"
        pd.DataFrame({"totally": [1, 2], "wrong": [3, 4]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="features missing"):
            score(path, show_examples=0, output_dir=tmp_path)

    def test_absent_file_raises_filenotfound(self, tmp_path) -> None:
        from model.score_dataset import score

        with pytest.raises(FileNotFoundError):
            score(tmp_path / "does_not_exist.csv", show_examples=0, output_dir=tmp_path)

    def test_raw_cic_format_scores_directly(self, tmp_path) -> None:
        """A raw CSV with un-normalised headers must work without pre-cleaning."""
        from model.score_dataset import score

        source = PATHS.raw_dir / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        if not source.exists():
            pytest.skip("raw CSV not present in data/raw/")

        summary = score(source, max_flows=2000, show_examples=0, output_dir=tmp_path)
        assert summary["rows_scored"] > 0
        assert summary["labelled"] is True

    def test_unseen_classes_are_reported_not_hidden(self, tmp_path) -> None:
        """The central honesty check for this tool.

        A file containing attack types the model was never trained on must have
        them counted and reported, never silently dropped or folded into the
        accuracy figure.
        """
        pytest.importorskip("pandas")
        import pandas as pd

        from model.score_dataset import score

        source = PATHS.raw_dir / "Wednesday-workingHours.pcap_ISCX.csv"
        if not source.exists():
            pytest.skip("Wednesday CSV not in data/raw/ (contains DoS variants)")

        summary = score(source, max_flows=80_000, show_examples=0, output_dir=tmp_path)

        assert "unseen_classes" in summary, "unseen attack types were not reported"
        assert summary["unseen_classes"], "no unseen classes counted"
        for label, stats in summary["unseen_classes"].items():
            assert stats["flows"] > 0
            assert 0.0 <= stats["miss_rate"] <= 1.0
            assert stats["flagged"] + stats["missed_as_benign"] == stats["flows"]

        # Accuracy must exclude unseen classes: mixing them in would conflate
        # "the model was wrong" with "the model was never taught this".
        assert summary["n_known_class_flows"] < summary["rows_scored"]

    def test_accuracy_excludes_unseen_classes(self, tmp_path) -> None:
        from model.score_dataset import score

        source = PATHS.raw_dir / "Wednesday-workingHours.pcap_ISCX.csv"
        if not source.exists():
            pytest.skip("Wednesday CSV not in data/raw/")

        summary = score(source, max_flows=80_000, show_examples=0, output_dir=tmp_path)
        # Every per-class support figure must be within the known-class subset.
        total_support = sum(v["support"] for v in summary["per_class"].values())
        assert total_support == summary["n_known_class_flows"]


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
class TestScorerSafety:
    def test_scorer_arms_the_passive_guard(self) -> None:
        import model.score_dataset  # noqa: F401
        from detection.passive_guard import is_armed

        assert is_armed() is True

    def test_guard_armed_before_heavy_imports(self) -> None:
        source = (Path(__file__).resolve().parent.parent
                  / "model" / "score_dataset.py").read_text(encoding="utf-8")
        guard_position = source.index("enforce_passive_mode(")
        pandas_position = source.index("import pandas")
        assert guard_position < pandas_position

    def test_scorer_has_no_network_calls(self) -> None:
        source = (Path(__file__).resolve().parent.parent
                  / "model" / "score_dataset.py").read_text(encoding="utf-8")
        for forbidden in (".send(", ".connect(", "socket.socket(",
                          "requests.", "urlopen("):
            assert forbidden not in source

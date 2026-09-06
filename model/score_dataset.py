"""
Score any flow dataset with the trained model.

Answers the question "what does the model say about traffic it has never seen?"
for a CSV that is not the training set.

Two modes, selected automatically:

  UNLABELLED  no threat_class column. Reports what the model predicts, the
              confidence distribution, and the severity breakdown. This is the
              real deployment case: traffic arrives, the model classifies it,
              and nobody knows the answers in advance.

  LABELLED    threat_class present. Additionally reports accuracy, a confusion
              matrix, per-class precision/recall/F1, and -- importantly -- how
              the model handles labels it was never trained on.

Usage
-----
    python -m model.score_dataset --input data/processed/holdout.csv
    python -m model.score_dataset --input traffic.csv --max-flows 50000
    python -m model.score_dataset --input traffic.csv --save-alerts
    python -m model.score_dataset --raw data/raw/SomeCapture.csv

`--raw` cleans a raw CIC-IDS2017-format CSV first, so an unfamiliar file can be
scored in one command.

Why this exists separately from model/evaluate.py
------------------------------------------------
`evaluate.py` scores the held-out split of the TRAINING dataset. It answers "did
training work?". This tool scores an arbitrary external dataset and answers "does
the model generalise?". The second question is the one that matters, and it has a
different, less flattering answer -- see the unseen-class section of the report.

SAFETY: reads local CSV, writes local files. Opens no sockets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.passive_guard import enforce_passive_mode  # noqa: E402

enforce_passive_mode(verbose=False)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FEATURE_PROFILE, PATHS, SEVERITY  # noqa: E402
from detection.alert_schema import Alert  # noqa: E402
from detection import ood as ood_module  # noqa: E402
from detection.severity import assess  # noqa: E402
from model.predict import predict_threat  # noqa: E402
from preprocessing.clean_data import TARGET_COLUMN  # noqa: E402
from preprocessing.feature_config import (  # noqa: E402
    FEATURE_PROFILES,
    IDENTITY_COLUMNS,
    LABEL_MAP,
    TARGET_CLASSES,
    normalize_column,
)

RULE = "=" * 76
THIN = "-" * 76


def _load_artefacts(profile: str) -> tuple[object, object, list[str], dict]:
    """Load one profile's trained model, encoder, and feature contract."""
    artefacts = PATHS.profile_artefacts(profile)
    for role in ("model", "features", "label_encoder"):
        path = artefacts[role]
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found in "
                f"{PATHS.profile_dir(profile).relative_to(PATHS.root)}\n"
                f"Run first:  python -m model.train --profile {profile}"
            )
    contract = json.loads(artefacts["features"].read_text(encoding="utf-8"))
    model = joblib.load(artefacts["model"])
    encoder = joblib.load(artefacts["label_encoder"])
    # Batch scoring benefits from every core, unlike the one-at-a-time replay
    # path where thread dispatch overhead dominates.
    if hasattr(model, "n_jobs"):
        model.n_jobs = -1
    return model, encoder, list(contract["features"]), contract


def _read_header(path: Path) -> set[str]:
    """Return the file's normalised column names, without reading the body.

    Used to pick a profile before any model is loaded, so a mismatched file
    produces a "use profile X instead" message rather than a wall of missing
    features.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    for encoding in ("utf-8", "latin-1"):
        try:
            head = pd.read_csv(path, encoding=encoding, nrows=0)
            return {normalize_column(c) for c in head.columns}
        except UnicodeDecodeError:
            continue
    return set()


def _read_input(path: Path, features: list[str],
                max_flows: int | None) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV and return (frame, missing_features).

    Column names are normalised on read, so a raw CIC-IDS2017 file with its
    inconsistent leading spaces works without pre-processing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False,
                                nrows=max_flows)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not decode {path.name}: {last_error}")

    frame.columns = [normalize_column(c) for c in frame.columns]

    # A raw file may carry the label under its original name.
    #
    # Out-of-scope labels ('DoS Hulk', 'Heartbleed', ...) map to None in
    # LABEL_MAP because clean_data drops them from TRAINING data. Here they must
    # be PRESERVED: a flow whose true class the model was never trained on is
    # exactly what the unseen-class section reports on, and mapping it to NaN
    # would silently discard that finding.
    if TARGET_COLUMN not in frame.columns and "Label" in frame.columns:
        raw = frame["Label"].astype("string").str.strip()
        mapped = raw.map(LABEL_MAP)
        # Where the map yields nothing, keep the original label text.
        frame[TARGET_COLUMN] = mapped.fillna(raw)
        frame["_raw_label"] = raw

    # Rows with no label at all cannot contribute to any accuracy figure.
    if TARGET_COLUMN in frame.columns:
        unlabelled_rows = frame[TARGET_COLUMN].isna()
        if unlabelled_rows.all():
            frame = frame.drop(columns=[TARGET_COLUMN])
        elif unlabelled_rows.any():
            frame = frame[~unlabelled_rows]

    missing = [f for f in features if f not in frame.columns]
    return frame, missing


def _prepare_matrix(frame: pd.DataFrame,
                    features: list[str]) -> tuple[np.ndarray, pd.Index]:
    """Build the model input matrix, returning it with the surviving row index.

    Rows containing NaN or infinity in any model feature are dropped and
    reported. Silently imputing them would produce predictions that look valid
    but rest on invented data.
    """
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    usable = numeric.notna().all(axis=1)
    clean = numeric[usable]
    return clean.to_numpy(dtype="float64"), clean.index


def score(
    input_path: Path,
    profile: str = FEATURE_PROFILE,
    max_flows: int | None = None,
    save_alerts: bool = False,
    save_report: bool = False,
    show_examples: int = 3,
    output_dir: Path | None = None,
    auto_profile: bool = False,
) -> dict:
    """Score a dataset with one profile's model. Returns a summary dict.

    profile selects which model and feature contract to use. auto_profile=True
    switches to whichever profile the file's columns actually satisfy, which is
    what makes `--auto-profile` on the command line possible.

    output_dir defaults to reports/. Tests pass a temp directory: output
    filenames are derived from the input stem, so a test run scoring a truncated
    slice of a file would otherwise overwrite the real summary for that same
    file and silently invalidate figures quoted elsewhere.
    """
    t0 = time.time()
    PATHS.ensure_dirs()
    reports_dir = output_dir or PATHS.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(RULE)
    emit("SCORING AN EXTERNAL DATASET")
    emit(RULE)
    emit(f"  input      : {input_path}")

    # --- profile selection ------------------------------------------------
    # Read the header first so the file's columns can be checked against each
    # profile's contract BEFORE a model is loaded. This is what lets the tool
    # say "use BIDIRECTIONAL" instead of failing with 13 missing features.
    header = _read_header(input_path)
    satisfied = [name for name, feats in FEATURE_PROFILES.items()
                 if all(f in header for f in feats)]

    if auto_profile:
        if not satisfied:
            emit(f"\n  ERROR: this file satisfies no profile's feature contract.")
            emit(f"  columns present: {len(header)}")
            for name, feats in FEATURE_PROFILES.items():
                gap = [f for f in feats if f not in header]
                emit(f"    {name}: missing {len(gap)} of {len(feats)}")
            raise ValueError("no profile contract satisfied by this file")
        # Prefer the configured default when the file satisfies several.
        chosen = (FEATURE_PROFILE if FEATURE_PROFILE in satisfied
                  else satisfied[0])
        if chosen != profile:
            emit(f"  profile    : {chosen}  (auto-selected; "
                 f"requested {profile} is not satisfied by this file)")
        profile = chosen

    model, encoder, features, contract = _load_artefacts(profile)

    model_path = PATHS.model_file(profile)
    emit(f"  model      : {model_path.relative_to(PATHS.root)}")
    emit(f"  profile    : {contract.get('profile')}  "
         f"({contract.get('feature_count', len(features))} features)")
    emit(f"  classes    : {', '.join(str(c) for c in encoder.classes_)}")
    emit(f"  trained at : {contract.get('trained_at', 'unknown')}")
    if len(satisfied) > 1:
        emit(f"  note       : this file also satisfies "
             f"{', '.join(p for p in satisfied if p != profile)}")

    frame, missing = _read_input(input_path, features, max_flows)
    emit(f"  rows read  : {len(frame):,}")

    if missing:
        emit(f"\n  ERROR: {len(missing)} of {len(features)} features required by "
             f"the {profile} contract are absent from this file:")
        for name in missing:
            emit(f"    {name}")
        if satisfied:
            emit(f"\n  This file DOES satisfy: {', '.join(satisfied)}.")
            emit(f"  Score it with:")
            emit(f"    python -m model.score_dataset --input {input_path.name} "
                 f"--profile {satisfied[0]}")
            emit(f"  or let the tool choose:")
            emit(f"    python -m model.score_dataset --input {input_path.name} "
                 f"--auto-profile")
        else:
            emit("\n  The file satisfies no configured profile. It is probably "
                 "not in")
            emit("  CIC-IDS2017 flow format, or it needs cleaning first:")
            emit(f"    python -m preprocessing.clean_data --file "
                 f"{input_path.name}")
        raise ValueError(f"{len(missing)} required features missing")

    labelled = TARGET_COLUMN in frame.columns
    emit(f"  mode       : {'LABELLED' if labelled else 'UNLABELLED'}")
    if not labelled:
        emit("               No ground truth, so accuracy cannot be measured.")
        emit("               This is the real deployment case.")

    # --- prepare ----------------------------------------------------------
    matrix, index = _prepare_matrix(frame, features)
    dropped = len(frame) - len(matrix)
    if dropped:
        emit(f"\n  dropped {dropped:,} rows with NaN/infinity in a model feature "
             f"({dropped / len(frame):.2%})")
    if len(matrix) == 0:
        raise ValueError("No usable rows after cleaning.")

    # --- predict ----------------------------------------------------------
    emit(f"\n  scoring {len(matrix):,} flows...")
    t_predict = time.time()
    encoded = model.predict(matrix)
    probabilities = model.predict_proba(matrix)
    predict_seconds = time.time() - t_predict

    predictions = encoder.inverse_transform(encoded)
    confidence = probabilities.max(axis=1)

    emit(f"  done in {predict_seconds:.2f}s "
         f"({len(matrix) / max(predict_seconds, 1e-9):,.0f} flows/s, batched)")

    # --- distribution check -----------------------------------------------
    # Runs per row against the profile's OOD reference. Without this the scorer
    # would report the classifier's answers as findings even when the input has
    # nothing in common with the training data -- exactly the failure the
    # synthetic fixture in tests/fixtures/ demonstrates.
    reliabilities: list[str] = []
    ood_scores = np.zeros(len(matrix))
    ood_feature_counter: Counter = Counter()
    for position in range(len(matrix)):
        assessment = ood_module.assess(matrix[position], profile, features)
        reliabilities.append(assessment.reliability)
        ood_scores[position] = assessment.ood_score
        ood_feature_counter.update(assessment.ood_features)

    reliability_array = np.array(reliabilities)
    unreliable = reliability_array == ood_module.UNRELIABLE

    # The REPORTED verdict, which is what an operator would act on. Distinct
    # from `predictions`, which is the classifier's raw answer.
    reported = np.where(unreliable, ood_module.OOD_THREAT_CLASS, predictions)

    # --- severity ---------------------------------------------------------
    decisions = [
        assess(reported[i], confidence[i], reliability=reliabilities[i],
               model_prediction=predictions[i])
        for i in range(len(matrix))
    ]
    severities = [d.severity for d in decisions]
    severity_counts = Counter(severities)
    alertable = sum(1 for d in decisions if d.is_alertable)
    review_only = sum(1 for d in decisions if d.needs_review and not d.is_alertable)

    # --- what the model predicted ----------------------------------------
    emit(f"\n{THIN}\n  WHAT THE MODEL PREDICTED\n{THIN}")
    emit("  The classifier's raw answers, before the distribution check.")
    predicted_counts = Counter(predictions)
    emit(f"  {'class':<12} {'flows':>10} {'share':>8} {'mean conf':>10}")
    emit(f"  {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 10}")
    for cls in list(TARGET_CLASSES) + sorted(
            set(predicted_counts) - set(TARGET_CLASSES)):
        count = predicted_counts.get(cls, 0)
        if not count:
            emit(f"  {cls:<12} {0:>10,} {0.0:>7.2%} {'-':>10}")
            continue
        mask = predictions == cls
        emit(f"  {cls:<12} {count:>10,} {count / len(predictions):>7.2%} "
             f"{confidence[mask].mean():>10.4f}")

    # --- reliability ------------------------------------------------------
    emit(f"\n{THIN}\n  CAN THOSE ANSWERS BE TRUSTED?\n{THIN}")
    emit("  Separate question from confidence: does this input resemble anything")
    emit("  the model was trained on?")
    emit("")
    reliability_counts = Counter(reliabilities)
    emit(f"  {'reliability':<14} {'flows':>10} {'share':>8}")
    emit(f"  {'-' * 14} {'-' * 10} {'-' * 8}")
    for tier in ood_module.RELIABILITY_ORDER[1:] + (ood_module.UNKNOWN,):
        count = reliability_counts.get(tier, 0)
        if count:
            emit(f"  {tier:<14} {count:>10,} {count / len(matrix):>7.2%}")

    if unreliable.any():
        emit(f"\n  {int(unreliable.sum()):,} flows "
             f"({unreliable.mean():.2%}) are outside the training distribution.")
        emit(f"  Their reported class becomes {ood_module.OOD_THREAT_CLASS}; the "
             f"classifier's answer is kept in 'model_prediction'.")
        said = Counter(str(cls) for cls in predictions[unreliable])
        breakdown = ", ".join(f"{cls} {count:,}"
                              for cls, count in said.most_common(4))
        emit(f"  What the classifier had said for those flows: {breakdown}")
        emit(f"  Their mean classifier confidence: "
             f"{confidence[unreliable].mean():.3f}  <- high confidence, "
             f"unreliable answer")

    if ood_feature_counter:
        emit(f"\n  features most often outside their training range:")
        for name, count in ood_feature_counter.most_common(6):
            emit(f"    {name:<30} {count:>8,} flows "
                 f"({count / len(matrix):6.2%})")

    emit(f"\n  reported verdict after the distribution check:")
    for cls, count in Counter(str(c) for c in reported).most_common():
        emit(f"    {cls:<14} {count:>10,} {count / len(matrix):>7.2%}")

    emit(f"\n  severity:")
    for level in SEVERITY.ladder:
        if severity_counts.get(level):
            emit(f"    {level:<10} {severity_counts[level]:>10,}")
    emit(f"    {'actionable':<10} {alertable:>10,}  "
         f"({alertable / len(predictions):.2%}, MEDIUM and above)")
    if review_only:
        emit(f"    {'review':<10} {review_only:>10,}  "
             f"({review_only / len(predictions):.2%}, not alertable but "
             f"reliability degraded)")

    truth = frame.loc[index, TARGET_COLUMN].to_numpy() if labelled else None

    # Flows whose true class is one the model can actually predict. Everything
    # else is an attack type it was never trained on and therefore cannot get
    # right by construction.
    known = np.isin(truth, list(encoder.classes_)) if labelled else None

    # --- confidence distribution -----------------------------------------
    emit(f"\n{THIN}\n  CONFIDENCE DISTRIBUTION\n{THIN}")
    if labelled:
        emit("  Accuracy is computed on flows whose true class the model knows.")
        emit("  Including unseen attack types here would conflate 'the model was")
        emit("  wrong' with 'the model was never taught this class', which are")
        emit("  different failures with different fixes.")
        emit("")
    bands = [(0.0, 0.5, "LOW"), (0.5, 0.75, "MEDIUM"),
             (0.75, 0.9, "HIGH"), (0.9, 1.01, "CRITICAL")]
    emit(f"  {'band':<10} {'range':<14} {'flows':>10} {'share':>8}"
         + (f" {'accuracy':>10} {'n known':>9}" if labelled else ""))
    emit(f"  {'-' * 10} {'-' * 14} {'-' * 10} {'-' * 8}"
         + (f" {'-' * 10} {'-' * 9}" if labelled else ""))

    band_rows = []
    for low, high, name in bands:
        mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        row = {"band": name, "n": count,
               "share": count / len(confidence) if len(confidence) else 0.0}
        text = (f"  {name:<10} [{low:.2f}, {min(high, 1.0):.2f})".ljust(27)
                + f" {count:>10,} {row['share']:>7.2%}")
        if labelled:
            scoreable = mask & known
            n_scoreable = int(scoreable.sum())
            row["n_known"] = n_scoreable
            if n_scoreable:
                band_accuracy = float(
                    (predictions[scoreable] == truth[scoreable]).mean())
                row["accuracy"] = band_accuracy
                text += f" {band_accuracy:>10.4f} {n_scoreable:>9,}"
            else:
                text += f" {'n/a':>10} {0:>9,}"
        emit(text)
        band_rows.append(row)

    summary: dict = {
        "input": str(input_path),
        "profile": profile,
        "profile_feature_count": len(features),
        "rows_read": int(len(frame)),
        "rows_scored": int(len(matrix)),
        "rows_dropped": int(dropped),
        "labelled": labelled,
        "predicted_counts": {k: int(v) for k, v in predicted_counts.items()},
        "reported_counts": {str(k): int(v)
                            for k, v in Counter(str(c)
                                                for c in reported).items()},
        "severity_counts": {k: int(v) for k, v in severity_counts.items()},
        "alertable": int(alertable),
        "review_only": int(review_only),
        "mean_confidence": float(confidence.mean()),
        "confidence_bands": band_rows,
        # Distribution assessment, kept as its own block so a consumer cannot
        # mistake reliability for classifier confidence.
        "reliability_counts": {k: int(v)
                               for k, v in Counter(reliabilities).items()},
        "out_of_distribution_rate": float(unreliable.mean()),
        "mean_ood_score": float(ood_scores.mean()),
        "ood_feature_counts": {k: int(v)
                               for k, v in ood_feature_counter.most_common()},
        "ood_confidence_when_unreliable": (
            float(confidence[unreliable].mean()) if unreliable.any() else None),
        "predict_seconds": round(predict_seconds, 3),
        "flows_per_second": round(len(matrix) / max(predict_seconds, 1e-9), 1),
    }

    # --- accuracy, if labels exist ---------------------------------------
    if labelled:
        emit(f"\n{THIN}\n  ACCURACY AGAINST GROUND TRUTH\n{THIN}")

        n_known, n_unknown = int(known.sum()), int((~known).sum())
        if n_known:
            accuracy = float((predictions[known] == truth[known]).mean())
            emit(f"  flows whose true class the model knows : {n_known:,}")
            emit(f"  accuracy on those flows               : {accuracy:.4f}")
            summary["accuracy_known_classes"] = accuracy
            summary["n_known_class_flows"] = n_known

            emit(f"\n  confusion matrix (rows = true, columns = predicted)")
            true_labels = sorted(set(truth[known]),
                                 key=lambda c: list(encoder.classes_).index(c))
            pred_labels = list(encoder.classes_)
            width = max(len(str(l)) for l in true_labels) + 2
            emit("  " + " " * width
                 + "".join(f"{str(l)[:9]:>10}" for l in pred_labels)
                 + "      total")
            for true_label in true_labels:
                mask = truth == true_label
                row = [int(((predictions == p) & mask).sum())
                       for p in pred_labels]
                emit(f"  {str(true_label):<{width}}"
                     + "".join(f"{v:>10,}" for v in row)
                     + f"  {sum(row):>9,}")

            emit(f"\n  per class (restricted to the {n_known:,} known-class "
                 f"flows above)")
            emit(f"  {'class':<12} {'precision':>10} {'recall':>8} {'F1':>8} "
                 f"{'support':>9}")
            emit(f"  {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 9}")
            # Restricted to known-class flows so these figures are consistent
            # with the accuracy and confusion matrix above. Counting unseen
            # attack types as false positives here would silently blend "the
            # model confused two classes it knows" with "the model was never
            # taught this class" -- see the separate figure below.
            k_pred, k_truth = predictions[known], truth[known]
            per_class = {}
            for cls in encoder.classes_:
                tp = int(((k_pred == cls) & (k_truth == cls)).sum())
                fp = int(((k_pred == cls) & (k_truth != cls)).sum())
                fn = int(((k_pred != cls) & (k_truth == cls)).sum())
                support = int((k_truth == cls).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
                per_class[str(cls)] = {"precision": precision, "recall": recall,
                                       "f1": f1, "support": support}
                if support or tp + fp:
                    emit(f"  {str(cls):<12} {precision:>10.4f} {recall:>8.4f} "
                         f"{f1:>8.4f} {support:>9,}")
                else:
                    emit(f"  {str(cls):<12} {'-':>10} {'-':>8} {'-':>8} "
                         f"{0:>9,}  not present in this dataset")
            summary["per_class"] = per_class

            # Operationally the most important single number, and it is only
            # computable when unseen classes ARE counted: of everything the
            # model waved through as benign, how much genuinely was benign.
            called_benign = predictions == SEVERITY.benign_label
            if called_benign.any() and n_unknown:
                truly_benign = int((truth[called_benign]
                                    == SEVERITY.benign_label).sum())
                purity = truly_benign / int(called_benign.sum())
                emit(f"\n  Of the {int(called_benign.sum()):,} flows the model "
                     f"called {SEVERITY.benign_label}, {purity:.1%} genuinely "
                     f"were.")
                emit(f"  The other {int(called_benign.sum()) - truly_benign:,} "
                     f"are attack types outside the model's classes, counted in "
                     f"full below.")
                summary["benign_prediction_purity"] = purity

            # False positives on benign traffic are the number an operator
            # actually cares about: it sets the alert-fatigue baseline.
            benign_mask = truth == SEVERITY.benign_label
            if benign_mask.any():
                false_positives = int((predictions[benign_mask]
                                       != SEVERITY.benign_label).sum())
                rate = false_positives / int(benign_mask.sum())
                emit(f"\n  false positives on benign traffic: "
                     f"{false_positives:,} of {int(benign_mask.sum()):,} "
                     f"({rate:.3%})")
                summary["benign_false_positive_rate"] = rate
                if false_positives:
                    fp_conf = confidence[benign_mask
                                         & (predictions != SEVERITY.benign_label)]
                    emit(f"    their confidence: mean {fp_conf.mean():.3f}, "
                         f"median {np.median(fp_conf):.3f}, "
                         f"{int((fp_conf >= 0.90).sum()):,} in the CRITICAL band")

        # --- the important part: classes the model was never trained on ---
        if n_unknown:
            emit(f"\n{THIN}\n  ATTACK TYPES THE MODEL WAS NEVER TRAINED ON\n{THIN}")
            emit(f"  {n_unknown:,} flows carry a true label outside the model's "
                 f"{len(encoder.classes_)} classes.")
            emit("  A supervised classifier can only answer 'which of my classes")
            emit("  is this closest to?', never 'is this abnormal?'. So these")
            emit("  flows CANNOT be classified correctly -- the question is")
            emit("  whether they are at least flagged as something suspicious.")
            emit("")
            emit(f"  {'true label':<22} {'flows':>9} {'flagged':>9} "
                 f"{'missed as BENIGN':>18}")
            emit(f"  {'-' * 22} {'-' * 9} {'-' * 9} {'-' * 18}")

            unseen = {}
            for label in sorted(set(truth[~known]), key=str):
                mask = truth == label
                total = int(mask.sum())
                if not total:
                    continue
                as_benign = int((predictions[mask]
                                 == SEVERITY.benign_label).sum())
                flagged = total - as_benign
                unseen[str(label)] = {
                    "flows": total, "flagged": flagged,
                    "missed_as_benign": as_benign,
                    "miss_rate": as_benign / total,
                }
                emit(f"  {str(label):<22} {total:>9,} "
                     f"{flagged / total:>8.1%} {as_benign / total:>17.1%}")
            summary["unseen_classes"] = unseen

            total_unseen = sum(v["flows"] for v in unseen.values())
            total_missed = sum(v["missed_as_benign"] for v in unseen.values())
            if total_unseen:
                emit(f"\n  overall: {total_missed:,} of {total_unseen:,} "
                     f"unseen-attack flows "
                     f"({total_missed / total_unseen:.1%}) were labelled BENIGN.")
                summary["unseen_miss_rate"] = total_missed / total_unseen

                # Does the distribution check rescue any of them? Measured
                # rather than assumed: on real unseen attacks it does not, and
                # the report must say so plainly instead of implying coverage.
                unseen_mask = ~known
                rescued = int((unseen_mask & unreliable).sum())
                emit(f"\n  of those, the distribution check flagged "
                     f"{rescued:,} ({rescued / total_unseen:.2%}) as "
                     f"out-of-distribution.")
                summary["unseen_ood_rescue_rate"] = rescued / total_unseen

                if rescued / total_unseen < 0.05:
                    emit("")
                    emit("  The distribution check does NOT help here, and that")
                    emit("  is expected: these flows are novel by LABEL, not by")
                    emit("  feature distribution. A slow-HTTP DoS flow genuinely")
                    emit("  resembles an ordinary slow HTTP flow on these")
                    emit("  features, so there is nothing distributionally")
                    emit("  unusual to detect.")
                emit("")
                emit("  This is a structural limit, not a tuning problem. Fixing")
                emit("  it requires training data for these classes. The")
                emit("  distribution check covers a different failure mode:")
                emit("  input from a different pipeline, network, or tool")
                emit("  version.")

    # --- example alerts ---------------------------------------------------
    if show_examples:
        emit(f"\n{THIN}\n  HIGHEST-CONFIDENCE DETECTIONS\n{THIN}")
        # Rank on the REPORTED verdict, so OOD flows appear here too.
        threat_mask = reported != SEVERITY.benign_label
        if not threat_mask.any():
            emit("  No non-benign predictions in this dataset.")
        else:
            order = np.argsort(-confidence)
            shown = 0
            for position in order:
                if not threat_mask[position]:
                    continue
                row = frame.loc[index[position]].to_dict()
                result = predict_threat(row, profile=profile)
                identity = {k: row.get(k) for k in
                            list(IDENTITY_COLUMNS) + ["Destination Port"]
                            if k in row}
                alert = Alert.build(
                    result, f"S{position:06d}", identity,
                    ground_truth=(str(row[TARGET_COLUMN]) if labelled
                                  and TARGET_COLUMN in row else None),
                    observation_mode="BATCH_SCORING",
                )
                emit(f"  {alert.summary()}")
                for item in result["evidence_detail"][:2]:
                    emit(f"      {item['feature']} = {item['display']}"
                         + (f"  ({item['comparison']})"
                            if item["comparison"] else ""))
                shown += 1
                if shown >= show_examples:
                    break

    # --- outputs ----------------------------------------------------------
    stem = f"{input_path.stem}_{PATHS.profile_slug(profile)}"
    predictions_path = reports_dir / f"scored_{stem}.csv"
    output = pd.DataFrame({
        # The verdict an operator would act on.
        "reported_threat": reported,
        # The classifier's raw answer, retained even when the verdict overrides
        # it, so the two can be compared row by row.
        "model_prediction": predictions,
        "confidence": np.round(confidence, 4),
        "reliability": reliability_array,
        "ood_score": np.round(ood_scores, 4),
        "severity": severities,
        "is_alertable": [d.is_alertable for d in decisions],
        "needs_review": [d.needs_review for d in decisions],
    }, index=index)
    for column in list(IDENTITY_COLUMNS) + ["Destination Port"]:
        if column in frame.columns:
            output[column] = frame.loc[index, column]
    if labelled:
        output["ground_truth"] = frame.loc[index, TARGET_COLUMN]
        # Correctness is judged against the CLASSIFIER's answer, not the
        # reported verdict: UNKNOWN_OOD never matches a dataset label, so
        # scoring the verdict would count every OOD row as a miss and understate
        # the classifier itself.
        output["correct"] = output["ground_truth"] == output["model_prediction"]
    output.to_csv(predictions_path, index=False)
    emit(f"\n  per-flow predictions : {predictions_path}")
    summary["predictions_file"] = str(predictions_path)

    if save_alerts:
        alerts_path = reports_dir / f"alerts_{stem}.jsonl"
        written = review_written = 0
        with alerts_path.open("w", encoding="utf-8") as stream:
            for position in range(len(matrix)):
                decision = decisions[position]
                # Write BOTH queues. Writing only alertable flows would drop
                # every DEGRADED-but-benign row, which on the OOD fixture is 9
                # of 20 -- exactly the flows the review queue exists to keep
                # visible. The dashboard filters the two apart on load.
                if not (decision.is_alertable or decision.needs_review):
                    continue
                row = frame.loc[index[position]].to_dict()
                result = predict_threat(row, profile=profile)
                identity = {k: row.get(k) for k in
                            list(IDENTITY_COLUMNS) + ["Destination Port"]
                            if k in row}
                alert = Alert.build(
                    result, f"S{position:06d}", identity,
                    ground_truth=(str(row[TARGET_COLUMN]) if labelled
                                  and TARGET_COLUMN in row else None),
                    observation_mode="BATCH_SCORING",
                )
                stream.write(alert.to_json_line())
                written += 1
                if not decision.is_alertable:
                    review_written += 1
        emit(f"  alert stream         : {alerts_path}  ({written:,} records: "
             f"{written - review_written:,} alerts, {review_written:,} review)")
        emit("  Load it in the dashboard by copying it over runtime/alerts.jsonl")
        summary["alerts_file"] = str(alerts_path)
        summary["alerts_written"] = int(written)
        summary["review_written"] = int(review_written)

    summary_path = reports_dir / f"scored_{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str),
                            encoding="utf-8")
    emit(f"  summary              : {summary_path}")

    if save_report:
        report_path = reports_dir / f"scored_{stem}.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  text report          : {report_path}")

    emit(f"\n  total elapsed: {time.time() - t0:.1f}s")
    emit(RULE)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score an external flow dataset with a profile's model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, metavar="CSV",
                        help="Processed or raw CSV to score.")
    source.add_argument("--raw", type=Path, metavar="CSV",
                        help="Raw CIC-IDS2017-format CSV; cleaned first.")
    parser.add_argument("--profile", default=FEATURE_PROFILE,
                        choices=sorted(FEATURE_PROFILES),
                        help=f"Which model to score with "
                             f"(default: {FEATURE_PROFILE}). Each profile has "
                             f"its own feature contract.")
    parser.add_argument("--auto-profile", action="store_true",
                        help="Pick whichever profile this file's columns "
                             "satisfy, instead of failing on a mismatch.")
    parser.add_argument("--max-flows", type=int, default=None, metavar="N",
                        help="Score only the first N rows.")
    parser.add_argument("--save-alerts", action="store_true",
                        help="Write a JSONL alert stream the dashboard can read.")
    parser.add_argument("--save-report", action="store_true",
                        help="Write the full text report to reports/.")
    parser.add_argument("--examples", type=int, default=3, metavar="N",
                        help="Highest-confidence detections to show (default 3).")
    args = parser.parse_args()

    try:
        if args.raw:
            from preprocessing.clean_data import clean_dataset

            print(RULE)
            print(f"CLEANING {args.raw.name} FIRST")
            print(RULE)
            cleaned = PATHS.processed_dir / f"scored_input_{args.raw.stem}.csv"
            clean_dataset(files=[args.raw.name], max_per_class=None,
                          out_path=cleaned, verbose=True)
            print()
            target = cleaned
        else:
            target = args.input

        score(
            input_path=target,
            profile=args.profile,
            max_flows=args.max_flows,
            save_alerts=args.save_alerts,
            save_report=args.save_report,
            show_examples=args.examples,
            auto_profile=args.auto_profile,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

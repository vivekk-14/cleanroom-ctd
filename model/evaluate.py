"""
Evaluate the saved model on a held-out test split.

Reloads the artefacts from disk -- model.pkl, features.json, label_encoder.pkl
-- and scores them against a stratified split of the processed dataset,
reproduced with the same random_state that train.py used. This verifies the
saved artefacts, not an in-memory model that happened to work during training.

Reported: accuracy, balanced accuracy, per-class precision/recall/F1, macro and
weighted averages, confusion matrix (counts and row-normalised), confidence
calibration, and measured single-flow latency.

Every number is computed here from a real prediction. Nothing is copied from
metrics.json or hardcoded.

Usage
-----
    python -m model.evaluate
    python -m model.evaluate --plot        # save confusion matrix PNG
    python -m model.evaluate --save-report # write reports/evaluation.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_PROFILE, PATHS, TRAINING  # noqa: E402
from preprocessing.clean_data import TARGET_COLUMN  # noqa: E402
from preprocessing.feature_config import FEATURE_PROFILES  # noqa: E402

RULE = "=" * 76
THIN = "-" * 76


def _load(profile: str) -> tuple[object, object, list[str], dict]:
    """Load one profile's saved artefacts, with actionable errors if absent."""
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
    return model, encoder, list(contract["features"]), contract


def _confidence_table(probabilities: np.ndarray, y_true: np.ndarray,
                      y_pred: np.ndarray) -> list[dict]:
    """Bucket predictions by confidence and measure accuracy in each bucket.

    This is the check that matters for the severity policy: severity is derived
    from confidence, so if confidence does not track correctness then a
    CRITICAL label means nothing. Buckets follow the severity bands.
    """
    confidence = probabilities.max(axis=1)
    bands = [(0.0, 0.5, "LOW"), (0.5, 0.75, "MEDIUM"),
             (0.75, 0.9, "HIGH"), (0.9, 1.01, "CRITICAL")]
    rows = []
    for low, high, name in bands:
        mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        rows.append({
            "band": name,
            "range": f"[{low:.2f}, {high:.2f})" if high <= 1.0
                     else f"[{low:.2f}, 1.00]",
            "n": count,
            "share": count / len(confidence) if len(confidence) else 0.0,
            "accuracy": float((y_true[mask] == y_pred[mask]).mean())
                        if count else float("nan"),
        })
    return rows


def evaluate(profile: str = FEATURE_PROFILE, plot: bool = False,
             save_report: bool = False) -> dict:
    """Score the saved model and print a full report."""
    PATHS.ensure_dirs()
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    model, encoder, features, contract = _load(profile)

    emit(RULE)
    emit("MODEL EVALUATION")
    emit(RULE)
    model_path = PATHS.model_file(profile)
    emit(f"  model      : {model_path.relative_to(PATHS.root)} "
         f"({model_path.stat().st_size / 1_048_576:.2f} MB)")
    emit(f"  profile    : {contract.get('profile')}")
    emit(f"  features   : {len(features)}")
    emit(f"  classes    : {', '.join(str(c) for c in encoder.classes_)}")
    emit(f"  trained at : {contract.get('trained_at', 'unknown')}")

    if not PATHS.processed_csv.exists():
        raise FileNotFoundError(
            f"{PATHS.processed_csv} not found.\n"
            f"Run first:  python -m preprocessing.clean_data"
        )

    frame = pd.read_csv(PATHS.processed_csv, low_memory=False)
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(
            f"The processed dataset lacks features the model needs: {missing}\n"
            f"The dataset and model are out of sync. Re-run clean_data then train."
        )

    X = frame[features].to_numpy(dtype="float64")
    y = encoder.transform(frame[TARGET_COLUMN].to_numpy())

    # Reproduce train.py's split exactly. Same data, same test_size, same
    # random_state, same stratification -> the same held-out rows, none of which
    # the model saw during fitting.
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TRAINING.test_size,
        stratify=y if TRAINING.stratify else None,
        random_state=TRAINING.random_state,
    )
    emit(f"\n  held-out test rows: {len(X_test):,} of {len(X):,} "
         f"(test_size={TRAINING.test_size}, random_state="
         f"{TRAINING.random_state})")
    emit("  This is the same split train.py used, reproduced from the same seed,")
    emit("  so none of these rows were seen during fitting.")

    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_seconds = time.time() - t0
    probabilities = model.predict_proba(X_test)

    labels = np.arange(len(encoder.classes_))
    names = [str(c) for c in encoder.classes_]

    accuracy = float(accuracy_score(y_test, y_pred))
    balanced = float(balanced_accuracy_score(y_test, y_pred))

    emit(f"\n{THIN}\n  OVERALL\n{THIN}")
    emit(f"  accuracy              : {accuracy:.4f}")
    emit(f"  balanced accuracy     : {balanced:.4f}   "
         f"(mean per-class recall)")
    emit(f"  precision (macro)     : "
         f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    emit(f"  recall    (macro)     : "
         f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    emit(f"  F1        (macro)     : "
         f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    emit(f"  precision (weighted)  : "
         f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    emit(f"  recall    (weighted)  : "
         f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    emit(f"  F1        (weighted)  : "
         f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    emit("")
    emit("  Macro treats every class equally; weighted weights by support.")
    emit("  With Botnet at ~1.3% of rows, macro is the honest headline number:")
    emit("  weighted metrics are dominated by the three large classes.")

    emit(f"\n{THIN}\n  PER CLASS\n{THIN}")
    emit(classification_report(y_test, y_pred, labels=labels,
                               target_names=names, digits=4, zero_division=0))

    # --- confusion matrix -------------------------------------------------
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    width = max(len(n) for n in names) + 2

    emit(f"{THIN}\n  CONFUSION MATRIX (counts)\n{THIN}")
    emit("  rows = true class, columns = predicted class")
    emit("  " + " " * width + "".join(f"{n[:9]:>10}" for n in names) + "     total")
    for i, name in enumerate(names):
        row = "".join(f"{v:>10,}" for v in matrix[i])
        emit(f"  {name:<{width}}{row}  {matrix[i].sum():>8,}")

    emit(f"\n{THIN}\n  CONFUSION MATRIX (row-normalised, % of each true class)\n{THIN}")
    emit("  " + " " * width + "".join(f"{n[:9]:>10}" for n in names))
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    for i, name in enumerate(names):
        row = "".join(f"{v * 100:>9.2f}%" for v in normalised[i])
        emit(f"  {name:<{width}}{row}")

    # Name the actual error modes instead of leaving the matrix to be read.
    emit("\n  Largest confusions:")
    errors = [
        (matrix[i][j], names[i], names[j])
        for i in range(len(names)) for j in range(len(names)) if i != j
    ]
    errors.sort(reverse=True)
    total_errors = sum(count for count, _, _ in errors)
    if total_errors == 0:
        emit("    none: no misclassifications on the held-out set")
    else:
        for count, true_name, pred_name in errors[:5]:
            if count == 0:
                continue
            share = count / max(matrix[names.index(true_name)].sum(), 1)
            emit(f"    {count:>5,}  {true_name} predicted as {pred_name} "
                 f"({share:.2%} of all {true_name})")
        emit(f"    {total_errors:,} misclassified of {len(y_test):,} "
             f"({total_errors / len(y_test):.3%})")

    # --- confidence calibration -------------------------------------------
    emit(f"\n{THIN}\n  CONFIDENCE vs CORRECTNESS\n{THIN}")
    emit("  Severity is derived from confidence, so confidence must track")
    emit("  correctness for a CRITICAL label to mean anything.")
    emit("")
    emit(f"  {'band':<10} {'range':<14} {'flows':>8} {'share':>8} {'accuracy':>10}")
    emit(f"  {'-' * 10} {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 10}")
    confidence_rows = _confidence_table(probabilities, y_test, y_pred)
    for row in confidence_rows:
        accuracy_text = ("n/a" if row["accuracy"] != row["accuracy"]
                         else f"{row['accuracy']:.4f}")
        emit(f"  {row['band']:<10} {row['range']:<14} {row['n']:>8,} "
             f"{row['share']:>7.2%} {accuracy_text:>10}")

    # --- timing -----------------------------------------------------------
    warmup = X_test[:1]
    model.predict_proba(warmup)
    n_probe = min(300, len(X_test))
    t0 = time.time()
    for i in range(n_probe):
        model.predict_proba(X_test[i:i + 1])
    single_ms = (time.time() - t0) / n_probe * 1000

    emit(f"\n{THIN}\n  MEASURED PERFORMANCE\n{THIN}")
    emit(f"  batch predict     : {len(X_test):,} flows in "
         f"{predict_seconds:.3f}s "
         f"({len(X_test) / max(predict_seconds, 1e-9):,.0f} flows/s)")
    emit(f"  single flow       : {single_ms:.2f} ms "
         f"(mean of {n_probe} calls, single-threaded)")
    emit(f"  throughput        : ~{1000 / single_ms:.0f} flows/s one at a time")
    emit("")
    emit("  The replay engine classifies one flow at a time, so single-flow")
    emit("  latency is the figure that bounds the demo. Measured on this")
    emit("  machine; it will differ on other hardware.")

    # --- feature importance ----------------------------------------------
    if hasattr(model, "feature_importances_"):
        emit(f"\n{THIN}\n  FEATURE IMPORTANCE (global, from the fitted forest)\n{THIN}")
        ranked = sorted(zip(features, model.feature_importances_),
                        key=lambda pair: -pair[1])
        for name, importance in ranked:
            emit(f"  {importance:.4f}  {name:<30} {'#' * int(importance * 120)}")
        emit("")
        emit("  Global importance across the training set. It is NOT a")
        emit("  per-prediction attribution: it does not explain any individual")
        emit("  alert. See detection/evidence.py for per-alert evidence.")

    results = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "f1_macro": float(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted",
                                      zero_division=0)),
        "test_rows": int(len(X_test)),
        "single_flow_ms": round(single_ms, 2),
        "confidence_bands": confidence_rows,
    }

    # --- optional outputs -------------------------------------------------
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")  # no display needed
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            for axis, data, title, fmt in (
                (axes[0], matrix, "Confusion matrix (counts)", ",d"),
                (axes[1], normalised * 100,
                 "Row-normalised (% of true class)", ".1f"),
            ):
                image = axis.imshow(data, cmap="Blues")
                axis.set_xticks(range(len(names)), names, rotation=45,
                                ha="right")
                axis.set_yticks(range(len(names)), names)
                axis.set_xlabel("Predicted")
                axis.set_ylabel("True")
                axis.set_title(title)
                threshold = data.max() / 2
                for i in range(len(names)):
                    for j in range(len(names)):
                        axis.text(j, i, format(data[i][j], fmt),
                                  ha="center", va="center", fontsize=8,
                                  color="white" if data[i][j] > threshold
                                  else "black")
                figure.colorbar(image, ax=axis, fraction=0.046)

            figure.suptitle(
                f"{contract.get('profile')}  |  accuracy {accuracy:.4f}  |  "
                f"macro F1 {results['f1_macro']:.4f}  |  "
                f"{len(X_test):,} held-out flows"
            )
            figure.tight_layout()
            slug = PATHS.profile_slug(profile)
            output = PATHS.reports_dir / f"confusion_matrix_{slug}.png"
            figure.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(figure)
            emit(f"\n  confusion matrix saved: {output}")
        except ImportError:
            emit("\n  matplotlib not installed; skipping --plot")

    if save_report:
        output = (PATHS.reports_dir
                  / f"evaluation_{PATHS.profile_slug(profile)}.txt")
        output.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  report saved: {output}")

    print(RULE)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on a held-out split."
    )
    parser.add_argument("--profile", default=FEATURE_PROFILE,
                        choices=sorted(FEATURE_PROFILES),
                        help=f"Which profile's model to evaluate "
                             f"(default: {FEATURE_PROFILE}).")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate every trained profile.")
    parser.add_argument("--plot", action="store_true",
                        help="Save a confusion-matrix PNG to reports/.")
    parser.add_argument("--save-report", action="store_true",
                        help="Save the full text report to reports/.")
    args = parser.parse_args()

    try:
        if args.all:
            for name in FEATURE_PROFILES:
                if PATHS.model_file(name).exists():
                    evaluate(profile=name, plot=args.plot,
                             save_report=args.save_report)
                    print()
                else:
                    print(f"  skipping {name}: not trained")
        else:
            evaluate(profile=args.profile, plot=args.plot,
                     save_report=args.save_report)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

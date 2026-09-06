"""
Train the flow classifier.

    data/processed/flows_clean.csv
      -> select the feature profile's columns
      -> label-encode threat_class
      -> stratified train/test split
      -> RandomForestClassifier (class_weight='balanced_subsample')
      -> evaluate on the held-out test set
      -> write model/model.pkl, features.json, label_encoder.pkl, metrics.json

Every number written to metrics.json comes from an actual fit on real data.
Nothing is hardcoded or estimated.

Usage
-----
    python -m model.train
    python -m model.train --profile BIDIRECTIONAL
    python -m model.train --compare            # train both, print comparison
    python -m model.train --trees 100 --depth 0    # depth 0 = unlimited

No preprocessing scaler is used. Random forests split on thresholds per
feature, so they are invariant to monotonic rescaling; a StandardScaler would
add a moving part and a failure mode without changing the decision boundary.
This also means predict.py needs only the feature order to build a valid input
vector, which keeps the Student A / Student B contract small.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET, FEATURE_PROFILE, PATHS, SEVERITY, TRAINING  # noqa: E402
from detection.ood import build_reference as build_ood_reference  # noqa: E402
from preprocessing.clean_data import TARGET_COLUMN  # noqa: E402
from preprocessing.feature_config import (  # noqa: E402
    FEATURE_PROFILES,
    TARGET_CLASSES,
    get_features,
)

RULE = "=" * 76

# Sentinel for "argument omitted, use the config default". Needed because
# max_depth=None is itself a meaningful sklearn value meaning "unlimited",
# so None cannot double as "not specified".
_DEFAULT = object()


def _load_processed(path: Path) -> pd.DataFrame:
    """Load the processed CSV, with an actionable message if it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}\n"
            f"Run this first:  python -m preprocessing.clean_data"
        )
    frame = pd.read_csv(path, low_memory=False)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"{path.name} has no {TARGET_COLUMN!r} column. "
            f"Re-run preprocessing.clean_data to regenerate it."
        )
    return frame


def _select_features(frame: pd.DataFrame, profile: str) -> list[str]:
    """Return the profile's features that exist in the frame, reporting gaps.

    Missing features are a hard error rather than a silent subset: a model
    quietly trained on 14 of 18 features would still score well here and then
    disagree with predict.py's feature ordering, which is exactly the kind of
    bug that surfaces during a demo.
    """
    wanted = get_features(profile)
    present = [f for f in wanted if f in frame.columns]
    missing = [f for f in wanted if f not in frame.columns]

    if missing:
        raise ValueError(
            f"Profile {profile!r} needs {len(wanted)} features but "
            f"{len(missing)} are absent from the processed dataset:\n"
            f"  missing: {missing}\n"
            f"  present: {present}\n"
            f"Check preprocessing/feature_config.py and re-run clean_data."
        )
    return present


def _atomic_dump(obj: object, path: Path, compress: int = 0) -> None:
    """joblib.dump via a temporary file plus os.replace.

    A direct dump that is interrupted midway leaves a truncated model.pkl that
    loads without error but predicts garbage. Writing to a sibling temp file
    and renaming makes the swap atomic on the same filesystem, so the artefact
    on disk is always either the old complete model or the new complete model.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(obj, tmp, compress=compress)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(text: str, path: Path) -> None:
    """Same atomicity guarantee for JSON artefacts."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def train(
    profile: str = FEATURE_PROFILE,
    n_estimators: int | None = None,
    max_depth: int | None | object = _DEFAULT,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """Train one model and return its metrics dictionary.

    max_depth:
        omitted -> use TRAINING.max_depth from config
        int     -> that depth
        None    -> unlimited depth
    """
    t_start = time.time()
    PATHS.ensure_dirs()

    n_estimators = n_estimators or TRAINING.n_estimators
    depth = TRAINING.max_depth if max_depth is _DEFAULT else max_depth

    if verbose:
        print(RULE)
        print(f"TRAINING  profile={profile}")
        print(RULE)

    # --- data -------------------------------------------------------------
    frame = _load_processed(PATHS.processed_csv)
    features = _select_features(frame, profile)

    X = frame[features].to_numpy(dtype="float64")
    y_text = frame[TARGET_COLUMN].to_numpy()

    # Guard against non-finite values reaching the fit. clean_data should have
    # removed them; this catches a hand-edited or externally supplied CSV
    # rather than letting sklearn raise a less obvious error later.
    if not np.isfinite(X).all():
        bad_cols = [features[i] for i in
                    np.where(~np.isfinite(X).all(axis=0))[0]]
        raise ValueError(
            f"Non-finite values present in {bad_cols}. "
            f"Re-run preprocessing.clean_data."
        )

    # Fix class order so the label encoder, confusion matrix, and dashboard
    # colours agree between runs. LabelEncoder would otherwise sort
    # alphabetically, giving BENIGN, Botnet, DDoS, PortScan, which reads oddly
    # in a confusion matrix.
    observed = [c for c in TARGET_CLASSES if c in set(y_text)]
    extra = sorted(set(y_text) - set(TARGET_CLASSES))
    if extra:
        print(f"  WARNING: unexpected classes in data: {extra}")
        observed += extra

    encoder = LabelEncoder()
    encoder.fit(observed)
    y = encoder.transform(y_text)

    if verbose:
        print(f"  rows      : {len(X):,}")
        print(f"  features  : {len(features)}")
        print(f"  classes   : {', '.join(encoder.classes_)}")
        counts = pd.Series(y_text).value_counts()
        for cls in encoder.classes_:
            n = int(counts.get(cls, 0))
            print(f"    {cls:<12} {n:>8,}  ({n / len(X) * 100:5.2f}%)")

    if len(observed) < 2:
        raise ValueError(
            f"Only {len(observed)} class present. Classification needs >= 2."
        )

    # --- split ------------------------------------------------------------
    # Stratified: Botnet is ~1.3% of the capped set, so an unstratified split
    # could leave the test set with a handful of Botnet rows (or none), making
    # its recall meaningless.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAINING.test_size,
        stratify=y if TRAINING.stratify else None,
        random_state=TRAINING.random_state,
    )
    if verbose:
        print(f"\n  train/test: {len(X_train):,} / {len(X_test):,} "
              f"(test_size={TRAINING.test_size}, stratified={TRAINING.stratify})")

    # --- fit --------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=depth,
        min_samples_leaf=TRAINING.min_samples_leaf,
        class_weight=TRAINING.class_weight,
        n_jobs=TRAINING.n_jobs,
        random_state=TRAINING.random_state,
    )
    if verbose:
        print(f"  forest    : {n_estimators} trees, depth={depth}, "
              f"class_weight={TRAINING.class_weight}")
        print("  fitting...", end=" ", flush=True)

    t_fit = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t_fit
    if verbose:
        print(f"done in {fit_seconds:.1f}s")

    # --- OOD reference ----------------------------------------------------
    # Built from the TRAINING split only. Using all rows would leak held-out
    # data into the reference and make the measured false-OOD rate in
    # detection/ood.py look better than it is.
    benign_encoded = None
    if SEVERITY.benign_label in list(encoder.classes_):
        benign_encoded = int(
            encoder.transform([SEVERITY.benign_label])[0])

    ood_reference = build_ood_reference(
        X_train=X_train,
        features=features,
        y_train=y_train,
        benign_label=benign_encoded,
    )
    novelty_model = ood_reference.pop("_novelty_model", None)

    if verbose:
        print(f"  OOD ref   : {len(ood_reference['bounds'])} feature bounds "
              f"at q={ood_reference['quantile']}, "
              f"UNRELIABLE at >={ood_reference['min_violations_for_ood']} "
              f"violations")
        if "novelty" in ood_reference:
            novelty = ood_reference["novelty"]
            print(f"              advisory novelty detector on "
                  f"{novelty['n_rows']:,} benign rows "
                  f"(contamination {novelty['contamination']})")

    # --- evaluate ---------------------------------------------------------
    # Single-threaded for prediction. MEASURED: for one row at a time, thread
    # dispatch overhead dominates (21 ms with n_jobs=-1 vs 13.8 ms with 1).
    model.n_jobs = 1

    t_pred = time.time()
    y_pred = model.predict(X_test)
    batch_seconds = time.time() - t_pred

    accuracy = float(accuracy_score(y_test, y_pred))
    balanced = float(balanced_accuracy_score(y_test, y_pred))

    metrics = {
        "profile": profile,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accuracy": accuracy,
        # Balanced accuracy = mean per-class recall. With a 1.3% class present,
        # plain accuracy can look excellent while that class is ignored, so both
        # are reported.
        "balanced_accuracy": balanced,
        "precision_macro": float(precision_score(y_test, y_pred,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred,
                                           average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro",
                                   zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred,
                                                    average="weighted",
                                                    zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted",
                                              zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted",
                                      zero_division=0)),
        "per_class": {},
        "confusion_matrix": {
            "labels": list(encoder.classes_),
            "rows_are_true_columns_are_predicted": True,
            "matrix": confusion_matrix(
                y_test, y_pred, labels=np.arange(len(encoder.classes_))
            ).tolist(),
        },
        "classification_report": classification_report(
            y_test, y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=list(encoder.classes_),
            digits=4, zero_division=0,
        ),
        "feature_importance": {},
        "dataset": {
            "processed_file": PATHS.processed_csv.name,
            "total_rows": int(len(X)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "class_counts": {k: int(v) for k, v in
                             pd.Series(y_text).value_counts().items()},
        },
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": depth,
            "min_samples_leaf": TRAINING.min_samples_leaf,
            "class_weight": TRAINING.class_weight,
            "test_size": TRAINING.test_size,
            "stratify": TRAINING.stratify,
            "random_state": TRAINING.random_state,
        },
        "timing": {
            "fit_seconds": round(fit_seconds, 2),
            "test_set_predict_seconds": round(batch_seconds, 3),
            "test_set_rows": int(len(X_test)),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    per_class_p = precision_score(y_test, y_pred, average=None,
                                  labels=np.arange(len(encoder.classes_)),
                                  zero_division=0)
    per_class_r = recall_score(y_test, y_pred, average=None,
                               labels=np.arange(len(encoder.classes_)),
                               zero_division=0)
    per_class_f = f1_score(y_test, y_pred, average=None,
                           labels=np.arange(len(encoder.classes_)),
                           zero_division=0)
    for i, cls in enumerate(encoder.classes_):
        metrics["per_class"][cls] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f[i]),
            "support": int((y_test == i).sum()),
        }

    metrics["feature_importance"] = {
        f: float(imp) for f, imp in
        sorted(zip(features, model.feature_importances_),
               key=lambda pair: -pair[1])
    }

    # --- single-flow latency ----------------------------------------------
    # The replay engine classifies one flow at a time, so per-row latency is
    # the number that matters for the "is it fast enough" claim. Measured here
    # rather than asserted.
    warmup = X_test[:1]
    model.predict_proba(warmup)
    n_probe = min(200, len(X_test))
    t_single = time.time()
    for i in range(n_probe):
        model.predict_proba(X_test[i:i + 1])
    single_ms = (time.time() - t_single) / n_probe * 1000
    metrics["timing"]["single_flow_predict_ms"] = round(single_ms, 2)
    metrics["timing"]["single_flow_probe_count"] = n_probe
    metrics["timing"]["throughput_flows_per_second"] = round(1000 / single_ms, 1)

    if verbose:
        print(f"\n  accuracy          : {accuracy:.4f}")
        print(f"  balanced accuracy : {balanced:.4f}")
        print(f"  macro F1          : {metrics['f1_macro']:.4f}")
        print(f"  weighted F1       : {metrics['f1_weighted']:.4f}")
        print(f"  single-flow latency: {single_ms:.2f} ms "
              f"(~{1000 / single_ms:.0f} flows/s, single-threaded)")
        print()
        print(metrics["classification_report"])

        print("  confusion matrix (rows = true, columns = predicted)")
        labels = list(encoder.classes_)
        width = max(len(l) for l in labels) + 2
        header = " " * width + "".join(f"{l[:9]:>10}" for l in labels)
        print(f"    {header}")
        for i, label in enumerate(labels):
            row = "".join(f"{v:>10,}" for v in
                          metrics["confusion_matrix"]["matrix"][i])
            print(f"    {label:<{width}}{row}")

        print("\n  feature importance")
        for feat, imp in list(metrics["feature_importance"].items()):
            bar = "#" * int(imp * 100)
            print(f"    {imp:.4f}  {feat:<28} {bar}")

    # --- save -------------------------------------------------------------
    if save:
        directory = PATHS.ensure_profile_dir(profile)
        model_path = PATHS.model_file(profile)
        features_path = PATHS.features_file(profile)
        encoder_path = PATHS.label_encoder_file(profile)
        metrics_path = PATHS.metrics_file(profile)
        ood_path = PATHS.ood_reference_file(profile)
        novelty_path = PATHS.novelty_model_file(profile)

        _atomic_dump(model, model_path, compress=TRAINING.compress)
        _atomic_dump(encoder, encoder_path, compress=TRAINING.compress)

        # features.json is the interface contract. predict.py reads ONLY this
        # to build its input vector, so the feature order can never drift out
        # of sync with the trained model.
        contract = {
            "profile": profile,
            "feature_count": len(features),
            "features": features,
            "classes": list(encoder.classes_),
            "target_column": TARGET_COLUMN,
            "requires_scaling": False,
            "trained_at": metrics["trained_at"],
            "model_file": model_path.name,
            "label_encoder_file": encoder_path.name,
            "ood_reference_file": ood_path.name,
            "notes": (
                "Feature order in 'features' is the exact column order the "
                "model was fitted on. predict_threat() must build its input "
                "vector in this order, and validate_features() enforces it. No "
                "scaler is required: random forests split on per-feature "
                "thresholds and are invariant to monotonic rescaling. This "
                "contract applies ONLY to this profile: a record satisfying a "
                "different profile's contract cannot be scored by this model."
            ),
        }
        _atomic_write_text(json.dumps(contract, indent=2), features_path)
        _atomic_write_text(json.dumps(metrics, indent=2), metrics_path)
        _atomic_write_text(json.dumps(ood_reference, indent=2), ood_path)

        if novelty_model is not None:
            _atomic_dump(novelty_model, novelty_path,
                         compress=TRAINING.compress)
        else:
            # A stale detector from an earlier run would be applied to a model
            # it was not fitted alongside.
            novelty_path.unlink(missing_ok=True)

        size_mb = model_path.stat().st_size / 1_048_576
        if verbose:
            print(f"\n  saved to {directory.relative_to(PATHS.root)}/")
            print(f"    {model_path.name:<22} {size_mb:.2f} MB")
            print(f"    {features_path.name:<22} "
                  f"{len(features)} features, profile={profile}")
            print(f"    {encoder_path.name:<22} "
                  f"{len(encoder.classes_)} classes")
            print(f"    {metrics_path.name:<22} full metrics")
            print(f"    {ood_path.name:<22} "
                  f"{len(ood_reference['bounds'])} distribution bounds")
            if novelty_model is not None:
                print(f"    {novelty_path.name:<22} advisory novelty detector")

    if verbose:
        print(f"\n  total elapsed: {time.time() - t_start:.1f}s")
        print(RULE)

    metrics["_features"] = features
    return metrics


def compare_profiles(verbose: bool = True) -> None:
    """Train every profile and print a side-by-side comparison.

    Every profile now SAVES its own artefacts, because each lives in its own
    directory and cannot overwrite another. Previously only the default profile
    was persisted, since they shared one model.pkl -- which is precisely the
    coupling that allowed a record from one profile to be scored by the other's
    model.
    """
    order = [p for p in FEATURE_PROFILES if p != FEATURE_PROFILE] + [FEATURE_PROFILE]
    results: dict[str, dict] = {}

    for profile in order:
        results[profile] = train(profile=profile, save=True, verbose=verbose)
        print()

    print(RULE)
    print("PROFILE COMPARISON")
    print(RULE)
    print(f"  {'profile':<24} {'feats':>6} {'accuracy':>9} {'bal.acc':>9} "
          f"{'macro F1':>9} {'fit s':>7} {'ms/flow':>8}")
    print(f"  {'-' * 24} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 7} {'-' * 8}")
    for profile, m in results.items():
        star = " *" if profile == FEATURE_PROFILE else "  "
        print(f"  {profile:<22}{star} {len(m['_features']):>6} "
              f"{m['accuracy']:>9.4f} {m['balanced_accuracy']:>9.4f} "
              f"{m['f1_macro']:>9.4f} {m['timing']['fit_seconds']:>7.1f} "
              f"{m['timing']['single_flow_predict_ms']:>8.2f}")
    print("  * = default profile, used when no --profile is given")
    print("  All profiles saved to their own model/profiles/<slug>/ directory.")

    # Per-class F1 exposes what the aggregate numbers hide: the profiles differ
    # mainly on the rare class.
    all_classes = sorted({c for m in results.values() for c in m["per_class"]},
                         key=lambda c: TARGET_CLASSES.index(c)
                         if c in TARGET_CLASSES else 99)
    print(f"\n  per-class F1")
    print(f"  {'profile':<24}" + "".join(f"{c:>12}" for c in all_classes))
    print(f"  {'-' * 24}" + "".join(f"{'-' * 12}" for _ in all_classes))
    for profile, m in results.items():
        row = "".join(f"{m['per_class'].get(c, {}).get('f1', float('nan')):>12.4f}"
                      for c in all_classes)
        print(f"  {profile:<24}{row}")

    best = max(results.items(), key=lambda kv: kv[1]["f1_macro"])
    print(f"\n  Highest macro F1: {best[0]} ({best[1]['f1_macro']:.4f})")
    if best[0] != FEATURE_PROFILE:
        print(f"  NOTE: {best[0]} scores higher than the default "
              f"{FEATURE_PROFILE}, but the default is chosen on threat-model")
        print("  grounds, not score: a unidirectional tap may not expose "
              "reverse-path features at all.")
    print(RULE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the flow classifier.")
    parser.add_argument(
        "--profile", default=FEATURE_PROFILE, choices=sorted(FEATURE_PROFILES),
        help=f"Feature profile to train (default: {FEATURE_PROFILE}). Each "
             f"profile is a separate model with its own feature contract.",
    )
    parser.add_argument("--trees", type=int, default=None,
                        help=f"Number of trees (default: {TRAINING.n_estimators}).")
    parser.add_argument("--depth", type=int, default=None,
                        help=f"Max depth, 0 = unlimited "
                             f"(default: {TRAINING.max_depth}).")
    parser.add_argument("--compare", action="store_true",
                        help="Train every profile and print a comparison.")
    parser.add_argument("--all", action="store_true",
                        help="Train every profile, without the comparison table.")
    parser.add_argument("--no-save", action="store_true",
                        help="Evaluate without writing artefacts to disk.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        if args.compare:
            compare_profiles(verbose=not args.quiet)
        elif args.all:
            for profile in FEATURE_PROFILES:
                train(profile=profile, save=not args.no_save,
                      verbose=not args.quiet)
                print()
            print(f"  trained profiles: "
                  f"{', '.join(PATHS.trained_profiles())}")
        else:
            # --depth 0 means unlimited; the flag being absent means
            # "use the config default", which the sentinel expresses.
            if args.depth is None:
                depth: int | None | object = _DEFAULT
            elif args.depth == 0:
                depth = None
            else:
                depth = args.depth

            train(
                profile=args.profile,
                n_estimators=args.trees,
                max_depth=depth,
                save=not args.no_save,
                verbose=not args.quiet,
            )
            if not args.no_save:
                print(f"\n  Next step:  python -m model.evaluate "
                      f"--profile {args.profile}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

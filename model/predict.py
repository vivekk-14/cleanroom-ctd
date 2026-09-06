"""
Prediction engine -- the interface between Student A's ML work and Student B's
streaming/dashboard work.

THIS IS THE CONTRACT. Student B needs only this:

    from model.predict import predict_threat

    result = predict_threat(flow_features)   # dict of feature -> value

    result == {
        "threat":       "DDoS",              # str, one of the model's classes
        "confidence":   0.9612,              # float in [0, 1]
        "severity":     "CRITICAL",          # str
        "evidence":     {feature: value},    # dict, 2-4 supporting values
        # --- plus supporting detail ---
        "base_severity":      "HIGH",
        "escalated":          True,
        "severity_reason":    "...",
        "is_alertable":       True,
        "evidence_detail":    [ {...}, ... ],
        "evidence_method":    "...",
        "aggregate_context":  "..." | None,
        "class_probabilities": {"BENIGN": 0.03, "DDoS": 0.96, ...},
        "model_ready":        True,
        "missing_features":   [],
        "profile":            "STRICT_UNIDIRECTIONAL",
    }

Guarantees
----------
1. predict_threat() NEVER raises. A malformed flow yields a result with
   model_ready or a diagnostic field set, not a traceback. A replay loop or
   dashboard must not die on one bad row.

2. It works BEFORE the model is trained. With no model.pkl on disk it returns
   threat="UNKNOWN", confidence=0.0, model_ready=False. Student B can therefore
   build and test the entire dashboard before Student A finishes training.

3. Feature order is read from model/features.json, which train.py writes. The
   caller passes a plain dict in any order; ordering is handled here. The order
   can never drift out of sync with the trained model.

4. Artefacts are loaded once and cached. Loading model.pkl per flow would add
   tens of milliseconds to every prediction.

No scaler is involved: random forests split on per-feature thresholds, so they
are invariant to monotonic rescaling. features.json records
requires_scaling=false to make that explicit.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURE_PROFILE, PATHS  # noqa: E402
from detection import ood as ood_module  # noqa: E402
from detection import severity as severity_module  # noqa: E402
from detection.evidence import build_evidence  # noqa: E402
from preprocessing.feature_config import (  # noqa: E402
    FEATURE_PROFILES,
    FeatureContractError,
    profiles_satisfied_by,
    validate_features,
)

# Returned when no trained model is available. Chosen so the dashboard renders
# something honest rather than an empty row or a fabricated class.
UNKNOWN_THREAT = "UNKNOWN"

# Value substituted for a feature the caller did not supply. Zero is used
# because every feature in both profiles is a count, size, rate or duration for
# which zero is a real, in-range observation, so it cannot push the tree into an
# unseen region of feature space. `missing_features` always lists what was
# filled, so a substitution is never invisible.
MISSING_FILL = 0.0

# Largest magnitude a feature value may have.
#
# scikit-learn's tree code casts its input to float32 internally. Anything above
# float32's maximum (~3.4e38) becomes inf during that cast, and the estimator
# then rejects the row -- reported as an overflow RuntimeWarning followed by a
# failed prediction. Values that large cannot be genuine flow measurements: the
# largest value anywhere in the real dataset is about 2.1e9 (Flow Bytes/s), so
# rejecting beyond float32 range discards only corrupt input. Such values are
# reported through `missing_features` rather than silently truncated.
MAX_FEATURE_MAGNITUDE = 3.0e38


@dataclass
class _Artefacts:
    """Cached model artefacts."""

    model: Any
    encoder: Any
    features: list[str]
    profile: str
    classes: list[str]


# One cache entry per profile. Two profiles can be loaded simultaneously without
# evicting each other, which matters when a caller scores the same flow under
# both to compare, and prevents a shared slot from ever handing back the wrong
# model for a requested profile.
_CACHE: dict[str, _Artefacts] = {}
_LOAD_ERROR: dict[str, str] = {}


def _resolve_profile(profile: str | None) -> str:
    """Return a validated profile name, defaulting to the configured one."""
    name = (profile or FEATURE_PROFILE).strip().upper()
    if name not in FEATURE_PROFILES:
        valid = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(f"Unknown profile {profile!r}. Valid options: {valid}")
    return name


def _load_artefacts(profile: str | None = None) -> _Artefacts | None:
    """Load and cache one profile's model, encoder and feature contract.

    Returns None if anything is missing or unreadable, recording why in
    _LOAD_ERROR[profile]. Never raises for a missing artefact: an untrained
    project must still run. An unknown profile name DOES raise, because that is
    a programming error rather than a missing file.
    """
    name = _resolve_profile(profile)

    if name in _CACHE:
        return _CACHE[name]
    if name in _LOAD_ERROR:
        return None

    artefacts = PATHS.profile_artefacts(name)
    required = ("model", "features", "label_encoder")
    missing = [artefacts[role].name for role in required
               if not artefacts[role].exists()]
    if missing:
        _LOAD_ERROR[name] = (
            f"artefacts for profile {name} not found: {', '.join(missing)} "
            f"(expected in {PATHS.profile_dir(name).relative_to(PATHS.root)}). "
            f"Run: python -m model.train --profile {name}"
        )
        return None

    try:
        import joblib  # imported here so the module loads without sklearn

        contract = json.loads(artefacts["features"].read_text(encoding="utf-8"))
        features = list(contract["features"])
        model = joblib.load(artefacts["model"])
        encoder = joblib.load(artefacts["label_encoder"])

        # Single-threaded prediction. MEASURED: for one row at a time, thread
        # dispatch costs more than the work itself (about 21 ms with n_jobs=-1
        # against 12 ms with n_jobs=1).
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        # The contract must name the profile it was saved for. A directory whose
        # features.json claims a different profile means the artefacts were
        # copied or renamed by hand, and nothing downstream could be trusted.
        stored_profile = str(contract.get("profile", "")).upper()
        if stored_profile != name:
            _LOAD_ERROR[name] = (
                f"artefact mismatch: {artefacts['features'].name} in the {name} "
                f"directory declares profile {stored_profile!r}. Re-run: "
                f"python -m model.train --profile {name}"
            )
            return None

        # The fitted model's feature count must match the contract's, and both
        # must match the profile definition in feature_config. Three-way, because
        # a stale contract that agrees with a stale model would otherwise pass.
        expected_from_model = getattr(model, "n_features_in_", len(features))
        expected_from_config = len(FEATURE_PROFILES[name])
        if not (expected_from_model == len(features) == expected_from_config):
            _LOAD_ERROR[name] = (
                f"artefact mismatch for {name}: model expects "
                f"{expected_from_model} features, features.json lists "
                f"{len(features)}, feature_config defines "
                f"{expected_from_config}. Re-run: "
                f"python -m model.train --profile {name}"
            )
            return None

        if features != list(FEATURE_PROFILES[name]):
            _LOAD_ERROR[name] = (
                f"artefact mismatch for {name}: the saved feature ORDER differs "
                f"from feature_config. The model was trained before the profile "
                f"was edited. Re-run: python -m model.train --profile {name}"
            )
            return None

        _CACHE[name] = _Artefacts(
            model=model,
            encoder=encoder,
            features=features,
            profile=name,
            classes=[str(c) for c in encoder.classes_],
        )
        return _CACHE[name]

    except Exception as exc:  # noqa: BLE001
        _LOAD_ERROR[name] = (
            f"{type(exc).__name__} while loading {name} artefacts: {exc}"
        )
        return None


def reset_cache(profile: str | None = None) -> None:
    """Drop cached artefacts so the next call reloads from disk.

    With no argument, clears every profile. Used by tests, and after retraining
    inside a long-running process.
    """
    if profile is None:
        _CACHE.clear()
        _LOAD_ERROR.clear()
        return
    name = _resolve_profile(profile)
    _CACHE.pop(name, None)
    _LOAD_ERROR.pop(name, None)


def is_model_ready(profile: str | None = None) -> bool:
    """True if the given profile's model is loaded or loadable."""
    try:
        return _load_artefacts(profile) is not None
    except ValueError:
        return False


def available_profiles() -> list[str]:
    """Profiles that have a usable trained model on disk."""
    return [name for name in FEATURE_PROFILES if is_model_ready(name)]


def model_status(profile: str | None = None) -> dict:
    """Describe artefact availability for one profile.

    Also lists every profile's readiness, so the dashboard can show which models
    exist without calling this once per profile.
    """
    name = _resolve_profile(profile)
    artefacts = _load_artefacts(name)

    all_profiles = {
        candidate: {
            "trained": candidate in _CACHE or (
                PATHS.model_file(candidate).exists()
                and PATHS.features_file(candidate).exists()
            ),
            "n_features": len(FEATURE_PROFILES[candidate]),
            "is_default": candidate == FEATURE_PROFILE,
        }
        for candidate in FEATURE_PROFILES
    }

    if artefacts is None:
        return {
            "model_ready": False,
            "profile": name,
            "error": _LOAD_ERROR.get(name),
            "hint": f"Run: python -m model.train --profile {name}",
            "profiles": all_profiles,
        }
    return {
        "model_ready": True,
        "profile": artefacts.profile,
        "n_features": len(artefacts.features),
        "features": list(artefacts.features),
        "classes": list(artefacts.classes),
        "model_file": str(PATHS.model_file(name)),
        "profile_dir": str(PATHS.profile_dir(name).relative_to(PATHS.root)),
        "profiles": all_profiles,
    }


def get_expected_features(profile: str | None = None) -> list[str]:
    """The exact feature names, in order, that a profile's model expects.

    Falls back to the profile definition in feature_config when the model is not
    trained yet, so a caller can still discover the contract. Returns [] only for
    an unknown profile.
    """
    try:
        name = _resolve_profile(profile)
    except ValueError:
        return []
    artefacts = _load_artefacts(name)
    if artefacts is not None:
        return list(artefacts.features)
    return list(FEATURE_PROFILES[name])


def _coerce(value: Any) -> float | None:
    """Convert a single feature value to a usable float, or None if impossible.

    Handles the shapes that arrive in practice: Python numbers, numpy scalars,
    strings from a CSV, None, NaN and infinity.

    Rejects NaN, infinity, and magnitudes beyond float32 range. sklearn casts to
    float32 internally, so an out-of-range value would become inf there and the
    prediction would fail; catching it here turns a hard failure into a reported
    missing feature.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if number in (float("inf"), float("-inf")):
        return None
    if abs(number) > MAX_FEATURE_MAGNITUDE:
        return None
    return number


def _unknown_result(reason: str, detail: str | None = None,
                    features: dict | None = None,
                    model_ready: bool = False,
                    missing: list[str] | None = None,
                    profile: str | None = None,
                    contract_ok: bool = True) -> dict:
    """Build a well-formed result for the case where no prediction was possible.

    Same keys as a successful result, so callers need no special-casing.

    `model_ready` reports the state of the MODEL, not of this call. An unusable
    input row must not be reported as a missing model: the dashboard uses this
    flag to decide whether to tell the user to run training, and that advice
    would be wrong. `error` explains what went wrong either way.

    `contract_ok` is False when the input did not satisfy the profile's feature
    contract, which is a different failure from a missing model and needs a
    different fix.
    """
    result = {
        "threat": UNKNOWN_THREAT,
        "confidence": 0.0,
        "severity": "INFO",
        "evidence": dict(list((features or {}).items())[:3]),
        "model_prediction": None,
        "model_confidence": 0.0,
        "confidence_means": (
            "No classification was produced, so there is no confidence to "
            "report."
        ),
        "base_severity": "INFO",
        "escalated": False,
        "severity_reason": reason,
        "is_alertable": False,
        # A record that could not be scored is not an alert, but it must not be
        # silently dropped either: a contract violation or a missing model is an
        # operational problem someone has to see.
        "needs_review": True,
        "evidence_detail": [],
        "evidence_method": (
            "No classification was produced, so no evidence was generated."
        ),
        "aggregate_context": None,
        "class_probabilities": {},
        "model_ready": model_ready,
        "missing_features": list(missing or []),
        "profile": profile,
        "contract_satisfied": contract_ok,
        # Distribution status is unknown: no vector was ever built, so no
        # distribution check could run. Reported as None rather than guessed.
        "in_distribution": None,
        "reliability": ood_module.UNKNOWN,
        "ood_score": 0.0,
        "ood_features": [],
        "ood_violations": [],
        "ood_reason": (
            "No prediction was made, so no distribution check was performed."
        ),
        "novelty_score": None,
        "novelty_flagged": None,
        "ood_reference_available": False,
        "ood_features_checked": 0,
    }
    if detail:
        result["error"] = detail
    return result


def predict_threat(
    flow_features: dict[str, Any],
    profile: str | None = None,
    *,
    strict: bool = True,
) -> dict:
    """Classify one flow using the given profile's model.

    Args:
        flow_features: mapping of canonical feature name to numeric value. Key
            order is irrelevant; ordering is imposed from the profile's contract.
            Extra keys (identity columns, other profiles' features) are ignored.
        profile: which model to use. Defaults to config.FEATURE_PROFILE
            ("STRICT_UNIDIRECTIONAL"). Each profile is a separate model with its
            own feature contract.
        strict: when True (default), a record missing any contract feature is
            REFUSED. When False, absent features are filled with 0.0 and listed
            in `missing_features`.

    Returns:
        The standard prediction dict. Never raises.

    Why strict=True is the default
    -----------------------------
    Filling absent features with zeros and predicting anyway produced a real
    failure: a 14-feature BIDIRECTIONAL record was scored by the 18-feature
    STRICT_UNIDIRECTIONAL model, 13 features were silently zero-filled, and the
    model returned confident classifications computed mostly from invented data.
    Refusing is the only safe default. `strict=False` remains available for the
    genuine case of a single dropped column in otherwise valid telemetry.
    """
    try:
        name = _resolve_profile(profile)
    except ValueError as exc:
        return _unknown_result(str(exc), detail=str(exc), contract_ok=False)

    if not isinstance(flow_features, dict):
        return _unknown_result(
            "Input was not a dict of feature values.",
            detail=f"got {type(flow_features).__name__}",
            model_ready=is_model_ready(name),
            profile=name, contract_ok=False,
        )

    artefacts = _load_artefacts(name)
    if artefacts is None:
        return _unknown_result(
            f"No trained model for profile {name}, so this flow was not "
            f"classified.",
            detail=_LOAD_ERROR.get(name),
            features={k: v for k, v in flow_features.items()
                      if isinstance(v, (int, float))},
            profile=name,
        )

    # --- feature contract -------------------------------------------------
    # Enforced BEFORE any vector is built, so a mismatched record cannot reach
    # the model at all.
    if strict:
        try:
            ordered = validate_features(flow_features, artefacts.features)
        except FeatureContractError as exc:
            suggestion = profiles_satisfied_by(flow_features)
            detail = str(exc)
            if suggestion:
                detail += (
                    f"\nThis record satisfies: {', '.join(suggestion)}. "
                    f"Call predict_threat(flow, profile='{suggestion[0]}')."
                )
            return _unknown_result(
                f"Input does not satisfy the {name} feature contract: "
                f"{len(exc.missing)} of {len(artefacts.features)} features "
                f"absent. No prediction was attempted.",
                detail=detail,
                features={k: v for k, v in flow_features.items()
                          if isinstance(v, (int, float))},
                model_ready=True, missing=exc.missing,
                profile=name, contract_ok=False,
            )
    else:
        ordered = {name_: flow_features.get(name_)
                   for name_ in artefacts.features}

    # --- build the input vector in the contract's exact order --------------
    row = np.empty(len(artefacts.features), dtype="float64")
    missing: list[str] = []
    for i, feature_name in enumerate(artefacts.features):
        coerced = _coerce(ordered.get(feature_name))
        if coerced is None:
            row[i] = MISSING_FILL
            missing.append(feature_name)
        else:
            row[i] = coerced

    # Present-but-unusable values (NaN, infinity, non-numeric text) survive the
    # contract check, so this remains necessary even in strict mode.
    if len(missing) == len(artefacts.features):
        return _unknown_result(
            "No usable feature values were found in the input, so no "
            "prediction was attempted.",
            detail=(f"all {len(missing)} contract features were present but "
                    f"none held a finite number below "
                    f"{MAX_FEATURE_MAGNITUDE:.1e}."),
            model_ready=True, missing=missing, profile=name,
        )

    # --- predict ----------------------------------------------------------
    try:
        probabilities = artefacts.model.predict_proba(row.reshape(1, -1))[0]
    except Exception as exc:  # noqa: BLE001
        return _unknown_result(
            "The model failed to score this flow.",
            detail=f"{type(exc).__name__}: {exc}",
            model_ready=True, missing=missing, profile=name,
        )

    best = int(np.argmax(probabilities))
    model_prediction = str(artefacts.encoder.inverse_transform([best])[0])
    model_confidence = float(probabilities[best])

    # --- distribution check -----------------------------------------------
    # Runs AFTER the classifier and is kept strictly separate from it. The
    # classifier answers "which class is this closest to?"; this answers "have I
    # seen anything like this before?". Conflating them is what makes a
    # confident BENIGN on unfamiliar input dangerous.
    ood = ood_module.assess(row, name, artefacts.features)

    # The reported threat class. When the input is outside the training
    # distribution the classifier's answer is not withdrawn -- it is preserved in
    # `model_prediction` -- but it is no longer presented as the verdict.
    if ood.reliability == ood_module.UNRELIABLE:
        threat = ood_module.OOD_THREAT_CLASS
    else:
        threat = model_prediction

    # --- severity and evidence -------------------------------------------
    decision = severity_module.assess(
        threat, model_confidence,
        reliability=ood.reliability,
        model_prediction=model_prediction,
    )
    evidence = build_evidence(model_prediction, flow_features, profile=name)

    result: dict[str, Any] = {
        # The four keys the problem statement requires.
        "threat": threat,
        "confidence": round(model_confidence, 4),
        "severity": decision.severity,
        "evidence": evidence.as_simple_dict(),

        # --- classifier output, kept separate from the verdict ---
        # `confidence` above is the CLASSIFIER's confidence in
        # `model_prediction`. It is NOT a probability that the flow is an
        # attack, and when in_distribution is False it should not be read as
        # supporting the reported threat class at all.
        "model_prediction": model_prediction,
        "model_confidence": round(model_confidence, 4),
        "confidence_means": (
            "The trained classifier's probability for 'model_prediction'. It "
            "measures the model's internal certainty, not the reliability of "
            "that answer. Read it together with 'in_distribution'."
        ),

        # Severity derivation, so the dashboard can explain the label.
        "base_severity": decision.base_severity,
        "escalated": decision.escalated,
        "severity_reason": decision.reason,
        "is_alertable": decision.is_alertable,
        # needs_review is a SEPARATE queue from is_alertable. A benign-looking
        # flow with one out-of-range feature is not an actionable alert, but it
        # must not disappear either.
        "needs_review": decision.needs_review,
        # Evidence detail.
        "evidence_detail": evidence.items,
        "evidence_method": evidence.method,
        "aggregate_context": evidence.aggregate_note,
        # Full distribution: lets the dashboard show the runner-up class, which
        # is what makes a low-confidence prediction interpretable.
        "class_probabilities": {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(artefacts.encoder.classes_, probabilities)
        },
        # Provenance.
        "model_ready": True,
        "missing_features": missing,
        "profile": artefacts.profile,
        "contract_satisfied": True,
    }
    # Distribution assessment: in_distribution, reliability, ood_score,
    # ood_features, ood_violations, ood_reason, novelty_score.
    result.update(ood.to_dict())
    return result


def predict_batch(rows: Iterable[dict[str, Any]],
                  profile: str | None = None,
                  *, strict: bool = True) -> list[dict]:
    """Classify many flows with one profile.

    Convenience wrapper for offline scoring. It calls predict_threat per row and
    therefore has the same per-row latency; it is NOT a vectorised fast path.
    The replay engine handles one flow at a time by design, so a batched
    implementation would add complexity the demo never exercises.
    """
    return [predict_threat(row, profile, strict=strict) for row in rows]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 76)
    print("PREDICTION ENGINE - self-test")
    print("=" * 76)

    status = model_status()
    print(f"\n  model_ready : {status['model_ready']}")
    if not status["model_ready"]:
        print(f"  error       : {status['error']}")
        print(f"  hint        : {status['hint']}")
        print("\n  Demonstrating the untrained-model path "
              "(Student B can build against this):")
        result = predict_threat({"Flow Duration": 1000.0})
        print(json.dumps(result, indent=2)[:700])
        raise SystemExit(0)

    print(f"  profile     : {status['profile']}")
    print(f"  features    : {status['n_features']}")
    print(f"  classes     : {', '.join(status['classes'])}")

    # Score real rows from the processed dataset, so nothing here is invented.
    if PATHS.processed_csv.exists():
        import pandas as pd

        frame = pd.read_csv(PATHS.processed_csv, low_memory=False)
        print(f"\n  scoring one real flow of each class from "
              f"{PATHS.processed_csv.name}")

        for cls in ["BENIGN", "DDoS", "PortScan", "Botnet"]:
            subset = frame[frame["threat_class"] == cls]
            if subset.empty:
                continue
            record = subset.iloc[0].to_dict()
            result = predict_threat(record)

            mark = "correct" if result["threat"] == cls else \
                   f"MISCLASSIFIED (true {cls})"
            print(f"\n  {'-' * 72}")
            print(f"  true={cls}  ->  predicted={result['threat']}  [{mark}]")
            print(f"    confidence : {result['confidence']:.4f}")
            print(f"    severity   : {result['severity']} "
                  f"(base {result['base_severity']}"
                  f"{', escalated' if result['escalated'] else ''})")
            print(f"    alertable  : {result['is_alertable']}")
            print(f"    reason     : {result['severity_reason']}")
            print(f"    evidence   :")
            for item in result["evidence_detail"]:
                print(f"      {item['feature']} = {item['display']}")
                if item["comparison"]:
                    print(f"        {item['comparison']}")
            probs = ", ".join(f"{k}={v:.3f}" for k, v in
                              result["class_probabilities"].items())
            print(f"    all probs  : {probs}")

        # --- accuracy on a real sample -----------------------------------
        sample = frame.sample(min(400, len(frame)), random_state=7)
        correct = 0
        t0 = time.time()
        for _, record in sample.iterrows():
            outcome = predict_threat(record.to_dict())
            correct += outcome["threat"] == record["threat_class"]
        elapsed = time.time() - t0
        print(f"\n  {'-' * 72}")
        print(f"  {len(sample)} real flows scored one at a time")
        print(f"    agreement with labels : {correct}/{len(sample)} "
              f"({correct / len(sample):.2%})")
        print(f"    total time            : {elapsed:.2f}s")
        print(f"    per flow              : {elapsed / len(sample) * 1000:.2f} ms")
        print(f"    throughput            : {len(sample) / elapsed:.0f} flows/s")
        print("  NOTE: these rows include the training set, so this figure is")
        print("  not a generalisation estimate. For held-out metrics see")
        print("  model/metrics.json or run: python -m model.evaluate")

    # --- robustness -------------------------------------------------------
    print(f"\n  {'-' * 72}")
    print("  Robustness -- none of these may raise:")
    cases: list[tuple[str, Any]] = [
        ("empty dict", {}),
        ("wrong key names", {"foo": 1, "bar": 2}),
        ("raw CSV header with leading space", {" Flow Duration": 100}),
        ("None values", {f: None for f in get_expected_features()}),
        ("string numbers", {f: "42" for f in get_expected_features()}),
        ("NaN values", {f: float("nan") for f in get_expected_features()}),
        ("infinity", {f: float("inf") for f in get_expected_features()}),
        ("negative values", {f: -1 for f in get_expected_features()}),
        ("huge values", {f: 1e18 for f in get_expected_features()}),
        ("extra unused keys",
         {**{f: 1 for f in get_expected_features()}, "unused": "x"}),
        ("not a dict", ["not", "a", "dict"]),
        ("None input", None),
    ]
    for label, payload in cases:
        try:
            outcome = predict_threat(payload)  # type: ignore[arg-type]
            n_missing = len(outcome["missing_features"])
            print(f"    [ok] {label:<38} -> {outcome['threat']:<9} "
                  f"conf={outcome['confidence']:.3f} "
                  f"ready={outcome['model_ready']} missing={n_missing}")
        except Exception as exc:  # noqa: BLE001
            print(f"    [RAISED] {label}: {type(exc).__name__}: {exc}")

    print("\n  Contract keys returned:")
    keys = sorted(predict_threat({f: 1.0 for f in get_expected_features()}))
    for key in keys:
        print(f"    {key}")
    print("=" * 76)

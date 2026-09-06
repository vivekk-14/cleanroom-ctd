"""
Central configuration for the passive threat-detection prototype.

Everything tunable lives here so that neither Student A (ML) nor Student B
(streaming/dashboard) has to edit the other's modules to change behaviour.

Import style used throughout the project:

    from config import PATHS, TRAINING, SEVERITY

Nothing in this file performs I/O beyond creating output directories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# Resolved from this file's location, NOT from the current working directory,
# so scripts behave the same whether launched from the project root or from
# inside a subfolder.
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Paths:
    """All filesystem locations used by the project.

    Deliberately relative to PROJECT_ROOT. No developer-specific absolute
    paths appear anywhere in this project, so a teammate can clone the repo,
    drop the CSVs into data/raw/, and everything resolves.
    """

    root: Path = PROJECT_ROOT

    # Input data
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    processed_csv: Path = PROJECT_ROOT / "data" / "processed" / "flows_clean.csv"

    # Model artefacts.
    #
    # Each feature profile is a SEPARATE model with its own feature contract,
    # stored under model/profiles/<slug>/. A 14-feature BIDIRECTIONAL model and
    # an 18-feature STRICT_UNIDIRECTIONAL model are different artefacts; they
    # are not interchangeable, and a single shared model.pkl made it possible
    # to feed one profile's data to the other's model. Per-profile directories
    # make that mistake unrepresentable.
    model_dir: Path = PROJECT_ROOT / "model"
    profiles_dir: Path = PROJECT_ROOT / "model" / "profiles"

    # Runtime alert stream (written by streaming/replay.py, read by dashboard)
    runtime_dir: Path = PROJECT_ROOT / "runtime"
    alerts_file: Path = PROJECT_ROOT / "runtime" / "alerts.jsonl"
    replay_status_file: Path = PROJECT_ROOT / "runtime" / "replay_status.json"

    # Evaluation output (confusion matrix image, text report)
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- per-profile artefact paths ---------------------------------------
    @staticmethod
    def profile_slug(profile: str) -> str:
        """Directory-safe name for a profile.

        >>> _Paths.profile_slug("STRICT_UNIDIRECTIONAL")
        'strict_unidirectional'
        """
        return profile.strip().lower().replace(" ", "_")

    def profile_dir(self, profile: str) -> Path:
        """Directory holding one profile's artefacts."""
        return self.profiles_dir / self.profile_slug(profile)

    def model_file(self, profile: str) -> Path:
        return self.profile_dir(profile) / "model.pkl"

    def features_file(self, profile: str) -> Path:
        """The feature contract: exact names, order, and count."""
        return self.profile_dir(profile) / "features.json"

    def label_encoder_file(self, profile: str) -> Path:
        return self.profile_dir(profile) / "label_encoder.pkl"

    def metrics_file(self, profile: str) -> Path:
        """Measured performance plus training provenance.

        Serves the role of the 'metadata.json' in the layout sketch. Kept as one
        file rather than two, because splitting it would duplicate the profile
        name, feature count, and training timestamp across both, and two copies
        of the same fact eventually disagree.
        """
        return self.profile_dir(profile) / "metrics.json"

    def ood_reference_file(self, profile: str) -> Path:
        """Distribution reference for out-of-distribution detection."""
        return self.profile_dir(profile) / "ood_reference.json"

    def novelty_model_file(self, profile: str) -> Path:
        """Pickled IsolationForest for the advisory novelty score.

        Stored separately from ood_reference.json because JSON cannot hold a
        fitted estimator.
        """
        return self.profile_dir(profile) / "novelty.pkl"

    def profile_artefacts(self, profile: str) -> dict[str, Path]:
        """Every artefact path for a profile, keyed by role."""
        return {
            "model": self.model_file(profile),
            "features": self.features_file(profile),
            "label_encoder": self.label_encoder_file(profile),
            "metrics": self.metrics_file(profile),
            "ood_reference": self.ood_reference_file(profile),
            "novelty": self.novelty_model_file(profile),
        }

    def trained_profiles(self) -> list[str]:
        """Profile slugs that have a model.pkl on disk."""
        if not self.profiles_dir.exists():
            return []
        return sorted(
            directory.name for directory in self.profiles_dir.iterdir()
            if (directory / "model.pkl").exists()
        )

    def ensure_dirs(self) -> None:
        """Create output directories if missing. Safe to call repeatedly."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.model_dir,
            self.profiles_dir,
            self.runtime_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_profile_dir(self, profile: str) -> Path:
        directory = self.profile_dir(profile)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


PATHS = _Paths()


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Dataset:
    """CIC-IDS2017 ingestion settings.

    Only the three CSVs that contain our four target classes are used for the
    MVP. Verified label counts in these files (measured, not estimated):

        Friday-...-DDos.csv      DDoS 128027   BENIGN  97718
        Friday-...-PortScan.csv  PortScan 158930   BENIGN 127537
        Friday-...-Morning.csv   Bot 1966      BENIGN 189067

    To train on more files, add filenames to `subset_files` or set
    `use_all_csvs = True`. Labels with no LABEL_MAP entry are dropped with a
    printed warning, so adding files never silently corrupts the class set.
    """

    subset_files: tuple[str, ...] = (
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    )

    # If True, glob every *.csv in data/raw/ instead of using subset_files.
    use_all_csvs: bool = False

    # Encoding fallback chain. MEASURED: Thursday-...-WebAttacks.csv is NOT
    # valid UTF-8 (raises UnicodeDecodeError); it decodes as latin-1. The
    # three subset files are valid UTF-8, but the fallback costs nothing and
    # keeps clean_data.py working if more files are added later.
    encodings: tuple[str, ...] = ("utf-8", "latin-1")

    # Per-class row cap, applied AFTER cleaning and deduplication.
    #
    # Why per-class and not a single global MAX_ROWS: Botnet is 0.28% of this
    # subset (1966 of ~703k rows). A global head/random cap starves it and the
    # model never learns the class. Capping each class independently preserves
    # rare classes while still bounding train time.
    #
    # MEASURED at 50000: 151,952 training rows, ~5s RandomForest fit,
    # macro F1 0.9890. Set to None to disable capping and use every row.
    max_rows_per_class: int | None = 50_000

    # Seed for every sampling/splitting operation, so runs are reproducible.
    random_state: int = 42

    # pandas read_csv chunk size. The subset loads comfortably in RAM on a
    # student laptop (~700k rows x 19 cols), but chunking keeps peak memory
    # low and makes progress visible. Set to None to read whole files.
    chunk_size: int | None = 200_000


DATASET = _Dataset()


# ---------------------------------------------------------------------------
# Feature profile
# ---------------------------------------------------------------------------
# Which feature set the model trains and predicts on. Defined in
# preprocessing/feature_config.py.
#
#   "STRICT_UNIDIRECTIONAL"  (default) Forward-direction features only.
#                            Matches the problem statement's threat model: a
#                            data diode or one-way tap yields a single
#                            direction, so reverse-path counters may not exist.
#
#   "BIDIRECTIONAL"          Includes backward/reverse features. Kept for
#                            comparison only. Scores WORSE on this dataset
#                            (see model/metrics.json after training both).
#
# MEASURED on the 3-file subset, 75/25 stratified split, 60 trees, depth 20:
#   STRICT_UNIDIRECTIONAL  accuracy 0.9989  macro F1 0.9549  Botnet F1 0.8211
#   BIDIRECTIONAL          accuracy 0.9979  macro F1 0.9140  Botnet F1 0.6583
FEATURE_PROFILE: str = os.environ.get("CTD_FEATURE_PROFILE", "STRICT_UNIDIRECTIONAL")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Training:
    """RandomForest hyperparameters and split settings.

    Defaults come from a measured sweep on the 3-file subset:

        trees  depth   train    macro F1   1-row latency   model size
        100    None    7.6 s    0.9871     21.0 ms         1.7 MB
         60      20    5.4 s    0.9890     13.8 ms         1.0 MB   <-- chosen
         40      16    3.7 s    0.9888     10.9 ms         0.6 MB

    60/20 is the best accuracy-per-millisecond point. 13.8 ms per flow is
    ~72 flows/sec single-threaded, comfortably faster than the 0.1 s replay
    delay, so the pipeline is not prediction-bound.
    """

    test_size: float = 0.25
    stratify: bool = True

    n_estimators: int = 60
    max_depth: int | None = 20
    min_samples_leaf: int = 1

    # 'balanced_subsample' reweights per bootstrap sample. Necessary here:
    # Botnet is ~1.3% of the capped training set. Without it, the model can
    # score high accuracy while ignoring Botnet entirely.
    class_weight: str | None = "balanced_subsample"

    # -1 uses all cores for TRAINING. Prediction is forced to single-threaded
    # in predict.py: for one row at a time, thread dispatch overhead exceeds
    # the work (measured 21 ms multi-threaded vs 13.8 ms single).
    n_jobs: int = -1

    random_state: int = 42

    # joblib compression for model.pkl. 3 gives ~1 MB instead of ~8 MB with
    # no measurable load-time penalty.
    compress: int = 3


TRAINING = _Training()


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Severity:
    """Confidence -> severity bands, plus optional threat-type escalation.

    Base bands (from the problem statement, section 10):

        confidence <  0.50            LOW
        0.50 <= confidence <  0.75    MEDIUM
        0.75 <= confidence <  0.90    HIGH
        confidence >= 0.90            CRITICAL

    Bands are half-open [lower, upper) so every confidence in [0, 1] maps to
    exactly one band with no gaps or overlaps.

    Threat-type escalation
    ----------------------
    Confidence alone measures how sure the model is, not how much damage the
    threat could do. A 0.80-confidence DDoS on critical infrastructure matters
    more than a 0.80-confidence port scan. `escalate` bumps a threat up one
    band; `benign_is_informational` forces all BENIGN flows to INFO regardless
    of confidence, so the analyst's alert list is not flooded with
    high-confidence "nothing is wrong" rows.

    Escalation is applied ONCE and cannot exceed CRITICAL. Every alert records
    both the pre- and post-escalation severity in its `severity_reason` field,
    so the dashboard can show exactly why a severity was assigned. Nothing is
    hidden.
    """

    # (lower_bound_inclusive, severity_name)
    bands: tuple[tuple[float, str], ...] = (
        (0.00, "LOW"),
        (0.50, "MEDIUM"),
        (0.75, "HIGH"),
        (0.90, "CRITICAL"),
    )

    ladder: tuple[str, ...] = ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")

    # Threat classes escalated one band above their confidence band.
    # DDoS: direct availability impact on the production network.
    # Botnet: implies an already-compromised internal host (C2 beacon).
    # PortScan is NOT escalated; it is reconnaissance, not yet impact.
    escalate: tuple[str, ...] = ("DDoS", "Botnet")

    # BENIGN never produces a severity-ranked alert.
    benign_is_informational: bool = True
    benign_label: str = "BENIGN"
    informational_severity: str = "INFO"

    # Severities that count toward the dashboard's "actionable alerts" tiles.
    alert_severities: tuple[str, ...] = ("MEDIUM", "HIGH", "CRITICAL")

    # Severity for a flow the model cannot be trusted on (see _OOD below).
    #
    # HIGH, not CRITICAL: an unreliable prediction warrants human review, but it
    # is not evidence of an attack. Ranking it above a confirmed high-confidence
    # DDoS would invert the analyst's priorities.
    #
    # This severity is assigned regardless of what the classifier predicted,
    # INCLUDING when it predicted BENIGN. That override is the entire point: the
    # dangerous failure is a confident BENIGN on input the model has never seen,
    # and the normal BENIGN -> INFO suppression would hide exactly that case.
    ood_severity: str = "HIGH"

    # A DEGRADED prediction (one feature out of range) keeps its normal
    # severity but is marked in the alert. Set to True to suppress it from the
    # actionable queue instead.
    degraded_is_informational: bool = False


SEVERITY = _Severity()


# ---------------------------------------------------------------------------
# Out-of-distribution detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _OOD:
    """Settings for detection/ood.py.

    Every default below was chosen by measurement, not intuition. The sweep is
    recorded in the detection/ood.py module docstring; the short version:

    Per-feature quantile bounds, fraction of rows reaching UNRELIABLE:

        q       held-out   Wed BENIGN   unseen atk   synthetic fixture
        0.001     0.250%       0.576%       0.744%          50.00%
        0.005     ~3.5%        ~6.9%       ~17.3%          95.00%
        0.01      ~6.9%       ~13.0%       ~39.1%          95.00%

    q=0.001 is the operating point: a 0.25% false-OOD rate on real held-out
    traffic is affordable, while 0.005 would put ~1,300 of Monday's 529,918
    benign flows into a review queue for no measured benefit on real attacks.

    The looser bounds do catch more unseen attacks, but by then the false
    positive rate has grown faster than the detection rate -- 17.3% detection at
    6.9% false positives is not a usable trade for a monitoring system.
    """

    # Lower tail fraction. Bounds are [quantile, 1 - quantile] of the training
    # split, so 0.001 keeps the central 99.8%.
    quantile: float = 0.001

    # Features that must be outside their bounds before a flow is called
    # UNRELIABLE. One violation gives DEGRADED instead.
    #
    # MEASURED at q=0.001: >=1 flags 0.850% of held-out rows, >=2 flags 0.250%.
    # Requiring two keeps single-feature noise out of the review queue while
    # still catching 50% of the synthetic fixture (95% reach DEGRADED).
    min_violations_for_ood: int = 2

    # --- advisory novelty detector ---
    # IsolationForest fitted on BENIGN training rows only.
    #
    # MEASURED: at contamination=0.05 it flags 47.20% of the unseen real DoS
    # attacks that the classifier calls BENIGN -- the only signal tested with
    # any traction on novel attack CLASSES. But it costs 4.75% false positives
    # on benign traffic, which on Monday's capture is ~25,000 review items.
    #
    # So it is fitted and reported, but does NOT change the verdict by default.
    # Flip novelty_drives_verdict to True to accept that trade.
    fit_novelty: bool = True
    novelty_contamination: float = 0.01
    novelty_estimators: int = 100
    novelty_min_rows: int = 1_000
    novelty_drives_verdict: bool = False

    random_state: int = 42

    # Feature-validation gate, applied before any prediction.
    #
    # Without this, a 14-column file scored against the 18-feature profile had
    # 13 features silently filled with 0.0 and still produced a confident
    # answer. Absent features are a contract violation, not a data quality
    # issue: the caller has supplied the wrong kind of record entirely.
    max_missing_features: int = 0


OOD = _OOD()


# ---------------------------------------------------------------------------
# Streaming / replay
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Replay:
    """Replay engine settings.

    SAFETY: the replay engine reads rows from a CSV on local disk and writes
    JSON lines to local disk. It opens no sockets and transmits nothing. This
    is enforced at runtime by detection/passive_guard.py, not merely asserted.
    """

    # Seconds between processed flows. 0.1 -> ~10 flows/sec, slow enough to
    # watch during a demo. Prediction takes ~14 ms, so the delay dominates.
    # Set to 0.0 for a fast batch run.
    delay_seconds: float = 0.1

    # Rows to replay before stopping. None = entire processed file.
    max_flows: int | None = None

    # Shuffle rows before replay. The processed CSV is grouped by source file,
    # so without shuffling the demo shows a long run of one class before the
    # next. Shuffling makes the threat-distribution chart populate immediately.
    shuffle: bool = True

    # Write BENIGN classifications to the alert stream too. Needed for an
    # honest "total flows observed" count and a meaningful distribution chart.
    # The dashboard filters them out of the actionable alert table.
    include_benign: bool = True

    # Truncate runtime/alerts.jsonl on start. False appends across runs.
    truncate_on_start: bool = True

    # Flush to disk every N flows. The dashboard tails the file, so small
    # values make alerts appear promptly; 1 = flush every line.
    flush_every: int = 1


REPLAY = _Replay()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Dashboard:
    """Streamlit dashboard settings."""

    page_title: str = "Passive Threat Detection - Unidirectional IP Traffic"
    page_icon: str = "\N{SATELLITE ANTENNA}"

    # Auto-refresh interval in seconds while replay is running.
    refresh_seconds: float = 2.0

    # Most recent alerts to show in the live table. Caps render cost; the
    # metric tiles and distribution chart still use the full stream.
    max_table_rows: int = 300

    # Read only the last N lines of alerts.jsonl per refresh.
    tail_lines: int = 5_000

    # Fixed colours so a class keeps its colour across refreshes.
    threat_colors: dict[str, str] = field(
        default_factory=lambda: {
            "BENIGN": "#2E9E5B",
            "DDoS": "#D62728",
            "PortScan": "#FF8C1A",
            "Botnet": "#8B2BE2",
            "UNKNOWN": "#7F7F7F",
            # Distinct blue-grey: an OOD flow is not a threat class, it is an
            # absence of a trustworthy answer, and the colour should not imply
            # it sits on the same scale as the four real classes.
            "UNKNOWN_OOD": "#4A6FA5",
        }
    )

    severity_colors: dict[str, str] = field(
        default_factory=lambda: {
            "INFO": "#2E9E5B",
            "LOW": "#9EC5FE",
            "MEDIUM": "#FFC107",
            "HIGH": "#FF8C1A",
            "CRITICAL": "#D62728",
        }
    )

    reliability_colors: dict[str, str] = field(
        default_factory=lambda: {
            "RELIABLE": "#2E9E5B",
            "DEGRADED": "#FFC107",
            "UNRELIABLE": "#4A6FA5",
            "UNKNOWN": "#7F7F7F",
        }
    )

    # One-line plain-English gloss per tier, shown beside the counts so a
    # reader does not have to already know what the tiers mean.
    reliability_labels: dict[str, str] = field(
        default_factory=lambda: {
            "RELIABLE": "input resembles training data",
            "DEGRADED": "one feature outside training range",
            "UNRELIABLE": "outside training distribution",
            "UNKNOWN": "no reference available",
        }
    )


DASHBOARD = _Dashboard()


# ---------------------------------------------------------------------------
# Operating posture (displayed by the dashboard)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Posture:
    """Strings the dashboard renders to state the system's operating mode.

    These are not decoration. The problem statement requires the monitoring
    enclave to be observably read-only, and the dashboard must say so.
    """

    mode: str = "PASSIVE MONITORING"
    access: str = "READ ONLY"
    assurance: str = "No outbound network actions are performed."
    detail: str = (
        "This system consumes a one-way copy of flow records. It does not "
        "probe, connect, handshake, inject, block, or send mitigation "
        "commands to the monitored network, and it does not decrypt payloads."
    )


POSTURE = _Posture()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # `python config.py` prints the resolved configuration. Useful first
    # command when setting up on a new machine.
    PATHS.ensure_dirs()

    print("=" * 68)
    print("Passive Threat Detection - resolved configuration")
    print("=" * 68)
    print(f"  project root      : {PATHS.root}")
    print(f"  raw data dir      : {PATHS.raw_dir}")
    print(f"  processed csv     : {PATHS.processed_csv}")
    print(f"  profiles dir      : {PATHS.profiles_dir}")
    print(f"  alert stream      : {PATHS.alerts_file}")
    print()
    print(f"  default profile   : {FEATURE_PROFILE}")
    print(f"  max rows / class  : {DATASET.max_rows_per_class}")
    print(f"  forest            : {TRAINING.n_estimators} trees, "
          f"depth {TRAINING.max_depth}, class_weight={TRAINING.class_weight}")
    print(f"  OOD bounds        : q={OOD.quantile}, UNRELIABLE at "
          f">={OOD.min_violations_for_ood} violations")
    print(f"  replay delay      : {REPLAY.delay_seconds}s")
    print()
    print(f"  posture           : {POSTURE.mode} / {POSTURE.access}")
    print(f"                      {POSTURE.assurance}")
    print()

    print("  raw CSV files expected:")
    missing_raw = 0
    for name in DATASET.subset_files:
        path = PATHS.raw_dir / name
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            print(f"    [ok]      {name}  ({size_mb:.1f} MB)")
        else:
            missing_raw += 1
            print(f"    [MISSING] {name}")

    # Per-profile artefacts. model_file() and friends take a profile argument,
    # so each must be resolved per profile rather than printed directly -- an
    # earlier version of this block printed the bound method itself.
    print()
    print("  trained model artefacts:")
    trained = PATHS.trained_profiles()
    for profile in ("STRICT_UNIDIRECTIONAL", "BIDIRECTIONAL"):
        slug = PATHS.profile_slug(profile)
        artefacts = PATHS.profile_artefacts(profile)
        present = [role for role, path in artefacts.items() if path.exists()]
        if slug in trained:
            size_mb = artefacts["model"].stat().st_size / 1_048_576
            print(f"    [ok]      {profile:<22} {size_mb:.2f} MB, "
                  f"{len(present)}/{len(artefacts)} artefacts")
            print(f"                                     "
                  f"{artefacts['model'].parent.relative_to(PATHS.root)}/")
        else:
            print(f"    [MISSING] {profile:<22} "
                  f"train with: python -m model.train --profile {profile}")

    print()
    print("  next step:")
    if missing_raw:
        print(f"    Place the {missing_raw} missing CSV(s) in "
              f"{PATHS.raw_dir.relative_to(PATHS.root)}/")
        print("    See README section 6 for the download link.")
    elif not PATHS.processed_csv.exists():
        print("    python -m preprocessing.clean_data")
    elif not trained:
        print("    python -m model.train --compare")
    else:
        print("    python -m model.evaluate      (then: streamlit run "
              "dashboard/app.py)")
    print("=" * 68)

"""
Replay engine -- simulates a one-way traffic feed from the processed dataset.

    data/processed/flows_clean.csv
      -> one row at a time
      -> feature dict
      -> predict_threat()
      -> Alert
      -> appended to runtime/alerts.jsonl

SAFETY
------
This engine reads a CSV from local disk and appends JSON lines to local disk.
It opens no sockets and transmits nothing. detection.passive_guard is armed
before any other project import, so any attempt to reach a non-loopback address
-- from this file or from anything it imports -- raises PassiveModeViolation
immediately.

The replay engine can NEVER put packets on a network. There is no packet
transmission code here, and the guard makes that enforceable rather than merely
stated.

Why JSON Lines and not a queue or socket
----------------------------------------
The dashboard runs as a separate Streamlit process. An append-only file is the
simplest transport that decouples them: the replay engine appends, the dashboard
tails. No broker, no port, no shared memory, and a half-written final line is
simply skipped by the reader. It also means the alert stream survives the
process, so a demo can be replayed and reviewed afterwards.

Usage
-----
    python -m streaming.replay                     # config defaults
    python -m streaming.replay --delay 0.05        # faster
    python -m streaming.replay --max-flows 200     # short demo
    python -m streaming.replay --delay 0 --quiet   # fast batch scoring
    python -m streaming.replay --alerts-only       # skip BENIGN flows
"""

from __future__ import annotations

# The guard is armed BEFORE any project import so that no imported module can
# open a socket during its own import. Ordering here is deliberate, not
# accidental; do not move these three lines.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.passive_guard import enforce_passive_mode  # noqa: E402

enforce_passive_mode(verbose=False)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pandas as pd  # noqa: E402

from config import DATASET, FEATURE_PROFILE, PATHS, POSTURE, REPLAY  # noqa: E402
from detection.alert_schema import Alert  # noqa: E402
from detection.passive_guard import posture_report  # noqa: E402
from model.predict import get_expected_features, model_status, predict_threat  # noqa: E402
from preprocessing.clean_data import CAPTURE_TS_COLUMN, TARGET_COLUMN  # noqa: E402
from preprocessing.feature_config import (  # noqa: E402
    FEATURE_PROFILES,
    IDENTITY_COLUMNS,
)

RULE = "=" * 76

# Set by the SIGINT handler so the loop can finish the current flow, write a
# final status file, and close the output cleanly. Killing the process mid-write
# would leave a truncated JSON line for the dashboard to skip.
_STOP_REQUESTED = False


def _handle_interrupt(signum, frame) -> None:  # noqa: ANN001, ARG001
    global _STOP_REQUESTED
    if _STOP_REQUESTED:
        # Second Ctrl+C: the user wants out now.
        raise KeyboardInterrupt
    _STOP_REQUESTED = True
    print("\n  [stopping after the current flow; press Ctrl+C again to force]")


class StatusWriter:
    """Writes runtime/replay_status.json for the dashboard to read.

    Written atomically via a temp file and os.replace, so the dashboard never
    reads a half-written JSON object.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temp = path.with_suffix(".json.tmp")

    def write(self, payload: dict) -> None:
        try:
            self.temp.write_text(json.dumps(payload, indent=2, default=str),
                                 encoding="utf-8")
            os.replace(self.temp, self.path)
        except OSError:
            # A status write failure must not stop the replay; the dashboard
            # falls back to reading alerts.jsonl directly.
            pass


def _split_row(record: dict, feature_names: list[str]) -> tuple[dict, dict]:
    """Split a CSV row into (features_for_model, identity_for_alert).

    The model must never see identity fields. Training on IP addresses would
    teach the classifier which lab host ran the attack rather than what attack
    traffic looks like, and the model would be useless on any other network.
    """
    identity = {
        key: record.get(key)
        for key in list(IDENTITY_COLUMNS) + ["Destination Port",
                                             CAPTURE_TS_COLUMN]
        if key in record
    }
    features = {name: record.get(name) for name in feature_names}
    return features, identity


def replay(
    profile: str | None = None,
    delay: float | None = None,
    max_flows: int | None = None,
    shuffle: bool | None = None,
    include_benign: bool | None = None,
    truncate: bool | None = None,
    source: Path | None = None,
    quiet: bool = False,
) -> dict:
    """Replay flows from the processed CSV, emitting alerts to JSON Lines.

    Returns a summary dict of what was processed.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False

    delay = REPLAY.delay_seconds if delay is None else delay
    max_flows = REPLAY.max_flows if max_flows is None else max_flows
    shuffle = REPLAY.shuffle if shuffle is None else shuffle
    include_benign = (REPLAY.include_benign if include_benign is None
                      else include_benign)
    truncate = REPLAY.truncate_on_start if truncate is None else truncate
    source = source or PATHS.processed_csv
    profile = profile or FEATURE_PROFILE

    PATHS.ensure_dirs()

    if not source.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {source}\n"
            f"Run first:  python -m preprocessing.clean_data"
        )

    status = model_status(profile)
    feature_names = get_expected_features(profile)

    print(RULE)
    print("PASSIVE FLOW REPLAY")
    print(RULE)
    print(f"  posture     : {POSTURE.mode} / {POSTURE.access}")
    print(f"                {POSTURE.assurance}")
    guard = posture_report()
    print(f"  guard       : armed={guard['passive_mode_armed']}, "
          f"outbound attempts blocked={guard['outbound_attempts_blocked']}")
    print(f"  source      : {source.name}")
    print(f"  output      : {PATHS.alerts_file}")

    if not status["model_ready"]:
        print(f"\n  WARNING: no trained model. {status['error']}")
        print("  Flows will be replayed and alerts emitted with "
              "threat='UNKNOWN'.")
        print("  The dashboard will work; classifications will not.")
        print("  Fix with:  python -m model.train\n")
    else:
        print(f"  model       : {status['profile']}, "
              f"{status['n_features']} features, "
              f"classes {', '.join(status['classes'])}")
        print(f"                "
              f"{PATHS.model_file(profile).relative_to(PATHS.root)}")

    # --- load ------------------------------------------------------------
    frame = pd.read_csv(source, low_memory=False)
    has_labels = TARGET_COLUMN in frame.columns

    if not include_benign and has_labels:
        before = len(frame)
        frame = frame[frame[TARGET_COLUMN] != "BENIGN"]
        print(f"  filter      : BENIGN excluded "
              f"({before - len(frame):,} rows skipped)")

    if shuffle:
        # The processed file is already shuffled by clean_data, but reshuffling
        # with a different seed per run keeps consecutive demos from being
        # identical, and guarantees a class mix from the first few flows so the
        # distribution chart is meaningful immediately.
        frame = frame.sample(frac=1.0, random_state=None).reset_index(drop=True)

    if max_flows is not None:
        frame = frame.head(max_flows)

    total = len(frame)
    print(f"  flows       : {total:,}")
    print(f"  delay       : {delay}s per flow", end="")
    if delay > 0:
        print(f"  (~{1 / delay:.1f} flows/s target)")
    else:
        print("  (no delay: as fast as possible)")
    if has_labels:
        print("  labels      : present, so live accuracy is shown "
              "(demo only; production traffic has no ground truth)")
    print(RULE)

    # --- output stream ---------------------------------------------------
    mode = "w" if truncate else "a"
    if truncate and PATHS.alerts_file.exists():
        print(f"  truncating existing {PATHS.alerts_file.name}")

    status_writer = StatusWriter(PATHS.replay_status_file)
    previous_handler = signal.signal(signal.SIGINT, _handle_interrupt)

    counts: Counter = Counter()
    severities: Counter = Counter()
    correct = 0
    labelled = 0
    started = time.time()
    processed = 0

    try:
        with PATHS.alerts_file.open(mode, encoding="utf-8") as stream:
            status_writer.write({
                "state": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "total_flows_planned": total,
                "flows_processed": 0,
                "source_file": source.name,
                "delay_seconds": delay,
                "model_ready": status["model_ready"],
                "posture": {
                    "mode": POSTURE.mode,
                    "access": POSTURE.access,
                    "assurance": POSTURE.assurance,
                },
            })

            for position, (_, series) in enumerate(frame.iterrows(), start=1):
                if _STOP_REQUESTED:
                    break

                record = series.to_dict()
                features, identity = _split_row(record, feature_names)
                ground_truth = (str(record[TARGET_COLUMN]) if has_labels
                                else None)

                prediction = predict_threat(features, profile=profile)

                alert = Alert.build(
                    prediction=prediction,
                    flow_id=f"F{position:06d}",
                    identity=identity,
                    ground_truth=ground_truth,
                    observation_mode="PASSIVE_REPLAY",
                )

                stream.write(alert.to_json_line())
                if REPLAY.flush_every and position % REPLAY.flush_every == 0:
                    stream.flush()

                processed = position
                counts[alert.threat_class] += 1
                severities[alert.severity] += 1
                if alert.correct is not None:
                    labelled += 1
                    correct += int(alert.correct)

                if not quiet:
                    print(f"  {alert.summary()}")

                # Refresh status periodically. Every flow would rewrite the file
                # 10x/second for no benefit; every 10 keeps the dashboard's
                # progress figure current without the churn.
                if position % 10 == 0 or position == total:
                    elapsed = time.time() - started
                    status_writer.write({
                        "state": "RUNNING",
                        "flows_processed": position,
                        "total_flows_planned": total,
                        "elapsed_seconds": round(elapsed, 1),
                        "flows_per_second": round(position / max(elapsed, 1e-9), 2),
                        "threat_counts": dict(counts),
                        "severity_counts": dict(severities),
                        "live_accuracy": (round(correct / labelled, 4)
                                          if labelled else None),
                        "model_ready": status["model_ready"],
                        "source_file": source.name,
                        "posture": {
                            "mode": POSTURE.mode,
                            "access": POSTURE.access,
                            "assurance": POSTURE.assurance,
                        },
                    })

                if delay > 0 and position < total:
                    time.sleep(delay)

    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    elapsed = time.time() - started

    summary = {
        "flows_processed": processed,
        "flows_planned": total,
        "elapsed_seconds": round(elapsed, 2),
        "flows_per_second": round(processed / max(elapsed, 1e-9), 2),
        "threat_counts": dict(counts),
        "severity_counts": dict(severities),
        "alertable": sum(v for k, v in severities.items()
                         if k in ("MEDIUM", "HIGH", "CRITICAL")),
        "live_accuracy": (round(correct / labelled, 4) if labelled else None),
        "labelled_flows": labelled,
        "alerts_file": str(PATHS.alerts_file),
        "outbound_attempts_blocked":
            posture_report()["outbound_attempts_blocked"],
    }

    status_writer.write({
        "state": "STOPPED" if _STOP_REQUESTED else "COMPLETE",
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **summary,
        "model_ready": status["model_ready"],
        "source_file": source.name,
        "posture": {
            "mode": POSTURE.mode,
            "access": POSTURE.access,
            "assurance": POSTURE.assurance,
        },
    })

    # --- summary ---------------------------------------------------------
    print(RULE)
    print("REPLAY COMPLETE" if not _STOP_REQUESTED else "REPLAY STOPPED")
    print(RULE)
    print(f"  flows processed : {processed:,} of {total:,}")
    print(f"  elapsed         : {elapsed:.1f}s "
          f"({summary['flows_per_second']:.1f} flows/s)")
    if delay > 0:
        print(f"                    (rate is set by the {delay}s replay delay, "
              f"not by model speed)")

    print(f"\n  classified as:")
    for threat, count in counts.most_common():
        print(f"    {threat:<10} {count:>7,}  "
              f"({count / max(processed, 1):6.2%})")

    print(f"\n  severity:")
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
        if severities.get(level):
            print(f"    {level:<10} {severities[level]:>7,}")
    print(f"    {'actionable':<10} {summary['alertable']:>7,}  "
          f"(MEDIUM and above)")

    if summary["live_accuracy"] is not None:
        print(f"\n  agreement with dataset labels: "
              f"{summary['live_accuracy']:.2%} "
              f"({correct:,}/{labelled:,})")
        print("  Ground truth exists only because this replays a labelled")
        print("  dataset. Production traffic has none.")

    print(f"\n  alerts written  : {PATHS.alerts_file}")
    print(f"  outbound network actions: "
          f"{summary['outbound_attempts_blocked']} attempted, "
          f"0 performed")
    print(f"\n  View the dashboard:  "
          f"streamlit run dashboard/app.py")
    print(RULE)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay processed flow records through the detector."
    )
    parser.add_argument("--profile", default=FEATURE_PROFILE,
                        choices=sorted(FEATURE_PROFILES),
                        help=f"Which model to classify with "
                             f"(default: {FEATURE_PROFILE}).")
    parser.add_argument("--delay", type=float, default=None, metavar="SECONDS",
                        help=f"Delay between flows "
                             f"(default: {REPLAY.delay_seconds}).")
    parser.add_argument("--max-flows", type=int, default=None, metavar="N",
                        help="Stop after N flows (default: all).")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Replay in file order instead of shuffling.")
    parser.add_argument("--alerts-only", action="store_true",
                        help="Skip BENIGN flows.")
    parser.add_argument("--append", action="store_true",
                        help="Append to alerts.jsonl instead of truncating.")
    parser.add_argument("--source", type=Path, default=None, metavar="CSV",
                        help="Alternative processed CSV to replay.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print a line per flow.")
    args = parser.parse_args()

    try:
        replay(
            profile=args.profile,
            delay=args.delay,
            max_flows=args.max_flows,
            shuffle=False if args.no_shuffle else None,
            include_benign=False if args.alerts_only else None,
            truncate=False if args.append else None,
            source=args.source,
            quiet=args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

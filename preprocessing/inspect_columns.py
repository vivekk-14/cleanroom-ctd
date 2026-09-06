"""
Inspect the raw CIC-IDS2017 CSVs before training.

RUN THIS FIRST. It reads the actual files on disk and reports what is really
in them, so no downstream module has to assume anything about column names,
encodings, or labels.

What it reports per file
------------------------
  * file size and encoding actually required to decode it
  * raw column count and the exact raw header strings (with whitespace shown)
  * duplicate column names after normalisation
  * label values with exact row counts
  * which labels map to a target class and which will be dropped
  * fully-empty rows
  * per-feature availability for both feature profiles
  * NaN / +inf / -inf counts for the columns that need them

Usage
-----
    python -m preprocessing.inspect_columns
    python -m preprocessing.inspect_columns --full-scan
    python -m preprocessing.inspect_columns --file Friday-...-DDos.pcap_ISCX.csv

By default label counting reads only the label column, which is fast (a few
seconds for 260 MB). --full-scan additionally loads the feature columns to
count NaN/inf, which takes longer.

This script only reads files. It writes nothing and opens no sockets.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python preprocessing/inspect_columns.py` as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET, PATHS  # noqa: E402
from preprocessing.feature_config import (  # noqa: E402
    ALL_PROFILE_FEATURES,
    DROP_COLUMNS,
    FEATURE_PROFILES,
    IDENTITY_COLUMNS,
    INFINITY_PRONE_COLUMNS,
    LABEL_COLUMN,
    LABEL_MAP,
    normalize_column,
)

RULE = "=" * 76
THIN = "-" * 76


def detect_encoding(path: Path) -> tuple[str | None, str | None]:
    """Return (working_encoding, error_message).

    Tries each encoding in config order and returns the first that decodes a
    small sample. This matters: one CIC-IDS2017 file is not valid UTF-8.
    """
    for enc in DATASET.encodings:
        try:
            pd.read_csv(path, encoding=enc, nrows=200, low_memory=False)
            return enc, None
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # malformed CSV, not an encoding problem
            return None, f"{type(exc).__name__}: {exc}"
    return None, f"none of {DATASET.encodings} could decode this file"


def read_raw_header(path: Path, encoding: str) -> list[str]:
    """Read the header line exactly as stored, without pandas mangling.

    pandas silently renames duplicate columns ('X' -> 'X.1'), which would hide
    the fact that the file genuinely contains the same header twice. Reading
    the line directly shows the truth.
    """
    with path.open("r", encoding=encoding, newline="") as handle:
        line = handle.readline().rstrip("\r\n")
    return line.split(",")


def inspect_file(path: Path, full_scan: bool) -> dict:
    """Inspect one CSV and print a report. Returns a summary dict."""
    print(f"\n{RULE}")
    print(f"FILE: {path.name}")
    print(f"      {path.stat().st_size / 1_048_576:.1f} MB")
    print(RULE)

    encoding, error = detect_encoding(path)
    if encoding is None:
        print(f"  ERROR: cannot read this file -- {error}")
        return {"file": path.name, "readable": False, "error": error}

    note = "" if encoding == DATASET.encodings[0] else "  <-- NOT utf-8"
    print(f"\n  Encoding required: {encoding}{note}")

    # --- header -------------------------------------------------------------
    raw_header = read_raw_header(path, encoding)
    normalized = [normalize_column(c) for c in raw_header]

    print(f"  Raw column count : {len(raw_header)}")

    variant = {85: "TrafficLabelling (has IP/port/timestamp identity columns)",
               79: "MachineLearningCVE (features + label only, no identity)"}
    print(f"  Layout           : {variant.get(len(raw_header), 'unrecognised')}")

    # Leading/trailing whitespace is the single biggest footgun in this
    # dataset, so quantify it explicitly.
    spaced = [c for c in raw_header if c != c.strip()]
    print(f"\n  Columns with leading/trailing whitespace: "
          f"{len(spaced)} of {len(raw_header)}")
    if spaced:
        for c in spaced[:3]:
            print(f"    {c!r}  ->  {normalize_column(c)!r}")
        if len(spaced) > 3:
            print(f"    ... and {len(spaced) - 3} more")
    clean = [c for c in raw_header if c == c.strip()]
    if clean:
        print(f"  Columns WITHOUT whitespace: {len(clean)}, e.g. {clean[0]!r}")
        print("  -> Header whitespace is inconsistent; normalisation is required.")

    # --- duplicates ---------------------------------------------------------
    dupes = {name: n for name, n in Counter(normalized).items() if n > 1}
    if dupes:
        print(f"\n  DUPLICATE column names after normalisation: {dupes}")
        for name in dupes:
            positions = [i for i, c in enumerate(normalized) if c == name]
            print(f"    {name!r} at header positions {positions}")
        print("  -> pandas appends '.1' to the second occurrence; "
              "clean_data.py drops it.")

    # --- identity columns ---------------------------------------------------
    present_identity = [c for c in IDENTITY_COLUMNS if c in normalized]
    missing_identity = [c for c in IDENTITY_COLUMNS if c not in normalized]
    print(f"\n  Identity columns present: {len(present_identity)}"
          f"/{len(IDENTITY_COLUMNS)}")
    if present_identity:
        print(f"    {', '.join(present_identity)}")
    if missing_identity:
        print(f"    MISSING: {', '.join(missing_identity)}")
        print("    -> alerts from this file would need placeholder identifiers")

    # --- feature availability ----------------------------------------------
    print(f"\n{THIN}\n  FEATURE AVAILABILITY\n{THIN}")
    profile_status: dict[str, dict] = {}
    for profile, feats in FEATURE_PROFILES.items():
        missing = [f for f in feats if f not in normalized]
        profile_status[profile] = {
            "total": len(feats),
            "missing": missing,
        }
        if missing:
            print(f"  {profile}: {len(feats) - len(missing)}/{len(feats)} present")
            for f in missing:
                print(f"      MISSING: {f!r}")
        else:
            print(f"  {profile}: all {len(feats)} features present  [ok]")

    # --- labels -------------------------------------------------------------
    print(f"\n{THIN}\n  LABELS\n{THIN}")
    if LABEL_COLUMN not in normalized:
        print(f"  ERROR: no {LABEL_COLUMN!r} column found.")
        return {"file": path.name, "readable": True, "labels": {}}

    raw_label_col = raw_header[normalized.index(LABEL_COLUMN)]
    labels = pd.read_csv(
        path, encoding=encoding, usecols=[raw_label_col], low_memory=False
    )[raw_label_col]

    total_rows = len(labels)
    empty_rows = int(labels.isna().sum())
    print(f"  Total data rows      : {total_rows:,}")
    if empty_rows:
        print(f"  Rows with NaN label  : {empty_rows:,}  <-- dropped by cleaning")

    counts = labels.value_counts()
    print(f"\n  {'raw label':<34} {'rows':>10}   maps to")
    print(f"  {'-' * 34} {'-' * 10}   {'-' * 12}")
    label_summary: dict[str, tuple[int, str | None]] = {}
    for raw_label, n in counts.items():
        target = LABEL_MAP.get(str(raw_label).strip(), "__UNMAPPED__")
        if target == "__UNMAPPED__":
            verdict = "NOT IN LABEL_MAP -> dropped"
        elif target is None:
            verdict = "dropped (out of MVP scope)"
        elif target == raw_label:
            verdict = target
        else:
            verdict = f"{target}  (renamed)"
        # repr() so non-ASCII characters such as the latin-1 en-dash in
        # 'Web Attack \x96 XSS' are visible rather than mojibake.
        print(f"  {repr(str(raw_label)):<34} {n:>10,}   {verdict}")
        label_summary[str(raw_label)] = (int(n), target)

    kept = sum(n for n, t in label_summary.values()
               if t is not None and t != "__UNMAPPED__")
    print(f"\n  Rows kept for the MVP: {kept:,} of {total_rows:,} "
          f"({kept / max(total_rows, 1) * 100:.1f}%)")

    # --- numeric health -----------------------------------------------------
    if full_scan:
        print(f"\n{THIN}\n  NUMERIC HEALTH (full scan)\n{THIN}")
        wanted_raw = [
            raw for raw, norm in zip(raw_header, normalized)
            if norm in ALL_PROFILE_FEATURES and norm not in DROP_COLUMNS
        ]
        # Deduplicate while preserving order, in case of repeated headers.
        wanted_raw = list(dict.fromkeys(wanted_raw))
        frame = pd.read_csv(
            path, encoding=encoding, usecols=wanted_raw, low_memory=False
        )
        frame.columns = [normalize_column(c) for c in frame.columns]

        problems = []
        for col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            n_nan = int(series.isna().sum())
            finite = np.isfinite(series.to_numpy(dtype="float64", na_value=np.nan))
            n_inf = int((~finite & series.notna().to_numpy()).sum())
            non_numeric = int(series.isna().sum() - frame[col].isna().sum())
            if n_nan or n_inf or non_numeric:
                problems.append((col, n_nan, n_inf, non_numeric))

        if problems:
            print(f"  {'column':<32} {'NaN':>9} {'+/-inf':>9} {'non-numeric':>12}")
            print(f"  {'-' * 32} {'-' * 9} {'-' * 9} {'-' * 12}")
            for col, n_nan, n_inf, non_num in problems:
                print(f"  {col:<32} {n_nan:>9,} {n_inf:>9,} {non_num:>12,}")
            print("  -> clean_data.py coerces to numeric, replaces inf with "
                  "NaN, then drops affected rows.")
        else:
            print("  No NaN, infinity, or non-numeric values found.")
    else:
        prone = [c for c in INFINITY_PRONE_COLUMNS if c in normalized]
        if prone:
            print(f"\n  Infinity-prone columns present: {', '.join(prone)}")
            print("  (pass --full-scan to count NaN/inf exactly)")

    return {
        "file": path.name,
        "readable": True,
        "encoding": encoding,
        "raw_columns": len(raw_header),
        "duplicates": dupes,
        "total_rows": total_rows,
        "empty_rows": empty_rows,
        "labels": label_summary,
        "profiles": profile_status,
        "identity_missing": missing_identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report the real structure of the raw CIC-IDS2017 CSVs."
    )
    parser.add_argument(
        "--full-scan", action="store_true",
        help="Also load feature columns to count NaN/inf exactly (slower).",
    )
    parser.add_argument(
        "--file", action="append", default=None, metavar="NAME",
        help="Inspect only this filename in data/raw/. Repeatable.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Inspect every *.csv in data/raw/, not just the configured subset.",
    )
    args = parser.parse_args()

    PATHS.ensure_dirs()

    if args.file:
        names = args.file
    elif args.all:
        names = sorted(p.name for p in PATHS.raw_dir.glob("*.csv"))
    else:
        names = list(DATASET.subset_files)

    print(RULE)
    print("RAW DATASET INSPECTION")
    print(RULE)
    print(f"  data/raw : {PATHS.raw_dir}")
    print(f"  files    : {len(names)}")
    print("  mode     : read-only. This script writes nothing.")

    if not names:
        print("\n  No CSV files found.")
        print(f"  Place the CIC-IDS2017 CSVs in: {PATHS.raw_dir}")
        return 1

    summaries, missing = [], []
    for name in names:
        path = PATHS.raw_dir / name
        if not path.exists():
            missing.append(name)
            continue
        summaries.append(inspect_file(path, args.full_scan))

    # --- combined view ------------------------------------------------------
    print(f"\n{RULE}\nCOMBINED SUMMARY\n{RULE}")

    if missing:
        print(f"\n  MISSING FILES ({len(missing)}):")
        for name in missing:
            print(f"    {name}")
        print(f"    Expected in {PATHS.raw_dir}")

    readable = [s for s in summaries if s.get("readable")]
    if not readable:
        print("\n  No readable files. Cannot continue.")
        return 1

    total = sum(s["total_rows"] for s in readable)
    empty = sum(s["empty_rows"] for s in readable)
    print(f"\n  Files read       : {len(readable)}")
    print(f"  Total rows       : {total:,}")
    if empty:
        print(f"  Empty rows       : {empty:,}")

    combined: Counter[str] = Counter()
    for summary in readable:
        for raw_label, (n, target) in summary["labels"].items():
            if target and target != "__UNMAPPED__":
                combined[target] += n

    if combined:
        print(f"\n  Target class totals (before dedup/cap):")
        kept_total = sum(combined.values())
        for cls, n in combined.most_common():
            print(f"    {cls:<12} {n:>10,}  ({n / kept_total * 100:5.2f}%)")
        print(f"    {'TOTAL':<12} {kept_total:>10,}")

        rarest, rarest_n = combined.most_common()[-1]
        share = rarest_n / kept_total * 100
        if share < 5.0:
            print(f"\n  Class imbalance: {rarest} is {share:.2f}% of the data.")
            print("  -> config uses per-class row caps (not a global cap) and")
            print("     class_weight='balanced_subsample' so this class is not")
            print("     swamped during training.")

    encodings = {s["encoding"] for s in readable}
    if len(encodings) > 1:
        print(f"\n  Mixed encodings across files: {encodings}")
        print("  -> clean_data.py tries each encoding in order per file.")

    all_dupes: set[str] = set()
    for summary in readable:
        all_dupes.update(summary["duplicates"])
    if all_dupes:
        print(f"\n  Duplicate header names present: {sorted(all_dupes)}")
        print(f"  -> DROP_COLUMNS handles: {list(DROP_COLUMNS)}")

    print("\n  Feature profile readiness:")
    ready = True
    for profile in FEATURE_PROFILES:
        gaps = {s["file"]: s["profiles"][profile]["missing"]
                for s in readable if s["profiles"][profile]["missing"]}
        if gaps:
            ready = False
            print(f"    {profile}: INCOMPLETE")
            for fname, miss in gaps.items():
                print(f"      {fname}: missing {miss}")
        else:
            n = FEATURE_PROFILES[profile].__len__()
            print(f"    {profile}: all {n} features present in every file  [ok]")

    print(f"\n{RULE}")
    if ready and not missing:
        print("READY.  Next step:  python -m preprocessing.clean_data")
    else:
        print("NOT READY.  Resolve the issues above before cleaning.")
    print(RULE)
    return 0 if (ready and not missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())

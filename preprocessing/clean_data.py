"""
Clean the raw CIC-IDS2017 CSVs into a single analysis-ready file.

Pipeline
--------
    data/raw/*.csv
      -> per-file: encoding fallback, column normalisation, column selection
      -> drop fully-empty rows
      -> drop the duplicated 'Fwd Header Length' column
      -> coerce features to numeric, replace +/-inf with NaN
      -> map raw labels to the four target classes, drop unmapped rows
      -> concatenate all files
      -> global de-duplication
      -> drop rows with any NaN feature
      -> per-class row cap (stratified sampling)
      -> data/processed/flows_clean.csv  +  flows_clean.meta.json

Every step prints how many rows it removed, so the row count is auditable end
to end. Nothing is dropped silently.

Usage
-----
    python -m preprocessing.clean_data
    python -m preprocessing.clean_data --max-per-class 20000
    python -m preprocessing.clean_data --no-cap
    python -m preprocessing.clean_data --out data/processed/small.csv

This module only reads from data/raw/ and writes to data/processed/. It opens
no sockets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET, PATHS  # noqa: E402
from preprocessing.feature_config import (  # noqa: E402
    ALL_PROFILE_FEATURES,
    DROP_COLUMNS,
    EVIDENCE_EXTRA_COLUMNS,
    IDENTITY_COLUMNS,
    LABEL_COLUMN,
    LABEL_MAP,
    TARGET_CLASSES,
    map_label,
    normalize_column,
)

RULE = "=" * 76

# Sentinel distinguishing "caller said nothing, use the config default" from
# "caller explicitly requested no cap". A plain None cannot express both.
_USE_CONFIG_DEFAULT = object()

# The canonical label column in the processed output. Named distinctly from the
# dataset's 'Label' so nothing downstream can confuse a raw label with a mapped
# target class.
TARGET_COLUMN = "threat_class"

# Original capture timestamp, preserved as a STRING for provenance.
#
# Deliberately not parsed into a datetime. CIC-IDS2017 timestamps are
# inconsistent and ambiguous: the subset files store '7/7/2017 3:30' in
# 12-hour form with no AM/PM marker, so an afternoon capture at 13:00 is
# indistinguishable from 01:00. Other files in the full dataset use
# '03/07/2017 08:55:58'. Parsing would silently invent wrong times.
#
# Alerts carry their own observation timestamp generated at replay time; this
# column exists only to show which capture minute a flow came from.
CAPTURE_TS_COLUMN = "capture_timestamp"


def _wanted_columns() -> set[str]:
    """Canonical names to retain: features + evidence extras + identity + label."""
    return (
        set(ALL_PROFILE_FEATURES)
        | set(EVIDENCE_EXTRA_COLUMNS)
        | set(IDENTITY_COLUMNS)
        | {LABEL_COLUMN}
    )


def _read_one(path: Path, wanted: set[str], verbose: bool = True) -> pd.DataFrame | None:
    """Read a single CSV, keeping only wanted columns, normalised.

    Column selection uses a callable `usecols` that normalises each header
    before testing membership. This is what makes the inconsistent leading
    whitespace in the CIC-IDS2017 headers a non-issue: we never have to write
    ' Flow Duration' with its exact spacing anywhere.
    """
    last_error: Exception | None = None

    for encoding in DATASET.encodings:
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                usecols=lambda c: normalize_column(c) in wanted,
                low_memory=False,
                on_bad_lines="warn",
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            if verbose:
                print(f"    {encoding}: UnicodeDecodeError, trying next encoding")
            continue
        except ValueError as exc:
            # Raised when usecols matches nothing, i.e. an unexpected layout.
            print(f"    ERROR: {path.name} -- {exc}")
            return None

        frame.columns = [normalize_column(c) for c in frame.columns]

        # Drop the duplicated header. VERIFIED on the DDoS file: positions 40
        # and 61 are both 'Fwd Header Length' and are byte-identical across all
        # 225,745 rows, so removing the '.1' copy loses no information.
        # normalize_column() leaves the '.1' suffix intact, which is what makes
        # the duplicate droppable by name.
        to_drop = [c for c in frame.columns if c in DROP_COLUMNS]
        if to_drop:
            frame = frame.drop(columns=to_drop)
            if verbose:
                print(f"    dropped duplicate column(s): {to_drop}")

        if verbose:
            note = "" if encoding == DATASET.encodings[0] else f"  (needed {encoding})"
            print(f"    read {len(frame):,} rows x {len(frame.columns)} cols{note}")
        return frame

    print(f"    ERROR: could not decode {path.name}: {last_error}")
    return None


def _clean_one(frame: pd.DataFrame, feature_cols: list[str], stats: Counter,
               verbose: bool = True) -> pd.DataFrame:
    """Clean one file's frame: empty rows, numeric coercion, inf, labels."""
    start = len(frame)

    # --- fully empty rows --------------------------------------------------
    # Not hypothetical: Thursday-...-WebAttacks.csv contains 288,602 rows where
    # every single cell is NaN. Removing them first keeps later diagnostics
    # meaningful.
    empty_mask = frame.isna().all(axis=1)
    n_empty = int(empty_mask.sum())
    if n_empty:
        frame = frame[~empty_mask]
        stats["empty_rows"] += n_empty
        if verbose:
            print(f"    removed {n_empty:,} fully-empty rows")

    # --- rows with no label -----------------------------------------------
    if LABEL_COLUMN in frame.columns:
        no_label = frame[LABEL_COLUMN].isna()
        n_no_label = int(no_label.sum())
        if n_no_label:
            frame = frame[~no_label]
            stats["missing_label"] += n_no_label
            if verbose:
                print(f"    removed {n_no_label:,} rows with no label")

    # --- numeric coercion + infinity --------------------------------------
    # errors='coerce' turns any malformed value ('', 'N/A', stray text) into
    # NaN. pandas 3.0 removed errors='ignore', so 'coerce' is the only
    # non-raising option, which is what we want here.
    present = [c for c in feature_cols if c in frame.columns]
    numeric = frame[present].apply(pd.to_numeric, errors="coerce")

    # Count non-numeric values introduced by coercion, before touching inf, so
    # the two problems are reported separately.
    n_coerced = int((numeric.isna() & frame[present].notna()).to_numpy().sum())
    if n_coerced:
        stats["non_numeric_cells"] += n_coerced
        if verbose:
            print(f"    coerced {n_coerced:,} non-numeric cells to NaN")

    # +/-inf comes from rate columns divided by a zero-microsecond duration.
    # MEASURED on the DDoS file: 'Flow Bytes/s' 30 inf, 'Flow Packets/s' 34.
    inf_mask = np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan))
    n_inf = int(inf_mask.sum())
    if n_inf:
        numeric = numeric.mask(pd.DataFrame(inf_mask, index=numeric.index,
                                            columns=numeric.columns))
        stats["infinite_cells"] += n_inf
        if verbose:
            cols = numeric.columns[inf_mask.any(axis=0)].tolist()
            print(f"    replaced {n_inf:,} +/-inf values with NaN in {cols}")

    frame = frame.copy()
    frame[present] = numeric

    # --- label mapping ----------------------------------------------------
    # The label column is OPTIONAL. Labelled data (CIC-IDS2017, for training and
    # evaluation) has it; genuinely unlabelled traffic handed to us for scoring
    # does not. Without this branch the pipeline raised KeyError: 'Label' and
    # could only ever process data whose answers were already known -- which
    # would make the tool useless for its actual purpose.
    if LABEL_COLUMN in frame.columns:
        raw_labels = frame[LABEL_COLUMN].astype("string").str.strip()
        frame[TARGET_COLUMN] = raw_labels.map(map_label)

        unmapped = frame[TARGET_COLUMN].isna()
        n_unmapped = int(unmapped.sum())
        if n_unmapped:
            # Report exactly which labels were discarded and how many of each.
            discarded = Counter(raw_labels[unmapped].dropna().tolist())
            for label, count in discarded.most_common():
                known = label in LABEL_MAP
                reason = "out of MVP scope" if known else "NOT IN LABEL_MAP"
                stats[f"dropped_label:{label}"] += count
                if verbose:
                    print(f"    dropped {count:,} rows labelled {label!r} "
                          f"({reason})")
            frame = frame[~unmapped]

        frame = frame.drop(columns=[LABEL_COLUMN])
    else:
        stats["unlabelled_files"] += 1
        if verbose:
            print(f"    no {LABEL_COLUMN!r} column: treating this as unlabelled "
                  f"traffic to be scored")

    if CAPTURE_TS_COLUMN not in frame.columns and "Timestamp" in frame.columns:
        frame = frame.rename(columns={"Timestamp": CAPTURE_TS_COLUMN})

    stats["rows_in"] += start
    stats["rows_kept"] += len(frame)
    return frame


def clean_dataset(
    files: list[str] | None = None,
    max_per_class: int | None | object = _USE_CONFIG_DEFAULT,
    out_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full cleaning pipeline and write the processed CSV.

    max_per_class:
        omitted  -> use DATASET.max_rows_per_class from config
        int      -> cap each class at that many rows
        None     -> no cap, keep every row

    Returns the cleaned DataFrame. Raises FileNotFoundError if no input file
    could be read.
    """
    t0 = time.time()
    PATHS.ensure_dirs()

    names = files if files is not None else list(DATASET.subset_files)
    if DATASET.use_all_csvs and files is None:
        names = sorted(p.name for p in PATHS.raw_dir.glob("*.csv"))

    out_path = out_path or PATHS.processed_csv
    wanted = _wanted_columns()
    feature_cols = sorted(set(ALL_PROFILE_FEATURES) | set(EVIDENCE_EXTRA_COLUMNS))

    print(RULE)
    print("CLEANING RAW FLOW RECORDS")
    print(RULE)
    print(f"  source : {PATHS.raw_dir}")
    print(f"  output : {out_path}")
    print(f"  files  : {len(names)}")
    print()

    stats: Counter = Counter()
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for name in names:
        path = PATHS.raw_dir / name
        if not path.exists():
            missing.append(name)
            print(f"  [MISSING] {name}")
            continue

        print(f"  {name}")
        raw = _read_one(path, wanted, verbose)
        if raw is None:
            continue
        frames.append(_clean_one(raw, feature_cols, stats, verbose))
        print()

    if missing:
        print(f"  WARNING: {len(missing)} file(s) not found in {PATHS.raw_dir}:")
        for name in missing:
            print(f"    {name}")
        print()

    if not frames:
        raise FileNotFoundError(
            f"No readable CSV files in {PATHS.raw_dir}. "
            f"Expected: {', '.join(names)}"
        )

    # --- combine ----------------------------------------------------------
    print(RULE)
    print("COMBINING AND DE-DUPLICATING")
    print(RULE)

    data = pd.concat(frames, ignore_index=True)
    del frames
    print(f"  combined                     : {len(data):,} rows")

    # --- global de-duplication --------------------------------------------
    # Must run AFTER concatenation: identical flows appear both within and
    # across files. MEASURED on this subset: 115,069 of 703,245 rows (16.4%)
    # are exact duplicates.
    #
    # Duplicates are compared on FEATURES + LABEL only, excluding identity
    # columns. Two flows with the same behaviour but different ephemeral source
    # ports are the same training example; keeping both inflates the row count
    # and lets identical records land in both the train and test split, which
    # would make the reported accuracy optimistic.
    dedup_cols = [c for c in data.columns
                  if c not in IDENTITY_COLUMNS and c != CAPTURE_TS_COLUMN]
    before = len(data)
    data = data.drop_duplicates(subset=dedup_cols, keep="first")
    n_dupes = before - len(data)
    stats["duplicates"] = n_dupes
    print(f"  removed exact duplicates     : {n_dupes:,} "
          f"({n_dupes / max(before, 1) * 100:.1f}%)")
    print(f"    (compared on {len(dedup_cols)} feature/label columns; "
          f"identity columns excluded)")

    # --- drop rows with unusable features ---------------------------------
    present_features = [c for c in ALL_PROFILE_FEATURES if c in data.columns]
    before = len(data)
    bad = data[present_features].isna().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        # Attribute the loss to specific columns so a future feature choice
        # that costs many rows is visible rather than mysterious.
        culprits = data.loc[bad, present_features].isna().sum()
        culprits = culprits[culprits > 0].sort_values(ascending=False)
        data = data[~bad]
        print(f"  removed rows with NaN/inf    : {n_bad:,}")
        for col, count in culprits.items():
            print(f"    {col}: {count:,} rows")
    else:
        print("  removed rows with NaN/inf    : 0")
    stats["unusable_rows"] = n_bad

    print(f"  clean rows                   : {len(data):,}")

    if data.empty:
        raise ValueError("Cleaning removed every row. Check the input files.")

    # Whether the output carries ground-truth labels. Everything below that
    # depends on classes is conditional on this: an unlabelled scoring dataset
    # has no class distribution to report and nothing to stratify a cap on.
    labelled = TARGET_COLUMN in data.columns

    # --- class distribution before capping ---------------------------------
    if labelled:
        print()
        print(RULE)
        print("CLASS DISTRIBUTION")
        print(RULE)
        counts = data[TARGET_COLUMN].value_counts()
        total = len(data)
        print(f"  {'class':<12} {'rows':>10}  {'share':>7}")
        print(f"  {'-' * 12} {'-' * 10}  {'-' * 7}")
        for cls in TARGET_CLASSES:
            n = int(counts.get(cls, 0))
            print(f"  {cls:<12} {n:>10,}  {n / total * 100:6.2f}%")
        unexpected = set(counts.index) - set(TARGET_CLASSES)
        for cls in sorted(unexpected):
            print(f"  {cls:<12} {int(counts[cls]):>10,}  <-- unexpected class")
    else:
        print()
        print(RULE)
        print("UNLABELLED DATASET")
        print(RULE)
        print("  No ground-truth labels present. This output is for SCORING")
        print("  only; it cannot be used to train or to measure accuracy.")
        print(f"  Score it with:  python -m model.score_dataset --input "
              f"{out_path.name}")

    # --- per-class cap ----------------------------------------------------
    cap = (DATASET.max_rows_per_class
           if max_per_class is _USE_CONFIG_DEFAULT else max_per_class)

    if cap is not None and not labelled:
        # Capping is a class-balancing operation, so it is meaningless without
        # labels. Silently applying a global head/sample instead would quietly
        # discard most of the traffic the user asked to have scored.
        print(f"\n  Per-class cap of {cap:,} skipped: no labels to balance on.")
        print("  Every row is kept. Use --max-flows on the scorer to limit work.")
        cap = None

    if cap is not None:
        print()
        print(f"  Applying per-class cap of {cap:,} rows")
        print("  (per-class, not a single global cap: Botnet is 0.28% of this")
        print("   subset, so a global cap would sample it almost out of existence)")

        pieces = []
        for cls, group in data.groupby(TARGET_COLUMN, sort=False):
            if len(group) > cap:
                pieces.append(group.sample(cap, random_state=DATASET.random_state))
                print(f"    {cls:<12} {len(group):>10,} -> {cap:,}")
            else:
                pieces.append(group)
                print(f"    {cls:<12} {len(group):>10,} -> unchanged (below cap)")
        data = pd.concat(pieces, ignore_index=True)
    elif labelled:
        print("\n  No per-class cap applied (--no-cap).")

    # Shuffle so the file is not ordered by class. train.py splits stratified
    # regardless, but an unordered file makes head-of-file inspection and
    # replay demos representative.
    data = data.sample(frac=1.0, random_state=DATASET.random_state).reset_index(drop=True)

    # Stable column order: identity, capture time, then features alphabetically,
    # then the target if present. Keeps diffs between runs readable.
    ordered = (
        [c for c in IDENTITY_COLUMNS if c in data.columns]
        + ([CAPTURE_TS_COLUMN] if CAPTURE_TS_COLUMN in data.columns else [])
        + sorted(c for c in data.columns
                 if c not in IDENTITY_COLUMNS
                 and c not in (CAPTURE_TS_COLUMN, TARGET_COLUMN))
        + ([TARGET_COLUMN] if labelled else [])
    )
    data = data[ordered]

    # --- write ------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576

    meta = {
        "generated_by": "preprocessing/clean_data.py",
        "source_files": [n for n in names if n not in missing],
        "missing_files": missing,
        "rows_read": int(stats["rows_in"]),
        "rows_written": int(len(data)),
        "removed": {
            "fully_empty_rows": int(stats["empty_rows"]),
            "missing_label_rows": int(stats["missing_label"]),
            "exact_duplicates": int(stats["duplicates"]),
            "nan_or_inf_rows": int(stats["unusable_rows"]),
            "non_numeric_cells_coerced": int(stats["non_numeric_cells"]),
            "infinite_cells_replaced": int(stats["infinite_cells"]),
        },
        "dropped_labels": {
            key.split(":", 1)[1]: int(value)
            for key, value in stats.items() if key.startswith("dropped_label:")
        },
        "max_rows_per_class": cap,
        "random_state": DATASET.random_state,
        "labelled": labelled,
        "class_counts": ({k: int(v) for k, v in
                          data[TARGET_COLUMN].value_counts().items()}
                         if labelled else None),
        "columns": list(data.columns),
        "identity_columns_present": [c for c in IDENTITY_COLUMNS
                                     if c in data.columns],
        "target_column": TARGET_COLUMN if labelled else None,
        "notes": [
            "Column names normalised: leading/trailing whitespace stripped.",
            "Duplicate 'Fwd Header Length' column dropped; verified identical "
            "to the retained copy on all rows.",
            "capture_timestamp kept as a string, not parsed: CIC-IDS2017 uses "
            "12-hour times with no AM/PM marker, so parsing would be ambiguous.",
            "Flag columns are binary 0/1 presence indicators, not counts.",
        ] + ([] if labelled else [
            "NO LABEL COLUMN in the source. This file is for scoring only: it "
            "cannot be used for training, and accuracy cannot be measured "
            "against it.",
        ]),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- summary ----------------------------------------------------------
    print()
    print(RULE)
    print("DONE")
    print(RULE)
    print(f"  rows read     : {stats['rows_in']:,}")
    print(f"  rows written  : {len(data):,}")
    print(f"  output        : {out_path}  ({size_mb:.1f} MB)")
    print(f"  metadata      : {meta_path.name}")
    print(f"  elapsed       : {time.time() - t0:.1f}s")
    print()
    if labelled:
        print("  final class counts:")
        for cls, n in data[TARGET_COLUMN].value_counts().items():
            print(f"    {cls:<12} {n:>8,}")
        print()
        print("  Next step:  python -m model.train")
    else:
        print("  No labels: this dataset is for scoring, not training.")
        print()
        print(f"  Next step:  python -m model.score_dataset --input {out_path}")
    print(RULE)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean raw CIC-IDS2017 CSVs into one processed file."
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None, metavar="N",
        help=f"Row cap per class (config default: {DATASET.max_rows_per_class}).",
    )
    parser.add_argument(
        "--no-cap", action="store_true",
        help="Keep every row; ignore the per-class cap.",
    )
    parser.add_argument(
        "--file", action="append", default=None, metavar="NAME",
        help="Process only this filename in data/raw/. Repeatable.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="PATH",
        help="Output CSV path (default: data/processed/flows_clean.csv).",
    )
    parser.add_argument("--quiet", action="store_true", help="Less per-file detail.")
    args = parser.parse_args()

    if args.no_cap and args.max_per_class is not None:
        parser.error("--no-cap and --max-per-class are mutually exclusive.")

    if args.no_cap:
        cap: int | None | object = None
    elif args.max_per_class is not None:
        cap = args.max_per_class
    else:
        cap = _USE_CONFIG_DEFAULT

    try:
        clean_dataset(
            files=args.file,
            max_per_class=cap,
            out_path=args.out,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

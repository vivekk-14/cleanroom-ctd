"""
Feature configuration for CIC-IDS2017 flow records.

This module is the single source of truth for:

  1. How raw CSV column names are normalised.
  2. Which normalised names the model uses (two selectable profiles).
  3. How raw dataset labels map onto the project's four target classes.
  4. Which columns are identity/metadata rather than model features.

WHY THIS FILE IS NECESSARY
--------------------------
The CIC-IDS2017 CSV headers are inconsistent. Verified by reading the actual
file headers on disk:

    ' Destination Port'              <- leading space
    ' Flow Duration'                 <- leading space
    'Total Length of Fwd Packets'    <- NO leading space
    'Flow Bytes/s'                   <- NO leading space
    ' Flow Packets/s'                <- leading space

Roughly two-thirds of columns carry a leading space and the rest do not, with
no discernible pattern. Any code that hardcodes these names is one typo away
from a KeyError. Every column name is therefore normalised on load, and
features are referenced by canonical name only.

Also verified in the real files:

  * ' Fwd Header Length' appears TWICE (header positions 40 and 61). pandas
    renames the second occurrence to ' Fwd Header Length.1'. It is dropped.
  * Flag columns (' SYN Flag Count', ' ACK Flag Count', ...) are BINARY 0/1
    presence indicators, not counts. Measured: unique() == [0, 1]. See the
    note on SYN_FLAG_IS_BINARY below.
  * 'Flow Bytes/s' and ' Flow Packets/s' contain +inf (division by a
    zero-microsecond flow duration) and NaN.
  * The label column is ' Label' with a leading space.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. Column-name normalisation
# ---------------------------------------------------------------------------


def normalize_column(name: str) -> str:
    """Convert a raw CSV column name to its canonical form.

    Rules applied in order:
      1. Strip leading/trailing whitespace  (' Flow Duration' -> 'Flow Duration')
      2. Collapse internal whitespace runs  ('Flow  Duration' -> 'Flow Duration')
      3. Strip non-breaking spaces and BOM  (seen in some redistributions)

    Case is preserved. The dataset's own capitalisation is kept so that
    canonical names remain recognisable to anyone familiar with CIC-IDS2017,
    and so they read well in dashboard evidence panels.

    >>> normalize_column(' Flow Duration')
    'Flow Duration'
    >>> normalize_column('Total Length of Fwd Packets')
    'Total Length of Fwd Packets'
    >>> normalize_column('\\ufeff Destination Port ')
    'Destination Port'
    """
    cleaned = name.replace("\ufeff", "").replace("\xa0", " ")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def normalize_columns(names: list[str]) -> list[str]:
    """Normalise a list of column names, preserving order."""
    return [normalize_column(n) for n in names]


# ---------------------------------------------------------------------------
# 2. Identity / metadata columns
# ---------------------------------------------------------------------------
# Present only in the TrafficLabelling variant of CIC-IDS2017 (85 columns).
# The MachineLearningCVE variant (79 columns) lacks them.
#
# These are carried through cleaning so alerts can show real src/dst IPs, but
# they are NEVER given to the model. Training on IP addresses or timestamps
# would let the classifier memorise which lab host ran the attack rather than
# learn traffic behaviour, and the model would be worthless on any other
# network.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Protocol",
    "Timestamp",
)

LABEL_COLUMN: str = "Label"

# Dropped outright.
#   'Fwd Header Length.1' : pandas-generated name for the duplicated column.
#   'Fwd Header Length.1' with a leading space is normalised to the same name.
DROP_COLUMNS: tuple[str, ...] = (
    "Fwd Header Length.1",
)


# ---------------------------------------------------------------------------
# 3. Feature profiles
# ---------------------------------------------------------------------------
# STRICT_UNIDIRECTIONAL -- the default, and the profile that matches the
# problem statement.
#
# Rationale: a hardware data diode or a one-way SPAN copy delivers traffic in
# a single direction. Reverse-path counters (Total Backward Packets, Bwd
# Packet Length Mean, Bwd IAT *, Down/Up Ratio) may be unavailable or
# unreliable in that deployment. Training on them would produce a model that
# cannot run in the environment the problem describes.
#
# CIC-IDS2017 was generated from bidirectional captures, so its "Fwd" features
# are what a forward-direction-only observer would compute. Restricting to
# them is the closest honest approximation of the target deployment that this
# dataset permits. This limitation is documented in the README.
#
# MEASURED (3-file subset, deduplicated, 75/25 stratified, 60 trees depth 20):
#   accuracy 0.9989   macro F1 0.9549   Botnet F1 0.8211
#
# Note this profile SCORES BETTER than the bidirectional one, chiefly because
# forward packet-length statistics and the TCP initial window size separate
# botnet C2 beacons from benign traffic more sharply than reverse-path volume
# does.
STRICT_UNIDIRECTIONAL: tuple[str, ...] = (
    # --- flow-level scalars observable from one direction ---
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    # --- forward packet-size distribution ---
    # Top discriminators after Destination Port. Port scans send near-empty
    # SYN probes (mean ~0 bytes); DDoS floods show a tight size distribution.
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    # --- forward rate ---
    "Fwd Packets/s",
    # --- forward inter-arrival timing ---
    # Botnet C2 beacons are periodic: low IAT Std relative to IAT Mean.
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    # --- forward header / TCP behaviour ---
    "Fwd Header Length",
    "Fwd PSH Flags",
    "act_data_pkt_fwd",       # forward packets carrying >0 payload bytes
    "min_seg_size_forward",   # smallest observed forward segment
    "Init_Win_bytes_forward", # TCP initial window; strong OS/stack signal
    # --- destination ---
    # Highest importance feature (0.17). See the caveat in the README: on this
    # dataset it partly encodes the lab's port assignments (botnet C2 on 8080).
    "Destination Port",
)

# BIDIRECTIONAL -- comparison profile only. This is the feature list from the
# original problem statement, section 6.
#
# Retained so the team can demonstrate the measured difference rather than
# assert it. Two of its features could not be justified:
#
#   'SYN Flag Count' -- MEASURED feature importance 0.0010, effectively
#       useless. The column is a binary 0/1 flag-present indicator, and in the
#       DDoS file every single DDoS row has SYN=0 while 7500 BENIGN rows have
#       SYN=1. It carries almost no signal for these classes.
#
#   'ACK Flag Count' -- also binary; importance 0.045. Weak but non-trivial.
#
# Both are kept in this profile because the problem statement asked for them,
# and their low importance is itself a finding worth showing.
#
# MEASURED (identical protocol to above):
#   accuracy 0.9979   macro F1 0.9140   Botnet F1 0.6583
BIDIRECTIONAL: tuple[str, ...] = (
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow IAT Mean",
    "Flow IAT Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "Destination Port",
)

FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "STRICT_UNIDIRECTIONAL": STRICT_UNIDIRECTIONAL,
    "BIDIRECTIONAL": BIDIRECTIONAL,
}

DEFAULT_PROFILE: str = "STRICT_UNIDIRECTIONAL"


def get_features(profile: str = DEFAULT_PROFILE) -> list[str]:
    """Return the canonical feature list for `profile`.

    Raises ValueError on an unknown profile name, listing the valid options,
    rather than failing later with a confusing KeyError during training.
    """
    if profile not in FEATURE_PROFILES:
        valid = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(
            f"Unknown feature profile {profile!r}. Valid options: {valid}"
        )
    return list(FEATURE_PROFILES[profile])


# ---------------------------------------------------------------------------
# Feature-contract validation
# ---------------------------------------------------------------------------
class FeatureContractError(ValueError):
    """Raised when input features do not satisfy a model's feature contract.

    Subclasses ValueError so existing `except ValueError` handlers still catch
    it, while allowing callers to distinguish a contract violation from any
    other bad argument.
    """

    def __init__(self, message: str, missing: list[str], extra: list[str],
                 expected: list[str]) -> None:
        super().__init__(message)
        self.missing = missing
        self.extra = extra
        self.expected = expected


def validate_features(
    flow_features: dict,
    expected_features: list[str],
    *,
    allow_extra: bool = True,
) -> dict:
    """Check input against a feature contract and return an ordered subset.

    Args:
        flow_features: mapping of feature name to value. Key order is irrelevant.
        expected_features: the exact features the model was fitted on, in the
            order it expects them.
        allow_extra: if True (default) unrelated keys are ignored; if False their
            presence is an error.

    Returns:
        A new dict containing exactly `expected_features`, inserted in that
        order. On Python 3.7+ dicts preserve insertion order, so iterating the
        result yields the model's feature order without a second lookup.

    Raises:
        FeatureContractError: if any expected feature is absent, or if extra keys
            are present and allow_extra is False.

    Why extra keys are permitted by default
    ---------------------------------------
    The replay engine passes whole CSV rows, which legitimately carry identity
    columns (Source IP, Flow ID, capture_timestamp) that the model must never
    see. Treating those as an error would force every caller to pre-filter, and
    a caller that filters wrongly is exactly the failure this function exists to
    prevent. The returned dict contains only contract features, so extras cannot
    reach the model regardless.

    This is the check that catches the mistake found in testing: a 14-feature
    BIDIRECTIONAL CSV was passed to an 18-feature STRICT_UNIDIRECTIONAL model,
    and 13 absent features were silently filled with zeros. The model returned
    confident predictions computed largely from invented data.

    >>> validate_features({"b": 2, "a": 1}, ["a", "b"])
    {'a': 1, 'b': 2}
    >>> validate_features({"a": 1}, ["a", "b"])
    Traceback (most recent call last):
        ...
    preprocessing.feature_config.FeatureContractError: ...
    """
    if not isinstance(flow_features, dict):
        raise FeatureContractError(
            f"Expected a dict of feature values, got "
            f"{type(flow_features).__name__}.",
            missing=list(expected_features), extra=[],
            expected=list(expected_features),
        )

    missing = [name for name in expected_features if name not in flow_features]
    extra = [name for name in flow_features if name not in expected_features]

    if missing:
        raise FeatureContractError(
            f"Feature contract violated: {len(missing)} of "
            f"{len(expected_features)} required features are absent.\n"
            f"  missing : {missing}\n"
            f"  received: {sorted(flow_features)[:8]}"
            f"{' ...' if len(flow_features) > 8 else ''}\n"
            f"Check that the input matches the model's profile. A "
            f"{len(flow_features)}-feature record cannot be scored by a "
            f"{len(expected_features)}-feature model.",
            missing=missing, extra=extra, expected=list(expected_features),
        )

    if extra and not allow_extra:
        raise FeatureContractError(
            f"Feature contract violated: {len(extra)} unexpected keys present "
            f"and allow_extra=False.\n  extra: {extra[:10]}",
            missing=[], extra=extra, expected=list(expected_features),
        )

    # Rebuilt in contract order. Never reuse the caller's ordering.
    return {name: flow_features[name] for name in expected_features}


def describe_contract_mismatch(flow_features: dict, profile: str) -> str:
    """Explain, in prose, why a record does not fit a profile.

    Used by the scoring tool and the dashboard to turn a contract failure into
    an actionable message instead of a stack trace.
    """
    expected = get_features(profile)
    if not isinstance(flow_features, dict):
        return f"Input is {type(flow_features).__name__}, not a dict."

    missing = [f for f in expected if f not in flow_features]
    if not missing:
        return f"Record satisfies the {profile} contract ({len(expected)} features)."

    # Suggest a profile that the record does satisfy, if one exists.
    alternatives = [
        name for name, features in FEATURE_PROFILES.items()
        if name != profile and all(f in flow_features for f in features)
    ]
    lines = [
        f"Record does not satisfy the {profile} contract: "
        f"{len(missing)} of {len(expected)} features absent.",
        f"Missing: {', '.join(missing)}",
    ]
    if alternatives:
        lines.append(
            f"This record DOES satisfy: {', '.join(alternatives)}. "
            f"Score it with that profile instead."
        )
    else:
        lines.append(
            "It satisfies no configured profile. The file is probably not in "
            "CIC-IDS2017 flow format."
        )
    return "\n".join(lines)


def profiles_satisfied_by(flow_features: dict) -> list[str]:
    """Every profile whose contract the given record fully satisfies."""
    if not isinstance(flow_features, dict):
        return []
    return [
        name for name, features in FEATURE_PROFILES.items()
        if all(f in flow_features for f in features)
    ]


# Union of every column any profile might need. Used as the `usecols` argument
# to pandas.read_csv so that a single cleaning run produces a processed file
# usable by both profiles, without re-reading 260 MB of CSV.
ALL_PROFILE_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(  # dict.fromkeys preserves order while de-duplicating
        list(STRICT_UNIDIRECTIONAL) + list(BIDIRECTIONAL)
    )
)


# ---------------------------------------------------------------------------
# 4. Evidence-support columns
# ---------------------------------------------------------------------------
# Columns kept in the processed file purely so detection/evidence.py can quote
# them in alerts. They are NOT model inputs.
#
# 'Flow Packets/s' and 'Flow Bytes/s' are the most human-legible way to show
# "this was a flood", so they are quoted in evidence even when the strict
# profile does not train on them.
EVIDENCE_EXTRA_COLUMNS: tuple[str, ...] = (
    "Flow Packets/s",
    "Flow Bytes/s",
    "Total Backward Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
)


# ---------------------------------------------------------------------------
# 5. Label mapping
# ---------------------------------------------------------------------------
# Maps raw ' Label' values onto the four MVP target classes.
#
# All keys below were read from the actual CSVs, with exact row counts:
#
#   BENIGN                       2373847  (across all 8 files)
#   DoS Hulk                      231073
#   PortScan                      158930
#   DDoS                          128027
#   DoS GoldenEye                  10293
#   FTP-Patator                     7938
#   SSH-Patator                     5897
#   DoS slowloris                   5796
#   DoS Slowhttptest                5499
#   Bot                             1966   <- maps to 'Botnet'
#   Web Attack - Brute Force        1507   (en-dash U+0096 in latin-1)
#   Web Attack - XSS                 652
#   Infiltration                      36
#   Web Attack - Sql Injection        21
#   Heartbleed                        11
#
# For the MVP only the four target classes are mapped. Everything else maps to
# None and is DROPPED by clean_data.py, which prints a count of what it
# discarded. Nothing is silently thrown away.
#
# To extend later (DoS, brute force, web attacks), add entries here. No other
# module needs changing: train.py derives its class list from the data.
LABEL_MAP: dict[str, str | None] = {
    # --- target classes ---
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Botnet",          # dataset says 'Bot', project class is 'Botnet'
    # --- present in the full dataset, out of scope for the MVP ---
    "DoS Hulk": None,
    "DoS GoldenEye": None,
    "DoS slowloris": None,
    "DoS Slowhttptest": None,
    "Heartbleed": None,
    "FTP-Patator": None,
    "SSH-Patator": None,
    "Infiltration": None,
    "Web Attack \x96 Brute Force": None,  # latin-1 en-dash, as stored
    "Web Attack \x96 XSS": None,
    "Web Attack \x96 Sql Injection": None,
    "Web Attack - Brute Force": None,     # UTF-8 hyphen variants, seen in
    "Web Attack - XSS": None,             # some redistributions
    "Web Attack - Sql Injection": None,
    "Web Attack \u2013 Brute Force": None,  # true en-dash variants
    "Web Attack \u2013 XSS": None,
    "Web Attack \u2013 Sql Injection": None,
}

# Canonical class order. Fixed so that confusion matrices, the label encoder,
# and dashboard colour assignments stay consistent between runs.
TARGET_CLASSES: tuple[str, ...] = ("BENIGN", "DDoS", "PortScan", "Botnet")


def map_label(raw_label: str) -> str | None:
    """Map a raw dataset label to a target class, or None to drop the row.

    Unrecognised labels return None rather than raising, so a new CSV with an
    unexpected label reduces the dataset instead of crashing the pipeline.
    clean_data.py reports every unmapped label it encounters.
    """
    if raw_label is None:
        return None
    return LABEL_MAP.get(str(raw_label).strip())


# ---------------------------------------------------------------------------
# 6. Dataset quirks, recorded as constants
# ---------------------------------------------------------------------------
# Referenced by evidence.py and the README so the same facts are not restated
# (and potentially contradicted) in several places.

# Flag columns are binary presence indicators, not counts. Measured:
# ' SYN Flag Count'.unique() == [0, 1] on the DDoS file.
#
# Consequence: the problem statement's example evidence of
# {"SYN Flag Count": 17892} is not representable in this dataset, and
# "abnormal SYN/ACK ratio" cannot be computed from these columns. evidence.py
# uses forward packet rate and payload-size statistics instead, and says so.
SYN_FLAG_IS_BINARY: bool = True

FLAG_COLUMNS: tuple[str, ...] = (
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
)

# Columns known to contain +/-inf from divide-by-zero-duration flows.
# Measured on the DDoS file: 'Flow Bytes/s' 30 inf + 4 NaN,
# ' Flow Packets/s' 34 inf.
INFINITY_PRONE_COLUMNS: tuple[str, ...] = (
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
)

# IANA protocol numbers -> names, for alert display. Measured distribution in
# the DDoS file: 6 (TCP) 192820, 17 (UDP) 32871, 0 54.
#
# Protocol 0 is HOPOPT. Here it marks flows the CICFlowMeter tool could not
# attribute to TCP or UDP, so it is displayed as 'OTHER' rather than a
# misleading protocol name.
PROTOCOL_NAMES: dict[int, str] = {
    0: "OTHER",
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    58: "ICMPv6",
}


def protocol_name(proto: object) -> str:
    """Map an IANA protocol number to a display name.

    Returns 'UNKNOWN' for values that are missing or not integer-like, so
    alert rendering never raises on malformed input.
    """
    try:
        return PROTOCOL_NAMES.get(int(float(proto)), f"PROTO-{int(float(proto))}")
    except (TypeError, ValueError):
        return "UNKNOWN"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("Feature configuration")
    print("=" * 68)
    for name, feats in FEATURE_PROFILES.items():
        marker = "  (default)" if name == DEFAULT_PROFILE else ""
        print(f"\n{name}{marker} - {len(feats)} features")
        for i, f in enumerate(feats, 1):
            print(f"  {i:2d}. {f}")

    print(f"\nUnion of all profiles: {len(ALL_PROFILE_FEATURES)} columns")
    print(f"Identity columns kept for alerts: {len(IDENTITY_COLUMNS)}")
    print(f"Extra evidence columns: {len(EVIDENCE_EXTRA_COLUMNS)}")

    print(f"\nTarget classes: {', '.join(TARGET_CLASSES)}")
    mapped = {k: v for k, v in LABEL_MAP.items() if v is not None}
    dropped = [k for k, v in LABEL_MAP.items() if v is None]
    print(f"Labels mapped to a class: {len(mapped)}")
    for k, v in mapped.items():
        arrow = "" if k == v else "   <-- renamed"
        print(f"  {k!r} -> {v}{arrow}")
    print(f"Labels dropped for the MVP: {len(dropped)}")

    print("\nNormalisation self-check:")
    for raw in (" Destination Port", "Total Length of Fwd Packets",
                " Flow Packets/s", "\ufeff Flow Duration "):
        print(f"  {raw!r:36s} -> {normalize_column(raw)!r}")
    print("=" * 68)

"""
Streamlit dashboard for the passive threat-detection prototype.

Reads runtime/alerts.jsonl (written by streaming/replay.py) and renders live
metrics, threat distribution, an alert table, and per-alert evidence.

    streamlit run dashboard/app.py

The dashboard is strictly a READER. It never writes to the alert stream, never
triggers a replay, and never sends anything to the monitored network. It also
offers no blocking, mitigation, or response controls -- deliberately, because
the monitoring enclave cannot communicate back to the production network.

Coupling
--------
This file imports nothing from model/train.py. It reads the JSON Lines stream
and, for the model-status panel, calls model.predict.model_status(). The
dashboard therefore works before any model is trained: alerts show
threat='UNKNOWN' and a banner explains why.
"""

from __future__ import annotations

# Arm the passive guard before anything else, so nothing this process imports
# can open an outbound socket. Loopback stays permitted, which is what lets
# Streamlit serve HTTP on localhost.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.passive_guard import enforce_passive_mode  # noqa: E402

enforce_passive_mode(verbose=False)

import json  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from config import (  # noqa: E402
    DASHBOARD,
    FEATURE_PROFILE,
    PATHS,
    POSTURE,
    SEVERITY,
)
from detection.alert_schema import parse_alert_line  # noqa: E402
from detection.passive_guard import posture_report  # noqa: E402
from detection.severity import describe_policy, severity_rank  # noqa: E402
from model.predict import model_status  # noqa: E402

st.set_page_config(
    page_title=DASHBOARD.page_title,
    page_icon=DASHBOARD.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_alerts(path: Path, tail_lines: int) -> list[dict]:
    """Read up to `tail_lines` alerts from the end of the JSONL stream.

    Reads the whole file then slices. For the hackathon's data volumes (a few
    thousand lines) this is far simpler than seeking backwards, and the cost is
    negligible. Malformed or partially written lines are skipped rather than
    raising: the replay engine may be mid-write when this runs.
    """
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as stream:
            lines = stream.readlines()
    except OSError:
        return []

    records = []
    for line in lines[-tail_lines:]:
        record = parse_alert_line(line)
        if record is not None:
            records.append(record)
    return records


def read_status(path: Path) -> dict:
    """Read runtime/replay_status.json, or return an empty dict."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from alert records, guaranteeing expected columns.

    Defaults matter for schema v1 records left over from an earlier run: the OOD
    columns must exist so every panel can filter on them, and `reliability` must
    default to UNKNOWN rather than RELIABLE -- an un-checked flow is not a
    trusted flow.
    """
    if not records:
        return pd.DataFrame(columns=[
            "flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
            "protocol", "threat_class", "confidence", "severity",
            "is_alertable", "needs_review", "model_prediction",
            "reliability", "in_distribution", "ood_score",
            "ground_truth", "correct",
        ])
    frame = pd.DataFrame(records)
    for column, default in (
        ("threat_class", "UNKNOWN"), ("severity", "INFO"),
        ("confidence", 0.0), ("is_alertable", False),
        ("protocol", "UNKNOWN"), ("ground_truth", None), ("correct", None),
        # v2 fields.
        ("needs_review", False), ("model_prediction", None),
        ("reliability", "UNKNOWN"), ("in_distribution", None),
        ("ood_score", 0.0), ("ood_features", None),
    ):
        if column not in frame.columns:
            frame[column] = default
    return frame


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[bool, float]:
    """Render sidebar controls and the operating-posture panel."""
    st.sidebar.title("Monitoring enclave")

    # Posture first: it is the point of the project, not a footnote.
    st.sidebar.success(f"**{POSTURE.mode}**\n\n{POSTURE.access}")
    st.sidebar.info(f"**{POSTURE.assurance}**")

    guard = posture_report()
    icon = "OK" if guard["passive_mode_armed"] else "NOT ARMED"
    st.sidebar.markdown(
        f"**Passive guard:** {icon}  \n"
        f"Outbound attempts blocked: `{guard['outbound_attempts_blocked']}`  \n"
        f"Loopback operations allowed: `{guard['loopback_operations_allowed']}`"
    )
    with st.sidebar.expander("What the guard enforces"):
        st.caption(guard["rule"])
        st.code("\n".join(guard["guarded_operations"]), language="text")
        st.caption(guard["scope_note"])

    st.sidebar.divider()
    st.sidebar.subheader("View")
    auto_refresh = st.sidebar.toggle(
        "Auto-refresh", value=True,
        help="Re-read the alert stream periodically while a replay is running.",
    )
    interval = st.sidebar.slider(
        "Refresh interval (s)", 1.0, 10.0, DASHBOARD.refresh_seconds, 0.5,
        disabled=not auto_refresh,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Model")

    # Each profile is a separate model with its own feature contract. The
    # dashboard reads alerts that already record which profile produced them, so
    # this selector only changes which model's details are DISPLAYED -- it does
    # not reclassify anything.
    trained = [name for name, info in model_status()["profiles"].items()
               if info["trained"]]
    if len(trained) > 1:
        chosen = st.sidebar.selectbox(
            "Profile", trained,
            index=trained.index(FEATURE_PROFILE)
            if FEATURE_PROFILE in trained else 0,
            help="Which trained model's contract to inspect. Alerts record the "
                 "profile that produced them.",
        )
    else:
        chosen = trained[0] if trained else FEATURE_PROFILE

    status = model_status(chosen)
    if status["model_ready"]:
        st.sidebar.markdown(
            f"Profile: `{status['profile']}`  \n"
            f"Features: `{status['n_features']}`  \n"
            f"Classes: `{', '.join(status['classes'])}`"
        )
        with st.sidebar.expander(f"Feature contract ({status['n_features']})"):
            st.caption(
                "The exact features, in the exact order, this model was fitted "
                "on. A record missing any of them is refused rather than "
                "zero-filled."
            )
            st.code("\n".join(status["features"]), language="text")

        others = [p for p in status["profiles"] if p != chosen]
        if others:
            with st.sidebar.expander("Other profiles"):
                for name in others:
                    info = status["profiles"][name]
                    mark = "trained" if info["trained"] else "not trained"
                    st.caption(f"{name}: {info['n_features']} features, {mark}")
    else:
        st.sidebar.error("No trained model")
        st.sidebar.code("python -m model.train", language="bash")

    with st.sidebar.expander("Severity policy"):
        st.code(describe_policy(), language="text")

    st.sidebar.divider()
    st.sidebar.caption(
        "This dashboard only reads local files. It offers no blocking or "
        "mitigation controls, because the enclave cannot send anything back to "
        "the monitored network."
    )
    return auto_refresh, interval


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_metrics(frame: pd.DataFrame, status: dict) -> None:
    """Top metric tiles: the four numbers that summarise the whole run.

    Deliberately four, not nine. A judge should be able to read this in a few
    seconds: how much traffic, how much is actionable, how much needs a human,
    and how much the model could not be trusted on.
    """
    total = len(frame)
    alertable = int(frame["is_alertable"].sum()) if total else 0
    # Review queue EXCLUDES alerts, so the two tiles do not double-count.
    review = (int((frame["needs_review"].fillna(False)
                   & ~frame["is_alertable"].fillna(False)).sum())
              if total else 0)
    ood = int((frame["reliability"] == "UNRELIABLE").sum()) if total else 0

    columns = st.columns(4)
    columns[0].metric("Flows observed", f"{total:,}")
    columns[1].metric("Alerts", f"{alertable:,}",
                      help="Actionable queue: severity MEDIUM or above.")
    columns[2].metric("Review", f"{review:,}",
                      help="Not actionable, but the classification is less "
                           "trustworthy than its confidence suggests. Kept "
                           "visible rather than suppressed.")
    columns[3].metric(
        "Out of distribution", f"{ood:,}",
        delta=(f"{ood / total:.1%} of traffic" if total and ood else None),
        delta_color="inverse",
        help="Input unlike anything in training. The model's answer is not "
             "trusted for these flows.",
    )

    state = status.get("state")
    if state == "RUNNING":
        planned = status.get("total_flows_planned") or 0
        processed = status.get("flows_processed") or 0
        rate = status.get("flows_per_second")
        label = (f"Replay running: {processed:,} of {planned:,} flows"
                 + (f" ({rate:.1f}/s)" if rate else ""))
        st.progress(min(processed / planned, 1.0) if planned else 0.0, text=label)
    elif state in ("COMPLETE", "STOPPED"):
        accuracy = status.get("live_accuracy")
        suffix = ""
        if accuracy is not None:
            suffix = (f" | agreement with dataset labels "
                      f"{accuracy:.2%} of {status.get('labelled_flows', 0):,}")
        st.caption(f"Replay {state.lower()}: "
                   f"{status.get('flows_processed', 0):,} flows"
                   f"{suffix}")


def render_reliability(frame: pd.DataFrame, records: list[dict]) -> None:
    """The OOD / reliability panel.

    This is the panel that distinguishes the system from a bare classifier, so
    it sits directly under the metric tiles rather than at the bottom of the
    page. Two columns: the tier counts, and the most recent OOD event in full.
    """
    st.subheader("Reliability")
    st.caption(
        "**Confidence** answers: how sure is the classifier? "
        "**Reliability** answers: do we trust that answer for this input? "
        "They are separate, and they can disagree."
    )

    if frame.empty:
        st.info("No flows yet.")
        return

    left, right = st.columns([2, 3])

    # --- tier counts ------------------------------------------------------
    with left:
        counts = frame["reliability"].value_counts()
        order = [t for t in ("RELIABLE", "DEGRADED", "UNRELIABLE", "UNKNOWN")
                 if t in counts.index]
        total = len(frame)

        for tier in order:
            count = int(counts[tier])
            share = count / total
            label = DASHBOARD.reliability_labels.get(tier, tier)
            colour = DASHBOARD.reliability_colors.get(tier, "#888")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"margin-bottom:6px'>"
                f"<div style='width:10px;height:26px;border-radius:3px;"
                f"background:{colour}'></div>"
                f"<div style='flex:1'>"
                f"<b>{tier}</b> &nbsp;<span style='color:#888'>{label}</span>"
                f"</div>"
                f"<div style='text-align:right;font-variant-numeric:tabular-nums'>"
                f"<b>{count:,}</b> <span style='color:#888'>{share:.1%}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        unreliable = int(counts.get("UNRELIABLE", 0))
        if unreliable:
            said = frame.loc[frame["reliability"] == "UNRELIABLE",
                             "model_prediction"].value_counts()
            mean_confidence = frame.loc[frame["reliability"] == "UNRELIABLE",
                                        "confidence"].mean()
            breakdown = ", ".join(f"{cls} {int(n)}" for cls, n in said.items())
            st.warning(
                f"For those {unreliable:,} flows the classifier said "
                f"**{breakdown}** at **{mean_confidence:.1%}** mean confidence. "
                f"High confidence, untrustworthy answer."
            )
        else:
            st.success("All observed flows lie within the training distribution.")

    # --- most recent OOD event -------------------------------------------
    with right:
        ood_records = [r for r in records
                       if r.get("reliability") == "UNRELIABLE"]
        if not ood_records:
            st.caption("No out-of-distribution events in this stream.")
            return

        latest = ood_records[-1]
        st.markdown("**Most recent out-of-distribution event**")

        fields = [
            ("Reported", f"`{latest.get('threat_class')}`"),
            ("Model said", f"`{latest.get('model_prediction')}`  "
                           f"at {latest.get('confidence', 0):.1%} confidence"),
            ("Reliability", f"`{latest.get('reliability')}`"),
            ("OOD score", f"{latest.get('ood_score', 0):.4f}"),
            ("Needs review", "**YES**" if latest.get("needs_review") else "no"),
            ("Flow", f"{latest.get('src_ip')}:{latest.get('src_port')} -> "
                     f"{latest.get('dst_ip')}:{latest.get('dst_port')}"),
        ]
        st.markdown("\n".join(f"- **{name}:** {value}"
                              for name, value in fields))

        features = latest.get("ood_features") or []
        if features:
            st.markdown("**Features outside the training range**")
            violations = {v.get("feature"): v
                          for v in (latest.get("ood_violations") or [])}
            for name in features:
                violation = violations.get(name)
                if violation:
                    low, high = violation.get("training_range", [0, 0])
                    st.markdown(
                        f"- `{name}` = **{violation.get('value'):,.4g}** "
                        f"({violation.get('direction')} the training range "
                        f"[{low:,.4g}, {high:,.4g}], "
                        f"{violation.get('excursion_range_widths', 0):,.1f}x "
                        f"the range width beyond it)"
                    )
                else:
                    st.markdown(f"- `{name}`")

        if latest.get("ood_reason"):
            with st.expander("Why this flow was flagged"):
                st.caption(latest["ood_reason"])



def render_distribution(frame: pd.DataFrame) -> None:
    """Threat-class and severity charts."""
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Threat distribution")
        if frame.empty:
            st.info("No flows yet.")
        else:
            counts = frame["threat_class"].value_counts()
            # Fixed order so bars do not jump between refreshes.
            order = [c for c in ("BENIGN", "DDoS", "PortScan", "Botnet",
                                 "UNKNOWN") if c in counts.index]
            order += [c for c in counts.index if c not in order]
            values = [int(counts[c]) for c in order]
            total = sum(values)

            figure = go.Figure(go.Bar(
                x=order, y=values,
                marker_color=[DASHBOARD.threat_colors.get(c, "#888")
                              for c in order],
                text=[f"{v:,}<br>{v / total:.1%}" for v in values],
                textposition="outside",
                hovertemplate="%{x}: %{y:,} flows<extra></extra>",
            ))
            figure.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="flows", xaxis_title=None, showlegend=False,
            )
            st.plotly_chart(figure)

    with right:
        st.subheader("Severity")
        if frame.empty:
            st.info("No flows yet.")
        else:
            counts = frame["severity"].value_counts()
            order = [s for s in SEVERITY.ladder if s in counts.index]
            values = [int(counts[s]) for s in order]
            figure = go.Figure(go.Pie(
                labels=order, values=values, hole=0.45,
                marker_colors=[DASHBOARD.severity_colors.get(s, "#888")
                               for s in order],
                sort=False,
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            ))
            figure.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                 legend=dict(orientation="v", x=1.0, y=0.5))
            st.plotly_chart(figure)


def render_timeline(frame: pd.DataFrame) -> None:
    """Threat class over the sequence of observed flows."""
    if frame.empty or "flow_id" not in frame.columns:
        return
    st.subheader("Observation timeline")

    working = frame.copy()
    # flow_id is 'F000123'; the numeric part is the observation sequence.
    working["sequence"] = (
        working["flow_id"].astype(str).str.lstrip("F")
        .apply(lambda text: int(text) if text.isdigit() else 0)
    )
    working = working.sort_values("sequence")

    figure = go.Figure()
    for threat in working["threat_class"].unique():
        subset = working[working["threat_class"] == threat]
        figure.add_trace(go.Scatter(
            x=subset["sequence"], y=subset["confidence"],
            mode="markers", name=str(threat),
            marker=dict(size=6,
                        color=DASHBOARD.threat_colors.get(str(threat), "#888")),
            hovertemplate=("flow %{x}<br>confidence %{y:.3f}"
                           f"<br>{threat}<extra></extra>"),
        ))
    # Severity band boundaries, so the chart shows why a flow got its label.
    for lower, name in SEVERITY.bands[1:]:
        figure.add_hline(y=lower, line_dash="dot", line_color="#bbb",
                         annotation_text=name, annotation_position="right",
                         annotation_font_size=10)
    figure.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="flow sequence", yaxis_title="confidence",
        yaxis_range=[0, 1.05],
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(figure)


def render_alert_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Filterable alert and review table. Returns the filtered frame."""
    st.subheader("Alerts and review queue")
    if frame.empty:
        st.info(
            "No flows observed yet.\n\n"
            "Start a replay in another terminal:\n"
            "`python -m streaming.replay`"
        )
        return frame

    controls = st.columns([2, 2, 2, 2, 2])
    queue = controls[0].radio(
        "Queue", ["Alerts", "Review", "Both", "All flows"], index=2,
        horizontal=False,
        help="Alerts are actionable. Review holds flows whose classification is "
             "less trustworthy than its confidence suggests.",
    )
    threats = sorted(frame["threat_class"].unique())
    chosen_threats = controls[1].multiselect("Threat", threats, default=threats)
    levels = [s for s in SEVERITY.ladder if s in set(frame["severity"])]
    chosen_levels = controls[2].multiselect("Severity", levels, default=levels)
    tiers = [t for t in ("RELIABLE", "DEGRADED", "UNRELIABLE", "UNKNOWN")
             if t in set(frame["reliability"])]
    chosen_tiers = controls[3].multiselect("Reliability", tiers, default=tiers)
    minimum_confidence = controls[4].slider("Min confidence", 0.0, 1.0, 0.0,
                                            0.05)

    alertable = frame["is_alertable"].fillna(False)
    review = frame["needs_review"].fillna(False)
    if queue == "Alerts":
        queue_mask = alertable
    elif queue == "Review":
        queue_mask = review & ~alertable
    elif queue == "Both":
        queue_mask = alertable | review
    else:
        queue_mask = pd.Series(True, index=frame.index)

    filtered = frame[
        queue_mask
        & frame["threat_class"].isin(chosen_threats)
        & frame["severity"].isin(chosen_levels)
        & frame["reliability"].isin(chosen_tiers)
        & (frame["confidence"] >= minimum_confidence)
    ]

    if filtered.empty:
        st.warning("No flows match the current filters.")
        return filtered

    # Most severe first, then most recent.
    display = filtered.copy()
    display["_rank"] = display["severity"].apply(severity_rank)
    display = display.sort_values(["_rank", "flow_id"], ascending=[False, False])
    display = display.head(DASHBOARD.max_table_rows)

    columns = ["flow_id", "src_ip", "src_port", "dst_ip", "dst_port",
               "protocol", "threat_class", "model_prediction", "confidence",
               "reliability", "severity", "needs_review"]
    if display["ground_truth"].notna().any():
        columns += ["ground_truth", "correct"]
    columns = [c for c in columns if c in display.columns]

    st.dataframe(
        display[columns],
        hide_index=True, height=340,
        column_config={
            "flow_id": st.column_config.TextColumn("Flow", width="small"),
            "src_ip": st.column_config.TextColumn("Source IP"),
            "src_port": st.column_config.NumberColumn("Sport", width="small",
                                                      format="%d"),
            "dst_ip": st.column_config.TextColumn("Destination IP"),
            "dst_port": st.column_config.NumberColumn("Dport", width="small",
                                                      format="%d"),
            "protocol": st.column_config.TextColumn("Proto", width="small"),
            "threat_class": st.column_config.TextColumn(
                "Reported", help="The verdict. UNKNOWN_OOD means the model's "
                                 "answer was not trustworthy for this input."),
            "model_prediction": st.column_config.TextColumn(
                "Model said", help="The classifier's raw answer, kept even when "
                                   "the verdict overrides it."),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.3f",
                help="The classifier's certainty. NOT a measure of whether the "
                     "answer is reliable."),
            "reliability": st.column_config.TextColumn(
                "Reliability", help="Whether this input resembles the training "
                                    "data."),
            "severity": st.column_config.TextColumn("Severity"),
            "needs_review": st.column_config.CheckboxColumn("Review"),
            "ground_truth": st.column_config.TextColumn("Label (demo)"),
            "correct": st.column_config.CheckboxColumn("Match"),
        },
    )
    shown = len(display)
    if shown < len(filtered):
        st.caption(f"Showing the {shown:,} most severe of {len(filtered):,} "
                   f"matching flows.")
    return filtered


def render_detail(frame: pd.DataFrame, records: list[dict]) -> None:
    """Per-alert detail: severity derivation, evidence, flow features."""
    st.subheader("Alert detail")
    if frame.empty:
        return

    by_id = {r.get("flow_id"): r for r in records}
    options = [fid for fid in frame["flow_id"].tolist() if fid in by_id]
    if not options:
        return

    def describe(flow_id: str) -> str:
        record = by_id[flow_id]
        mark = ""
        if record.get("reliability") == "UNRELIABLE":
            mark = "  [OOD]"
        elif record.get("reliability") == "DEGRADED":
            mark = "  [degraded]"
        return (f"{flow_id}  |  {record.get('threat_class')}  "
                f"{record.get('confidence', 0):.1%}  "
                f"{record.get('severity')}{mark}  |  "
                f"{record.get('src_ip')} -> {record.get('dst_ip')}:"
                f"{record.get('dst_port')}")

    chosen = st.selectbox("Select a flow", options, format_func=describe)
    alert = by_id[chosen]

    # An out-of-distribution flow gets its own banner above everything else:
    # reading the confidence figure without this context is the exact mistake
    # the OOD layer exists to prevent.
    if alert.get("reliability") == "UNRELIABLE":
        st.error(
            f"**Out of distribution.** The classifier said "
            f"**{alert.get('model_prediction')}** at "
            f"**{alert.get('confidence', 0):.1%}** confidence, but this input "
            f"lies outside the range seen during training, so that confidence "
            f"does not support the answer. Reported as "
            f"`{alert.get('threat_class')}` and routed to review."
        )
    elif alert.get("reliability") == "DEGRADED":
        st.warning(
            "**Reliability degraded.** One feature lies outside the training "
            "range; treat this classification with caution."
        )

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Classification**")
        st.metric(str(alert.get("threat_class")),
                  f"{alert.get('confidence', 0):.2%} confidence")
        if alert.get("model_prediction") and \
                alert.get("model_prediction") != alert.get("threat_class"):
            st.caption(f"Classifier's own answer: "
                       f"**{alert['model_prediction']}**. The verdict differs "
                       f"because the input is out of distribution.")
        if alert.get("confidence_means"):
            st.caption(alert["confidence_means"])

        severity = str(alert.get("severity"))
        colour = DASHBOARD.severity_colors.get(severity, "#888")
        reliability = str(alert.get("reliability", "UNKNOWN"))
        reliability_colour = DASHBOARD.reliability_colors.get(reliability, "#888")
        st.markdown(
            f"<div style='display:flex;gap:8px'>"
            f"<div style='padding:8px 12px;border-radius:6px;"
            f"background:{colour};color:white;font-weight:600'>{severity}</div>"
            f"<div style='padding:8px 12px;border-radius:6px;"
            f"background:{reliability_colour};color:white;font-weight:600'>"
            f"{reliability}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if alert.get("escalated"):
            st.caption(f"Escalated from {alert.get('base_severity')} "
                       f"because of the threat class.")
        st.caption(alert.get("severity_reason", ""))

        probabilities = alert.get("class_probabilities") or {}
        if probabilities:
            st.markdown("**Model output across all classes**")
            ordered = dict(sorted(probabilities.items(),
                                  key=lambda kv: -kv[1]))
            figure = go.Figure(go.Bar(
                x=list(ordered.values()), y=list(ordered.keys()),
                orientation="h",
                marker_color=[DASHBOARD.threat_colors.get(k, "#888")
                              for k in ordered],
                text=[f"{v:.3f}" for v in ordered.values()],
                textposition="auto",
            ))
            figure.update_layout(height=170,
                                 margin=dict(l=10, r=10, t=10, b=10),
                                 xaxis_range=[0, 1], showlegend=False,
                                 xaxis_title="probability")
            st.plotly_chart(figure)

        st.markdown("**Flow**")
        st.code(
            f"flow_id   : {alert.get('flow_id')}\n"
            f"observed  : {alert.get('timestamp')}\n"
            f"source    : {alert.get('src_ip')}:{alert.get('src_port')}\n"
            f"dest      : {alert.get('dst_ip')}:{alert.get('dst_port')}\n"
            f"protocol  : {alert.get('protocol')}\n"
            f"capture   : {alert.get('capture_timestamp')}\n"
            f"mode      : {alert.get('observation_mode')}",
            language="text",
        )

    with right:
        st.markdown("**Supporting evidence**")
        details = alert.get("evidence_detail") or []
        if details:
            for item in details:
                st.markdown(f"`{item.get('feature')}` = "
                            f"**{item.get('display')}**")
                if item.get("comparison"):
                    st.caption(f"{item['comparison']}")
                st.caption(item.get("why", ""))
                importance = item.get("importance")
                if importance:
                    st.caption(f"Global model importance: {importance:.4f}")
                st.markdown("")
        elif alert.get("evidence"):
            st.json(alert["evidence"])
        else:
            st.caption("No evidence generated for this flow.")

        if alert.get("evidence_method"):
            st.info(alert["evidence_method"])
        if alert.get("aggregate_context"):
            st.warning(f"**Aggregate context, not evaluated per-flow:** "
                       f"{alert['aggregate_context']}")

    if alert.get("ground_truth") is not None:
        match = alert.get("correct")
        judged = alert.get("model_prediction") or alert.get("threat_class")
        text = (f"Dataset label: **{alert['ground_truth']}** -- "
                f"{'matches' if match else 'does NOT match'} the classifier's "
                f"answer (**{judged}**).")
        (st.success if match else st.error)(
            text + "  \nCorrectness is judged against the classifier's answer, "
            "not the reported verdict: UNKNOWN_OOD never matches a dataset "
            "label. Ground truth exists only because this is a replay of a "
            "labelled dataset; production traffic has none."
        )

    with st.expander("Raw alert record (JSON)"):
        st.json(alert)


def render_footer(records: list[dict], status: dict) -> None:
    """Data-provenance and limitations panel."""
    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Data provenance**")
        source = status.get("source_file", "unknown")
        st.caption(
            f"Alerts read from `{PATHS.alerts_file.name}`, produced by "
            f"replaying `{source}`. Flow records originate from the "
            f"CIC-IDS2017 public dataset (prerecorded capture). No live or "
            f"malicious traffic is generated by this system."
        )
        if records:
            unknown = sum(1 for r in records if r.get("threat_class") == "UNKNOWN")
            if unknown:
                st.warning(f"{unknown:,} flows are UNKNOWN: no model was "
                           f"available when they were processed.")

    with right:
        st.markdown("**Known limitations**")
        st.caption(
            "Each flow is classified independently; there is no cross-flow "
            "correlation, so aggregate behaviour such as flood volume or scan "
            "breadth is not measured. Evidence is feature-based, not a "
            "per-prediction attribution method such as SHAP. Destination Port "
            "is the highest-importance feature, so the model partly reflects "
            "the port assignments of the source capture and would need "
            "retraining for another network."
        )
        st.caption(
            "**Reliability checking detects distribution shift, not novel "
            "attacks.** It flags traffic whose observed feature distribution "
            "differs from the training distribution. It does not guarantee "
            "detection of previously unseen attack labels when those attacks "
            "remain statistically similar to benign traffic -- measured, unseen "
            "DoS variants average 0.03 out-of-range features, the same as "
            "benign traffic."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    auto_refresh, interval = render_sidebar()

    st.title("AI-Based Detection of Cyber Threats in Unidirectional IP Traffic")

    # The operating posture is the problem statement's central constraint, so it
    # is a banner at the top of the page rather than a line of caption text.
    posture_left, posture_right = st.columns([1, 2])
    posture_left.success(f"### {POSTURE.mode}\n**{POSTURE.access}**")
    posture_right.info(f"**{POSTURE.assurance}**\n\n{POSTURE.detail}")

    status = read_status(PATHS.replay_status_file)
    records = read_alerts(PATHS.alerts_file, DASHBOARD.tail_lines)
    frame = to_frame(records)

    model = model_status()
    if not model["model_ready"]:
        st.error(
            f"**No trained model.** {model['error']}\n\n"
            f"The dashboard is functional, but flows cannot be classified. "
            f"Train the model with `python -m model.train`."
        )

    if not records and not status:
        st.info(
            "**Waiting for the alert stream.**\n\n"
            "Run these in order, in a separate terminal:\n\n"
            "```\n"
            "python -m preprocessing.clean_data\n"
            "python -m model.train\n"
            "python -m streaming.replay\n"
            "```"
        )

    render_metrics(frame, status)
    st.divider()
    render_reliability(frame, records)
    st.divider()
    render_distribution(frame)
    render_timeline(frame)
    st.divider()
    filtered = render_alert_table(frame)
    st.divider()
    render_detail(filtered, records)
    render_footer(records, status)

    st.caption(
        f"Last read {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC - "
        f"{len(records):,} alerts in view"
    )

    if auto_refresh and status.get("state") == "RUNNING":
        # Rerun only while a replay is active. Polling a finished stream would
        # burn CPU during a demo for no new data.
        import time as _time
        _time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()

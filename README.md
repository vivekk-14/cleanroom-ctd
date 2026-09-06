# AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

A passive machine-learning detector for one-way network traffic feeds. It
consumes flow records, classifies each flow, assigns a severity, and explains
its reasoning on a live dashboard.

**This MVP performs passive analysis and does not communicate back with the
monitored production network.** It never probes, connects, handshakes, injects
packets, blocks traffic, sends mitigation commands, or decrypts payloads. That
claim is enforced at runtime by `detection/passive_guard.py` and verified by 76
tests in `tests/test_passive_safety.py`.

---

## 1. Problem statement

Critical infrastructure networks may use passive traffic mirroring or a hardware
data diode to send a one-way copy of network traffic into a monitoring enclave.
The enclave can observe traffic but cannot communicate back to the production
network. Detection must therefore work entirely from passive observation.

The system must not send probes, initiate connections, perform handshakes, send
mitigation commands, block traffic, modify production traffic, or decrypt
encrypted payloads. It observes traffic, extracts metadata, detects suspicious
behaviour, classifies threats, calculates confidence and severity, and generates
alerts.

## 2. What this prototype does

```
CIC-IDS2017 CSV
      ↓
   clean  →  feature contract  →  RandomForest  →  distribution check
                                       ↓                   ↓
                              model_prediction      in_distribution?
                              model_confidence         reliability
                                       ↓                   ↓
                                    ╰──── severity ────╯
                                            ↓
                              alert queue  /  review queue
                                            ↓
                    dashboard  ←  alerts.jsonl  ←  alert schema v2
```

Four threat classes: `BENIGN`, `DDoS`, `PortScan`, `Botnet`, plus
`UNKNOWN_OOD` for input the model cannot be trusted on.

Measured on a held-out split of 37,987 flows the model never saw during
training:

| Metric | Value |
|---|---|
| Accuracy | **0.9987** |
| Balanced accuracy | **0.9921** |
| Macro F1 | **0.9888** |
| Weighted F1 | **0.9987** |
| Single-flow latency | **10.23 ms** (~98 flows/s, single-threaded) |
| Training time | **5.8 s** |

Those figures cover the four trained classes. On three *entirely unseen* capture
days the model held 0.9991–0.9999 accuracy with a 0.008–0.088% false-positive
rate — but it labelled 96–100% of **attack types it was never trained on** as
benign. Section 11 measures this in full; it is the most important caveat in this
document.

The more interesting property is not the accuracy figure. It is that the system
**separates how confident the classifier is from whether that confidence can be
trusted** for a given input. On a hand-written test file the model was 91.3%
confident and completely wrong; instead of reporting that as a finding, the
system reports `UNKNOWN_OOD` and routes the flow to review. Section 12 covers
this, including what it does and does not detect.

Every number in this README was produced by running the code in this repository.
Regenerate them all with `python -m model.train` followed by
`python -m model.evaluate`. Nothing here is estimated or copied from a paper.

---

## 3. Architecture

```
cyber-threat-detection/
  config.py                    all tunable settings in one place
  requirements.txt
  README.md
  .gitignore

  data/
    raw/                       CIC-IDS2017 CSVs (not committed)
    processed/
      flows_clean.csv          cleaned, labelled, capped
      flows_clean.meta.json    audit trail of what cleaning removed

  preprocessing/
    feature_config.py          canonical names, 2 feature profiles, label map
    inspect_columns.py         reports what is REALLY in the raw CSVs
    clean_data.py              multi-file cleaning; labels optional

  model/
    train.py                   fit, evaluate, save artefacts
    evaluate.py                reload artefacts, score held-out split
    score_dataset.py           score any external CSV, labelled or not
    predict.py                 predict_threat()  <-- the team interface
    profiles/
      strict_unidirectional/   18-feature model, its own contract
        model.pkl              RandomForest (0.91 MB)
        features.json          exact feature order the model expects
        label_encoder.pkl      class name <-> integer mapping
        metrics.json           full measured metrics
        ood_reference.json     per-feature training-range bounds
        novelty.pkl            advisory IsolationForest
      bidirectional/           14-feature model, same artefact set

  detection/
    severity.py                confidence -> severity, reliability first
    ood.py                     distribution check: is this input familiar?
    evidence.py                per-class explainable evidence
    alert_schema.py            the standard alert record (v2)
    passive_guard.py           runtime enforcement of read-only operation

  streaming/
    replay.py                  CSV -> prediction -> alerts.jsonl

  dashboard/
    app.py                     Streamlit + Plotly, reads alerts.jsonl

  runtime/
    alerts.jsonl               append-only alert stream
    replay_status.json         replay progress, for the dashboard

  reports/
    confusion_matrix.png       from model.evaluate --plot
    scored_*.csv               per-flow predictions from model.score_dataset
    scored_*.summary.json      machine-readable scoring summaries

  tests/
    fixtures/
      ood_synthetic_attacks.csv  IMMUTABLE out-of-distribution fixture
    test_prediction.py        102 tests: contract, severity, evidence, schema
    test_passive_safety.py     77 tests: static + runtime safety
    test_ood.py                56 tests: distribution check + the invariant
    test_profiles.py           40 tests: per-profile contracts and routing
    test_scoring.py            23 tests: external datasets, unlabelled input
```

Modules are independent. `dashboard/app.py` imports nothing from `model/train.py`.
The only coupling between the ML side and the dashboard side is
`predict_threat()` and the JSON Lines alert file.

### Why JSON Lines between replay and dashboard

The replay engine and the dashboard are separate OS processes. An append-only
file is the simplest transport that decouples them: no broker, no port, no
shared memory. A partially written final line is skipped by the reader and
picked up on the next refresh. The stream also survives the process, so a demo
can be reviewed after the fact.

---

## 4. Technology stack

Pinned in `requirements.txt` so both team members get identical behaviour.

| Package | Version | Used for |
|---|---|---|
| pandas | 3.0.0 | CSV loading, cleaning |
| numpy | 2.4.1 | numeric arrays |
| scikit-learn | 1.8.0 | RandomForestClassifier, metrics |
| joblib | 1.5.3 | model serialisation |
| streamlit | 1.63.0 | dashboard |
| plotly | 7.0.0 | charts |
| matplotlib | 3.11.0 | confusion-matrix PNG (evaluation only) |
| pytest | 9.1.1 | tests |

Verified on Python 3.14.2, Windows 11. No Docker, Kafka, Spark, GPU, or cloud
services. Everything runs locally on a student laptop.

**scapy is deliberately not a dependency.** The prototype reads flow records
from CSV. Not depending on a packet library removes the possibility of packet
transmission entirely, and a test asserts scapy stays out of
`requirements.txt`.

---

## 5. Dataset

**CIC-IDS2017**, `TrafficLabelling` variant (85 columns), from the Canadian
Institute for Cybersecurity. Three of the eight CSVs contain the four target
classes:

| File | Labels present | Rows |
|---|---|---|
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | DDoS 128,027 / BENIGN 97,718 | 225,745 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | PortScan 158,930 / BENIGN 127,537 | 286,467 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Bot 1,966 / BENIGN 189,067 | 191,033 |

The `TrafficLabelling` variant was chosen over `MachineLearningCVE` (79 columns)
because it includes Source IP, Destination IP, ports, protocol, and timestamp.
Alerts therefore carry real addresses instead of placeholders.

### Label mapping

The dataset says `Bot`; this project's class is `Botnet`. `LABEL_MAP` in
`preprocessing/feature_config.py` handles the rename and explicitly drops the 11
out-of-scope labels (`DoS Hulk`, `FTP-Patator`, `Heartbleed`, the `Web Attack`
variants, and others). Unmapped labels reduce the dataset with a printed count;
nothing is discarded silently.

### Five dataset quirks this code handles

All five were found by reading the actual files, not assumed.

**1. Column headers are inconsistently spaced.** 70 of 85 columns have a leading
space; 15 do not, with no pattern.

```
' Destination Port'              leading space
' Flow Duration'                 leading space
'Total Length of Fwd Packets'    NO leading space
'Flow Bytes/s'                   NO leading space
' Flow Packets/s'                leading space
```

Any code hardcoding these names is one typo from a `KeyError`. Every header is
normalised on load, and features are referenced by canonical name only.

**2. `Fwd Header Length` appears twice**, at header positions 40 and 61. pandas
renames the second to `Fwd Header Length.1`. Verified byte-identical to the
first across all 225,745 rows of the DDoS file, so the duplicate is dropped.

**3. One file is not UTF-8.** `Thursday-...-WebAttacks.csv` raises
`UnicodeDecodeError` and needs `latin-1`. It also contains 288,602 rows where
every cell is NaN, and its labels use a latin-1 en-dash
(`'Web Attack \x96 XSS'`). The three files used here are valid UTF-8, but the
fallback chain and empty-row handling are in place for when more files are
added.

**4. Infinity is present.** Rate columns divided by a zero-microsecond duration
produce `+inf`. Measured across the three files: 1,007 infinite cells, all in
`Flow Bytes/s` and `Flow Packets/s`.

**5. Flag columns are binary, not counts.** `SYN Flag Count` and
`ACK Flag Count` have `unique() == [0, 1]` — they are flag-present indicators.
This has consequences for evidence generation (section 15) and it is what the
out-of-distribution check catches when a hand-written file assigns a SYN count of
8,000 (section 12).

Run the inspector yourself:

```bash
python -m preprocessing.inspect_columns --full-scan
```

---

## 6. Installation

```bash
git clone <your-repo-url>
cd cyber-threat-detection

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

Download CIC-IDS2017 from
<https://www.unb.ca/cic/datasets/ids-2017.html>, then copy these three files
from the `TrafficLabelling` folder into `data/raw/`:

```
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
```

Confirm the setup:

```bash
python config.py
```

It prints every resolved path and marks each expected CSV `[ok]` or
`[MISSING]`. All paths are relative to the project root, so there are no
machine-specific absolute paths anywhere in the repository.

---

## 7. Running the pipeline

Four commands, in order.

### Step 1 — Clean the data

```bash
python -m preprocessing.clean_data
```

Measured output on the three-file subset:

```
combined                     : 703,245 rows
removed exact duplicates     : 88,084 (12.5%)
removed rows with NaN/inf    : 451
clean rows                   : 614,710

class distribution before capping
  BENIGN          394,057   64.10%
  DDoS            128,012   20.82%
  PortScan         90,694   14.75%
  Botnet            1,947    0.32%

per-class cap of 50,000
  BENIGN   394,057 -> 50,000
  DDoS     128,012 -> 50,000
  PortScan  90,694 -> 50,000
  Botnet     1,947 -> unchanged (below cap)

rows written : 151,947    output: 35.0 MB    elapsed: 25.8s
```

**Per-class caps, not a global row cap.** The spec suggested
`MAX_ROWS = 200000`. Botnet is 0.32% of this subset, so a global cap would
sample it almost out of existence and the model would never learn the class.
Capping each class independently bounds training time while preserving rare
classes. Configure with `DATASET.max_rows_per_class`, or override:

```bash
python -m preprocessing.clean_data --max-per-class 20000
python -m preprocessing.clean_data --no-cap
```

**De-duplication runs after concatenation**, comparing features and label but
excluding identity columns. Two flows with identical behaviour but different
ephemeral source ports are the same training example. Keeping both would let
identical records land in the train and test split, making reported accuracy
optimistic.

`flows_clean.meta.json` records every removal so the row count is auditable end
to end.

### Step 2 — Train

```bash
python -m model.train
```

Writes `model.pkl`, `features.json`, `label_encoder.pkl`, and `metrics.json`.
Each is written to a temp file then `os.replace`d, so an interrupted run cannot
leave a truncated `model.pkl` that loads without error but predicts garbage.

Compare the two feature profiles:

```bash
python -m model.train --compare
```

### Step 3 — Evaluate

```bash
python -m model.evaluate
python -m model.evaluate --plot --save-report
```

Reloads the saved artefacts from disk and scores them against the same held-out
split, reproduced from the same random seed. This validates what is actually on
disk, not an in-memory model that happened to work during training.

### Step 4 — Demo

Two terminals.

```bash
# terminal 1
streamlit run dashboard/app.py

# terminal 2
python -m streaming.replay
```

The dashboard opens at <http://localhost:8501> and updates as flows are
processed.

Useful replay options:

```bash
python -m streaming.replay --max-flows 200      # short demo
python -m streaming.replay --delay 0.05         # faster
python -m streaming.replay --delay 0 --quiet    # fast batch scoring
python -m streaming.replay --alerts-only        # skip BENIGN
```

**The replay engine never puts packets on a network.** It reads a local CSV and
appends to a local file. `passive_guard` is armed before any other project
import.

---

## 8. Feature profiles

Two profiles are selectable via `FEATURE_PROFILE` in `config.py`.

### STRICT_UNIDIRECTIONAL (default, 18 features)

Forward-direction features only:

```
Flow Duration                Fwd IAT Mean
Total Fwd Packets            Fwd IAT Std
Total Length of Fwd Packets  Fwd IAT Max
Fwd Packet Length Max        Fwd IAT Min
Fwd Packet Length Min        Fwd Header Length
Fwd Packet Length Mean       Fwd PSH Flags
Fwd Packet Length Std        act_data_pkt_fwd
Fwd Packets/s                min_seg_size_forward
Destination Port             Init_Win_bytes_forward
```

**Why this is the default.** A hardware data diode or a one-way SPAN copy
delivers traffic in a single direction. Reverse-path counters
(`Total Backward Packets`, `Bwd Packet Length Mean`, `Down/Up Ratio`) may be
unavailable or unreliable in that deployment. Training on them would produce a
model that cannot run in the environment the problem statement describes.

### BIDIRECTIONAL (14 features)

The feature list from the original specification, retained for comparison. It
includes backward features and both binary flag columns.

### Measured comparison

Identical protocol: same processed dataset, 75/25 stratified split,
`random_state=42`, 60 trees, depth 20.

| Profile | Features | Accuracy | Balanced acc | Macro F1 | Botnet F1 |
|---|---|---|---|---|---|
| **STRICT_UNIDIRECTIONAL** | 18 | **0.9987** | **0.9921** | **0.9888** | **0.9575** |
| BIDIRECTIONAL | 14 | 0.9962 | 0.9813 | 0.9672 | 0.8750 |

The honest threat model also scores better. Botnet F1 rises from 0.8750 to
0.9575, chiefly because forward packet-length statistics and TCP initial window
size separate C2 beacons from benign traffic more sharply than reverse-path
volume does.

Reproduce with `python -m model.train --compare`.

---

## 9. Model

`RandomForestClassifier`, 60 trees, max depth 20,
`class_weight='balanced_subsample'`, `random_state=42`.

Chosen from a measured sweep:

| Trees | Depth | Train | Macro F1 | Latency | Size |
|---|---|---|---|---|---|
| 100 | unlimited | 7.6 s | 0.9871 | 21.0 ms | 1.7 MB |
| **60** | **20** | **5.4 s** | **0.9890** | **13.8 ms** | **1.0 MB** |
| 40 | 16 | 3.7 s | 0.9888 | 10.9 ms | 0.6 MB |

60/20 is the best accuracy-per-millisecond point.

`class_weight='balanced_subsample'` matters here: Botnet is 1.28% of the capped
training set. Without reweighting, a model can score 98.7% accuracy while
ignoring the class entirely.

**No scaler is used.** Random forests split on per-feature thresholds and are
invariant to monotonic rescaling. A `StandardScaler` would add a moving part and
a failure mode without changing the decision boundary. `features.json` records
`requires_scaling: false` to make this explicit.

**Prediction is forced single-threaded.** Measured: for one row at a time,
thread dispatch overhead exceeds the work — 21 ms with `n_jobs=-1` against
12 ms with `n_jobs=1`.

---

## 10. Model evaluation

All figures below are from `python -m model.evaluate` on the 37,987-flow
held-out split, reproduced from `random_state=42`. None of these rows were seen
during fitting.

### Overall

```
accuracy              : 0.9987
balanced accuracy     : 0.9921    (mean per-class recall)
precision (macro)     : 0.9856
recall    (macro)     : 0.9921
F1        (macro)     : 0.9888
precision (weighted)  : 0.9987
recall    (weighted)  : 0.9987
F1        (weighted)  : 0.9987
```

Macro treats every class equally; weighted weights by support. With Botnet at
1.28% of rows, **macro is the honest headline number** — weighted metrics are
dominated by the three large classes.

### Per class

```
              precision    recall  f1-score   support

      BENIGN     0.9983    0.9976    0.9980     12500
      Botnet     0.9441    0.9713    0.9575       487
        DDoS     0.9999    0.9996    0.9998     12500
    PortScan     0.9999    0.9998    0.9999     12500

    accuracy                         0.9987     37987
   macro avg     0.9856    0.9921    0.9888     37987
weighted avg     0.9987    0.9987    0.9987     37987
```

### Confusion matrix

Rows are the true class, columns the prediction.

```
              BENIGN    Botnet      DDoS  PortScan     total
BENIGN        12,470        28         1         1    12,500
Botnet            14       473         0         0       487
DDoS               5         0    12,495         0    12,500
PortScan           2         0         0    12,498    12,500
```

51 misclassified of 37,987 (**0.134%**). The error modes:

- 28 BENIGN predicted as Botnet (0.22% of BENIGN) — false positives, the cost of
  weighting a 1.28% class up
- 14 Botnet predicted as BENIGN (2.87% of Botnet) — missed beacons, the more
  serious error
- 5 DDoS and 2 PortScan as BENIGN

Botnet is the weakest class in both directions. With 487 test examples against
12,500 for each other class, that is expected, and it is the honest limit of
what this dataset supports.

### Confidence calibration

Severity is derived from confidence, so confidence must track correctness for a
`CRITICAL` label to mean anything:

| Band | Range | Flows | Share | Accuracy |
|---|---|---|---|---|
| LOW | [0.00, 0.50) | 0 | 0.00% | n/a |
| MEDIUM | [0.50, 0.75) | 67 | 0.18% | 0.7761 |
| HIGH | [0.75, 0.90) | 145 | 0.38% | 0.9034 |
| CRITICAL | [0.90, 1.00] | 37,775 | 99.44% | 0.9994 |

Accuracy rises monotonically with confidence, which is what makes the severity
mapping defensible. Note that 99.44% of flows land in the top band — the model
is decisive on this dataset, which is itself a sign that the classes are
cleanly separable here and would likely be less so on production traffic.

### Measured performance

```
batch predict : 37,987 flows in 0.434s   (87,539 flows/s)
single flow   : 10.23 ms                 (~98 flows/s, single-threaded)
```

The replay engine classifies one flow at a time, so single-flow latency bounds
the demo. At 10.23 ms the pipeline is not prediction-bound: the 0.1 s replay
delay dominates by a factor of ten. Measured on this machine (Python 3.14.2,
Windows 11); it will differ on other hardware.

### Feature importance

```
0.1576  Destination Port
0.1389  Total Length of Fwd Packets
0.1356  Fwd Packet Length Max
0.1148  Fwd Packet Length Mean
0.0632  Init_Win_bytes_forward
0.0631  Flow Duration
0.0570  Fwd Packets/s
0.0467  Fwd Header Length
0.0362  Fwd IAT Max
0.0315  Fwd IAT Std
0.0284  Total Fwd Packets
0.0254  Fwd IAT Mean
0.0239  Fwd IAT Min
0.0220  Fwd Packet Length Min
0.0187  act_data_pkt_fwd
0.0187  Fwd Packet Length Std
0.0168  min_seg_size_forward
0.0014  Fwd PSH Flags
```

This is **global** importance across the training set. It is not a
per-prediction attribution and does not explain any individual alert.

---

## 11. Scoring an external dataset

Section 10 measures whether *training* worked, on a held-out split of the same
dataset. This section measures whether the model **generalises** — the question
that actually matters, and the one with the less flattering answer.

```bash
python -m model.score_dataset --input traffic.csv
python -m model.score_dataset --raw data/raw/SomeCapture.csv
python -m model.score_dataset --input traffic.csv --save-alerts --save-report
python -m model.score_dataset --input traffic.csv --max-flows 50000
```

Two modes, selected automatically:

**UNLABELLED** — no `threat_class` column. Reports what the model predicts, the
confidence distribution, and the severity breakdown. This is the real deployment
case: traffic arrives, the model classifies it, nobody knows the answers.

**LABELLED** — additionally reports accuracy, a confusion matrix, per-class
metrics, the benign false-positive rate, and how the model handles attack types
it was never trained on.

`--raw` cleans a raw CIC-IDS2017-format CSV first, so an unfamiliar file can be
scored in one command. Column-name normalisation means the inconsistent leading
spaces described in section 5 are handled automatically.

Outputs land in `reports/`: per-flow predictions as CSV, a machine-readable
summary as JSON, optionally a full text report, and with `--save-alerts` a JSONL
alert stream the dashboard can display.

### Generalisation on three unseen capture days

CIC-IDS2017 has eight daily captures. The model was trained on three. Scoring the
others measures performance on traffic from different days, different times, and
different attack campaigns.

To reproduce, copy the extra CSVs into `data/raw/` (they are not kept there by
default, to avoid an extra 500 MB in the repository):

```bash
python -m model.score_dataset --input data/raw/Monday-WorkingHours.pcap_ISCX.csv
python -m model.score_dataset --input data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
python -m model.score_dataset --input data/raw/Wednesday-workingHours.pcap_ISCX.csv
```

| Day | Flows scored | Known-class accuracy | Benign FP rate | Throughput |
|---|---|---|---|---|
| Monday | 529,918 | 0.9991 | 0.088% | 342,000 flows/s |
| Tuesday | 445,909 | 0.9999 | 0.008% | 328,000 flows/s |
| Wednesday | 692,703 | 0.9995 | 0.054% | 312,000 flows/s |

The false-positive rate holds up on genuinely unseen benign traffic, which is the
number that determines whether an analyst can live with the system. On Monday's
529,918 benign flows the model raised 465 alerts; their mean confidence was 0.63,
so most landed in MEDIUM rather than CRITICAL.

Batch scoring runs at roughly 320,000 flows/s because it vectorises across all
cores. The 10.23 ms single-flow figure in section 10 applies to the replay path,
which classifies one flow at a time by design.

### The result that matters: attack types never trained on

Tuesday and Wednesday contain seven attack types outside the model's four
classes. Measured miss rates:

```
true label            flows    flagged   missed as BENIGN
DoS Hulk            231,073       4.3%             95.7%
DoS GoldenEye        10,293       0.0%            100.0%
DoS Slowhttptest      5,499       1.0%             99.0%
DoS slowloris         5,796       0.1%             99.9%
Heartbleed               11       0.0%            100.0%
FTP-Patator           7,938       0.1%             99.9%
SSH-Patator           5,897       0.0%            100.0%
```

**96.1% of unseen-attack flows on Wednesday, and 99.9% on Tuesday, were labelled
BENIGN.**

This is structural, not a tuning problem. A four-class supervised classifier
answers "which of my classes is this closest to?", never "is this abnormal?".
Nothing in the training data taught it that slow-HTTP DoS or SSH brute-forcing
exists, so it confidently maps them to the nearest thing it knows, which is
ordinary traffic.

Two consequences the tool reports explicitly:

**Accuracy excludes unseen classes.** Folding them in would conflate "the model
was wrong" with "the model was never taught this", which are different failures
with different fixes. The reported 0.9995 on Wednesday covers the 440,031 flows
whose true class the model can actually predict.

**Predicted-BENIGN purity is reported separately.** Of the 682,526 Wednesday
flows the model called BENIGN, only **64.4%** genuinely were. That single number
is the honest summary of this limitation, and it is the one to quote rather than
the accuracy figure.

Fixing this requires either training on those classes — straightforward, see
section 23 — or adding an unsupervised anomaly detector alongside the classifier
to flag traffic that is merely *unusual*. The second is the more valuable
direction, because it covers attacks nobody has labelled yet.

Note what this does **not** describe. Section 12 adds a distribution check that
refuses to trust the model on unfamiliar input, but measured, it does not help
here: these unseen DoS flows sit squarely inside the training distribution on
every feature. They are novel by label, not by feature values.

### Unlabelled traffic

`preprocessing/clean_data.py` accepts CSVs with no label column, so genuinely
unlabelled traffic can be prepared and scored:

```bash
python -m preprocessing.clean_data --file unknown_traffic.csv --out data/processed/unknown.csv
python -m model.score_dataset --input data/processed/unknown.csv --save-alerts
```

Two behaviours worth knowing. No labels are ever invented: absent ground truth
stays absent, and the metadata records `labelled: false`. And the per-class row
cap is **skipped** rather than silently replaced by a global sample — capping is a
class-balancing operation, so without classes it would just discard most of the
traffic you asked to have scored.

---

## 12. Confidence is not reliability

This is the most important design decision in the project, and it came out of a
failure rather than a plan.

```
Model confidence answers:
    "How confident is the classifier in its prediction?"

Reliability answers:
    "Do we trust this prediction, for this input?"

These are deliberately separate fields.
```

### The failure that motivated it

A hand-written 20-row CSV (`tests/fixtures/ood_synthetic_attacks.csv`) was
scored against the BIDIRECTIONAL model. Five rows were labelled DDoS and carried
`SYN Flag Count` values of 4,200–8,000. In the real dataset that column is a
binary 0/1 flag-present indicator — the maximum possible value is 1.

The model classified **all 20 rows as BENIGN at 0.913 mean confidence**, five of
them at exactly 1.000. Worse, because BENIGN is suppressed to INFO by design,
every one of those rows would have been dropped from the analyst's queue
entirely.

A random forest cannot extrapolate. Every threshold it learned for
`SYN Flag Count` sits at 0.5, so a value of 8,000 falls on the same side of that
split as 1 does. The feature carries no information, the prediction rests on
whatever the remaining features happen to say, and the confidence figure reports
the model's internal certainty about that — not whether the answer is worth
anything.

### What the system does now

```
Traffic
  ↓
Feature validation      contract check: right features present?
  ↓
Random Forest           model_prediction + model_confidence
  ↓
Distribution check      does this input resemble the training data?
  ↓
┌──────────────────────────────────────────────────────┐
│ RELIABLE     0 violations   verdict = prediction     │
│ DEGRADED     1 violation    verdict kept, flagged    │
│ UNRELIABLE  ≥2 violations   verdict = UNKNOWN_OOD    │
└──────────────────────────────────────────────────────┘
  ↓
Severity  (reliability is checked BEFORE the benign rule)
  ↓
Alert queue  /  Review queue
```

Per-feature bounds come from the 0.1st and 99.9th percentiles of the **training
split only**. Using every row would leak held-out data into the reference and
make the false-positive rate below look better than it is.

The classifier's answer is never discarded — it moves to `model_prediction` so an
analyst can see what the model thought and why it was not trusted:

```json
{
  "threat_class":     "UNKNOWN_OOD",
  "confidence":       0.8167,
  "model_prediction": "BENIGN",
  "model_confidence": 0.8167,
  "confidence_means": "The trained classifier's probability for
                       'model_prediction'. It measures the model's internal
                       certainty, not the reliability of that answer.",
  "in_distribution":  false,
  "reliability":      "UNRELIABLE",
  "ood_score":        0.8929,
  "ood_features":     ["SYN Flag Count", "Total Fwd Packets",
                       "Total Length of Fwd Packets", "ACK Flag Count"],
  "severity":         "HIGH",
  "needs_review":     true
}
```

`in_distribution` is tri-state. `null` means no reference was available, which is
a different state from `false`, and `validate()` rejects an integer `0` in that
field for exactly that reason.

### Rule ordering, and why it matters

Reliability is checked **before** the BENIGN → INFO suppression. Reversing those
two rules is what made the fixture's rows invisible in the first place:

| Reported | Model said | Conf | Reliability | Severity | Alert | Review |
|---|---|---|---|---|---|---|
| UNKNOWN_OOD | BENIGN | 0.98 | UNRELIABLE | HIGH | yes | yes |
| UNKNOWN_OOD | DDoS | 0.95 | UNRELIABLE | HIGH | yes | yes |
| BENIGN | BENIGN | 0.98 | DEGRADED | INFO | no | **yes** |
| BENIGN | BENIGN | 0.98 | RELIABLE | INFO | no | no |
| DDoS | DDoS | 0.82 | DEGRADED | CRITICAL | yes | yes |
| DDoS | DDoS | 0.82 | RELIABLE | CRITICAL | yes | no |

Severity for an out-of-distribution flow is **HIGH, not CRITICAL**. An
untrustworthy answer warrants human review, but it is not evidence of an attack,
and ranking it above a confirmed high-confidence DDoS would invert the analyst's
priorities.

### Two queues, not one threshold

`is_alertable` and `needs_review` are independent. That came from measuring the
alternative rather than guessing:

| | ≥2 violations (chosen) | ≥1 violation |
|---|---|---|
| Fixture rows caught | 10 / 20 | 19 / 20 |
| Monday false flags | 4,694 | 9,316 |

Lowering the threshold to 1 would have doubled the false-positive workload on one
real capture day. Instead, DEGRADED flows keep their normal severity but set
`needs_review`, so they appear in a lower-priority queue. Result on the fixture:
**10 actionable alerts, 9 review-only, 1 genuinely in-distribution, 0 invisible.**

### Measured cost and benefit

Per-feature bounds at q=0.001, UNRELIABLE at ≥2 violations:

| Dataset | Flagged UNRELIABLE | What it should be |
|---|---|---|
| Held-out CIC-IDS2017 | **0.250%** | low — these are in-distribution |
| Monday BENIGN (unseen day) | **0.887%** | low |
| Synthetic OOD fixture | **50.0%** | high (95% reach DEGRADED) |

### The limitation, stated precisely

> OOD detection identifies traffic whose observed feature distribution differs
> from the training distribution. It does **not** guarantee detection of
> previously unseen attack labels when those attacks remain statistically similar
> to benign traffic.

This is measured, not hedged. The unseen DoS variants from Wednesday's capture
average **0.03 out-of-range features — identical to benign traffic.** The
distribution check flags 0.74% of them, against a 0.25% baseline: no meaningful
lift. A slow-HTTP DoS flow genuinely resembles an ordinary slow HTTP flow on
these 14 features, so there is nothing distributionally unusual to find.

Those attacks are novel by **label**, not by feature distribution. The fix is
training data for those classes, not a better OOD detector.

An IsolationForest fitted on benign traffic alone does reach 47.2% of them, but
at 4.75% false positives — roughly 25,000 review items on Monday's capture. It is
therefore fitted and its score reported, but it does **not** change the verdict
by default (`OOD.novelty_drives_verdict = False`).

**This system does not solve zero-day detection.** It covers a different and more
common failure: input arriving from a different feature-extraction pipeline, a
different network, or a different tool version — and it refuses to present a
confident answer on input it has never seen anything like.

---

## 13. Prediction engine — the team interface

`predict_threat()` is the single, frozen contract between the ML side and the
dashboard side.

```python
from model.predict import predict_threat

result = predict_threat(flow_features)   # dict: feature name -> value
```

Returns:

```python
{
    "threat":     "DDoS",         # str, one of the model's classes
    "confidence": 0.9612,         # float in [0, 1]
    "severity":   "CRITICAL",     # str
    "evidence":   {               # dict, 2-4 supporting feature values
        "Fwd Packet Length Mean": 7.0,
        "Flow Duration": 1876595.0,
        "Init_Win_bytes_forward": 256.0,
    },

    # severity derivation, so the dashboard can explain the label
    "base_severity":   "HIGH",
    "escalated":       True,
    "severity_reason": "Confidence 96% maps to CRITICAL; ...",
    "is_alertable":    True,

    # evidence detail
    "evidence_detail":   [{feature, value, display, why, comparison, importance}],
    "evidence_method":   "Feature-based evidence: ...",
    "aggregate_context": "The flood exists in the aggregate ...",

    # model output and provenance
    "class_probabilities": {"BENIGN": 0.0288, "DDoS": 0.9612, ...},
    "model_ready":         True,
    "missing_features":    [],
    "profile":            "STRICT_UNIDIRECTIONAL",
}
```

### Four guarantees

**1. It never raises.** A malformed flow returns a well-formed result with a
diagnostic field set. Verified against 18 hostile inputs including `None`,
`[]`, `"string"`, NaN, infinity, `1e300`, nested dicts, and wrong key names. A
replay loop must not die on one bad row.

**2. It works before the model exists.** With no `model.pkl` on disk it returns
`threat="UNKNOWN"`, `confidence=0.0`, `model_ready=False`. Student B can build
and test the entire dashboard before Student A finishes training. Nothing
blocks.

**3. Feature order is read from `features.json`.** The caller passes a plain dict
in any order; ordering is handled internally. The order can never drift out of
sync with the trained model. A test asserts that reversing the dict's key order
produces an identical prediction.

**4. Artefacts are cached.** Loading `model.pkl` per flow would add tens of
milliseconds to every prediction.

An additional safeguard: values above float32 range are rejected rather than
silently truncated. sklearn casts to float32 internally, where `1e300` becomes
`inf` and the prediction fails. The largest genuine value anywhere in the
dataset is about 2.1e9, so this only discards corrupt input — and it is reported
through `missing_features`.

---

## 14. Severity

Confidence answers "how sure is the model?". Severity answers "how much should
an analyst care?". They are different questions.

### Rule 1 — BENIGN is always INFO

Whatever the confidence. A high-confidence BENIGN classification is not an
alert. Ranking it alongside real detections would flood the analyst's queue with
rows saying "nothing is wrong", which is how real SOC dashboards become useless.

### Rule 2 — confidence maps to a base band

```
0.00 <= confidence < 0.50    LOW
0.50 <= confidence < 0.75    MEDIUM
0.75 <= confidence < 0.90    HIGH
confidence >= 0.90           CRITICAL
```

Bands are half-open, so every value in [0, 1] maps to exactly one band with no
gaps or overlaps. A test walks all 1,001 values from 0.000 to 1.000 to confirm.

### Rule 3 — threat class escalates one band

| Class | Escalated | Reasoning |
|---|---|---|
| DDoS | Yes | Direct availability impact. In critical infrastructure, loss of availability *is* the damage, so a merely probable DDoS still warrants attention. |
| Botnet | Yes | Implies a host inside the monitored network is already compromised and beaconing to external C2. The breach has happened. |
| PortScan | **No** | Reconnaissance. It precedes impact rather than causing it, and scans are frequent enough that escalating them would dilute the higher bands. |

Escalation is applied at most once and cannot exceed CRITICAL.

Worked examples:

```
threat       conf   base      final     alertable  escalated
PortScan     0.42   LOW       LOW       False      -
PortScan     0.82   HIGH      HIGH      True       -
DDoS         0.42   LOW       MEDIUM    True       yes
DDoS         0.82   HIGH      CRITICAL  True       yes
DDoS         0.96   CRITICAL  CRITICAL  True       -    (already at top)
Botnet       0.55   MEDIUM    HIGH      True       yes
BENIGN       0.99   INFO      INFO      False      -
```

Every alert records both its pre- and post-escalation severity plus a
human-readable `severity_reason`, so the dashboard shows the derivation rather
than an unexplained label. All thresholds live in `config.SEVERITY`; editing
them changes behaviour with no code change.

Inspect the live policy:

```bash
python -m detection.severity
```

---

## 15. Explainable evidence

For each classification, `detection/evidence.py` selects the feature values that
best justify it and states in plain language why each is suspicious.

**Scope claim, stated precisely.** This is feature-based evidence. It is not
SHAP, not LIME, not counterfactual explanation. Two honest sources are combined:
the model's global feature importances, and per-class reference statistics
measured from the processed dataset. It does **not** attribute an individual
prediction to individual features. Random-forest global importance is not a
per-prediction attribution, and the wording in every alert reflects that limit.

### A finding that contradicts the specification

The spec suggested DDoS evidence of "very high packet rate, high SYN count".
Measured median `Fwd Packets/s` per class:

| Class | Median Fwd Packets/s |
|---|---|
| PortScan | 20,000.0 |
| Botnet | 57.7 |
| BENIGN | 39.6 |
| **DDoS** | **1.7** |

**DDoS is the slowest class in this dataset, slower than benign traffic.**
CIC-IDS2017's DDoS is a distributed HTTP flood: each individual flow is slow and
ordinary-looking, and the attack exists in the *aggregate* — 49,998 flows from a
single source IP to a single destination. A per-flow "high packet rate" rule
would have flagged PortScan as the flood.

The evidence rules therefore use what actually separates the classes: forward
payload size, flow duration, TCP initial window size, and inter-arrival
regularity.

Similarly, `SYN Flag Count` cannot support the suggested rule. It is binary 0/1
in this dataset, not a count, and its model importance is 0.0014. The spec's
example alert value of `"SYN Flag Count": 17892` is not representable here. It is
not used as evidence, and the reason is documented in the code.

### What each class actually looks like

Measured medians from the processed dataset:

| Feature | BENIGN | DDoS | PortScan | Botnet |
|---|---|---|---|---|
| Fwd Packets/s | 39.6 | 1.7 | 20,000.0 | 57.7 |
| Fwd Packet Length Mean | 39.0 | 7.0 | 0.0 | 6.0 |
| Flow Duration (µs) | 48,699 | 1,876,595 | 50 | 71,053 |
| Init_Win_bytes_forward | 122 | 256 | 29,200 | 8,192 |
| Fwd IAT Std | 0.0 | 908,248 | 0.0 | 1,653 |
| act_data_pkt_fwd | 1.0 | 3.0 | 0.0 | 0.0 |
| Destination Port | 80 | 80 | 3,527 | 8,080 |

**PortScan** — zero payload, ~50 µs flows, one forward packet, TCP initial
window 29,200 (raw scanner socket), ports with no standard service.

**DDoS** — small uniform payload (median 7 bytes, std 1.2 across 50,000 flows),
flows held open ~1.9 s, high forward IAT variance, initial window 256.

**Botnet** — initial window 8,192 (consistent implant fingerprint), fixed high
port 8080, low IAT variability (periodic beaconing), almost no payload-carrying
packets.

### Two safeguards

**Contradiction guard.** Evidence is only asserted when the value actually
supports the claim. A DDoS-classified flow with a benign-typical duration gets
neutral wording — "recorded for context: this value is not unusual" — instead of
the false statement "held open 38x longer than benign".

**Categorical features get categorical comparisons.** "Port 3527 is 44x the
benign median of 80" is nonsense. `Destination Port` is described as
`port 3527 (registered range, no standard service)` instead, with well-known
service names where they apply, including ICS ports (Modbus 502, IEC-104 2404,
DNP3 20000).

### Aggregate context

Behaviour that requires correlation across many flows **cannot** be asserted
from a single flow record. Rather than implying detection it does not perform,
each alert carries an explicit note:

> Per-flow features alone understate this threat. In the source capture the DDoS
> consists of 49,998 flows from a single source IP to a single destination — the
> flood exists in the aggregate. This prototype classifies each flow
> independently and does not perform cross-flow correlation.

Similar notes cover scan breadth (999 distinct destination ports against one
host) and beacon periodicity (1,242 of 1,947 botnet flows to a single external
IP).

Reference statistics are hardcoded constants so evidence generation stays a
fast, pure function during replay — it must not re-read a 35 MB CSV per flow.
Regenerate them after changing the dataset:

```bash
python -m detection.evidence --recompute
```

---

## 16. Alert schema

One JSON object per line in `runtime/alerts.jsonl`. Schema version 1.

```json
{
  "schema_version": 1,
  "timestamp": "2026-09-05T17:57:05+00:00",
  "flow_id": "F000002",
  "source_flow_id": "172.16.0.1-192.168.10.50-46094-445-6",
  "src_ip": "172.16.0.1",
  "dst_ip": "192.168.10.50",
  "src_port": 46094,
  "dst_port": 445,
  "protocol": "TCP",
  "threat_class": "PortScan",
  "confidence": 1.0,
  "severity": "CRITICAL",
  "base_severity": "CRITICAL",
  "escalated": false,
  "severity_reason": "Confidence 100% maps to CRITICAL. PortScan is not escalated: it indicates reconnaissance rather than realised impact.",
  "is_alertable": true,
  "evidence": {
    "Fwd Packet Length Mean": 0.0,
    "Init_Win_bytes_forward": 29200.0,
    "Flow Duration": 1094.0,
    "Fwd Packets/s": 2742.230347
  },
  "evidence_detail": [
    {
      "feature": "Fwd Packet Length Mean",
      "value": 0.0,
      "display": "0 bytes",
      "why": "Probe packets carry little or no payload: the sender is testing whether a port answers, not exchanging data.",
      "comparison": "zero, against a benign median of 39 bytes",
      "importance": 0.1148
    }
  ],
  "evidence_method": "Feature-based evidence: measured per-class reference statistics combined with the model's global feature importances. This is not a per-prediction attribution method such as SHAP.",
  "aggregate_context": "Scan breadth (many destination ports per source, many hosts) is an aggregate property...",
  "class_probabilities": {"BENIGN": 0.0, "Botnet": 0.0, "DDoS": 0.0, "PortScan": 1.0},
  "capture_timestamp": "7/7/2017 2:54",
  "observation_mode": "PASSIVE_REPLAY",
  "model_ready": true,
  "ground_truth": "PortScan",
  "correct": true
}
```

This is a real record from a demo run, not a mock-up. IP addresses are genuine
CIC-IDS2017 values.

### Design notes

**`timestamp` vs `capture_timestamp`.** `timestamp` is when this system observed
the flow, generated at replay time in UTC with an explicit offset.
`capture_timestamp` is the original dataset string, kept verbatim and
deliberately **never parsed**: CIC-IDS2017 records `7/7/2017 3:30` in 12-hour
form with no AM/PM marker, so an afternoon capture at 13:00 is indistinguishable
from 01:00. Parsing would silently invent wrong times.

**`ground_truth` and `correct` are demo-only.** They exist because this replays
a *labelled* dataset, which lets the dashboard show live agreement. Production
traffic has no ground truth, so both fields would be `null`. They are labelled as
such in the dashboard rather than presented as a detection result.

**Every field is defensively coerced.** Confidence is clamped to [0, 1]. Ports
outside 0–65535 become 0. NaN IPs become `0.0.0.0`. A malformed row must not stop
a live replay.

**Truncated lines are skipped, not raised.** The dashboard may read the file
while the replay engine is mid-write, so the final line can be incomplete.
`parse_alert_line()` returns `None` and the reader picks it up next refresh.

---

## 17. Dashboard

```bash
streamlit run dashboard/app.py
```

**Posture banner** — the first thing on the page, not a footnote:
`PASSIVE MONITORING`, `READ ONLY`, and
**"No outbound network actions are performed."**

**Four metric tiles** — deliberately four, not nine, so the state of a run is
readable in a few seconds:

```
┌──────────────────────────────────────────────────────────┐
│  Flows observed    Alerts     Review    Out of distribution│
│        60            35         11             10          │
└──────────────────────────────────────────────────────────┘
```

Review excludes alerts, so the two tiles never double-count the same flow.

**Reliability panel** — sits directly under the tiles, because it is what
distinguishes this from a bare classifier. It opens by stating the distinction in
one line, then shows the tier counts and the most recent OOD event in full:

```
RELIABLE     input resembles training data              1    1.7%
DEGRADED     one feature outside training range        49   81.7%
UNRELIABLE   outside training distribution             10   16.7%

  For those 10 flows the classifier said BENIGN 10 at 93.2% mean
  confidence. High confidence, untrustworthy answer.

Most recent out-of-distribution event
  Reported:      UNKNOWN_OOD
  Model said:    BENIGN  at 95.0% confidence
  Reliability:   UNRELIABLE
  OOD score:     0.8571
  Needs review:  YES
  Flow:          172.16.0.1:50009 -> 192.168.10.50:443

Features outside the training range
  SYN Flag Count = 5,500  (above the training range [0, 1],
                           5,499.0x the range width beyond it)
  Total Fwd Packets = 6,000  (above [1, 155], 37.9x beyond)
  Total Length of Fwd Packets = 4.2e+05  (above [0, 1.99e+04], 20.1x beyond)
```

Every one of those numbers is read from the alert record; none is hardcoded.

**Threat distribution** — Plotly bar chart across BENIGN / DDoS / PortScan /
Botnet / UNKNOWN_OOD with fixed colours so a class keeps its colour between
refreshes, plus a severity donut. `UNKNOWN_OOD` is deliberately blue-grey rather
than a warm alert colour: it is the *absence* of a trustworthy answer, not a
fifth threat class.

**Observation timeline** — confidence per flow over the observation sequence,
coloured by class, with severity band boundaries drawn as reference lines so the
chart shows *why* each flow got its label.

**Alert and review queues** — one table with a queue selector (Alerts / Review /
Both / All flows). Columns include both `Reported` and `Model said`, so a
divergence between the verdict and the classifier's answer is visible at a
glance, alongside confidence, reliability, severity, and a review checkbox.

**Alert detail** — select any flow to see the classification, the full
probability distribution across all classes, the severity derivation, every
evidence item with its comparison and reasoning, and the raw JSON. An
out-of-distribution flow gets a red banner *above* everything else stating what
the model said, how confident it was, and why that confidence does not support
the answer — reading the confidence figure without that context is the exact
mistake this layer exists to prevent.

The dashboard is strictly a reader. It never writes to the alert stream, never
triggers a replay, and **offers no blocking, mitigation, or response controls** —
deliberately, because the monitoring enclave cannot communicate back to the
production network. A test asserts no such controls exist in the source.

It also degrades honestly. Verified via Streamlit's `AppTest` harness in four
states, with **zero exceptions in each**: populated stream, empty stream, missing
model, and a legacy schema-v1 stream with no OOD fields at all (those flows
default to `reliability: UNKNOWN` rather than being silently treated as
trustworthy).

---

## 18. Safety and passive monitoring

The project's central claim is that it observes and never acts. That claim is
enforced, not merely asserted.

### Runtime enforcement

`detection/passive_guard.py` wraps the outbound operations of Python's `socket`
module. One rule:

> **An operation is permitted only if its address is loopback.**

Applied uniformly to `connect`, `connect_ex`, `bind`, `listen`, `accept`, `send`,
`sendall`, `sendto`, `sendmsg`, and `sendfile`. The address checked is whichever
one the operation actually uses — the argument for `connect`/`bind`/`sendto`, the
socket's own bound address for `listen`/`accept`, the connected peer for
`send`/`sendall`. Also blocked: `create_connection` to remote hosts,
`AF_PACKET`/`AF_LINK` raw sockets (the mechanism for packet injection), and
`getaddrinfo`/`gethostbyname` on real hostnames — resolving a hostname emits a
DNS query, which is an outbound action even though no application socket opens.
IP literals and `localhost` still resolve.

Any violation raises `PassiveModeViolation` immediately. A warning-and-continue
design would let the packet be sent, which is precisely what must not happen.

The guard is armed as the **first** action in `streaming/replay.py` and
`dashboard/app.py`, before any other project import, so nothing an imported
module does at import time can open a socket. A test asserts the arming call
precedes `import streamlit`.

### Why loopback is permitted

Streamlit is an HTTP server on localhost and asyncio needs a socketpair
self-pipe. Loopback traffic never reaches a network interface, so it cannot reach
the monitored network. Every loopback allowance is counted and displayed on the
dashboard, so the exemption is visible rather than hidden.

This distinction was found by testing, not assumed. An earlier version blocked
`listen()` outright, which made asyncio — and therefore Streamlit — impossible to
start, while adding no safety, because the socket in question never leaves the
host.

### Static enforcement

`tests/test_passive_safety.py` greps every project source file for forbidden
capabilities, so something pasted in later is caught too:

- packet injection (`sendp`, `srp`, `sr1`, scapy transmission)
- live capture and raw sockets (`sniff`, `AsyncSniffer`, `SOCK_RAW`, `AF_PACKET`)
- firewall and routing manipulation (`iptables`, `netsh advfirewall`, `pfctl`,
  `ip route del`)
- offensive tooling (`nmap`, `masscan`, `hydra`, `metasploit`, `sqlmap`, `hping`)
- shell execution (`subprocess`, `os.system`, `os.popen`, `pty.spawn`)
- outbound client libraries (`requests`, `urllib.request`, `httpx`, `ftplib`,
  `smtplib`, `paramiko`)
- payload decryption (`Cipher`, `AES.new`, `load_pem_private_key`,
  `SSLKEYLOGFILE`)

The offensive-tooling scan matches **code identifiers**, not raw text. This
matters: `DoS slowloris` is a legitimate CIC-IDS2017 label that `LABEL_MAP` must
list in order to drop it, and a naive substring search also matches `nmap`
inside `unmapped`. Naming a label is not the same as running a tool.

**No subprocess anywhere.** The guard patches sockets inside this interpreter
only; a subprocess would be outside its reach. So the project spawns none, and a
test enforces that.

### Limits of the guarantee, stated plainly

This is Python-level enforcement within one interpreter. It cannot stop a
subprocess, a C extension calling `send()` directly, or a compromised
interpreter. **Real deployment assurance comes from the hardware data diode,
which is a physical one-way path.** This guard makes the software's intent
enforceable and testable; it does not replace the diode.

Verify for yourself:

```bash
python -m detection.passive_guard        # 13-check self-test
python -m pytest tests/test_passive_safety.py -v
```

---

## 19. Tests

```bash
python -m pytest tests/ -q
```

```
295 passed, 3 skipped in 102.54s
```

The three skips are `AF_PACKET` (Linux-only) and two unseen-class tests that need
`Wednesday-workingHours.pcap_ISCX.csv` in `data/raw/`. Copy that file in and the
count becomes **297 passed, 1 skipped**. On a fresh clone before training,
model-dependent tests also skip rather than fail.

| File | Tests | Covers |
|---|---|---|
| `test_prediction.py` | 102 | column normalisation, label mapping, feature profiles, severity bands and escalation, evidence, the `predict_threat` contract, alert schema, integration |
| `test_passive_safety.py` | 77 | static code scan, guard arming, outbound blocking, loopback allowance, replay/dashboard safety, posture documentation |
| `test_ood.py` | 56 | the OOD reference artefact, all four verdict tiers, severity Rule 0 and its ordering, the UNRELIABLE↔UNKNOWN_OOD invariant, fixture integrity |
| `test_profiles.py` | 40 | per-profile artefacts, feature-contract enforcement, profile routing, cross-profile refusal |
| `test_scoring.py` | 23 | cleaning without labels, both scoring modes, unseen-class reporting, alert output, scorer safety |

Tests requiring a trained model are **skipped, not failed**, when `model.pkl` is
absent, so a fresh clone is green before training.

Some tests worth knowing about:

- `test_feature_order_is_respected` — reversing the input dict's key order must
  produce an identical prediction. Catches feature-ordering bugs, which would
  otherwise yield plausible but wrong answers.
- `test_identity_fields_do_not_change_the_prediction` — changing source or
  destination IP must not move the confidence by any amount.
- `test_strict_profile_excludes_backward_features` — asserts no `Bwd`,
  `Backward`, or `Down/Up` feature can enter the unidirectional profile.
- `test_bands_have_no_gaps` — walks all 1,001 confidence values from 0.000 to
  1.000.
- `test_contradictory_values_are_not_asserted` — a benign-looking value under an
  attack classification must not be described as attack-like.
- `test_method_does_not_overclaim` — the evidence method string must state that
  it is not SHAP.
- `test_unseen_classes_are_reported_not_hidden` — attack types outside the
  model's classes must be counted and reported, never silently dropped or folded
  into the accuracy figure.
- `test_cap_is_skipped_without_labels` — a per-class cap must not be applied to
  unlabelled data, where it would discard most of the traffic.
- `test_ood_benign_is_NOT_suppressed_to_info` — the single most important test in
  the suite. A confident BENIGN on out-of-distribution input must reach the
  analyst rather than being suppressed to INFO.
- `test_unreliable_without_ood_class_is_rejected` and its reverse — the
  UNRELIABLE ↔ UNKNOWN_OOD invariant, checked in both directions so a partial
  change to the OOD layer cannot report a trusted-looking class.
- `test_syn_flag_count_is_impossible_by_design` — asserts the OOD fixture still
  contains values that cannot occur in the real dataset. It exists to stop a
  well-meaning future edit from "fixing" the fixture until the model passes.
- `test_false_ood_rate_on_real_data` — pins the cost side of the tradeoff so the
  false-positive rate cannot creep upward unnoticed.

### Verified by mutation, not just by passing

A test that always passes is worse than no test. Two deliberate breakages were
introduced to confirm the suite actually detects them:

| Mutation | Result |
|---|---|
| Disable severity Rule 0 (`if reliability == "UNRELIABLE"` → `if False`) | **9 tests fail** |
| Disable the invariant check in `validate()` | **1 test fails** |
| Canary file with `subprocess.run`, `requests.get`, `os.system`, `sendp` | **4 tests fail** |

Each mutation was reverted immediately and the suite returned to green.

### A note on the static safety scan

The forbidden-code scans in `test_passive_safety.py` tokenise each file and blank
string literals and comments before matching, because this project legitimately
*discusses* the things it must not *do*: `.send(` appears inside a test asserting
nobody calls it, and `DoS slowloris` is a real CIC-IDS2017 label that `LABEL_MAP`
lists in order to drop it.

An earlier version of that transformation joined tokens with spaces, which turned
`sendp(None)` into `sendp ( None )` and silently made every call-syntax pattern
unmatchable — a false negative in a safety test, which is worse than the false
positive it was fixing. The scans were re-verified against a deliberately
malicious canary file containing `subprocess.run`, `requests.get`, `os.system`,
and `sendp`; all four were caught.

---

## 20. Team split

Two students, one frozen interface.

### Student A — ML and detection

`preprocessing/`, `model/`, `detection/`

Data cleaning, feature selection, model training and evaluation, the prediction
function, confidence, severity, and evidence logic.

### Student B — streaming and dashboard

`streaming/`, `dashboard/`

CSV replay, the streaming pipeline, alert display, Streamlit dashboard,
visualisations, integration.

### The interface

Student B imports exactly one thing from the ML side:

```python
from model.predict import predict_threat

result = predict_threat(flow_features)
```

Student B never touches `features.json`, never loads `model.pkl`, and never
imports anything from `model/train.py`. Feature ordering, missing-value filling,
and type coercion are all handled inside `predict_threat()`.

**Neither student blocks the other.** Before any model exists, `predict_threat()`
returns `model_ready: False` with `threat="UNKNOWN"` instead of crashing, so the
entire dashboard can be built and tested first. In the other direction, the ML
side can be developed and evaluated with no dashboard running at all.

The transport is equally decoupled: the replay engine appends to
`runtime/alerts.jsonl`; the dashboard tails it. Neither process imports the
other.

---

## 21. Demo script

### The pipeline being demonstrated

```
Replay CSV
    ↓
Feature contract validation
    ↓
Profile selection
    ↓
ML classification            model_prediction + model_confidence
    ↓
Distribution assessment      in_distribution + reliability
    ↓
Severity                     reliability checked before class
    ↓
Alert queue  /  Review queue
    ↓
Dashboard
```

### Part 1 — normal operation

```bash
# terminal 1
streamlit run dashboard/app.py

# terminal 2
python -m streaming.replay --max-flows 300 --delay 0.1
```

1. The dashboard opens with `PASSIVE MONITORING / READ ONLY` and
   "No outbound network actions are performed."
2. Replay starts. Flow records stream from the processed CSV.
3. Each flow is classified, then checked against the training distribution.
4. Alerts appear in the table, most severe first.
5. Confidence, reliability, and severity are shown per alert with the derivation.
6. Selecting an alert reveals the evidence features and why each is suspicious.
7. The threat distribution and reliability panels update live.

A real 300-flow run:

```
flows processed : 300 of 300
elapsed         : 35.0s (8.6 flows/s)
                  (rate is set by the 0.1s replay delay, not by model speed)

classified as:
  DDoS           101  (33.67%)
  PortScan        98  (32.67%)
  BENIGN          96  (32.00%)
  Botnet           5  ( 1.67%)

severity:
  CRITICAL       204
  INFO            96
  actionable     204  (MEDIUM and above)

agreement with dataset labels: 100.00% (300/300)

outbound network actions: 0 attempted, 0 performed
```

### Part 2 — the part worth showing

Score the deliberately out-of-distribution fixture:

```bash
python -m model.score_dataset \
    --input tests/fixtures/ood_synthetic_attacks.csv \
    --profile BIDIRECTIONAL
```

```
WHAT THE MODEL PREDICTED
  BENIGN    20  100.00%   mean conf 0.9133

CAN THOSE ANSWERS BE TRUSTED?
  RELIABLE       1    5.00%
  DEGRADED       9   45.00%
  UNRELIABLE    10   50.00%

  10 flows (50.00%) are outside the training distribution.
  What the classifier had said for those flows: BENIGN 10
  Their mean classifier confidence: 0.932  <- high confidence,
                                              unreliable answer

  features most often outside their training range:
    ACK Flag Count                 11 flows (55.00%)
    Total Length of Fwd Packets    10 flows (50.00%)
    SYN Flag Count                 10 flows (50.00%)

  reported verdict after the distribution check:
    UNKNOWN_OOD    10   50.00%
    BENIGN         10   50.00%

  actionable  10   review  9
```

The classifier was 91.3% confident and 100% wrong. The system does not report
that as a finding.

### What to say

The strongest claim here is not the accuracy figure. It is this:

> When the model encounters traffic outside what it learned, it does not blindly
> trust its own prediction. It separates classifier confidence from prediction
> reliability, and sends unfamiliar observations to review rather than silently
> passing them as benign.

And immediately after, the honest boundary:

> This detects distribution shift. It does not detect novel attacks that look
> statistically ordinary — measured, those are missed, and the fix is training
> data, not a better detector.

Replay shuffles on every run, so the class mix varies between demos. Botnet stays
scarce because it is 1.28% of the dataset — use `--alerts-only` or a larger
`--max-flows` if you want to be sure of showing one.

**No real malicious traffic is required or generated.** Everything replays a
prerecorded public capture.

For a fast run, use `--delay 0`. At the measured 10.23 ms per flow, the full
151,947-flow dataset takes roughly 26 minutes single-threaded, so bound it with
`--max-flows` for a demo.

---

## 22. Limitations

Stated plainly, because a prototype that overstates itself is worse than one that
does not.

**1. Each flow is classified independently.** There is no cross-flow
correlation. Aggregate behaviour — flood volume, scan breadth, beacon
periodicity — is not measured, even though those are the properties that most
strongly identify these attacks. Alerts carry an explicit `aggregate_context`
note saying so.

**2. `Destination Port` is the top feature at 0.1576 importance.** Removing it
collapses Botnet F1 from 0.9575 to **0.2175**, because CIC-IDS2017's botnet C2
sits on port 8080 in 1,261 of 1,966 flows. The model partly memorises the lab's
port assignments. A different network would need retraining.

**3. CIC-IDS2017 is a 2017 lab capture, not production traffic.** Attack
techniques have moved on, the class balance is artificial, and benign traffic
comes from a small simulated user population. High scores here do not transfer
directly to a real network.

**4. Forward-only features are an approximation.** CIC-IDS2017 was generated from
*bidirectional* captures. Its `Fwd` features are what a forward-direction
observer would compute, which is the closest honest approximation this dataset
permits — but it is not the same as flow records genuinely derived from a
one-way tap.

**5. Botnet has only 1,947 examples** (487 in the test split). Its F1 of 0.9575
rests on a small sample, and 2.87% of botnet flows are missed. That per-class
figure is the least reliable number in this README.

**6. Evidence is feature-based, not per-prediction attribution.** Global
importance plus measured class statistics. Not SHAP, not LIME.

**7. The passive guard is Python-level.** It does not constrain subprocesses or C
extensions. Deployment assurance requires the hardware diode.

**8. Only four threat classes — and unseen attacks are missed, not flagged.**
DGA, DNS tunnelling, encrypted malware, and data exfiltration are out of scope.
This is not merely a coverage gap: **measured**, the model labels 96–100% of
attack types it was never trained on as BENIGN (section 11). Of everything it
called BENIGN on Wednesday's capture, only 64.4% genuinely was. A supervised
classifier cannot answer "is this abnormal?", only "which of my classes is this
closest to?". If a judge asks what happens with a zero-day, the honest answer is
that it gets missed.

**8a. The distribution check does not fix limitation 8.** The reliability layer in
section 12 catches input whose feature distribution differs from training — a
hand-written file, a different extraction pipeline, another network. It does not
catch novel attacks that look statistically ordinary: the unseen DoS variants
average 0.03 out-of-range features, the same as benign traffic. Detection rate on
them is 0.74% against a 0.25% baseline. These are two different failure modes, and
only the first is addressed.

**9. Confidence is decisive to the point of being suspicious.** 99.44% of
held-out flows land in the CRITICAL band. On this dataset the classes are cleanly
separable; production traffic would produce a much flatter distribution, and the
MEDIUM/HIGH bands would then carry real weight. This is also why `confidence` and
`reliability` are separate fields — see section 12.

**10. No temporal validation.** The train/test split is random, not
chronological. Flows from the same attack burst can appear in both splits, which
inflates scores relative to a time-ordered evaluation. Section 11's cross-day
scoring partly compensates, since those captures are from different days
entirely.

**11. OOD bounds are per-feature, not joint.** A flow whose individual feature
values are each plausible but whose *combination* never occurs in training will
pass the check. Detecting that needs a density or distance model over the joint
distribution; the advisory IsolationForest is a partial step, but it is not used
to drive the verdict.

---

## 23. Future improvements

Roughly in order of value per unit of effort.

**Anomaly detection alongside classification.** Promoted to first place by the
section 11 results. A supervised classifier recognises only its four trained
classes and, measured, waves 96–100% of everything else through as BENIGN. An
unsupervised model (IsolationForest, autoencoder) trained on benign traffic alone
would flag flows that are merely *unusual*, covering the DoS variants, brute-force
attacks, and genuine zero-days the classifier cannot see. This is the single
change that would most improve real-world usefulness.

**Cross-flow correlation.** A sliding window keyed on source IP would measure
flows-per-second per source, distinct destination ports per source, and beacon
interval regularity — the features that actually characterise floods, scans, and
C2. Directly addresses limitations 1 and 2.

**Extend to the remaining threat classes.** The architecture already supports it:
add entries to `LABEL_MAP`, add features to a profile, retrain. No other module
changes, because `train.py` derives its class list from the data. Adding the DoS
variants and Patator attacks would immediately fix most of the unseen-class misses,
and the data is already on disk. DNS tunnelling and DGA need per-query features
(name length, entropy, subdomain count) that flow records do not carry, so they
would need a DNS-specific feature extractor.

**Chronological validation.** Split on `capture_timestamp` instead of randomly,
to get an honest estimate of performance on future traffic.

**Second dataset.** Validating on CSE-CIC-IDS2018 or UNSW-NB15 would show whether
the model generalises past one lab. `model/score_dataset.py` already accepts an
arbitrary CSV, so this is mostly a matter of obtaining the data and mapping its
column names.

**Drop or de-weight `Destination Port`.** Forces the model to learn behaviour
rather than lab topology. Expect materially lower scores and a genuinely more
transferable model.

**True per-prediction explanation.** SHAP on a random forest is tractable at this
scale and would upgrade the evidence from "these features matter globally" to
"these features drove *this* decision".

**Real PCAP ingestion.** Offline `scapy` reading of a capture file, computing
forward-direction flow features directly, would remove the dependency on
pre-computed CIC-IDS2017 CSVs. Must remain strictly offline file reading — no
live capture, no transmission.

---

## 24. Reference

### Commands

```bash
python config.py                                # verify setup
python -m preprocessing.inspect_columns         # inspect raw CSVs
python -m preprocessing.inspect_columns --full-scan
python -m preprocessing.clean_data              # clean
python -m preprocessing.clean_data --max-per-class 20000
python -m model.train                           # train
python -m model.train --compare                 # compare profiles
python -m model.evaluate --plot --save-report   # evaluate held-out split
python -m model.score_dataset --input FILE.csv  # score an external dataset
python -m model.score_dataset --raw FILE.csv --save-alerts
python -m model.predict                         # prediction self-test
python -m detection.severity                    # severity policy
python -m detection.ood                         # distribution-check policy
python -m detection.evidence                    # evidence examples
python -m detection.alert_schema                # schema demo
python -m detection.passive_guard               # safety self-test
python -m streaming.replay                      # replay
streamlit run dashboard/app.py                  # dashboard
python -m pytest tests/ -q                      # all tests
```

### Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `Processed dataset not found` | Cleaning not run | `python -m preprocessing.clean_data` |
| `model artefacts not found: model.pkl` | Not trained | `python -m model.train` |
| `[MISSING]` in `python config.py` | CSVs absent | Copy the three CSVs into `data/raw/` |
| `UnicodeDecodeError` | Non-UTF-8 CSV | Handled automatically; `latin-1` fallback |
| `KeyError: ' Flow Duration'` | Un-normalised header | Use canonical names; `normalize_column()` strips whitespace |
| `PassiveModeViolation` | Code tried to reach the network | Working as designed — remove the outbound call |
| Dashboard shows no alerts | Replay not started | `python -m streaming.replay` in a second terminal |
| `artefact mismatch: model expects N features` | Artefacts from different runs | `python -m model.train` |
| Botnet F1 much lower than reported | Per-class cap disabled or too small | Keep `max_rows_per_class` at 50,000 |
| `N required features missing` when scoring | CSV is not in CIC-IDS2017 flow format | Check the column list the error prints; clean the file first |
| Scoring reports everything as BENIGN | The dataset's attacks are outside the four trained classes | Expected; see section 11 |

### Environment these results were measured on

```
Python        3.14.2
scikit-learn  1.8.0
pandas        3.0.0
numpy         2.4.1
platform      Windows-11-10.0.26200-SP0
```

### Dataset citation

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating
a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th
International Conference on Information Systems Security and Privacy (ICISSP),
Portugal, January 2018.

---

## Statement of scope

This MVP performs passive analysis and does not communicate back with the
monitored production network. It does not probe, connect, handshake, inject
packets, block traffic, send mitigation commands, modify production traffic, or
decrypt payloads. It is a defensive monitoring prototype built for a hackathon,
evaluated on a public prerecorded dataset, and it is not production software.

On detection specifically: the system classifies flows into four trained classes
and reports when an input lies outside the distribution it was trained on. It
does not claim to detect unknown or zero-day attacks. Distribution checking
identifies traffic whose observed feature values differ from training; it does
not guarantee detection of previously unseen attack labels when those attacks
remain statistically similar to benign traffic. Both properties are measured in
sections 11 and 12, including where they fail.

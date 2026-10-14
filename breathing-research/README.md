# Breathing research with ESPectre motion scores

Record ESPectre's processed High Accuracy motion scores and inspect their periodicity offline. The tools use Python's standard library and make no device configuration changes. They do not collect raw CSI or modify the occupancy FSM.

Periodicity measurements are exploratory: they are not breathing classifications or confidence probabilities. Compare candidate rhythms with an independent breathing reference and matched empty-room recordings.

## Connect the device

1. Connect the computer to the same reachable network as the sensing ESP32-S3. If another ESP32 provides a SoftAP, connect both the computer and the sensing S3 to that SoftAP.
2. Keep the current ESPectre firmware, select High Accuracy in the device panel, and wait until sensing is ready.
3. Find the S3's Direct API address through **Find devices** in the web panel or `./espectre devices --chip s3 --json` from the `espectre/` directory, if its CLI dependencies are installed.

Use the complete address, including `http://`, the port, and `/espectre/v1`. The example below uses `http://192.168.4.2:62587/espectre/v1`; replace it with your S3's address, not the SoftAP gateway address. The firmware must advertise motion events through its HTTP event stream.

## Record scores

Run from the repository root:

```bash
python3 breathing-research/score_research.py record \
  --endpoint http://192.168.4.2:62587/espectre/v1 \
  --output breathing-research/data/sitting-01.jsonl \
  --participant p01 --session-id day01 \
  --posture sitting --condition quiet --position desk \
  --duration 300
```

This records five minutes of events with device metadata, device timestamps, receive timestamps, and experiment labels. Use anonymous participant IDs. Use the same session ID for related conditions and nodes collected in one experiment session. Record reference recording identifiers, orientation, and disturbances with `--notes`.

Supported postures are `empty`, `standing`, `sitting`, `lying_back`, `lying_left`, `lying_right`, and `other`. Conditions are `quiet`, `empty`, `fan`, `typing`, `movement`, and `other`. For an empty-room control, use `--participant none --posture empty --condition empty` and a new output filename. Record matched standing, sitting, and lying conditions at the same positions, and repeat on another day.

Existing output files are never overwritten. Choose a new filename for each recording. A disconnected stream leaves a partial recording and returns an error; missed events are not replayed. Recording uses one SSE connection; the current firmware supports at most two SSE clients, so close an extra monitoring panel if all connections are occupied.

## Analyze a recording

```bash
python3 breathing-research/score_research.py analyze \
  --input breathing-research/data/sitting-01.jsonl \
  --output breathing-research/reports/sitting-01.csv
```

Defaults use 30-second windows advanced every five seconds and search candidate frequencies from 0.1 to 0.6 Hz. Analysis splits history at device resets, gaps longer than one second, and sensing configuration events. It fits sinusoids to detrended scores using device timestamps.

The CSV includes the mean score, score standard deviation, candidate frequency, candidate cycles per minute, and sinusoid explained fraction. These describe signal structure, not the probability that someone is breathing. Constant scores have no candidate frequency. Overlapping windows are correlated; keep complete sessions together when later splitting training and test data.

Local recordings and reports are ignored by Git. No live hardware validation is implied by the offline checks.

## Train an exploratory presence model

The presence task has one label: occupied versus empty. Standing, sitting, lying, and typing are all occupied examples. Posture remains metadata and is not a model output. The first model is regularized logistic regression over six score statistics, using 30-second non-overlapping windows after trimming ten seconds from both ends of each uninterrupted sensing segment. It uses no synthetic data.

```bash
python3 breathing-research/train_presence.py \
  --data breathing-research/data \
  --selection breathing-research/reports/posture-review/selection.json \
  --output breathing-research/reports/presence-baseline-01
```

Omit `--selection` to train on every recording. With a selection manifest, the script also evaluates the full dataset and reports predictions on the excluded recordings. Every evaluation fold holds out an entire recording, fits feature scaling on training data only, and balances classes and recordings so long recordings do not dominate. The six features are log mean, log median, log 95th percentile, log standard deviation, zero-score fraction, and fraction of scores above 0.001.

Outputs include `model.json`, `evaluation.json`, and per-recording prediction CSVs. The model's sigmoid score is not a calibrated probability. Evaluation averages window scores per held-out recording at a fixed 0.5 threshold; it does not measure live FSM performance or false events per hour. Same-day recording holdouts do not establish transfer across days, people, or rooms. A fresh labeled day must be held out before relying on the model. No predictions alter the FSM.

## Live presence evidence

Run the saved model against your S3's processed motion stream:

```bash
python3 breathing-research/live_presence.py \
  --endpoint http://192.168.4.2:62587/espectre/v1
```

Replace the example address with the S3's Direct API address. The default model is the full-data Extra Trees model at `breathing-research/reports/presence-extra-trees/model.json`, trained on all 16 recordings. It uses 100 shallow trees and the same six score features. The trees are exported to JSON, so live inference still uses only the Python standard library. Use `--model PATH` for another compatible model; the original logistic model remains at `breathing-research/reports/presence-baseline-01/model.json`. Model artifacts are local and ignored by Git, so another checkout needs its own trained model file.

The detector waits ten seconds to settle and fills a 30-second window, so the first score normally arrives after about 40 seconds. It then evaluates a rolling window every five seconds. This overlapping live cadence differs from the non-overlapping training evaluation and has not been evaluated for false events per hour. Live inference has no end-of-recording trim because future samples are unavailable.

Outputs are `OCCUPIED_EVIDENCE`, `NO_OCCUPIED_EVIDENCE`, `WARMING_UP`, and `UNKNOWN`. A low score does not prove emptiness. Gaps, resets, sensing changes, and stale input clear the history; an interrupted SSE connection reports `UNKNOWN` and exits rather than retaining the previous prediction. Restart the command to reconnect. The script changes no device settings and sends no FSM or MQTT commands. Use `--json` for machine-readable status lines, `--duration 120` for a bounded live run, or Ctrl+C to stop.

Test the same inference logic offline with an existing recording:

```bash
python3 breathing-research/live_presence.py \
  --replay breathing-research/data/layingBack-00.jsonl --json
```

## Raw CSI and matched comparisons

The raw research path trains a separate occupied/empty model and compares its held-out predictions with a processed-score model trained on matching trials. This is a comparison of two inputs, not a synchronized fused model. It uses per-packet amplitude normalization and six features describing temporal channel-shape variation, adjacent-profile displacement, and spatial variation. It requires normalized HT20 int8 NPZ recordings with trustworthy per-packet `device_ticks_us`, a nominal 100 pps capture, and at least 85% slot coverage in each accepted window.

The NumPy analysis environment is separate from the ESPectre CLI environment:

```bash
python3 -m venv breathing-research/.venv
breathing-research/.venv/bin/python -m pip install -r breathing-research/requirements-raw.txt
```

For each trial, first record 60 seconds of processed scores with `score_research.py record`, as above. Then, keeping the boards, person, and position unchanged, record raw CSI using your working ESPectre CLI environment from `espectre/`:

```bash
./espectre collect --target http://192.168.4.2:62587/espectre/v1 \
  --label empty --duration 60 --pps 100 --detector lightweight

./espectre collect --target http://192.168.4.2:62587/espectre/v1 \
  --label static_presence --duration 60 --pps 100 --detector lightweight
```

These are separate commands for separate conditions: leave the room for `empty`, and stay in the desired occupied position for `static_presence`. In this checkout, the collector's host detector registry supports only `lightweight`, despite CLI help mentioning High Accuracy. The `--detector` flag chooses the host-side collection readiness check, not the device's sensing profile; keep High Accuracy selected on the device for processed-score recordings. The CLI prints the saved NPZ path. With a fixed duration, interrupting raw collection discards the recording. Raw collection pauses normal motion events and affects any FSM using them. Use a dedicated research S3 to keep the FSM's sensing nodes running. The collector switches the device to external traffic and leaves it there; restore the previous traffic mode in the panel after collection before resuming ordinary score monitoring. Do not run the live score reader on the same board during raw collection.

Copy `raw_sessions.example.json` to `raw_sessions.local.json`, replace the placeholder paths, and add one entry for every trial. Paths are relative to the manifest file. Keep `session_id`, posture, and condition consistent with the matching score recording, and give every pair a unique `experiment_id`. To make a recording distinguishable across days, include the day in its filename. For raw-only training, omit `score_file` from every entry.

```bash
breathing-research/.venv/bin/python breathing-research/raw_presence.py \
  --manifest breathing-research/raw_sessions.local.json \
  --output breathing-research/reports/raw-comparison-01

breathing-research/.venv/bin/python -m unittest discover -s breathing-research -v
```

Collect at least two independent empty and two occupied trials for a first exploratory recording holdout. Repeat both classes on another day; with multiple session IDs, the script holds out whole sessions instead. Every training fold must contain both labels. The output includes raw and processed model artifacts, capture-quality summaries, and `cross-examination.csv` showing where the models agree or disagree. Results from separate captures cannot establish synchronized fusion performance. Raw models are not compatible with `live_presence.py`, which expects processed score features.

## Compare small presence models

The optional scikit-learn experiment compares the existing logistic baseline with shallow Random Forest, Extra Trees, and gradient-boosted trees. It keeps the same six features, fixed 0.5 threshold, recording-balanced class weights, and predetermined parameters. It performs no parameter search and does not change the live detector.

```bash
breathing-research/.venv/bin/python -m pip install -r breathing-research/requirements-models.txt
breathing-research/.venv/bin/python breathing-research/compare_models.py \
  --selection breathing-research/reports/posture-review/selection.json \
  --output breathing-research/reports/model-comparison-01
```

Choose a new output directory for each run. Whole recordings are held out when only one session exists; with multiple sessions, whole sessions are held out. All folds require both classes in training. The report includes full-data and selected-subset results, per-recording predictions, and separately saved fitted models. The reduced subset is explicitly exploratory. Comparing several models on these same holdouts is not an independent final test: use a fresh labeled session before relying on a model. Tree artifacts are saved as local `.joblib` files; Random Forest and Extra Trees also have JSON exports supported by `live_presence.py`. For example, use `--model breathing-research/reports/model-comparison-01/full-extra_trees.json` after a new comparison run. Generating comparisons does not replace the active model; the currently selected Extra Trees export is stored separately under `reports/presence-extra-trees/`.

## Uploaded raw captures and live raw detection

The October 14 S3 upload has three empty and four static-presence captures. Their paths and occupied/empty labels are in the local `raw_sessions.local.json`. Posture was not recorded, so occupied captures are labeled `occupied` rather than assigned a guessed posture. Analysis excludes the first ten seconds of every static-presence capture, and also trims the last ten seconds. It centers 10 ms sampling slots on packet times so normal timestamp jitter does not create artificial missing slots; the 85% coverage requirement remains unchanged.

Train the raw-only Extra Trees model:

```bash
breathing-research/.venv/bin/python breathing-research/raw_presence.py \
  --manifest breathing-research/raw_sessions.local.json \
  --algorithm extra_trees \
  --output breathing-research/reports/raw-presence-01
```

Choose a new output directory for subsequent runs. The first uploaded-data evaluation holds out complete recordings and detects two of four occupied recordings, with two false occupied predictions among three empty recordings. These seven same-day recordings do not establish useful reliability. The processed-score model and its live script remain separate.

Run the raw-only live detector from the repository root:

```bash
breathing-research/.venv/bin/python breathing-research/live_raw_presence.py \
  --endpoint http://192.168.4.2:62587/espectre/v1
```

Replace the address with your S3's. The default model is `reports/raw-presence-01/raw-model.json` under this directory; use `--model PATH` for another compatible raw model. Keep CSI target rate at 100 pps. If the board is already in external mode, the script generates the required UDP traffic; stop other external generators first. In internal mode, the script uses the board's own traffic. The script does not change traffic settings. Opening the raw session pauses normal device motion events, so run it on a research board if your FSM must remain active. Ctrl+C closes the raw connection and stops its traffic generator.

Expect about 40 seconds of settling and window fill, followed by updates every five seconds. Missing or stale CSI produces `UNKNOWN`; low model output means `NO_OCCUPIED_EVIDENCE`, not confirmed emptiness. Use `--duration 120` for a bounded run or `--json` for machine-readable output. Replay is available with `--replay PATH_TO_CAPTURE.npz`; predictions on training recordings are demonstrations, not independent validation.

## Help and offline checks

```bash
python3 breathing-research/score_research.py record --help
python3 breathing-research/score_research.py analyze --help
python3 -m unittest discover -s breathing-research -v
```

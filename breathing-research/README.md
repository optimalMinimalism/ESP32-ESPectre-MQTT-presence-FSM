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

Replace the example address with the S3's Direct API address. The default model is `breathing-research/reports/presence-baseline-01/model.json`; use `--model PATH` for another compatible model. Model artifacts are local and ignored by Git, so another checkout needs its own trained model file.

The detector waits ten seconds to settle and fills a 30-second window, so the first score normally arrives after about 40 seconds. It then evaluates a rolling window every five seconds. This overlapping live cadence differs from the non-overlapping training evaluation and has not been evaluated for false events per hour. Live inference has no end-of-recording trim because future samples are unavailable.

Outputs are `OCCUPIED_EVIDENCE`, `NO_OCCUPIED_EVIDENCE`, `WARMING_UP`, and `UNKNOWN`. A low score does not prove emptiness. Gaps, resets, sensing changes, and stale input clear the history; an interrupted SSE connection reports `UNKNOWN` and exits rather than retaining the previous prediction. Restart the command to reconnect. The script changes no device settings and sends no FSM or MQTT commands. Use `--json` for machine-readable status lines, `--duration 120` for a bounded live run, or Ctrl+C to stop.

Test the same inference logic offline with an existing recording:

```bash
python3 breathing-research/live_presence.py \
  --replay breathing-research/data/layingBack-00.jsonl --json
```

## Help and offline checks

```bash
python3 breathing-research/score_research.py record --help
python3 breathing-research/score_research.py analyze --help
python3 -m unittest discover -s breathing-research -v
```

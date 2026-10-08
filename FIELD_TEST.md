# ESPectre four-node occupancy field test

This procedure takes four ESP32-S3 boards from flashing through a complete room-occupancy test. Commands rely on ESPectre's automatic serial-port detection. Specify `--port <SERIAL_PORT>` only when more than one compatible board is connected and the CLI cannot choose unambiguously.

The Wi-Fi and MQTT passwords must not be stored in this file.

## Test topology

All participants must be connected to the same LAN:

```text
S3_A (doorway) ─┐
S3_B (internal) ├── Wi-Fi AP ── MQTT broker
S3_C (internal) ┤               Aggregator computer
S3_D (internal) ┘
```

The access point must allow communication between wireless clients. Disable guest or client isolation. The MQTT broker must listen on a LAN-reachable interface and have a stable hostname or address.

## What to bring

- Four ESP32-S3 boards
- Four stable USB power supplies and cables
- A computer with this repository and its Python environment
- Access to the test Wi-Fi network
- A reachable MQTT broker, such as Mosquitto or the Home Assistant broker
- Labels for `S3_A`, `S3_B`, `S3_C`, and `S3_D`

Repository location:

```bash
cd /home/optimalminimalist/workMisc/movement-dectection-wifi/ESPectre/espectre
```

## 1. Build the firmware once

Build one Native firmware image for all four S3 boards:

```bash
./espectre native build --chip s3
```

If a suitable successful build already exists, this step can be skipped. The flash command uses the last successfully published S3 build artifact.

## 2. Flash and label each board

Connect only one board at a time. For a clean initial installation, run:

```bash
./espectre native flash --chip s3 --erase
```

Repeat for all four boards and label them immediately:

| Label | Role |
| --- | --- |
| `S3_A` | Doorway sensing path |
| `S3_B` | Internal sensing path |
| `S3_C` | Internal sensing path |
| `S3_D` | Internal sensing path |

`--erase` removes saved Wi-Fi and MQTT configuration. Omit it for later firmware updates when those settings should be retained.

If automatic serial selection is ambiguous, first disconnect the other boards or add the actual port reported by the operating system:

```bash
./espectre native flash --chip s3 --port <SERIAL_PORT> --erase
```

Do not copy a port name from another computer or session; USB port names are host- and connection-dependent.

## 3. Provision Wi-Fi

With one board connected at a time, run:

```bash
./espectre provision \
  --chip s3 \
  --frontend native \
  --ssid <WIFI_SSID> \
  --json
```

The CLI obtains the password from a hidden prompt unless the configured password environment variable is present. Repeat for every board.

If more than one compatible serial device is connected, add:

```text
--port <SERIAL_PORT>
```

Record the returned endpoint or address next to the physical board label. DHCP addresses can change, so treat the device ID—not the IP address—as its stable identity.

Holding the S3 `BOOT` button for approximately three seconds clears saved Wi-Fi settings and returns the board to provisioning mode.

## 4. Discover the boards and record their IDs

After all boards join the network, connect the test computer to the same LAN and run:

```bash
./espectre devices --frontend native --chip s3 --json
```

Match each physical label to its 16-character `device_id`. If necessary, power the boards one at a time while discovering them.

Record the mapping locally:

```text
S3_A = <DEVICE_ID_A>
S3_B = <DEVICE_ID_B>
S3_C = <DEVICE_ID_C>
S3_D = <DEVICE_ID_D>
```

If mDNS discovery is unavailable, obtain an address from the access point's client list and query it directly:

```bash
./espectre direct get device \
  --endpoint http://<DEVICE_ADDRESS>:62587/espectre/v1
```

## 5. Verify every detector independently

For each discovered endpoint, run:

```bash
./espectre direct get health \
  --endpoint http://<DEVICE_ADDRESS>:62587/espectre/v1

./espectre direct get sensing \
  --endpoint http://<DEVICE_ADDRESS>:62587/espectre/v1

./espectre direct get diagnostics \
  --endpoint http://<DEVICE_ADDRESS>:62587/espectre/v1
```

Confirm that:

- The device is online.
- Sensing is enabled.
- Calibration completes and the detector becomes ready.
- CSI traffic is being received.
- The movement score responds when someone crosses the sensing path.

For optional USB diagnostics, connect one board and let the CLI select it:

```bash
./espectre monitor --chip s3 --frontend native --reset
```

Use `--port <SERIAL_PORT>` only if selection is ambiguous. Stop the monitor with `Ctrl+C`.

## 6. Configure MQTT on every board

MQTT is disabled until it is configured. The safest method for an authenticated broker is the Device settings page because it does not place the password in shell history:

- <https://espectre.dev/tools/device-settings/>

For an anonymous broker on a trusted test LAN, configure each board through Direct HTTP:

```bash
./espectre direct patch mqtt \
  --endpoint http://<DEVICE_ADDRESS>:62587/espectre/v1 \
  --data '{
    "scheme": "mqtt",
    "host": "<MQTT_BROKER_HOST>",
    "port": 1883,
    "topic_prefix": "espectre/v1/devices"
  }'
```

Repeat for all four boards. The broker host must contain only a hostname or address, without `mqtt://`, credentials, a port, or a path.

Verify each saved configuration:

```bash
./espectre direct get mqtt \
  --endpoint http://<DEVICE_ADDRESS>:62587/espectre/v1
```

Expected evidence includes `configured: true` and the topic prefix `espectre/v1/devices`.

## 7. Install the boards in the room

Use propagation paths that cover different parts of the room:

- Place `S3_A` so its strongest sensing path crosses the doorway.
- Place `S3_B` across the main walking route.
- Place `S3_C` at a different angle across the sitting area.
- Place `S3_D` on another independent internal path.

Then:

1. Make the room empty.
2. Power all four boards in their final positions.
3. Keep the room empty and still for at least 60 seconds.
4. Confirm that all four detectors become ready.

Do not move a board after calibration. Restart or recalibrate it after changing its position.

## 8. Create the occupancy configuration

Create a local configuration from the example:

```bash
cp config/occupancy.example.yaml config/occupancy.yaml
```

Replace the placeholder IDs with the recorded values:

```yaml
nodes:
  - node_id: S3_A
    device_id: "<DEVICE_ID_A>"
    role: doorway

  - node_id: S3_B
    device_id: "<DEVICE_ID_B>"
    role: internal

  - node_id: S3_C
    device_id: "<DEVICE_ID_C>"
    role: internal

  - node_id: S3_D
    device_id: "<DEVICE_ID_D>"
    role: internal
```

Keep the initial timing and threshold defaults unchanged for the first complete test.

## 9. Start the occupancy aggregator

The room must be empty when the process starts because its in-memory state begins at `EMPTY`.

```bash
./espectre occupancy \
  --config config/occupancy.yaml \
  --broker <MQTT_BROKER_HOST> \
  --csv-log occupancy-test.csv \
  --jsonl-log occupancy-test.jsonl
```

Broker credentials can come from `MQTT_USERNAME` and `MQTT_PASSWORD`, or from `--username` and `--password`. Avoid placing real passwords in saved documentation or shared shell history.

The dashboard should show a score for every node within a few seconds. A `--` value indicates missing or stale telemetry. Check the device ID, MQTT configuration, broker connectivity, and detector readiness before continuing.

## 10. Execute the test sequence

### Test A: Empty room

Keep the room empty for 60 seconds.

Expected state:

```text
EMPTY
```

Isolated score changes must not confirm occupancy.

### Test B: Enter and walk

1. Walk through the doorway and cross `S3_A`'s sensing path.
2. Within eight seconds, cross paths covered by at least two internal nodes.
3. Continue walking for approximately 20 seconds.

Expected sequence:

```text
EMPTY
→ POSSIBLE_ENTRY
→ OCCUPIED_MOVING
```

### Test C: Remain still

Sit down and remain still for five minutes.

Expected sequence:

```text
OCCUPIED_MOVING
→ OCCUPIED_STILL
```

The state must not become `EMPTY`.

### Test D: Resume movement

Stand and walk across at least two internal paths.

Expected sequence:

```text
OCCUPIED_STILL
→ OCCUPIED_MOVING
```

### Test E: Approach the door without leaving

1. Walk from the interior toward the doorway.
2. Trigger the doorway path.
3. Return into the room and cross the internal paths again.

Expected sequence:

```text
OCCUPIED_MOVING
→ POSSIBLE_EXIT
→ OCCUPIED_MOVING
```

Renewed internal movement must cancel the possible exit.

### Test F: Leave the room

1. Walk from the interior toward the doorway.
2. Cross the doorway path and leave.
3. Keep the room empty.

Expected sequence:

```text
OCCUPIED_MOVING or OCCUPIED_STILL
→ POSSIBLE_EXIT
→ EMPTY
```

The final transition requires at least five seconds without internal activity. The four-second fusion window can add several seconds of visible delay.

### Test G: Open the door without entering

Start a new aggregator run with an empty room, then open the door or move near it without entering.

Expected sequence:

```text
EMPTY
→ POSSIBLE_ENTRY
→ EMPTY
```

The possible entry expires after eight seconds without strong internal activity.

### Test H: Unrelated Wi-Fi traffic

Generate ordinary network traffic without entering or leaving the room.

Expected result:

```text
The prior stable state is retained.
```

The disturbance must not satisfy both the multi-node and temporal-ordering rules.

## 11. Review the evidence

Inspect recent CSV evaluations:

```bash
tail -n 30 occupancy-test.csv
```

Inspect complete JSONL records:

```bash
tail -n 10 occupancy-test.jsonl
```

Review:

- `state_before` and `state_after`
- `reason`
- `occupancy_confidence`
- Per-node and fused scores
- Active and missing nodes
- Source-node state and device timestamp

A successful end-to-end run contains:

```text
EMPTY
→ POSSIBLE_ENTRY
→ OCCUPIED_MOVING
→ OCCUPIED_STILL
→ OCCUPIED_MOVING
→ POSSIBLE_EXIT
→ EMPTY
```

## Operational notes

- Restart the aggregator only when the room is known to be empty. State persistence is currently in memory, not across process restarts.
- DHCP addresses may change. Rediscover devices or use DHCP reservations instead of embedding addresses in the procedure.
- Serial port names may change after reconnecting a board. Let the CLI auto-detect the port whenever possible.
- Direct HTTP can work by address when mDNS discovery is blocked.
- Guest networks, VLAN separation, multicast filtering, and client isolation can prevent discovery or broker access.
- USB is required for flashing, provisioning, and serial diagnostics. After setup, each board only needs stable power and network access.

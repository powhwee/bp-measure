# Development Log: Sensor Prototyping Discussion
Date: 2026-02-08

## 1. Repository Analysis: `bp-measure`

### Overview
The project is a non-invasive blood pressure monitor estimating Systolic (SBP) and Diastolic (DBP) blood pressure from smartphone camera PPG signals. It runs entirely client-side using WebGPU.

### Repo Structure
*   **Python Trainer (`src/trainer/`)**:
    *   `data_loader.py`: Loads UCI Cuff-less BP dataset (MIMIC-II derived).
    *   `resnet1d.py`: ResNet-18 adapted for 1D signals with Squeeze-and-Excitation attention.
    *   `train.py`: Training loop using Huber loss.
    *   `export_onnx.py`: Exports trained model to ONNX.
*   **Web App (`web/`)**:
    *   **GPU Pipeline**:
        *   `SignalProcessor.ts` + `extract_rgb.wgsl`: Computes average RGB from camera feed via WebGPU compute shader (no CPU readback).
        *   `DSPEngine.ts` + `resample.wgsl`: FFT-based resampling to 125Hz.
        *   `InferenceEngine.ts`: Runs ONNX Runtime (WebGPU backend) on Z-score normalized data.

## 2. Evaluation of Camera-Based Method

### Assessment
*   **Research/Demo Quality**: Impressive engineering ("Zero-CPU" pipeline, signal parity).
*   **Clinical Viability**: **Low**.
*   **Key Issues**:
    *   **Domain Gap**: Training data is from invasive clinical sensors; inference is from noisy CMOS camera.
    *   **Confounders**: Ambient light, finger pressure, skin tone, variable frame rates.
    *   **Generalization**: High individual variation; "calibration" is just a linear offset.

## 3. Proposal: Hardware Upgrade (MAX30102)

### Why MAX30102?
*   **Dedicated Sensor**: Red (660nm) + IR (880nm) LEDs + optimized photodiode.
*   **Signal Quality**: Much cleaner morphology (dicrotic notch visibility).
*   **Consistency**: Fixed sample rate (e.g., 100Hz), ambient light cancellation.
*   **Closer to Ground Truth**: Reduces domain gap with clinical training data.

## 4. Prototyping Architecture (RPi + Mac)

### Setup
*   **Sensor**: MAX30102 connected to Raspberry Pi via I²C.
*   **Compute**: Mac M4 Max (running the WebGPU app).
*   **Connection**: WebSocket over WiFi.

```mermaid
graph LR
    A[MAX30102] -->|I2C| B[Raspberry Pi]
    B -->|WebSocket JSON| C[Mac M4 Max]
    C -->|WebGPU Pipeline| D[BP Estimate]
```

### Raspberry Pi Role
*   Acts as a "dumb" sensor relay.
*   Runs a simple Python script to read I²C data and stream via `websockets`.

**Server Script (`sensor_relay.py`):**
```python
import asyncio, json, websockets
from max30102 import MAX30102

sensor = MAX30102()
sensor.setup()

async def stream(websocket):
    while True:
        red, ir = sensor.read_sequential(100)
        await websocket.send(json.dumps({"ir": ir, "red": red, "fs": 100}))
        await asyncio.sleep(0.01)

asyncio.run(websockets.serve(stream, "0.0.0.0", 8765))
```

### Web App Changes
*   Replace `SignalProcessor.ts` (Camera) with a WebSocket client (`SensorInput.ts`).
*   Accumulate IR samples.
*   Feed into existing `DSPEngine` -> `InferenceEngine`.

## 5. Physical Enclosure

### Requirement
*   3D printable finger clip for bare MAX30102 sensor.
*   No microcontroller housing needed on the finger.

### Recommendation
*   **Best Option**: [Pulse Oximeter Finger Clip (MAX30102 module) by mobi_electronik](https://www.thingiverse.com/thing:5765498)
    *   **Pros**: Spring-loaded torsion design, open back for wire exit, recessed sensor pocket to reduce ambient light.
    *   **Requirements**: Torsion spring + M2 self-tapping screw.
*   **Alternative (Discarded)**: [MakerWorld MAX30102 Clip (1061909)](https://makerworld.com/en/models/1061909)
    *   **Cons**: Ultra-miniature snap-fit with no clear wire routing channel. Wires would have to exit awkwardly.
    *   **Decision**: Stick with the mobi_electronik design for better wire management and consistent pressure.

## 6. Wiring Diagram (MAX30102 to RPi)

```
MAX30102 Module     Raspberry Pi (GPIO Header)
─────────────────   ──────────────────────────────────
VIN  ──────────────  3.3V (Pin 1)
GND  ──────────────  GND  (Pin 6)
SDA  ──────────────  SDA  (GPIO 2, Pin 3)
SCL  ──────────────  SCL  (GPIO 3, Pin 5)
```
*   **Note**: Most MAX30102 modules have built-in pull-up resistors, so direct connection to Pi GPIO is fine.

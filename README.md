# Maxwell: Real-Time Biometric & Physiological Monitoring System

Maxwell is a computer vision and biomedical signal processing application that extracts cardiorespiratory and behavioral indicators of stress and fatigue from unconstrained webcam feeds. Designed for edge execution, the system fuses plane-orthogonal-to-skin (POS) chrominance models with kinematic facial tracking to provide continuous, non-contact physiological assessment.

---

## Technical Architecture & Signal Processing

### 1. Multi-Region Vascular rPPG
Traditional remote photoplethysmography (rPPG) often suffers from motion artifacts and illumination shifts. Maxwell mitigates this by sampling four distinct facial regions of interest (ROIs)—the left and right malar cheeks, nasal bridge, and forehead.
* **Chrominance Fusion:** The pipeline extracts raw RGB values alongside CIELAB a* hemoglobin chrominance channels to isolate blood volume pulse (BVP) signals from ambient light fluctuations.
* **Ballistocardiographic (BCG) Fusion:** Subtle vertical micro-motions of the nasal tip and upper lip are tracked over time and fused with the optical pulse signal.
* **Spectral Analysis:** Extracted time-series signals are detrended and passed through a 3rd-order Butterworth bandpass filter (0.80 Hz - 2.75 Hz). Welch’s Power Spectral Density (PSD) estimation is then applied to compute dominant cardiac frequency peaks, yielding real-time **Heart Rate (BPM)** and **Heart Rate Variability (HRV - RMSSD)** alongside a Signal Quality Index (SQI) confidence metric.

### 2. Behavioral Fatigue & Ergonomic Telemetry
* **PERCLOS & EAR:** Evaluates Eyelid Closure Percentage over rolling temporal windows and computes Eye Aspect Ratio to quantify acute ocular fatigue.
* **Mouth Aspect Ratio (MAR):** Continuously monitors vertical lip separation to detect and log frequency-based yawn events.
* **Ergonomic Posture & Distance:** Estimates camera-to-subject distance in centimeters using inter-ocular pixel spans and monitors head pitch/yaw deviations to flag cervical slouching.

### 3. Cloud Clinical Triage Integration
Aggregated numerical telemetry—including heart rate, HRV, respiration rate, stress index, and sleepiness metrics—can be bundled with natural language symptom queries. These payloads are routed through SerpApi’s Google Gemini integration endpoint to generate structured, evidence-based triage recommendations.

---

## System Requirements & Dependencies

* **Language:** Python 3.10+
* **Core Libraries:** OpenCV (`cv2`), MediaPipe Tasks API (`face_landmarker.task`)
* **Numerical & Signal Processing:** NumPy, SciPy (`scipy.signal`)
* **API & Networking:** Requests
* **Speech/Audio (Optional):** `SpeechRecognition`, `pyttsx3` / system text-to-speech utilities

---

## Installation & Deployment

### 1. Clone Repository & Set Up Environment
```bash
git clone https://github.com/YOUR_USERNAME/maxwell-health-companion.git
cd maxwell-health-companion

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install opencv-python mediapipe numpy scipy requests SpeechRecognition pyttsx3
```

### 3. Configure API Credentials
Export your SerpApi key as a secure shell environment variable:
```bash
export SERPAPI_API_KEY="your_actual_serpapi_key_here"
```

### 4. Execute Application
```bash
python maxwell_companion.py
```

---

## Operational Controls

* **`C`** : Calibrate neutral facial expression and posture baseline.
* **`T`** : Open on-screen text input buffer for symptom entry.
* **`SPACE`** : Initialize microphone speech-to-text listener.
* **`R`** : Execute proactive multi-vital health assessment via cloud API.
* **`B`** : Toggle guided box-breathing cadence visualizer.
* **`Q`** : Terminate application process.

---

## Privacy & Data Handling
All video frame processing, geometric landmark extraction, and rPPG digital signal processing occur locally on the host machine. No raw video frames, images, or personally identifiable visual data are stored or transmitted. Only aggregated scalar telemetry and optional user-submitted text are transmitted to the configured API endpoint.
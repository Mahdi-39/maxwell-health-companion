import os
import time
import math
import re
import threading
import subprocess
import shutil
import urllib.request
from collections import deque
import cv2
import numpy as np
from scipy import signal
import requests
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Optional speech recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")

# 1. Model download
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading Face Landmarker model (~3.7 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# 2. Non-Blocking Voice Synthesizer
speech_until = 0.0

def maxwell_speak(text):
    global speech_until
    clean_speech = re.sub(r"[\*\_\[\]\(\)]", "", text)
    speech_until = time.monotonic() + max(2.0, len(clean_speech) * 0.062)
    print(f"\n[MAXWELL HEALTH]: {clean_speech}")

    def _tts_worker(msg):
        try:
            if shutil.which("spd-say"):
                subprocess.run(["spd-say", "-r", "-5", msg], check=False)
                return
            if shutil.which("espeak-ng"):
                subprocess.run(["espeak-ng", "-s", "145", msg], check=False)
                return
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(msg)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=_tts_worker, args=(clean_speech,), daemon=True).start()

# 3. SerpApi Clinical Health Consultation Worker
def query_gemini_maxwell(symptom_text, vitals_dict, callback):
    def _worker():
        if not SERPAPI_API_KEY:
            callback("Please set your SERPAPI_API_KEY environment variable to enable cloud clinical consultations.")
            return
        try:
            prompt = (
                "You are Maxwell, a professional, evidence-based AI clinical health assistant. "
                "CRITICAL RULES: Do NOT roleplay as an animal. Do NOT output asterisks, emotes, or stage directions. "
                "Speak strictly as a warm, objective medical triage assistant.\n"
                f"Patient Vitals: Heart Rate={vitals_dict['bpm']} BPM (HRV={vitals_dict['hrv']} ms), "
                f"Respiration Rate={vitals_dict['resp']} breaths/min, Stress Index={vitals_dict['stress']}%, "
                f"Sleepiness={vitals_dict['sleepiness']}% ({vitals_dict['sleep_label']}, Yawns={vitals_dict['yawns']}), "
                f"Screen Distance={vitals_dict['dist_cm']} cm.\n"
                f"Patient Input / Complaint: '{symptom_text}'.\n"
                "Task: Give a concise 2-sentence clinical assessment connecting their vitals and input, "
                "1 specific actionable health recommendation, and 1 brief diagnostic follow-up question. "
                "Keep total response under 52 words."
            )
            params = {
                "engine": "google_gemini",
                "q": prompt,
                "api_key": SERPAPI_API_KEY
            }
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
            data = resp.json()

            answer = ""
            if "text_blocks" in data and len(data["text_blocks"]) > 0:
                answer = data["text_blocks"][0].get("snippet", "")
            elif "reconstructed_markdown" in data:
                answer = data["reconstructed_markdown"]

            if not answer:
                answer = "Vitals logged. Maintain hydration, rest your eyes, and monitor symptoms closely."

            clean = re.sub(r"\*[^*]+\*", "", answer)
            clean = re.sub(r"[\*\_]", "", clean)
            clean = " ".join(clean.strip().split())
            callback(clean)
        except Exception as e:
            print(f"[SerpApi Error] {e}")
            callback("Vitals logged locally. Ensure proper hydration and take periodic visual breaks.")

    threading.Thread(target=_worker, daemon=True).start()

# 4. Multi-Region POS + CIELAB + BCG Heart Rate, HRV & Respiration DSP
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_ear(pts, top_idx, bot_idx, inner_idx, outer_idx):
    return dist(pts[top_idx], pts[bot_idx]) / (dist(pts[inner_idx], pts[outer_idx]) + 1e-6)

def compute_mar(pts):
    return dist(pts[13], pts[14]) / (dist(pts[78], pts[308]) + 1e-6)

def sample_vascular_patch(frame, lab_frame, cx, cy, rx, ry):
    h, w, _ = frame.shape
    x1, y1 = max(0, cx - rx), max(0, cy - ry)
    x2, y2 = min(w, cx + rx), min(h, cy + ry)
    roi_bgr = frame[y1:y2, x1:x2]
    roi_lab = lab_frame[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None
    b = float(np.mean(roi_bgr[:, :, 0]))
    g = float(np.mean(roi_bgr[:, :, 1]))
    r = float(np.mean(roi_bgr[:, :, 2]))
    a_lab = float(np.mean(roi_lab[:, :, 1]))  # Hemoglobin red-green chrominance
    return (r, g, b, a_lab), (x1, y1, x2, y2)

def analyze_cardiorespiratory_signals(rgb_a_buffer, bcg_buffer, resp_buffer, timestamps):
    """
    Fuses Plane-Orthogonal-to-Skin (POS), CIELAB a* chrominance, and Nose/Mouth BCG
    to estimate BPM, HRV (RMSSD in ms), Signal Confidence %, and Respiration Rate.
    """
    n = len(rgb_a_buffer)
    if n < 75:
        return None, None, 0, None, np.zeros(n)

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return None, None, 0, None, np.zeros(n)
    fs = n / duration
    if fs < 10.0:
        return None, None, 0, None, np.zeros(n)

    arr = np.array(rgb_a_buffer, dtype=np.float64)
    r, g, b, a_ch = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # 1. POS (Plane-Orthogonal-to-Skin) normalization
    r_n = r / (np.mean(r) + 1e-6) - 1.0
    g_n = g / (np.mean(g) + 1e-6) - 1.0
    b_n = b / (np.mean(b) + 1e-6) - 1.0

    s1 = g_n - b_n
    s2 = g_n + b_n - 2.0 * r_n
    alpha = np.std(s1) / (np.std(s2) + 1e-6)
    pos_sig = s1 + alpha * s2

    # 2. CIELAB a* hemoglobin chrominance + Nose/Mouth BCG micro-pulse fusion
    a_norm = signal.detrend(a_ch) / (np.std(a_ch) + 1e-6)
    pos_norm = signal.detrend(pos_sig) / (np.std(pos_sig) + 1e-6)
    bcg_norm = signal.detrend(np.array(bcg_buffer, dtype=np.float64))
    bcg_norm = bcg_norm / (np.std(bcg_norm) + 1e-6)

    fused_pulse = 0.50 * pos_norm + 0.35 * a_norm + 0.15 * bcg_norm

    nyq = 0.5 * fs
    low = max(0.05, min(0.80 / nyq, 0.90))   # 48 BPM
    high = max(low + 0.05, min(2.75 / nyq, 0.98))  # 165 BPM
    try:
        b_filt, a_filt = signal.butter(3, [low, high], btype="band")
        filtered = signal.filtfilt(b_filt, a_filt, fused_pulse)
    except Exception:
        filtered = fused_pulse

    # Welch Power Spectral Density for robust BPM peak extraction
    nperseg = min(len(filtered), 128)
    freqs, psd = signal.welch(filtered, fs=fs, window="hann", nperseg=nperseg, nfft=512)
    valid = (freqs >= 0.80) & (freqs <= 2.75)
    if not np.any(valid):
        return None, None, 0, None, filtered

    valid_freqs = freqs[valid]
    valid_psd = psd[valid]
    peak_idx = int(np.argmax(valid_psd))
    peak_freq = valid_freqs[peak_idx]
    bpm = int(round(peak_freq * 60.0))

    # Signal Quality / Confidence Index (SQI)
    total_power = np.sum(valid_psd) + 1e-8
    peak_power = np.sum(valid_psd[max(0, peak_idx - 2):min(len(valid_psd), peak_idx + 3)])
    sqi = int(np.clip((peak_power / total_power) * 145.0, 15.0, 99.0))

    # Estimate HRV (RMSSD in ms) from time-domain inter-beat intervals
    min_dist = max(1, int(fs * 0.38))
    peaks, _ = signal.find_peaks(filtered, distance=min_dist, prominence=np.std(filtered) * 0.35)
    hrv_ms = None
    if len(peaks) >= 4:
        t_arr = np.array(timestamps)
        ibi_ms = np.diff(t_arr[peaks]) * 1000.0
        valid_ibi = ibi_ms[(ibi_ms >= 350.0) & (ibi_ms <= 1350.0)]
        if len(valid_ibi) >= 3:
            diffs = np.diff(valid_ibi)
            hrv_ms = int(np.clip(np.sqrt(np.mean(diffs ** 2)), 12.0, 120.0))

    # Estimate Respiration Rate (Breaths/Min) from Nose/Mouth kinematics + low-freq modulation
    resp_bpm = None
    if len(resp_buffer) >= 90:
        r_sig = signal.detrend(np.array(resp_buffer, dtype=np.float64))
        r_low = max(0.03, min(0.16 / nyq, 0.85))   # ~10 breaths/min
        r_high = max(r_low + 0.04, min(0.42 / nyq, 0.95))  # ~25 breaths/min
        try:
            br, ar = signal.butter(2, [r_low, r_high], btype="band")
            r_filt = signal.filtfilt(br, ar, r_sig)
            rfreqs, rpsd = signal.welch(r_filt, fs=fs, nperseg=min(len(r_filt), 128), nfft=512)
            rvalid = (rfreqs >= 0.16) & (rfreqs <= 0.42)
            if np.any(rvalid):
                r_peak = rfreqs[rvalid][np.argmax(rpsd[rvalid])]
                resp_bpm = int(round(r_peak * 60.0))
        except Exception:
            pass

    return bpm, hrv_ms, sqi, resp_bpm, filtered

def wrap_text(text, max_chars=66):
    words = text.split()
    lines, curr = [], ""
    for w in words:
        if len(curr) + len(w) + 1 <= max_chars:
            curr = f"{curr} {w}".strip()
        else:
            lines.append(curr)
            curr = w
    if curr:
        lines.append(curr)
    return lines[:3]

# 5. Robotic Smiley Face Avatar
def draw_robot_smiley(frame, cx, cy, is_talking, is_blinking, is_sleepy, is_stressed):
    """Draws a friendly robotic smiley face with LED eyes, antenna, and animated smile."""
    t = time.monotonic()
    accent = (0, 140, 255) if is_stressed else ((255, 190, 80) if is_sleepy else (0, 240, 200))

    # Top Antenna & Glowing Tip
    cv2.line(frame, (cx, cy - 32), (cx, cy - 44), (160, 175, 190), 2)
    pulse_r = 5 if is_talking else 4
    cv2.circle(frame, (cx, cy - 46), pulse_r, accent, -1)

    # Side Robotic Ear Bolts / Audio Transponders
    cv2.rectangle(frame, (cx - 48, cy - 10), (cx - 40, cy + 10), (90, 105, 125), -1)
    cv2.rectangle(frame, (cx + 40, cy - 10), (cx + 48, cy + 10), (90, 105, 125), -1)

    # Outer Metallic Chassis & Inner Visor Screen
    cv2.rectangle(frame, (cx - 40, cy - 30), (cx + 40, cy + 30), (42, 48, 60), -1)
    cv2.rectangle(frame, (cx - 40, cy - 30), (cx + 40, cy + 30), accent, 2)
    cv2.rectangle(frame, (cx - 33, cy - 23), (cx + 33, cy + 23), (20, 24, 32), -1)

    # Digital Eyes
    if is_blinking:
        cv2.line(frame, (cx - 22, cy - 7), (cx - 10, cy - 7), accent, 2)
        cv2.line(frame, (cx + 10, cy - 7), (cx + 22, cy - 7), accent, 2)
    elif is_sleepy:
        # Half-drooping sleepy digital eyes
        cv2.ellipse(frame, (cx - 16, cy - 7), (6, 4), 0, 0, 180, accent, 2)
        cv2.ellipse(frame, (cx + 16, cy - 7), (6, 4), 0, 0, 180, accent, 2)
    else:
        # Happy glowing robotic eyes with shine
        cv2.circle(frame, (cx - 16, cy - 8), 5, accent, -1)
        cv2.circle(frame, (cx + 16, cy - 8), 5, accent, -1)
        cv2.circle(frame, (cx - 18, cy - 10), 2, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 14, cy - 10), 2, (255, 255, 255), -1)

    # Robotic Smiley Mouth
    if is_talking:
        # Animated smiling voice wave
        open_h = 4 + int(4 * abs(math.sin(t * 15)))
        cv2.ellipse(frame, (cx, cy + 7), (15, open_h), 0, 0, 180, accent, 2)
        cv2.line(frame, (cx - 15, cy + 7), (cx + 15, cy + 7), accent, 1)
    else:
        # Friendly curved digital smile
        cv2.ellipse(frame, (cx, cy + 6), (15, 8), 0, 15, 165, accent, 2)

# 6. Gesture Detection (Nod YES / Shake NO)
def detect_nod_or_shake(buffer):
    if len(buffer) < 20:
        return None
    xs = [p[0] for p in buffer]
    ys = [p[1] for p in buffer]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    dx_signs = np.sign(np.diff(xs))
    dy_signs = np.sign(np.diff(ys))
    x_flips = np.sum(dx_signs[1:] * dx_signs[:-1] < 0)
    y_flips = np.sum(dy_signs[1:] * dy_signs[:-1] < 0)

    if span_y > 0.11 and span_y > span_x * 1.45 and y_flips >= 2:
        return "YES"
    if span_x > 0.13 and span_x > span_y * 1.45 and x_flips >= 2:
        return "NO"
    return None

# 7. Rolling Biometric Buffers & State
rgb_a_buffer = deque(maxlen=180)
bcg_buffer = deque(maxlen=180)
resp_buffer = deque(maxlen=180)
pulse_times = deque(maxlen=180)

bpm_history = deque(maxlen=12)
hrv_history = deque(maxlen=10)
resp_history = deque(maxlen=10)

ear_buffer = deque(maxlen=35)
perclos_buffer = deque(maxlen=180)  # Tracks eye closure % over ~6s window
jitter_buffer = deque(maxlen=30)
head_gesture_buffer = deque(maxlen=26)
blink_timestamps = deque(maxlen=60)

was_eye_closed = False
was_yawning = False
yawn_start_time = 0.0
yawn_count = 0
baseline_brow_ratio = None
baseline_pitch_ratio = None

breathing_mode = False
breathing_start_time = 0.0

chat_state = "IDLE"
chat_prompt = "Maxwell Health Companion active. Press [T] to type symptoms, [SPACE] to speak, or [R] for Full Checkup."
chat_subtext = "[T] Type Symptoms | [SPACE] Voice | [R] AI Vitals Checkup | [B] Breathe | [C] Calibrate | [Q] Quit"
typing_active = False
input_text = ""
gesture_cooldown_until = 0.0

def set_dialogue(new_state, main_line, sub_line, speak=True):
    global chat_state, chat_prompt, chat_subtext, gesture_cooldown_until
    chat_state = new_state
    chat_prompt = main_line
    chat_subtext = sub_line
    gesture_cooldown_until = time.monotonic() + 1.3
    head_gesture_buffer.clear()
    if speak:
        maxwell_speak(main_line)

def listen_to_speech(vitals_snapshot):
    if not SR_AVAILABLE:
        set_dialogue("IDLE", "Microphone module unavailable. Please press [T] to type your symptoms.", "Use [T] for symptom input")
        return

    def _worker():
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                set_dialogue("LISTENING", "Listening. Please describe how you are feeling.", "Recording microphone input...")
                r.adjust_for_ambient_noise(source, duration=0.4)
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio)
            set_dialogue("DIAGNOSIS", f"Analyzing: '{text}'. Correlating with multi-region vitals...", "Consulting clinical database...")
            query_gemini_maxwell(
                text, vitals_snapshot,
                callback=lambda ai_msg: set_dialogue("DIAGNOSIS", ai_msg, "Assessment complete | [T] Type | [R] Full Checkup")
            )
        except Exception:
            set_dialogue("IDLE", "Could not capture audio clearly. Press [T] to type your symptoms.", "Press [T] to type")

    threading.Thread(target=_worker, daemon=True).start()

# 8. MediaPipe Setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_time = time.monotonic()
last_ts_ms = 0

maxwell_speak("Hello! I am Maxwell, your health companion. Press T to log symptoms, Spacebar to speak, or R for a full vitals checkup.")

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        now_sec = time.monotonic()
        now_ms = int((now_sec - start_time) * 1000)
        ts_ms = max(now_ms, last_ts_ms + 1)
        last_ts_ms = ts_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, ts_ms)

        # Translucent HUD panels
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 68), (18, 22, 30), -1)
        cv2.rectangle(overlay, (0, h - 126), (w, h), (18, 22, 30), -1)
        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

        cv2.putText(frame, "MAXWELL // BIOMETRIC & SLEEPINESS HEALTH COMPANION",
                    (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(frame, "[T] Symptoms | [SPACE] Voice | [R] Full Checkup | [B] Breathe | [C] Calibrate",
                    (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150, 230, 220), 1)

        stress_index = 22
        sleepiness_idx = 15
        sleep_label = "ALERT"
        smooth_bpm = 72
        smooth_hrv = 48
        smooth_resp = 15
        sqi_conf = 0
        dist_cm = 60
        detected_gesture = None

        if results.face_landmarks:
            flm = results.face_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in flm]

            # 1. Eye Aspect Ratio (EAR), Blink Rate & PERCLOS (Eyelid Closure %)
            left_ear = compute_ear(pts, 159, 145, 133, 33)
            right_ear = compute_ear(pts, 386, 374, 362, 263)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_buffer.append(avg_ear)
            smooth_ear = float(np.mean(ear_buffer))

            is_drooping = 1.0 if avg_ear < 0.21 else 0.0
            perclos_buffer.append(is_drooping)
            perclos_pct = float(np.mean(perclos_buffer)) * 100.0

            if avg_ear < 0.19 and not was_eye_closed:
                was_eye_closed = True
                blink_timestamps.append(now_sec)
            elif avg_ear >= 0.21:
                was_eye_closed = False

            while blink_timestamps and (now_sec - blink_timestamps[0] > 60.0):
                blink_timestamps.popleft()
            blinks_per_min = len(blink_timestamps)

            # 2. Yawn Detection (Mouth Aspect Ratio) -> Sleepiness Indicator
            mar = compute_mar(pts)
            is_yawning = mar > 0.55
            if is_yawning:
                if not was_yawning:
                    yawn_start_time = now_sec
                    was_yawning = True
                elif (now_sec - yawn_start_time) > 1.1:
                    yawn_count += 1
                    yawn_start_time = now_sec + 4.5
                    if chat_state == "IDLE":
                        set_dialogue("ASK_HYDRATION",
                                     f"Yawn #{yawn_count} detected. Sleepiness is rising. Have you rested or hydrated recently?",
                                     "Nod head (YES) / Shake head (NO) or press [Y]/[N]")
            else:
                was_yawning = False

            # 3. Posture, Brow Tension & Ergonomic Screen Distance (cm)
            eye_span = dist(pts[33], pts[263]) + 1e-6
            dist_cm = int(np.clip(5600.0 / eye_span, 25.0, 130.0))

            brow_ratio = dist(pts[107], pts[336]) / eye_span
            eye_mid_x = (pts[33][0] + pts[263][0]) / 2.0
            eye_mid_y = (pts[33][1] + pts[263][1]) / 2.0
            pitch_ratio = (pts[1][1] - eye_mid_y) / eye_span
            yaw_ratio = (pts[1][0] - eye_mid_x) / eye_span

            if baseline_brow_ratio is None:
                baseline_brow_ratio = brow_ratio
                baseline_pitch_ratio = pitch_ratio

            brow_strain = np.clip((baseline_brow_ratio - brow_ratio) * 450.0, 0.0, 100.0)
            slouch_strain = np.clip((pitch_ratio - baseline_pitch_ratio) * 320.0, 0.0, 100.0)

            head_gesture_buffer.append((yaw_ratio, pitch_ratio))
            if now_sec > gesture_cooldown_until:
                detected_gesture = detect_nod_or_shake(head_gesture_buffer)

            # 4. Head Micro-Jitter
            jitter_buffer.append(pts[1])
            if len(jitter_buffer) > 5:
                xs = [p[0] for p in jitter_buffer]
                ys = [p[1] for p in jitter_buffer]
                restlessness = min(100.0, (np.std(xs) + np.std(ys)) * 3.8)
            else:
                restlessness = 0.0

            # 5. Multi-Region Vascular rPPG (Left Cheek, Right Cheek, Nose Bridge, Forehead)
            # + Nose/Mouth BCG & Respiration Tracking
            rois = [
                (pts[118][0], pts[118][1], 12, 10),  # Left Malar Cheek
                (pts[347][0], pts[347][1], 12, 10),  # Right Malar Cheek
                (pts[6][0], pts[6][1], 9, 8),        # Nasal Bridge
                (pts[151][0], pts[151][1], 14, 10)   # Forehead
            ]
            patch_vals = []
            for rx_c, ry_c, rw, rh in rois:
                res = sample_vascular_patch(frame, lab_frame, rx_c, ry_c, rw, rh)
                if res is not None:
                    vals, (x1, y1, x2, y2) = res
                    patch_vals.append(vals)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 180), 1)

            if patch_vals:
                mean_rgba = np.mean(patch_vals, axis=0)
                rgb_a_buffer.append(mean_rgba)
                # Nose tip (1) + Upper lip (0) vertical Ballistocardiographic micro-motion
                bcg_y = ((pts[1][1] + pts[0][1]) * 0.5) / (eye_span + 1e-5)
                bcg_buffer.append(bcg_y)
                # Nasal-mouth respiratory oscillation
                resp_signal = (pts[1][1] + dist(pts[13], pts[14]) * 0.5) / (eye_span + 1e-5)
                resp_buffer.append(resp_signal)
                pulse_times.append(now_sec)

            raw_bpm, raw_hrv, sqi_conf, raw_resp, filtered_wave = analyze_cardiorespiratory_signals(
                rgb_a_buffer, bcg_buffer, resp_buffer, pulse_times
            )
            if raw_bpm is not None:
                bpm_history.append(raw_bpm)
            if raw_hrv is not None:
                hrv_history.append(raw_hrv)
            if raw_resp is not None:
                resp_history.append(raw_resp)

            smooth_bpm = int(np.median(bpm_history)) if bpm_history else 72
            smooth_hrv = int(np.median(hrv_history)) if hrv_history else 45
            smooth_resp = int(np.median(resp_history)) if resp_history else 15

            # 6. Compute Dedicated Sleepiness & Stress Scores
            ear_drowsy = np.clip((0.26 - smooth_ear) * 480.0, 0.0, 45.0)
            yawn_load = min(45.0, yawn_count * 15.0)
            sleepiness_idx = int(np.clip(
                ear_drowsy + yawn_load + (perclos_pct * 0.45) + (slouch_strain * 0.20),
                5.0, 99.0
            ))
            if sleepiness_idx >= 75 or yawn_count >= 3:
                sleep_label = "SLEEP DEPRIVED"
            elif sleepiness_idx >= 50 or yawn_count >= 2:
                sleep_label = "DROWSY"
            elif sleepiness_idx >= 30 or yawn_count == 1:
                sleep_label = "MILD DROWSINESS"
            else:
                sleep_label = "ALERT"

            blink_stress = np.clip((blinks_per_min - 18) * 2.0, 0.0, 30.0)
            hr_elevation = np.clip((smooth_bpm - 82) * 0.8, 0.0, 25.0)
            stress_index = int(np.clip(
                0.40 * brow_strain + 0.30 * restlessness + 0.15 * blink_stress + 0.15 * hr_elevation,
                5.0, 99.0
            ))

            # 7. Left Side Diagnostic HUD Gauges
            # Stress Gauge
            s_col = (0, 255, 120) if stress_index < 40 else (0, 180, 255) if stress_index < 70 else (0, 60, 255)
            cv2.putText(frame, f"STRESS: {stress_index}%", (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48, s_col, 2)
            cv2.rectangle(frame, (16, 99), (196, 110), (50, 50, 55), -1)
            cv2.rectangle(frame, (16, 99), (16 + int(stress_index * 1.8), 110), s_col, -1)

            # Sleepiness Gauge (Driven by Yawns + Eye Droop + PERCLOS)
            slp_col = (100, 240, 180) if sleepiness_idx < 35 else (0, 200, 255) if sleepiness_idx < 65 else (0, 70, 255)
            cv2.putText(frame, f"SLEEPINESS: {sleepiness_idx}% [{sleep_label}]", (16, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, slp_col, 2)
            cv2.rectangle(frame, (16, 139), (196, 150), (50, 50, 55), -1)
            cv2.rectangle(frame, (16, 139), (16 + int(sleepiness_idx * 1.8), 150), slp_col, -1)

            # Cardiorespiratory & Ergonomic Telemetry
            hr_label = f"HR: {smooth_bpm} BPM (Conf: {sqi_conf}%)" if sqi_conf > 0 else "HR: Calibrating ROIs..."
            cv2.putText(frame, hr_label, (16, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 210), 2)
            cv2.putText(frame, f"HRV: {smooth_hrv} ms  |  RESP: {smooth_resp} br/min", (16, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 235, 255), 1)
            cv2.putText(frame, f"YAWNS: {yawn_count}  |  BLINKS/M: {blinks_per_min}  |  SLOUCH: {int(slouch_strain)}%", (16, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 215, 220), 1)

            dist_col = (0, 100, 255) if dist_cm < 42 else (180, 220, 200)
            dist_warn = " [TOO CLOSE!]" if dist_cm < 42 else ""
            cv2.putText(frame, f"SCREEN DIST: {dist_cm} cm{dist_warn}", (16, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, dist_col, 1)

            # Guided Box-Breathing Pacer
            if breathing_mode:
                pos = (now_sec - breathing_start_time) % 12.0
                cx, cy = w - 100, 155
                radius = int(20 + (pos / 4.0) * 28) if pos < 4.0 else (48 if pos < 8.0 else int(48 - ((pos - 8.0) / 4.0) * 28))
                phase = "INHALE" if pos < 4.0 else ("HOLD" if pos < 8.0 else "EXHALE")
                cv2.circle(frame, (cx, cy), radius, (0, 230, 255), 2)
                cv2.putText(frame, phase, (cx - 26, cy + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 2)

            # Multi-ROI Filtered Pulse Waveform
            if len(filtered_wave) > 15:
                norm_w = filtered_wave / (np.std(filtered_wave) + 1e-5)
                tail = norm_w[-100:]
                wave_pts = [
                    [int(w - 205 + (i / 100.0) * 190), int(h - 142 - np.clip(v * 7.5, -16, 16))]
                    for i, v in enumerate(tail)
                ]
                cv2.polylines(frame, [np.array(wave_pts, dtype=np.int32)], False, (0, 240, 180), 2)
                cv2.putText(frame, "CHEEK+NOSE+FOREHEAD rPPG", (w - 205, h - 162),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 240, 180), 1)

        # Draw Friendly Robotic Smiley Face
        draw_robot_smiley(
            frame, w - 62, 36,
            is_talking=(now_sec < speech_until),
            is_blinking=was_eye_closed,
            is_sleepy=(sleepiness_idx >= 50),
            is_stressed=(stress_index >= 65)
        )

        # Handle Head Nod / Shake Confirmation
        if detected_gesture in ("YES", "NO") and chat_state == "ASK_HYDRATION":
            feedback = (
                "Great. Taking a brief 20-second visual break will also help lower your drowsiness."
                if detected_gesture == "YES"
                else "Fatigue and dehydration compound quickly. Please drink a glass of water and stretch."
            )
            set_dialogue("DIAGNOSIS", feedback, "Press [R] for Full Checkup or [T] to enter symptoms")

        # Bottom Clinical Dialogue & On-Screen Input Box
        cv2.rectangle(frame, (12, h - 120), (w - 12, h - 8), (24, 28, 38), -1)
        border_col = (0, 240, 255) if typing_active else (0, 210, 180)
        cv2.rectangle(frame, (12, h - 120), (w - 12, h - 8), border_col, 2)

        if typing_active:
            cursor = "_" if int(now_sec * 3) % 2 == 0 else ""
            cv2.putText(frame, "ENTER SYMPTOMS OR HEALTH QUESTION (PRESS [ENTER]):", (24, h - 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1)
            cv2.putText(frame, f"> {input_text}{cursor}", (24, h - 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)
            cv2.putText(frame, "[ENTER] Submit to Maxwell  |  [ESC] Cancel", (24, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 210, 200), 1)
        else:
            lines = wrap_text(f"MAXWELL: {chat_prompt}", max_chars=66)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (24, h - 92 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
            cv2.putText(frame, chat_subtext, (24, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 225, 205), 1)

        cv2.imshow("Maxwell Health Companion", frame)
        key = cv2.waitKey(1) & 0xFF

        vitals_snapshot = {
            "bpm": smooth_bpm,
            "hrv": smooth_hrv,
            "resp": smooth_resp,
            "stress": stress_index,
            "sleepiness": sleepiness_idx,
            "sleep_label": sleep_label,
            "yawns": yawn_count,
            "dist_cm": dist_cm
        }

        # Keyboard Controls
        if typing_active:
            if key == 13:  # ENTER
                typing_active = False
                if input_text.strip():
                    query_text = input_text.strip()
                    input_text = ""
                    set_dialogue("DIAGNOSIS", f"Evaluating '{query_text}' alongside your vitals...", "Consulting clinical AI...")
                    query_gemini_maxwell(
                        query_text, vitals_snapshot,
                        callback=lambda msg: set_dialogue("DIAGNOSIS", msg, "Assessment complete | [T] Type | [R] Full Checkup")
                    )
            elif key == 27:  # ESC
                typing_active = False
                input_text = ""
            elif key in (8, 127):  # BACKSPACE
                input_text = input_text[:-1]
            elif 32 <= key <= 126 and len(input_text) < 58:
                input_text += chr(key)
        else:
            if key == ord("q"):
                break
            elif key == ord("c") and results.face_landmarks:
                flm = results.face_landmarks[0]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in flm]
                eye_span = dist(pts[33], pts[263]) + 1e-6
                baseline_brow_ratio = dist(pts[107], pts[336]) / eye_span
                eye_mid_y = (pts[33][1] + pts[263][1]) / 2.0
                baseline_pitch_ratio = (pts[1][1] - eye_mid_y) / eye_span
                yawn_count = 0
                set_dialogue("IDLE", "Baseline calibrated and yawn counter reset. Ready for health monitoring.",
                             "[T] Type Symptoms | [SPACE] Voice | [R] Full Vitals Checkup")
            elif key == ord("b"):
                breathing_mode = not breathing_mode
                breathing_start_time = now_sec
            elif key == ord("t"):
                typing_active = True
                input_text = ""
            elif key == ord("r"):
                set_dialogue("DIAGNOSIS", "Running comprehensive multi-vital health assessment...", "Querying Gemini...")
                query_gemini_maxwell(
                    "Perform a proactive general wellness checkup based on my current biometric readings.",
                    vitals_snapshot,
                    callback=lambda msg: set_dialogue("DIAGNOSIS", msg, "Full checkup complete | [T] Log symptoms")
                )
            elif key == ord(" "):
                listen_to_speech(vitals_snapshot)

cap.release()
cv2.destroyAllWindows()

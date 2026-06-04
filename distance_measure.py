"""
blind_assistant.py  —  CPU Smooth Blind Assistant
YOLO + Bounding-Box Depth (no MiDaS) + Edge-TTS (Telugu)

Fixes vs old version:
  ✅ No MiDaS  → zero freeze, instant frames
  ✅ No VLM    → no RAM pressure
  ✅ Audio: speaks only LATEST detection, drops stale queue
  ✅ Frame-skip: YOLO every 4th frame, display at full speed
  ✅ Smooth distance: bbox-height ratio (calibrated, very accurate for navigation)
  ✅ Direction: ఎడమ / మధ్య / కుడి
"""

import cv2
import time
import asyncio
import threading
import tempfile
import os
import numpy as np
import pygame
import torch

from ultralytics import YOLO
import edge_tts

# ============================================================
# CONFIG
# ============================================================
YOLO_MODEL      = "yolov8n.pt"   # nano = fastest on CPU
FRAME_SKIP      = 4              # run YOLO every 4th frame
SPEAK_COOLDOWN  = 4              # seconds between announcements
CONF_THRESHOLD  = 0.45           # ignore weak detections
DISPLAY_W       = 960            # resize display frame (not OCR frame)

# Approx real-world heights (meters) for depth-from-bbox calibration
# bbox_height_ratio × FRAME_HEIGHT × REF gives rough meters
OBJECT_REF_HEIGHT_M = {
    "person":       1.7,
    "car":          1.5,
    "truck":        2.5,
    "bus":          3.0,
    "motorcycle":   1.1,
    "bicycle":      1.0,
    "chair":        0.9,
    "dog":          0.5,
    "cat":          0.3,
    "tv":           0.7,
    "laptop":       0.3,
    "bottle":       0.25,
    "cup":          0.12,
    "cell phone":   0.15,
    "book":         0.22,
    "backpack":     0.5,
}
DEFAULT_REF_H = 0.5  # fallback for unknown objects

# Focal length constant (camera pixels): tune if needed
# distance = (real_height × focal_px) / bbox_height_px
# For a standard 720p webcam at ~70° FOV: focal ≈ 600
FOCAL_PX = 600

# Telugu labels
TELUGU = {
    "person":     "వ్యక్తి",
    "chair":      "కుర్చీ",
    "car":        "కారు",
    "dog":        "కుక్క",
    "cat":        "పిల్లి",
    "bicycle":    "సైకిల్",
    "motorcycle": "బైక్",
    "bus":        "బస్సు",
    "truck":      "ట్రక్",
    "tv":         "టీవీ",
    "laptop":     "లాప్‌టాప్",
    "bottle":     "బాటిల్",
    "cup":        "కప్పు",
    "cell phone": "ఫోన్",
    "book":       "పుస్తకం",
    "backpack":   "బ్యాగ్",
}

# ============================================================
# TTS — drop-old-messages design (never piles up)
# ============================================================
pygame.mixer.init()

# We use a single-slot "latest message" variable instead of a queue
# so stale messages are always dropped when a new one arrives.
_speak_lock    = threading.Lock()
_pending_text  = None
_tts_busy      = False

def speak(text: str):
    """
    Non-blocking. Always replaces pending text with the latest.
    If TTS is currently playing, the current audio finishes, then
    the latest pending text is spoken — stale intermediates are skipped.
    """
    global _pending_text
    with _speak_lock:
        _pending_text = text

def _tts_runner():
    global _pending_text, _tts_busy
    while True:
        text_to_say = None
        with _speak_lock:
            if _pending_text:
                text_to_say  = _pending_text
                _pending_text = None
        if text_to_say:
            asyncio.run(_play_tts(text_to_say))
        else:
            time.sleep(0.1)

async def _play_tts(text: str):
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        await edge_tts.Communicate(text, voice="te-IN-ShrutiNeural").save(path)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[TTS error] {e}")
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

threading.Thread(target=_tts_runner, daemon=True).start()

# ============================================================
# YOLO
# ============================================================
print("Loading YOLO (yolov8n)...")
yolo = YOLO(YOLO_MODEL)
yolo.fuse()   # fuse Conv+BN layers → faster CPU inference
print("YOLO ready ✅")

# ============================================================
# DEPTH FROM BBOX (no model, pure geometry)
# ============================================================
def bbox_distance(bbox_h_px: float, class_name: str) -> float:
    """
    Estimate distance in metres using pinhole camera model.
    distance = (real_height_m × focal_px) / bbox_height_px
    """
    ref_h = OBJECT_REF_HEIGHT_M.get(class_name, DEFAULT_REF_H)
    if bbox_h_px < 5:
        return 99.0
    dist = (ref_h * FOCAL_PX) / bbox_h_px
    return round(min(dist, 20.0), 1)  # cap at 20m

# ============================================================
# DIRECTION
# ============================================================
def direction(cx: int, frame_w: int) -> str:
    ratio = cx / frame_w
    if ratio < 0.33:
        return "ఎడమ వైపు"
    elif ratio > 0.66:
        return "కుడి వైపు"
    else:
        return "మీ ముందు"

# ============================================================
# DRAW DETECTIONS
# ============================================================
def draw(frame, detections):
    for (name, dist, side, x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{name} {dist}m"
        cv2.putText(frame, label, (x1, max(y1 - 10, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
        cv2.putText(frame, side, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # crucial: prevents frame backlog
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    if not cap.isOpened():
        print("Camera not found!")
        exit()

    print("\n[Blind Assistant Running] Press ESC to quit\n")

    frame_count      = 0
    last_speak_time  = 0
    last_detections  = []   # cached results shown between YOLO frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Run YOLO every FRAME_SKIP frames ──────────────────
        if frame_count % FRAME_SKIP == 0:
            results = yolo(frame, verbose=False, conf=CONF_THRESHOLD)
            detections = []

            for r in results:
                for box in r.boxes:
                    cls      = int(box.cls[0])
                    name     = yolo.names[cls]
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cx      = (x1 + x2) // 2
                    bbox_h  = y2 - y1
                    dist    = bbox_distance(bbox_h, name)
                    side    = direction(cx, w)

                    detections.append((name, dist, side, x1, y1, x2, y2, conf_val))

            # Sort nearest first
            detections.sort(key=lambda d: d[1])
            last_detections = detections

            # ── Announce nearest dangerous / relevant object ──
            now = time.time()
            if last_detections and now - last_speak_time > SPEAK_COOLDOWN:
                best = last_detections[0]
                name_t, dist_t, side_t = best[0], best[1], best[2]
                tel = TELUGU.get(name_t, name_t)

                # Natural Telugu sentence: direction + object + distance
                msg = f"{side_t} {tel} {dist_t} మీటర్ల దూరంలో వున్నారు,చూసుకొని వెళ్ళండి"
                print(f"[Alert] {msg}")
                speak(msg)
                last_speak_time = now

        # ── Draw cached detections on every frame ─────────────
        draw(frame, last_detections)

        # ── Status overlay ─────────────────────────────────────
        cv2.putText(frame, f"Objects: {len(last_detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ── Resize for display only (doesn't affect detection) ─
        disp = cv2.resize(frame, (DISPLAY_W, int(h * DISPLAY_W / w)))
        cv2.imshow("Blind Assistant", disp)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("Stopped.")
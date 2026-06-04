"""
AI Telugu Face Recognition
--------------------------
FLOW:
  Press S → says "కెమెరా ముందు ఉండండి" → captures 5 photos → type name in terminal → saves
  Recognition → known: says name every 2 sec | unknown: says alert every 2 sec

INSTALL:
  pip install face_recognition opencv-python numpy edge-tts pygame
"""

import cv2
import os
import face_recognition
import numpy as np
import asyncio
import edge_tts
import pygame
import threading
import time
import uuid
import pickle
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

SAVE_DIR        = "saved_faces"
DB_FILE         = "face_db.pkl"
CAPTURE_PHOTOS  = 5
CAPTURE_GAP     = 0.5
TOLERANCE       = 0.50
SPEAK_INTERVAL  = 2.0
FRAME_SCALE     = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════

def to_rgb(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════

pygame.mixer.init()
_speaking      = False
_speaking_lock = threading.Lock()

def speak_telugu(text: str) -> None:
    global _speaking
    with _speaking_lock:
        if _speaking:
            return
        _speaking = True

    filename = f"_tmp_{uuid.uuid4().hex[:8]}.mp3"
    try:
        async def _gen():
            await edge_tts.Communicate(text, voice="te-IN-ShrutiNeural").save(filename)
        asyncio.run(_gen())
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[VOICE ERROR] {e}")
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except:
            pass
        with _speaking_lock:
            _speaking = False

def speak_async(text: str) -> None:
    threading.Thread(target=speak_telugu, args=(text,), daemon=True).start()

# ══════════════════════════════════════════════════════════════════
# FACE DATABASE
# ══════════════════════════════════════════════════════════════════

face_db: dict = {}

def load_db() -> None:
    global face_db
    if Path(DB_FILE).exists():
        with open(DB_FILE, "rb") as f:
            face_db = pickle.load(f)
        total = sum(len(v) for v in face_db.values())
        print(f"[DB] Loaded {len(face_db)} people, {total} encodings")
    else:
        face_db = {}
        print("[DB] No database found — fresh start")

def save_db() -> None:
    with open(DB_FILE, "wb") as f:
        pickle.dump(face_db, f)
    print(f"[DB] Saved {len(face_db)} people to {DB_FILE}")

def get_all_encodings() -> tuple:
    encodings, names = [], []
    for name, enc_list in face_db.items():
        for enc in enc_list:
            encodings.append(enc)
            names.append(name)
    return encodings, names

load_db()

# ══════════════════════════════════════════════════════════════════
# RECOGNITION WORKER
# ══════════════════════════════════════════════════════════════════

_rec_lock     = threading.Lock()
_rec_results  = []
_latest_small = [None]
_stop_worker  = threading.Event()
_pause_worker = threading.Event()   # pause recognition during save

def recognition_worker() -> None:
    known_encodings, known_names = get_all_encodings()
    print(f"[WORKER] Started with {len(known_names)} encodings")

    while not _stop_worker.is_set():
        if _pause_worker.is_set():
            time.sleep(0.1)
            continue

        frame_small = _latest_small[0]
        if frame_small is None:
            time.sleep(0.05)
            continue

        rgb_small = to_rgb(frame_small)

        try:
            locations = face_recognition.face_locations(rgb_small, model="hog")
        except Exception as e:
            print(f"[WORKER] Error: {e}")
            time.sleep(0.05)
            continue

        if not locations:
            with _rec_lock:
                _rec_results.clear()
            time.sleep(0.05)
            continue

        try:
            encodings = face_recognition.face_encodings(rgb_small, locations)
        except Exception as e:
            print(f"[WORKER] Encoding error: {e}")
            time.sleep(0.05)
            continue

        results = []
        for encoding, location in zip(encodings, locations):
            name, confidence = "Unknown", 0.0
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                min_idx   = int(np.argmin(distances))
                min_dist  = float(distances[min_idx])
                if min_dist <= TOLERANCE:
                    name       = known_names[min_idx]
                    confidence = round((1.0 - min_dist) * 100, 1)
            results.append((name, confidence, *location))

        with _rec_lock:
            _rec_results.clear()
            _rec_results.extend(results)

        known_encodings, known_names = get_all_encodings()
        time.sleep(0.05)

# ══════════════════════════════════════════════════════════════════
# SAVE PERSON — runs in its own thread so camera stays open
# ══════════════════════════════════════════════════════════════════

_saving = False   # flag so main loop knows to skip keys

def save_person_thread(cap: cv2.VideoCapture) -> None:
    global _saving, face_db
    _saving = True
    _pause_worker.set()   # stop recognition worker during capture

    try:
        # Step 1 — Announce
        print("\n[SAVE] కెమెరా ముందు ఉండండి — capturing 5 photos...")
        speak_telugu("కెమెరా ముందు ఉండండి")   # blocking so it finishes before capture
        time.sleep(0.5)

        encodings_captured = []
        frames_captured    = []   # store actual frames to save as photos later
        attempt = 0

        # Step 2 — Capture 5 photos
        while len(encodings_captured) < CAPTURE_PHOTOS and attempt < 50:
            attempt += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            rgb = to_rgb(frame)
            locations = face_recognition.face_locations(rgb, model="hog")

            if not locations:
                print(f"[SAVE] ⚠️  No face detected (attempt {attempt})")
                display = frame.copy()
                cv2.putText(display, "No face — move closer",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("AI Telugu Face Assistant", display)
                cv2.waitKey(1)
                time.sleep(0.3)
                continue

            enc_list = face_recognition.face_encodings(rgb, locations)
            if not enc_list:
                continue

            encodings_captured.append(enc_list[0])
            frames_captured.append(frame.copy())   # save this frame
            count = len(encodings_captured)
            print(f"[SAVE] Photo {count}/{CAPTURE_PHOTOS} captured")

            # Visual flash
            display = frame.copy()
            top, right, bottom, left = locations[0]
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(display, f"Capturing {count}/{CAPTURE_PHOTOS}",
                        (left, max(top - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("AI Telugu Face Assistant", display)
            cv2.waitKey(1)
            time.sleep(CAPTURE_GAP)

        if not encodings_captured:
            print("[SAVE] No face captured. Try again.")
            speak_async("ముఖం కనుగొనబడలేదు మళ్ళీ ప్రయత్నించండి")
            return

        # Step 3 — Ask name in terminal
        print(f"\n[SAVE] {len(encodings_captured)} photos captured!")
        print("[SAVE] >>> Terminal లో పేరు టైప్ చేయండి (type name and press Enter): ", end="", flush=True)
        person_name = input().strip().lower()

        if not person_name:
            print("[SAVE] No name entered — discarding.")
            return

        # Step 4 — Create saved_faces/<name>/ folder and save all 5 photos
        person_dir = os.path.join(SAVE_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = int(time.time())
        for i, frm in enumerate(frames_captured):
            img_path = os.path.join(person_dir, f"{person_name}_{timestamp}_{i+1}.jpg")
            cv2.imwrite(img_path, frm)
            print(f"[SAVE] 🖼️  Saved photo {i+1} → {img_path}")

        # Save encodings to DB
        if person_name not in face_db:
            face_db[person_name] = []
        face_db[person_name].extend(encodings_captured)

        save_db()
        print(f"[SAVE] '{person_name}' saved successfully — {len(frames_captured)} photos in saved_faces/{person_name}/")
        speak_async(f"{person_name} సేవ్ అయింది")

    finally:
        _pause_worker.clear()   # resume recognition
        _saving = False

# ══════════════════════════════════════════════════════════════════
# VOICE — speak every 2 seconds
# ══════════════════════════════════════════════════════════════════

_last_speak_time: dict = {}

def handle_voice(names_in_frame: set) -> None:
    now = time.time()
    for name in names_in_frame:
        last = _last_speak_time.get(name, 0.0)
        if now - last >= SPEAK_INTERVAL:
            _last_speak_time[name] = now
            if name == "Unknown":
                print("[VOICE] గుర్తు తెలియని వ్యక్తి")
                speak_async("గుర్తు తెలియని వ్యక్తి")
            else:
                print(f"[VOICE] {name}")
                speak_async(f"{name} వస్తున్నారు")

    gone = set(_last_speak_time.keys()) - names_in_frame
    for name in gone:
        del _last_speak_time[name]

# ══════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════

def draw_face(frame, top, right, bottom, left, name, confidence, scale):
    top    = int(top    / scale)
    right  = int(right  / scale)
    bottom = int(bottom / scale)
    left   = int(left   / scale)
    color  = (0, 220, 0) if name != "Unknown" else (0, 0, 220)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = f"{name}" if name == "Unknown" else f"{name}  {confidence}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (left, top - th - 14), (left + tw + 10, top), color, -1)
    cv2.putText(frame, label, (left + 5, top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

def draw_hud(frame, fps, face_count, people_names, saving=False):
    h, w = frame.shape[:2]
    F    = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (0, 0), (w, 42), (15, 15, 15), -1)
    cv2.putText(frame, "AI Telugu Face Recognition", (10, 28), F, 0.7, (0, 220, 220), 2)
    cv2.putText(frame, f"FPS:{fps:.0f}",      (w - 110, 28), F, 0.6, (100, 255, 100), 1)
    cv2.putText(frame, f"Faces:{face_count}", (w - 210, 28), F, 0.6, (200, 200, 200), 1)
    cv2.rectangle(frame, (0, h - 36), (w, h), (15, 15, 15), -1)
    known = [p for p in people_names if p != "Unknown"]
    if known:
        cv2.putText(frame, "Visible: " + ", ".join(known),
                    (10, h - 12), F, 0.55, (0, 220, 0), 1)
    hint = "SAVING... type name in terminal" if saving else "S=Save  Q=Quit"
    color = (0, 200, 255) if saving else (160, 160, 160)
    cv2.putText(frame, hint, (w - 320, h - 12), F, 0.55, color, 1)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═"*55)
    print("  AI Telugu Face Recognition")
    print("═"*55)
    print("  S → stand in front → auto 5 photos → type name")
    print("  Q → Quit")
    print("═"*55 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    worker = threading.Thread(target=recognition_worker, daemon=True, name="FaceWorker")
    worker.start()

    frame_count = 0
    fps         = 0.0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps       = 30.0 / max(time.time() - fps_timer, 1e-6)
            fps_timer = time.time()

        # Feed recognition worker
        if not _saving:
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            _latest_small[0] = small

        # Draw results
        with _rec_lock:
            results = list(_rec_results)

        names_this_frame = set()
        for item in results:
            name, confidence, top, right, bottom, left = item
            draw_face(frame, top, right, bottom, left, name, confidence, FRAME_SCALE)
            names_this_frame.add(name)

        if not _saving:
            handle_voice(names_this_frame)

        draw_hud(frame, fps, len(results), names_this_frame, saving=_saving)
        cv2.imshow("AI Telugu Face Assistant", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Press S — start save in background thread (keeps camera & window alive)
        if key == ord("s") and not _saving:
            threading.Thread(
                target=save_person_thread, args=(cap,), daemon=True
            ).start()

    _stop_worker.set()
    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye.")

if __name__ == "__main__":
    main()
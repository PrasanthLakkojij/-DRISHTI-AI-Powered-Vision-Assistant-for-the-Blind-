
"""
AI Telugu Place Recognition using CLIP
----------------------------------------
FEATURES:
✅ Live webcam
✅ CLIP understands scenes semantically
✅ No training needed — just add place names!
✅ Recognizes: kitchen, bedroom, bathroom, road, office, classroom etc
✅ Press S → save a custom place with 5 photos
✅ Telugu voice speaks "ఇది kitchen స్థలం"
✅ Speaks only when place changes

pip install torch torchvision
pip install transformers
pip install opencv-python numpy
pip install edge-tts pygame pillow
"""

import cv2
import os
import numpy as np
import asyncio
import edge_tts
import pygame
import threading
import time
import uuid
from PIL import Image

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

SAVE_DIR       = "saved_places"
CONF_THRESHOLD = 0.30      # minimum CLIP confidence to confirm a place
SPEAK_COOLDOWN = 6.0       # seconds before repeating same place
CHECK_EVERY    = 2.0       # run CLIP every 2 seconds (heavy model)

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────

pygame.mixer.init()
_speaking = False


def speak_telugu(text: str) -> None:
    global _speaking
    if _speaking:
        return
    _speaking = True
    filename = f"{uuid.uuid4()}.mp3"
    try:
        async def _make():
            await edge_tts.Communicate(
                text, voice="te-IN-ShrutiNeural"
            ).save(filename)
        asyncio.run(_make())
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[VOICE ERROR] {e}")
    finally:
        if os.path.exists(filename):
            try: os.remove(filename)
            except: pass
        _speaking = False


# ─────────────────────────────────────────
# LOAD CLIP
# ─────────────────────────────────────────

print("[CLIP] Loading model... (first time takes 1-2 min)")

from transformers import CLIPProcessor, CLIPModel
import torch

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("[CLIP] Model ready ✅")


# ─────────────────────────────────────────
# PLACE LABELS
# Add or remove any place names here!
# CLIP understands these without training.
# ─────────────────────────────────────────

DEFAULT_PLACES = [
    "kitchen",
    "bedroom",
    "bathroom",
    "living room",
    "dining room",
    "office",
    "classroom",
    "road",
    "corridor",
    "staircase",
    "entrance",
    "garden",
    "garage",
]

# Telugu translations for default places
TELUGU_NAMES = {
    "kitchen":      "వంటగది",
    "bedroom":      "పడక గది",
    "bathroom":     "బాత్రూమ్",
    "living room":  "హాలు",
    "dining room":  "భోజన గది",
    "office":       "కార్యాలయం",
    "classroom":    "తరగతి గది",
    "road":         "రోడ్డు",
    "corridor":     "వరండా",
    "staircase":    "మెట్లు",
    "entrance":     "ముఖద్వారం",
    "garden":       "తోట",
    "garage":       "గరాజ్",
}


# ─────────────────────────────────────────
# SAVED CUSTOM PLACES (from S key)
# Each saved place stores 5 reference images.
# CLIP compares current frame against them.
# ─────────────────────────────────────────

custom_places: dict = {}   # name → list of PIL images


def load_custom_places() -> None:
    """Load saved place photos from disk."""
    custom_places.clear()
    for place_name in os.listdir(SAVE_DIR):
        place_dir = os.path.join(SAVE_DIR, place_name)
        if not os.path.isdir(place_dir):
            continue
        images = []
        for img_file in sorted(os.listdir(place_dir)):
            if not img_file.endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(place_dir, img_file))
            if img is not None:
                images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if images:
            custom_places[place_name] = images
            print(f"[LOADED] Custom place '{place_name}' — {len(images)} photos")
    print(f"[DB] Custom places loaded: {len(custom_places)}")


load_custom_places()


# ─────────────────────────────────────────
# CLIP INFERENCE
# ─────────────────────────────────────────

def clip_match_labels(pil_image: Image.Image, labels: list[str]) -> tuple[str, float]:
    """
    Run CLIP zero-shot classification.
    Returns (best_label, confidence_score).
    """
    prompts = [f"a photo of a {label}" for label in labels]

    inputs = clip_processor(
        text=prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs    = clip_model(**inputs)
        logits     = outputs.logits_per_image       # shape: (1, num_labels)
        probs      = logits.softmax(dim=1)[0]       # shape: (num_labels,)

    best_idx   = probs.argmax().item()
    best_label = labels[best_idx]
    best_conf  = float(probs[best_idx])

    return best_label, best_conf


def clip_match_custom(pil_image: Image.Image) -> tuple[str, float]:
    """
    Match current frame against saved custom place photos using CLIP embeddings.
    Returns (best_place_name, similarity_score).
    """
    if not custom_places:
        return "", 0.0

    # Get embedding of current frame
    inputs_img = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        frame_emb = clip_model.get_image_features(**inputs_img)
        frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True)

    best_name  = ""
    best_score = 0.0

    for place_name, ref_images in custom_places.items():
        scores = []
        for ref_img in ref_images:
            inputs_ref = clip_processor(images=ref_img, return_tensors="pt")
            with torch.no_grad():
                ref_emb = clip_model.get_image_features(**inputs_ref)
                ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
            # cosine similarity
            sim = float((frame_emb * ref_emb).sum())
            scores.append(sim)
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_name  = place_name

    # cosine similarity > 0.80 means good match
    if best_score >= 0.80:
        return best_name, best_score
    return "", best_score


def recognise_place(pil_image: Image.Image) -> tuple[str, float, str]:
    """
    Full recognition pipeline:
    1. Check custom saved places first (higher priority)
    2. Fall back to default CLIP label matching

    Returns (place_name, confidence, source)
    source = "custom" or "default"
    """
    # ── Step 1: Custom places ──────────────────────────────────────────
    if custom_places:
        custom_name, custom_score = clip_match_custom(pil_image)
        if custom_name:
            return custom_name, custom_score, "custom"

    # ── Step 2: Default CLIP labels ───────────────────────────────────
    label, conf = clip_match_labels(pil_image, DEFAULT_PLACES)
    if conf >= CONF_THRESHOLD:
        return label, conf, "default"

    return "", 0.0, ""


# ─────────────────────────────────────────
# SAVE A NEW PLACE
# ─────────────────────────────────────────

def save_place(cap: cv2.VideoCapture, place_name: str) -> None:
    place_dir = os.path.join(SAVE_DIR, place_name)
    os.makedirs(place_dir, exist_ok=True)

    existing = len([f for f in os.listdir(place_dir) if f.endswith(".jpg")])

    print(f"\n[SAVE] Capturing 5 photos of '{place_name}'")
    print("[SAVE] Slowly move camera to cover different angles...")

    saved = 0
    for i in range(5):
        countdown = 2
        print(f"[SAVE] Photo {i+1}/5 in {countdown} seconds...")
        time.sleep(countdown)

        ret, frame = cap.read()
        if not ret:
            print(f"[SAVE] ⚠️  Camera failed for photo {i+1}")
            continue

        path = os.path.join(place_dir, f"{existing+i}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
        print(f"[SAVE] ✅ Photo {i+1} saved")

        # flash on screen
        cv2.putText(frame, f"Saved {i+1}/5", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("AI Telugu Place Assistant", frame)
        cv2.waitKey(200)

    print(f"[SAVE] ✅ Done! {saved}/5 photos saved for '{place_name}'")

    # Reload
    load_custom_places()

    threading.Thread(
        target=speak_telugu,
        args=(f"{place_name} స్థలం సేవ్ అయింది",),
        daemon=True,
    ).start()


# ─────────────────────────────────────────
# BACKGROUND RECOGNITION THREAD
# Runs CLIP every 2 seconds off the main thread
# so webcam never freezes
# ─────────────────────────────────────────

_result_lock  = threading.Lock()
_result       = {"place": "", "conf": 0.0, "source": ""}
_latest_frame = [None]
_stop_bg      = threading.Event()


def bg_recognition_worker() -> None:
    while not _stop_bg.is_set():
        time.sleep(CHECK_EVERY)
        frame = _latest_frame[0]
        if frame is None:
            continue
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            place, conf, source = recognise_place(pil_img)
            with _result_lock:
                _result["place"]  = place
                _result["conf"]   = conf
                _result["source"] = source
        except Exception as e:
            print(f"[CLIP ERROR] {e}")


if __name__ == "__main__":
    bg_thread = threading.Thread(
        target=bg_recognition_worker, daemon=True, name="CLIPWorker"
    )
    bg_thread.start()

    # ─────────────────────────────────────────
    # WEBCAM MAIN LOOP
    # ─────────────────────────────────────────

    cap = cv2.VideoCapture(0)

    print("\n✅ Webcam started")
    print("   Press S  → save a new place")
    print("   Press Q  → quit\n")

    last_spoken_place = ""
    last_spoken_time  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _latest_frame[0] = frame
        display = frame.copy()
        now     = time.time()

        # ── Get latest CLIP result ───────────────────────────────────────────
        with _result_lock:
            place  = _result["place"]
            conf   = _result["conf"]
            source = _result["source"]

        # ── Telugu name lookup ───────────────────────────────────────────────
        if place:
            te_name = TELUGU_NAMES.get(place, place)   # use English if no Telugu
            label   = f"ఇది {te_name} స్థలం  ({conf*100:.0f}%)"
            color   = (0, 210, 0)

            # Speak when place changes or cooldown passed
            if (
                place != last_spoken_place
                or now - last_spoken_time > SPEAK_COOLDOWN
            ):
                last_spoken_place = place
                last_spoken_time  = now
                speak_text        = f"ఇది {te_name} స్థలం"
                print(f"[PLACE] {speak_text}  conf={conf*100:.0f}%  src={source}")
                threading.Thread(
                    target=speak_telugu,
                    args=(speak_text,),
                    daemon=True,
                ).start()
        else:
            label = "స్థలం గుర్తు తెలియలేదు"
            color = (0, 60, 200)
            if last_spoken_place:
                last_spoken_place = ""

        # ── HUD bar ──────────────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (display.shape[1], 55), (15, 15, 15), -1)
        cv2.putText(display, label, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        # ── Show saved places list on screen ─────────────────────────────────
        if custom_places:
            saved_text = "Saved: " + ", ".join(custom_places.keys())
            cv2.putText(display, saved_text,
                        (10, display.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Instructions ──────────────────────────────────────────────────────
        cv2.putText(display, "S=Save Place  Q=Quit",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        cv2.imshow("AI Telugu Place Assistant", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            name = input("\n[SAVE] Place name (e.g. kitchen, my room): ").strip()
            if not name:
                print("[SAVE] ❌ No name. Skipping.")
                continue
            save_place(cap, name)

    # ─────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────

    _stop_bg.set()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye.")

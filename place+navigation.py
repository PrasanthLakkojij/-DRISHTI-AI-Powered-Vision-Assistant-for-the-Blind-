"""
╔══════════════════════════════════════════════════════════════════╗
║   DRISHTI — COMBINED PLACE SAVER + VISUAL NAVIGATION            ║
║                                                                  ║
║   CONTROLS:                                                      ║
║   S = Save current place (captures 5 photos)                    ║
║   W = Enter navigation mode (choose destination)                ║
║   R = Reset / change destination (during navigation)            ║
║   Q = Quit                                                       ║
║                                                                  ║
║   INSTALL:                                                       ║
║   pip install torch torchvision transformers                     ║
║   pip install opencv-contrib-python numpy pillow                 ║
║   pip install edge-tts pygame timm                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import asyncio, cv2, os, threading, time, uuid
import numpy as np
import pygame
from pathlib import Path
from PIL import Image

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

PLACES_DIR        = "saved_places"
CLIP_CHECK_EVERY  = 1.5        # seconds between CLIP inference
ORB_MIN_MATCHES   = 8
ARRIVAL_THRESHOLD = 0.82       # CLIP similarity to say "arrived"
SPEAK_COOLDOWN    = 3.5
DEPTH_SCALE       = 5.0
CONF_THRESHOLD    = 0.30       # min CLIP confidence for place recognition HUD

os.makedirs(PLACES_DIR, exist_ok=True)

# Telugu direction messages
NAV_MSG = {
    "left":      "ఎడమవైపు తిరగండి",
    "right":     "కుడివైపు తిరగండి",
    "forward":   "ముందుకు వెళ్ళండి",
    "arrived":   "మీరు గమ్యానికి చేరుకున్నారు",
    "searching": "స్థలం వెతుకుతున్నాం",
    "obstacle":  "ముందు అడ్డంకి ఉంది జాగ్రత్త",
}

DEFAULT_PLACES = [
    "kitchen", "bedroom", "bathroom", "living room", "dining room",
    "office", "classroom", "road", "corridor", "staircase",
    "entrance", "garden", "garage",
]

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
    fname = f"_nav_{uuid.uuid4().hex[:6]}.mp3"
    try:
        async def _g():
            import edge_tts
            await edge_tts.Communicate(
                text, voice="te-IN-ShrutiNeural"
            ).save(fname)
        asyncio.run(_g())
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.04)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[VOICE] {e}")
    finally:
        try:
            if os.path.exists(fname):
                os.remove(fname)
        except:
            pass
        with _speaking_lock:
            _speaking = False


def speak_bg(text: str) -> None:
    threading.Thread(target=speak_telugu, args=(text,), daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# LOAD CLIP
# ══════════════════════════════════════════════════════════════════

print("[CLIP] Loading model... (first time takes 1-2 min)")

import torch
from transformers import CLIPProcessor, CLIPModel

_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()

print("[CLIP] Model ready ✅")

# ══════════════════════════════════════════════════════════════════
# LOAD MiDaS DEPTH ESTIMATOR
# ══════════════════════════════════════════════════════════════════

print("[MiDaS] Loading depth model...")
try:
    midas_dir = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    loaded_locally = False

    if os.path.exists(midas_dir):
        try:
            import sys as _sys
            if midas_dir not in _sys.path:
                _sys.path.insert(0, midas_dir)

            from midas.midas_net_custom import MidasNet_small
            from torchvision.transforms import Compose
            from midas.transforms import Resize, NormalizeImage, PrepareForNet

            _midas = MidasNet_small(
                path=None, features=64, backbone="efficientnet_lite3",
                exportable=True, non_negative=True, blocks={'expand': True}
            )
            checkpoint_path = os.path.expanduser(
                "~/.cache/torch/hub/checkpoints/midas_v21_small_256.pt"
            )
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                _midas.load_state_dict(state_dict)
                _midas.eval()
                _midas_xfm = Compose([
                    lambda img: {"image": img / 255.0},
                    Resize(256, 256, resize_target=None, keep_aspect_ratio=True,
                           ensure_multiple_of=32, resize_method="upper_bound",
                           image_interpolation_method=cv2.INTER_CUBIC),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                    lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
                ])
                print("  [MiDaS] Loaded from local cache ✅")
                MIDAS_OK = True
                loaded_locally = True
            else:
                print(f"Checkpoint not found at {checkpoint_path}")
        except Exception as ex:
            print(f"Direct local MiDaS load failed: {ex}. Falling back...")

    if not loaded_locally:
        _midas     = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _midas_xfm = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
        _midas.eval()
        MIDAS_OK   = True
        print("[MiDaS] Ready ✅")
except Exception as e:
    print(f"[MiDaS] Not available ({e}) — depth disabled")
    MIDAS_OK = False

# ══════════════════════════════════════════════════════════════════
# ORB FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════

_orb     = cv2.ORB_create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
FLANN_INDEX_LSH = 6
_flann   = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_LSH, "table_number": 6,
     "key_size": 12, "multi_probe_level": 1},
    {"checks": 50},
)

# ══════════════════════════════════════════════════════════════════
# PLACE DATABASE
# ══════════════════════════════════════════════════════════════════

class PlaceDB:
    """Loads saved_places/<name>/*.jpg and precomputes CLIP + ORB features."""

    def __init__(self, root: str) -> None:
        self.root   = Path(root)
        self.places: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        self.places.clear()
        if not self.root.exists():
            print(f"[DB] ⚠️  '{self.root}' not found — create it first")
            return

        for place_dir in sorted(self.root.iterdir()):
            if not place_dir.is_dir():
                continue
            name  = place_dir.name
            imgs  = []; pils = []; kps = []; descs = []; embs = []

            for img_path in sorted(place_dir.glob("*.jpg")):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pil  = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                kp, desc = _orb.detectAndCompute(gray, None)
                if desc is not None:
                    kps.append(kp); descs.append(desc)

                inp = _clip_proc(images=pil, return_tensors="pt")
                with torch.no_grad():
                    emb = _clip_model.get_image_features(**inp)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embs.append(emb)
                imgs.append(bgr); pils.append(pil)

            if not imgs:
                continue

            avg_emb = torch.stack(embs).mean(dim=0)
            avg_emb = avg_emb / avg_emb.norm()

            self.places[name] = {
                "images": imgs, "pils": pils,
                "kps": kps, "descs": descs,
                "avg_emb": avg_emb, "all_embs": embs,
            }
            print(f"[DB] Loaded '{name}' — {len(imgs)} images")

        print(f"[DB] Total: {len(self.places)} places ✅")

    def reload(self) -> None:
        """Reload from disk (called after saving a new place)."""
        self._load()

    def names(self) -> list[str]:
        return list(self.places.keys())


print("\n[DB] Loading place database...")
DB = PlaceDB(PLACES_DIR)

if not DB.places:
    print("\n⚠️  No places in saved_places/ yet.")
    print("   Run the camera and press S to save a place first.\n")
else:
    print(f"[DB] Available places: {', '.join(DB.names())}\n")

# ══════════════════════════════════════════════════════════════════
# SAVE A NEW PLACE  (S key)
# ══════════════════════════════════════════════════════════════════

def save_place(cap: cv2.VideoCapture, place_name: str) -> None:
    place_dir = os.path.join(PLACES_DIR, place_name)
    os.makedirs(place_dir, exist_ok=True)

    existing = len([f for f in os.listdir(place_dir) if f.endswith(".jpg")])

    print(f"\n[SAVE] Capturing 5 photos of '{place_name}'")
    print("[SAVE] Slowly move camera to cover different angles...")

    saved = 0
    for i in range(5):
        print(f"[SAVE] Photo {i+1}/5 in 2 seconds...")
        time.sleep(2)

        ret, frame = cap.read()
        if not ret:
            print(f"[SAVE] ⚠️  Camera failed for photo {i+1}")
            continue

        path = os.path.join(place_dir, f"{existing + i}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
        print(f"[SAVE] ✅ Photo {i+1} saved")

        flash = frame.copy()
        cv2.putText(flash, f"Saved {i+1}/5", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("DRISHTI", flash)
        cv2.waitKey(200)

    print(f"[SAVE] ✅ Done! {saved}/5 photos saved for '{place_name}'")

    # Reload DB with the new place included
    DB.reload()
    print(f"[DB] Reloaded — available places: {', '.join(DB.names())}")

    speak_bg(f"{place_name} స్థలం సేవ్ అయింది")


# ══════════════════════════════════════════════════════════════════
# CLIP HELPERS
# ══════════════════════════════════════════════════════════════════

def clip_similarity(frame_bgr: np.ndarray, dest_name: str) -> float:
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = _clip_proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    dest_emb = DB.places[dest_name]["avg_emb"]
    return float((emb * dest_emb).sum())


def clip_match_labels(pil_image: Image.Image, labels: list[str]) -> tuple[str, float]:
    prompts = [f"a photo of a {label}" for label in labels]
    inputs  = _clip_proc(text=prompts, images=pil_image,
                         return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _clip_model(**inputs).logits_per_image
        probs  = logits.softmax(dim=1)[0]
    best_idx = probs.argmax().item()
    return labels[best_idx], float(probs[best_idx])


def clip_match_custom(pil_image: Image.Image) -> tuple[str, float]:
    if not DB.places:
        return "", 0.0
    inp = _clip_proc(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    best_name = ""; best_score = 0.0
    for name, data in DB.places.items():
        sim = float((emb * data["avg_emb"]).sum())
        if sim > best_score:
            best_score = sim; best_name = name
    return (best_name, best_score) if best_score >= 0.80 else ("", best_score)


def recognise_place(pil_image: Image.Image) -> tuple[str, float, str]:
    if DB.places:
        name, score = clip_match_custom(pil_image)
        if name:
            return name, score, "custom"
    label, conf = clip_match_labels(pil_image, DEFAULT_PLACES)
    if conf >= CONF_THRESHOLD:
        return label, conf, "default"
    return "", 0.0, ""


# ══════════════════════════════════════════════════════════════════
# ORB DIRECTION ESTIMATOR
# ══════════════════════════════════════════════════════════════════

def orb_direction(frame_bgr: np.ndarray, dest_name: str) -> dict:
    result = {"direction": "forward", "offset_x": 0, "offset_y": 0,
              "arrow_dst": None, "n_matches": 0, "confidence": 0.0}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp_cur, desc_cur = _orb.detectAndCompute(gray, None)
    if desc_cur is None or len(desc_cur) < 5:
        return result

    place = DB.places[dest_name]
    h, w  = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2

    best_matches = []; best_ref_kps = None

    for ref_desc, ref_kps in zip(place["descs"], place["kps"]):
        if ref_desc is None or len(ref_desc) < 5:
            continue
        try:
            raw  = _matcher.match(desc_cur, ref_desc)
            good = sorted(raw, key=lambda m: m.distance)[:50]
            good = [m for m in good if m.distance < 60]
            if len(good) > len(best_matches):
                best_matches = good; best_ref_kps = ref_kps
        except Exception:
            continue

    result["n_matches"] = len(best_matches)
    if len(best_matches) < ORB_MIN_MATCHES or best_ref_kps is None:
        return result

    result["confidence"] = min(1.0, len(best_matches) / 40.0)

    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    pts_ref = np.float32([best_ref_kps[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 5.0)

    if H is None:
        offset_x = float(np.mean(pts_cur[:,0,0])) - float(np.mean(pts_ref[:,0,0]))
        result["offset_x"]  = offset_x
        result["direction"] = "left" if offset_x < -30 else ("right" if offset_x > 30 else "forward")
        result["arrow_dst"] = (int(np.mean(pts_cur[:,0,0])), cy)
        return result

    ref_h, ref_w = DB.places[dest_name]["images"][0].shape[:2]
    ref_centre   = np.float32([[ref_w/2, ref_h/2]]).reshape(-1,1,2)
    try:
        proj = cv2.perspectiveTransform(ref_centre, H)
        px, py = int(proj[0,0,0]), int(proj[0,0,1])
    except Exception:
        px, py = cx, cy

    offset_x = px - cx; offset_y = py - cy
    result["offset_x"]  = offset_x
    result["offset_y"]  = offset_y
    result["arrow_dst"] = (max(30, min(w-30, px)), max(30, min(h-30, py)))

    THRESH = w * 0.12
    if abs(offset_x) < THRESH and abs(offset_y) < THRESH * 0.8:
        result["direction"] = "forward"
    elif offset_x < -THRESH:
        result["direction"] = "left"
    elif offset_x > THRESH:
        result["direction"] = "right"
    else:
        result["direction"] = "forward"

    return result


# ══════════════════════════════════════════════════════════════════
# MiDaS DEPTH
# ══════════════════════════════════════════════════════════════════

def estimate_depth_at(frame_bgr: np.ndarray, x: int, y: int) -> float:
    if not MIDAS_OK:
        return -1.0
    try:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = _midas_xfm(rgb)
        with torch.no_grad():
            pred = _midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame_bgr.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        depth_map = pred.cpu().numpy()
        h2, w2    = depth_map.shape
        raw       = float(depth_map[max(0, min(h2-1, y)), max(0, min(w2-1, x))])
        return round(DEPTH_SCALE / (raw / depth_map.max() + 1e-6), 1)
    except Exception:
        return -1.0


# ══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════

def draw_nav_arrow(frame, direction, arrow_dst, confidence):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    src = (cx, h - 80)
    if arrow_dst and confidence > 0.3:
        tip = arrow_dst
    elif direction == "left":
        tip = (cx - 180, cy)
    elif direction == "right":
        tip = (cx + 180, cy)
    else:
        tip = (cx, cy - 120)
    cv2.arrowedLine(frame, src, tip, (0, 0, 0), 12, tipLength=0.3)
    cv2.arrowedLine(frame, src, tip, (0, 255, 80), 6, tipLength=0.3)
    pulse = int(18 + 6 * abs(np.sin(time.time() * 3)))
    cv2.circle(frame, tip, pulse, (0, 255, 80), 3)
    cv2.circle(frame, tip, 6, (255, 255, 255), -1)


def draw_match_overlay(frame, dest_name, ref_thumb):
    if ref_thumb is None:
        return
    h, w = frame.shape[:2]
    tw, th = 160, 100
    thumb = cv2.resize(ref_thumb, (tw, th))
    roi = frame[10:10+th+10, w-tw-10:w-10]
    ov  = roi.copy()
    cv2.rectangle(ov, (0,0), (tw+10, th+10), (20,20,20), -1)
    cv2.addWeighted(ov, 0.6, roi, 0.4, 0, roi)
    frame[15:15+th, w-tw-5:w-5] = thumb
    cv2.putText(frame, f"Target: {dest_name}",
                (w-tw-5, 15+th+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)


def draw_hud_nav(frame, dest, sim, direction, n_matches, depth, confidence):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (0,0), (w, 50), (10,10,10), -1)
    status_color = (0,255,80) if sim > 0.75 else (0,180,255)
    cv2.putText(frame, f"Navigating to: {dest.upper()}", (10, 32), F, 0.8, status_color, 2)
    bar_x = w - 220
    cv2.putText(frame, "Match:", (bar_x, 22), F, 0.5, (180,180,180), 1)
    cv2.rectangle(frame, (bar_x+55, 10), (bar_x+160, 26), (50,50,50), -1)
    filled = int((bar_x+55) + sim * 105)
    bar_color = (0,255,80) if sim > ARRIVAL_THRESHOLD else (0,180,255)
    cv2.rectangle(frame, (bar_x+55, 10), (filled, 26), bar_color, -1)
    cv2.putText(frame, f"{sim*100:.0f}%", (bar_x+165, 22), F, 0.5, (220,220,220), 1)

    cv2.rectangle(frame, (0, h-48), (w, h), (10,10,10), -1)
    dir_te = {"left":"← ఎడమవైపు","right":"→ కుడివైపు",
              "forward":"↑ ముందుకు","arrived":"✓ చేరుకున్నారు"}.get(direction, "↑ ముందుకు")
    cv2.putText(frame, dir_te, (10, h-18), F, 0.85, (0,255,150), 2)
    cv2.putText(frame, f"ORB:{n_matches}pts", (w-300, h-18), F, 0.55, (180,180,180), 1)
    if depth > 0:
        cv2.putText(frame, f"Depth:{depth}m", (w-180, h-18), F, 0.55, (150,220,255), 1)
    cv2.putText(frame, "Q=Quit  R=Reset  S=Save Place", (w-260, h-36), F, 0.42, (120,120,120), 1)

    if direction == "arrived" or sim >= ARRIVAL_THRESHOLD:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-50), (w, h//2+50), (0,100,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"  {dest.upper()} చేరుకున్నారు!",
                    (w//2-200, h//2+15), F, 1.1, (0,255,100), 3)


def draw_hud_idle(frame, place, conf, source):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX
    if place:
        te_name = TELUGU_NAMES.get(place, place)
        label   = f"ఇది {te_name} స్థలం  ({conf*100:.0f}%)"
        color   = (0, 210, 0)
    else:
        label = "స్థలం గుర్తు తెలియలేదు"
        color = (0, 60, 200)
    cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.putText(frame, label, (10, 38), F, 0.85, color, 2)
    if DB.places:
        saved_text = "Saved: " + ", ".join(DB.names())
        cv2.putText(frame, saved_text, (10, h-35), F, 0.5, (180,180,180), 1)
    cv2.putText(frame, "S=Save Place  W=Navigate  Q=Quit",
                (10, h-12), F, 0.55, (150,150,150), 1)


def draw_feature_points(frame, dest_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps, _ = _orb.detectAndCompute(gray, None)
    if kps:
        for kp in kps[:30]:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)


# ══════════════════════════════════════════════════════════════════
# BACKGROUND WORKERS
# ══════════════════════════════════════════════════════════════════

# ── Idle recognition worker (shows place name on HUD) ─────────────
_idle_lock    = threading.Lock()
_idle_result  = {"place": "", "conf": 0.0, "source": ""}
_latest_frame = [None]
_idle_stop    = threading.Event()


def idle_recognition_worker() -> None:
    while not _idle_stop.is_set():
        time.sleep(CLIP_CHECK_EVERY)
        frame = _latest_frame[0]
        if frame is None:
            continue
        try:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            place, conf, source = recognise_place(pil)
            with _idle_lock:
                _idle_result["place"]  = place
                _idle_result["conf"]   = conf
                _idle_result["source"] = source
        except Exception as e:
            print(f"[IDLE CLIP] {e}")


# ── Navigation worker ─────────────────────────────────────────────
_nav_lock   = threading.Lock()
_nav_state  = {
    "sim": 0.0, "direction": "forward", "offset_x": 0,
    "arrow_dst": None, "n_matches": 0, "confidence": 0.0,
    "depth": -1.0, "arrived": False,
}
_nav_frame = [None]
_nav_dest  = [None]
_nav_stop  = threading.Event()


def nav_worker() -> None:
    last_speak_time = 0.0
    last_direction  = ""

    while not _nav_stop.is_set():
        time.sleep(CLIP_CHECK_EVERY)
        frame = _nav_frame[0]
        dest  = _nav_dest[0]
        if frame is None or dest is None or dest not in DB.places:
            continue
        try:
            sim      = clip_similarity(frame, dest)
            orb_res  = orb_direction(frame, dest)
            direction  = orb_res["direction"]
            arrow_dst  = orb_res["arrow_dst"]
            n_matches  = orb_res["n_matches"]
            confidence = orb_res["confidence"]
            offset_x   = orb_res["offset_x"]
            arrived    = sim >= ARRIVAL_THRESHOLD
            if arrived:
                direction = "arrived"

            depth = -1.0
            if arrow_dst and MIDAS_OK:
                depth = estimate_depth_at(frame, arrow_dst[0], arrow_dst[1])

            with _nav_lock:
                _nav_state["sim"]        = sim
                _nav_state["direction"]  = direction
                _nav_state["offset_x"]   = offset_x
                _nav_state["arrow_dst"]  = arrow_dst
                _nav_state["n_matches"]  = n_matches
                _nav_state["confidence"] = confidence
                _nav_state["depth"]      = depth
                _nav_state["arrived"]    = arrived

            now = time.time()
            if now - last_speak_time > SPEAK_COOLDOWN:
                if arrived:
                    speak_bg(f"{dest} చేరుకున్నారు")
                    last_speak_time = now
                elif direction != last_direction or now - last_speak_time > 6.0:
                    te_dir    = NAV_MSG.get(direction, "ముందుకు వెళ్ళండి")
                    depth_str = f" {depth:.0f} మీటర్లు" if depth > 0 else ""
                    speak_bg(f"{te_dir}{depth_str}")
                    last_speak_time = now
                    last_direction  = direction
        except Exception as e:
            print(f"[NAV WORKER] {e}")





# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# NON-BLOCKING INPUT QUEUE
# All input() calls run in a background thread so cv2.waitKey()
# never gets blocked and keys always register instantly.
# ══════════════════════════════════════════════════════════════════

import queue as _queue

_input_queue  = _queue.Queue()   # holds (tag, value) tuples
_input_busy   = threading.Event()  # set while input thread is waiting


def _ask(tag: str, prompt: str) -> None:
    """Run input() in background thread, post result to queue."""
    def _worker():
        _input_busy.set()
        val = input(prompt).strip()
        _input_queue.put((tag, val))
        _input_busy.clear()
    threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed!")
        return

    print("\n✅ Camera started")
    print("   S = Save a new place")
    print("   W = Enter navigation mode")
    print("   ESC = Exit nav → back to idle")
    print("   Q = Quit\n")

    # Start idle recognition worker
    idle_thread = threading.Thread(
        target=idle_recognition_worker, daemon=True, name="IdleWorker"
    )
    idle_thread.start()

    # State
    nav_mode          = False
    nav_dest          = ""
    ref_thumb         = None
    nav_thread        = None
    last_spoken_place = ""
    last_spoken_time  = 0.0
    arrived_time      = 0.0
    fps               = 0.0
    fps_timer         = time.time()
    frame_count       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        _latest_frame[0] = frame.copy()
        if nav_mode:
            _nav_frame[0] = frame.copy()

        if frame_count % 30 == 0:
            fps       = 30.0 / max(time.time() - fps_timer, 1e-6)
            fps_timer = time.time()

        display = frame.copy()
        now     = time.time()

        # ── Process any pending input answers ─────────────────────
        while not _input_queue.empty():
            tag, val = _input_queue.get_nowait()

            if tag == "save_name":
                if not val:
                    print("[SAVE] ❌ No name. Skipping.")
                else:
                    threading.Thread(
                        target=save_place, args=(cap, val), daemon=True
                    ).start()

            elif tag == "nav_dest":
                if val in DB.places:
                    nav_dest  = val
                    nav_mode  = True
                    ref_thumb = DB.places[val]["images"][0]
                    with _nav_lock:
                        _nav_state["arrived"]   = False
                        _nav_state["direction"] = "forward"
                        _nav_state["sim"]       = 0.0
                        _nav_state["n_matches"] = 0
                        _nav_state["confidence"]= 0.0
                        _nav_state["depth"]     = -1.0
                        _nav_state["arrow_dst"] = None
                    _nav_dest[0] = val
                    _nav_stop.clear()
                    if nav_thread is None or not nav_thread.is_alive():
                        nav_thread = threading.Thread(
                            target=nav_worker, daemon=True, name="NavWorker"
                        )
                        nav_thread.start()
                    arrived_time = 0.0
                    speak_bg(f"{val} వైపు నావిగేషన్ మొదలవుతుంది")
                    print(f"\n✅ Navigating to: {val}\n")
                else:
                    print(f"❌ '{val}' not found. Available: {', '.join(DB.names())}")
                    print("   Press W to try again.")

            elif tag == "reset_dest":
                if val in DB.places:
                    nav_dest     = val
                    _nav_dest[0] = val
                    ref_thumb    = DB.places[val]["images"][0]
                    arrived_time = 0.0
                    speak_bg(f"{val} వైపు నావిగేషన్ మొదలవుతుంది")
                    print(f"✅ Destination changed to: {val}")
                else:
                    print(f"❌ '{val}' not found")

        # ── NAV MODE ──────────────────────────────────────────────
        if nav_mode:
            with _nav_lock:
                state = dict(_nav_state)

            draw_feature_points(display, nav_dest)

            if not state["arrived"]:
                draw_nav_arrow(display, state["direction"],
                               state["arrow_dst"], state["confidence"])

            draw_match_overlay(display, nav_dest, ref_thumb)
            draw_hud_nav(display, nav_dest, state["sim"], state["direction"],
                         state["n_matches"], state["depth"], state["confidence"])

            # ── Auto-cycle back to idle after arrival ──────────────
            if state["arrived"]:
                if arrived_time == 0.0:
                    arrived_time = now
                    print(f"\n✅ Arrived at '{nav_dest}'! Returning to idle in 3 seconds...")

                if arrived_time != 0.0 and now - arrived_time >= 3.0:
                    nav_mode     = False
                    _nav_dest[0] = None
                    arrived_time = 0.0
                    with _nav_lock:
                        _nav_state["arrived"]   = False
                        _nav_state["direction"] = "forward"
                        _nav_state["sim"]       = 0.0
                    nav_dest  = ""
                    ref_thumb = None
                    print("[CYCLE] Back to idle. Press W to navigate again.\n")
            else:
                arrived_time = 0.0

        # ── IDLE MODE ─────────────────────────────────────────────
        else:
            with _idle_lock:
                place  = _idle_result["place"]
                conf   = _idle_result["conf"]
                source = _idle_result["source"]

            draw_hud_idle(display, place, conf, source)

            if place:
                te_name = TELUGU_NAMES.get(place, place)
                if place != last_spoken_place or now - last_spoken_time > 6.0:
                    last_spoken_place = place
                    last_spoken_time  = now
                    speak_bg(f"ఇది {te_name} స్థలం")
            else:
                last_spoken_place = ""

        # FPS
        cv2.putText(display, f"{fps:.0f}fps",
                    (display.shape[1]-75, display.shape[0]-52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.imshow("DRISHTI", display)
        key = cv2.waitKey(1) & 0xFF

        # ── Q: Quit ───────────────────────────────────────────────
        if key == ord("q"):
            break

        # ── S: Save place (only when input thread is free) ────────
        if key == ord("s") and not _input_busy.is_set():
            print()
            _ask("save_name", "[SAVE] Place name (e.g. kitchen, my room): ")

        # ── W: Navigate (only when input thread is free) ──────────
        if key == ord("w") and not nav_mode and not _input_busy.is_set():
            if not DB.places:
                print("\n⚠️  No saved places yet! Press S to save first.\n")
            else:
                print("\n" + "═"*60)
                print("  DRISHTI — VISUAL PLACE NAVIGATION  (GOD LEVEL)")
                print("═"*60)
                print(f"  Places available: {', '.join(DB.names())}")
                print("  Q = Quit   R = Reset destination")
                print("═"*60)
                print("\nAvailable places:")
                for i, name in enumerate(DB.names(), 1):
                    print(f"  {i}. {name}")
                _ask("nav_dest", "\nEnter destination name: ")

        # ── R: Reset destination ──────────────────────────────────
        if key == ord("r") and nav_mode and not _input_busy.is_set():
            print("\nAvailable places:")
            for i, name in enumerate(DB.names(), 1):
                print(f"  {i}. {name}")
            _ask("reset_dest", "New destination: ")

        # ── ESC: Exit nav → idle ──────────────────────────────────
        if key == 27 and nav_mode:
            nav_mode     = False
            nav_dest     = ""
            ref_thumb    = None
            arrived_time = 0.0
            _nav_dest[0] = None
            print("\n[NAV] Exited → back to idle. Press W to navigate again.\n")

    # ── Cleanup ───────────────────────────────────────────────────
    _idle_stop.set()
    _nav_stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("\nGoodbye.")


if __name__ == "__main__":
    main()
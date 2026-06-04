"""
╔══════════════════════════════════════════════════════════════════╗
║   DRISHTI — VISUAL PLACE NAVIGATION  (GOD LEVEL)                ║
║                                                                  ║
║   TECH STACK:                                                    ║
║   • CLIP  → semantic place recognition (where am I?)            ║
║   • ORB   → feature matching + direction estimation             ║
║   • MiDaS → real depth estimation (how far?)                    ║
║   • OpenCV homography → exact visual alignment arrows           ║
║   • edge-tts → Telugu voice navigation                          ║
║                                                                  ║
║   HOW IT WORKS:                                                  ║
║   1. Loads saved_places/ folder (your existing folder)          ║
║   2. You say/type destination: "kitchen"                        ║
║   3. CLIP finds which saved place looks like current frame      ║
║   4. ORB matches features → computes direction offset           ║
║   5. Homography tells: turn left / right / go forward           ║
║   6. MiDaS estimates distance in metres                         ║
║   7. Green arrow drawn on screen                                ║
║   8. Telugu voice: "ఎడమవైపు తిరగండి వంటగది దగ్గర ఉంది"          ║
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

PLACES_DIR        = "saved_places"       # your existing folder
CLIP_CHECK_EVERY  = 1.5                  # seconds between CLIP inference
ORB_MIN_MATCHES   = 8                    # minimum ORB matches for direction
ARRIVAL_THRESHOLD = 0.82                 # CLIP similarity to say "arrived"
SPEAK_COOLDOWN    = 3.5                  # seconds between nav instructions
DEPTH_SCALE       = 5.0                  # MiDaS depth → metres scaling factor

# Telugu direction messages
NAV_MSG = {
    "left":     "ఎడమవైపు తిరగండి",
    "right":    "కుడివైపు తిరగండి",
    "forward":  "ముందుకు వెళ్ళండి",
    "arrived":  "మీరు గమ్యానికి చేరుకున్నారు",
    "searching": "స్థలం వెతుకుతున్నాం",
    "obstacle": "ముందు అడ్డంకి ఉంది జాగ్రత్త",
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
            await edge_tts.Communicate(
                text, voice="te-IN-ShrutiNeural"
            ).save(fname)
        import edge_tts
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
            if os.path.exists(fname): os.remove(fname)
        except: pass
        with _speaking_lock:
            _speaking = False


def speak_bg(text: str) -> None:
    threading.Thread(target=speak_telugu, args=(text,), daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# LOAD CLIP
# ══════════════════════════════════════════════════════════════════

print("[CLIP] Loading... (first run takes ~60 sec)")
import torch
from transformers import CLIPProcessor, CLIPModel

_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()
print("[CLIP] Ready ✅")

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
            
            # Instantiate model directly
            _midas = MidasNet_small(
                path=None,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={'expand': True}
            )
            
            # Load cached weights directly
            checkpoint_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/midas_v21_small_256.pt")
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                _midas.load_state_dict(state_dict)
                _midas.eval()
                
                # Instantiate transforms Composer directly
                _midas_xfm = Compose([
                    lambda img: {"image": img / 255.0},
                    Resize(
                        256,
                        256,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                    lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
                ])
                
                print("  [MiDaS] Loaded DIRECTLY from local cache (bypassed PyTorch Hub) ✅")
                MIDAS_OK = True
                loaded_locally = True
            else:
                print(f"Checkpoint not found at {checkpoint_path}")
        except Exception as ex:
            print(f"Direct local MiDaS load failed: {ex}. Falling back to standard loaders...")
            
    if not loaded_locally:
        _midas      = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        _midas_xfm  = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
        _midas.eval()
        MIDAS_OK    = True
        print("[MiDaS] Ready ✅")
except Exception as e:
    print(f"[MiDaS] Not available ({e}) — depth disabled")
    MIDAS_OK    = False

# ══════════════════════════════════════════════════════════════════
# ORB FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════

_orb     = cv2.ORB_create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
FLANN_INDEX_LSH = 6
_flann   = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_LSH, "table_number": 6,
     "key_size": 12, "multi_probe_level": 1},
    {"checks": 50}
)

# ══════════════════════════════════════════════════════════════════
# PLACE DATABASE — loads your existing saved_places/ folder
# ══════════════════════════════════════════════════════════════════

class PlaceDB:
    """
    Loads all images from saved_places/<name>/*.jpg
    Precomputes:
      - CLIP embeddings (for recognition)
      - ORB keypoints + descriptors (for direction)
      - PIL images (for display)
    """

    def __init__(self, root: str) -> None:
        self.root    = Path(root)
        self.places  : dict[str, dict] = {}   # name → data
        self._load()

    def _load(self) -> None:
        if not self.root.exists():
            print(f"[DB] ⚠️  '{self.root}' not found — create it first")
            return

        for place_dir in sorted(self.root.iterdir()):
            if not place_dir.is_dir():
                continue
            name   = place_dir.name
            imgs   = []
            pils   = []
            kps    = []
            descs  = []
            embs   = []

            for img_path in sorted(place_dir.glob("*.jpg")):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pil  = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                # ORB
                kp, desc = _orb.detectAndCompute(gray, None)
                if desc is not None:
                    kps.append(kp)
                    descs.append(desc)

                # CLIP embedding
                inp = _clip_proc(images=pil, return_tensors="pt")
                with torch.no_grad():
                    emb = _clip_model.get_image_features(**inp)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embs.append(emb)

                imgs.append(bgr)
                pils.append(pil)

            if not imgs:
                continue

            # Average CLIP embedding for this place
            avg_emb = torch.stack(embs).mean(dim=0)
            avg_emb = avg_emb / avg_emb.norm()

            self.places[name] = {
                "images": imgs,
                "pils":   pils,
                "kps":    kps,
                "descs":  descs,
                "avg_emb": avg_emb,
                "all_embs": embs,
            }
            print(f"[DB] Loaded '{name}' — {len(imgs)} images")

        print(f"[DB] Total: {len(self.places)} places ✅")

    def names(self) -> list[str]:
        return list(self.places.keys())


print("\n[DB] Loading place database...")
DB = PlaceDB(PLACES_DIR)

if not DB.places:
    print("\n⚠️  No places found in saved_places/ — navigation disabled until places are saved")
    print("   Use 'save place <name>' command to save locations first")

print(f"[DB] Available places: {', '.join(DB.names()) if DB.places else 'none'}\n")


# ══════════════════════════════════════════════════════════════════
# CLIP RECOGNITION — where am I? how similar to destination?
# ══════════════════════════════════════════════════════════════════

def clip_similarity(frame_bgr: np.ndarray, dest_name: str) -> float:
    """
    Returns cosine similarity between current frame and destination place.
    Range: 0.0 (no match) → 1.0 (identical)
    """
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = _clip_proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    dest_emb = DB.places[dest_name]["avg_emb"]
    sim      = float((emb * dest_emb).sum())
    return sim


def clip_best_match(frame_bgr: np.ndarray) -> tuple[str, float]:
    """Find which saved place looks most like current frame."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = _clip_proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    best_name = ""
    best_sim  = -1.0
    for name, data in DB.places.items():
        sim = float((emb * data["avg_emb"]).sum())
        if sim > best_sim:
            best_sim  = sim
            best_name = name

    return best_name, best_sim


# ══════════════════════════════════════════════════════════════════
# ORB DIRECTION ESTIMATOR
# ══════════════════════════════════════════════════════════════════

def orb_direction(frame_bgr: np.ndarray, dest_name: str) -> dict:
    """
    Matches ORB features between current frame and best reference image.
    Uses homography to estimate:
      - horizontal offset → left / right
      - vertical offset   → forward / backward
      - matched region centre → where to draw arrow

    Returns dict:
      direction : "left" | "right" | "forward" | "arrived"
      offset_x  : pixel offset (negative=left, positive=right)
      offset_y  : pixel offset
      arrow_dst : (x, y) arrow tip on current frame
      n_matches : number of good matches
      confidence: 0.0 – 1.0
    """
    result = {
        "direction":  "forward",
        "offset_x":   0,
        "offset_y":   0,
        "arrow_dst":  None,
        "n_matches":  0,
        "confidence": 0.0,
    }

    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp_cur, desc_cur = _orb.detectAndCompute(gray, None)

    if desc_cur is None or len(desc_cur) < 5:
        return result

    place   = DB.places[dest_name]
    h, w    = frame_bgr.shape[:2]
    cx, cy  = w // 2, h // 2

    best_matches  = []
    best_ref_kps  = None

    # Match against every reference image — keep best
    for ref_desc, ref_kps in zip(place["descs"], place["kps"]):
        if ref_desc is None or len(ref_desc) < 5:
            continue
        try:
            raw = _matcher.match(desc_cur, ref_desc)
            good = sorted(raw, key=lambda m: m.distance)[:50]
            good = [m for m in good if m.distance < 60]
            if len(good) > len(best_matches):
                best_matches = good
                best_ref_kps = ref_kps
        except Exception:
            continue

    result["n_matches"] = len(best_matches)

    if len(best_matches) < ORB_MIN_MATCHES or best_ref_kps is None:
        return result

    result["confidence"] = min(1.0, len(best_matches) / 40.0)

    # Extract matched point pairs
    pts_cur = np.float32(
        [kp_cur[m.queryIdx].pt for m in best_matches]
    ).reshape(-1, 1, 2)
    pts_ref = np.float32(
        [best_ref_kps[m.trainIdx].pt for m in best_matches]
    ).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 5.0)

    if H is None:
        # Fallback: use centroid shift
        cx_cur = float(np.mean(pts_cur[:, 0, 0]))
        cx_ref = float(np.mean(pts_ref[:, 0, 0]))
        offset_x = cx_cur - cx_ref
        result["offset_x"]  = offset_x
        result["direction"] = "left" if offset_x < -30 else (
                              "right" if offset_x > 30 else "forward")
        result["arrow_dst"] = (int(cx_cur), cy)
        return result

    # Project reference image centre through homography
    ref_h, ref_w = DB.places[dest_name]["images"][0].shape[:2]
    ref_centre   = np.float32([[ref_w / 2, ref_h / 2]]).reshape(-1, 1, 2)
    try:
        proj = cv2.perspectiveTransform(ref_centre, H)
        px, py = int(proj[0, 0, 0]), int(proj[0, 0, 1])
    except Exception:
        px, py = cx, cy

    offset_x = px - cx
    offset_y = py - cy

    result["offset_x"]  = offset_x
    result["offset_y"]  = offset_y
    result["arrow_dst"] = (
        max(30, min(w - 30, px)),
        max(30, min(h - 30, py)),
    )

    THRESH = w * 0.12   # 12% of frame width
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
# MiDaS DEPTH — how far is destination region?
# ══════════════════════════════════════════════════════════════════

def estimate_depth_at(frame_bgr: np.ndarray, x: int, y: int) -> float:
    """Returns estimated depth in metres at pixel (x, y)."""
    if not MIDAS_OK:
        return -1.0
    try:
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp    = _midas_xfm(rgb)
        with torch.no_grad():
            pred = _midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        depth_map = pred.cpu().numpy()
        h, w      = depth_map.shape
        px        = max(0, min(w - 1, x))
        py        = max(0, min(h - 1, y))
        raw       = float(depth_map[py, px])
        # MiDaS returns inverse depth — higher = closer
        # Convert to metres heuristically
        metres    = DEPTH_SCALE / (raw / depth_map.max() + 1e-6)
        return round(metres, 1)
    except Exception:
        return -1.0


# ══════════════════════════════════════════════════════════════════
# NAVIGATION OVERLAY DRAWING
# ══════════════════════════════════════════════════════════════════

def draw_nav_arrow(frame: np.ndarray, direction: str,
                   arrow_dst: tuple | None, confidence: float) -> None:
    h, w  = frame.shape[:2]
    cx    = w // 2
    cy    = h // 2
    src   = (cx, h - 80)          # arrow tail — bottom centre

    # Choose tip based on direction
    if arrow_dst and confidence > 0.3:
        tip = arrow_dst
    elif direction == "left":
        tip = (cx - 180, cy)
    elif direction == "right":
        tip = (cx + 180, cy)
    else:
        tip = (cx, cy - 120)

    # Shadow
    cv2.arrowedLine(frame, src, tip, (0, 0, 0), 12, tipLength=0.3)
    # Main arrow
    arrow_color = (0, 255, 80)
    cv2.arrowedLine(frame, src, tip, arrow_color, 6, tipLength=0.3)

    # Pulsing circle at tip
    pulse = int(18 + 6 * abs(np.sin(time.time() * 3)))
    cv2.circle(frame, tip, pulse, arrow_color, 3)
    cv2.circle(frame, tip, 6, (255, 255, 255), -1)


def draw_match_overlay(frame: np.ndarray, dest_name: str,
                       best_ref_img: np.ndarray | None) -> None:
    """Draw a small thumbnail of the destination in corner."""
    if best_ref_img is None:
        return
    h, w = frame.shape[:2]
    thumb_w, thumb_h = 160, 100
    thumb = cv2.resize(best_ref_img, (thumb_w, thumb_h))
    # semi-transparent background
    roi = frame[10:10+thumb_h+10, w-thumb_w-10:w-10]
    ov  = roi.copy()
    cv2.rectangle(ov, (0,0), (thumb_w+10, thumb_h+10), (20,20,20), -1)
    cv2.addWeighted(ov, 0.6, roi, 0.4, 0, roi)
    frame[15:15+thumb_h, w-thumb_w-5:w-5] = thumb
    cv2.putText(frame, f"Target: {dest_name}",
                (w-thumb_w-5, 15+thumb_h+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)


def draw_hud(frame: np.ndarray, dest: str, sim: float,
             direction: str, n_matches: int, depth: float,
             confidence: float) -> None:
    h, w = frame.shape[:2]
    F    = cv2.FONT_HERSHEY_SIMPLEX

    # Top bar
    cv2.rectangle(frame, (0,0), (w, 50), (10,10,10), -1)

    status_color = (0,255,80) if sim > 0.75 else (0,180,255)
    cv2.putText(frame, f"Navigating to: {dest.upper()}",
                (10, 32), F, 0.8, status_color, 2)

    # Similarity bar
    bar_x = w - 220
    cv2.putText(frame, "Match:", (bar_x, 22), F, 0.5, (180,180,180), 1)
    cv2.rectangle(frame, (bar_x+55, 10), (bar_x+160, 26), (50,50,50), -1)
    filled = int((bar_x+55) + sim * 105)
    bar_color = (0,255,80) if sim > ARRIVAL_THRESHOLD else (0,180,255)
    cv2.rectangle(frame, (bar_x+55, 10), (filled, 26), bar_color, -1)
    cv2.putText(frame, f"{sim*100:.0f}%", (bar_x+165, 22), F, 0.5, (220,220,220), 1)

    # Bottom info bar
    cv2.rectangle(frame, (0, h-48), (w, h), (10,10,10), -1)

    dir_te = {
        "left":    "← ఎడమవైపు",
        "right":   "→ కుడివైపు",
        "forward": "↑ ముందుకు",
        "arrived": "✓ చేరుకున్నారు",
    }.get(direction, "↑ ముందుకు")

    cv2.putText(frame, dir_te, (10, h-18), F, 0.85, (0,255,150), 2)
    cv2.putText(frame, f"ORB:{n_matches}pts", (w-300, h-18), F, 0.55, (180,180,180), 1)

    if depth > 0:
        cv2.putText(frame, f"Depth:{depth}m", (w-180, h-18), F, 0.55, (150,220,255), 1)

    cv2.putText(frame, "Q=Quit  R=Reset", (w-160, h-36), F, 0.45, (120,120,120), 1)

    # Arrived banner
    if direction == "arrived" or sim >= ARRIVAL_THRESHOLD:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-50), (w, h//2+50), (0,100,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"✓  {dest.upper()} చేరుకున్నారు!",
                    (w//2 - 200, h//2 + 15), F, 1.1, (0,255,100), 3)


def draw_feature_points(frame: np.ndarray, dest_name: str) -> None:
    """Draw ORB keypoint matches on frame (debug visual)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps, _ = _orb.detectAndCompute(gray, None)
    if kps:
        # draw top 30 keypoints
        for kp in kps[:30]:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame, (x,y), 3, (0,200,255), -1)


# ══════════════════════════════════════════════════════════════════
# BACKGROUND NAVIGATION WORKER
# Runs CLIP + ORB + MiDaS off main thread → no camera freeze
# ══════════════════════════════════════════════════════════════════

_nav_lock   = threading.Lock()
_nav_state  = {
    "sim":        0.0,
    "direction":  "forward",
    "offset_x":   0,
    "arrow_dst":  None,
    "n_matches":  0,
    "confidence": 0.0,
    "depth":      -1.0,
    "arrived":    False,
}
_nav_frame  = [None]
_nav_dest   = [None]
_nav_stop   = threading.Event()


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
            # ── CLIP similarity ──────────────────────────────────────
            sim = clip_similarity(frame, dest)

            # ── ORB direction ────────────────────────────────────────
            orb_res = orb_direction(frame, dest)

            direction  = orb_res["direction"]
            arrow_dst  = orb_res["arrow_dst"]
            n_matches  = orb_res["n_matches"]
            confidence = orb_res["confidence"]
            offset_x   = orb_res["offset_x"]

            # Override direction with arrived if CLIP says so
            arrived = sim >= ARRIVAL_THRESHOLD
            if arrived:
                direction = "arrived"

            # ── MiDaS depth at arrow tip ─────────────────────────────
            depth = -1.0
            if arrow_dst and MIDAS_OK:
                depth = estimate_depth_at(frame, arrow_dst[0], arrow_dst[1])

            # ── Update shared state ──────────────────────────────────
            with _nav_lock:
                _nav_state["sim"]        = sim
                _nav_state["direction"]  = direction
                _nav_state["offset_x"]   = offset_x
                _nav_state["arrow_dst"]  = arrow_dst
                _nav_state["n_matches"]  = n_matches
                _nav_state["confidence"] = confidence
                _nav_state["depth"]      = depth
                _nav_state["arrived"]    = arrived

            # ── Voice instructions ───────────────────────────────────
            now = time.time()
            if now - last_speak_time > SPEAK_COOLDOWN:
                if arrived:
                    speak_bg(f"{dest} చేరుకున్నారు")
                    last_speak_time = now
                elif direction != last_direction or now - last_speak_time > 6.0:
                    te_dir = NAV_MSG.get(direction, "ముందుకు వెళ్ళండి")
                    depth_str = f" {depth:.0f} మీటర్లు" if depth > 0 else ""
                    speak_bg(f"{te_dir}{depth_str}")
                    last_speak_time = now
                    last_direction  = direction

        except Exception as e:
            print(f"[NAV WORKER] Error: {e}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═"*60)
    print("  DRISHTI — VISUAL PLACE NAVIGATION  (GOD LEVEL)")
    print("═"*60)
    print(f"  Places available: {', '.join(DB.names())}")
    print("  Q = Quit   R = Reset destination")
    print("═"*60)

    # Choose destination
    print("\nAvailable places:")
    for i, name in enumerate(DB.names(), 1):
        print(f"  {i}. {name}")

    dest = input("\nEnter destination name: ").strip().lower()
    if dest not in DB.places:
        print(f"❌ '{dest}' not found. Available: {', '.join(DB.names())}")
        return

    print(f"\n✅ Navigating to: {dest}")
    speak_bg(f"{dest} వైపు నావిగేషన్ మొదలవుతుంది")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed!"); return

    # Set destination and start worker
    _nav_dest[0] = dest
    worker = threading.Thread(target=nav_worker, daemon=True, name="NavWorker")
    worker.start()

    # Get reference thumbnail for corner display
    ref_thumb = DB.places[dest]["images"][0] if DB.places[dest]["images"] else None

    frame_count = 0
    fps         = 0.0
    fps_timer   = time.time()

    print("\n[NAV] Camera running. Point camera and walk toward destination.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        _nav_frame[0] = frame.copy()

        # FPS
        if frame_count % 30 == 0:
            fps       = 30.0 / max(time.time()-fps_timer, 1e-6)
            fps_timer = time.time()

        # Get latest nav state
        with _nav_lock:
            state = dict(_nav_state)

        display = frame.copy()

        # ── Feature points (subtle debug) ─────────────────────────────
        draw_feature_points(display, dest)

        # ── Navigation arrow ──────────────────────────────────────────
        if not state["arrived"]:
            draw_nav_arrow(
                display,
                state["direction"],
                state["arrow_dst"],
                state["confidence"],
            )

        # ── Reference thumbnail ───────────────────────────────────────
        draw_match_overlay(display, dest, ref_thumb)

        # ── HUD ───────────────────────────────────────────────────────
        draw_hud(
            display,
            dest,
            state["sim"],
            state["direction"],
            state["n_matches"],
            state["depth"],
            state["confidence"],
        )

        # FPS top-right
        cv2.putText(display, f"{fps:.0f}fps",
                    (display.shape[1]-75, display.shape[0]-52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.imshow("DRISHTI Navigation", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            # Reset — choose new destination
            _nav_dest[0] = None
            print("\nAvailable places:")
            for i, name in enumerate(DB.names(), 1):
                print(f"  {i}. {name}")
            new_dest = input("New destination: ").strip().lower()
            if new_dest in DB.places:
                dest         = new_dest
                _nav_dest[0] = dest
                ref_thumb    = DB.places[dest]["images"][0]
                speak_bg(f"{dest} వైపు నావిగేషన్ మొదలవుతుంది")
                print(f"✅ Destination changed to: {dest}")
            else:
                print(f"❌ '{new_dest}' not found")
                _nav_dest[0] = dest

    _nav_stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Navigation ended. Goodbye.")


if __name__ == "__main__":
    main()
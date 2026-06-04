"""
╔══════════════════════════════════════════════════════════════════════╗
║   DRISHTI COMBINED — YOLO + FACE + PLACE NAVIGATION                 ║
║                                                                      ║
║   DEFAULT MODE  : YOLO object detection + Face recognition          ║
║   VOICE (always): say "listen" → menu in Telugu                     ║
║   listen + cmd  : "listen స్థలం సేవ్" / "listen వెళ్ళు" etc.           ║
║   S key         : Save a face or place (keyboard fallback)          ║
║   W key         : Enter place-navigation mode (keyboard fallback)   ║
║   R key         : Reset nav destination (nav mode only)             ║
║   ESC key       : Exit nav → back to default mode                   ║
║   Q key         : Quit                                               ║
║                                                                      ║
║   INSTALL:                                                           ║
║   pip install torch torchvision transformers                         ║
║   pip install opencv-contrib-python numpy pillow                     ║
║   pip install edge-tts pygame timm                                   ║
║   pip install ultralytics face_recognition                           ║
║   pip install SpeechRecognition pyaudio                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import asyncio, cv2, os, threading, time, uuid, tempfile, pickle, queue
try:
    import speech_recognition as _sr
    SR_OK = True
except ImportError:
    SR_OK = False
    print('[Voice CMD] speech_recognition not installed — voice commands disabled')
    print('[Voice CMD] Install: pip install SpeechRecognition pyaudio')
import numpy as np
import pygame
from pathlib import Path
from PIL import Image

# ══════════════════════════════════════════════════════════════════
# GLOBAL MODE FLAGS  — only one module runs at a time
# ══════════════════════════════════════════════════════════════════

# Possible values: "default"  |  "nav"  |  "save_face"  |  "save_place"  |  "listening"  |  "ocr"
APP_MODE      = "default"
APP_MODE_LOCK = threading.Lock()

# ══════════════════════════════════════════════════════════════════
# LANGUAGE CONFIG — change at runtime via voice command
# ══════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES = {
    # voice name               : (edge-tts voice,         STT lang code, display name)
    "telugu"    : ("te-IN-ShrutiNeural",    "te-IN",  "తెలుగు"),
    "hindi"     : ("hi-IN-SwaraNeural",     "hi-IN",  "हिंदी"),
    "tamil"     : ("ta-IN-PallaviNeural",   "ta-IN",  "தமிழ்"),
    "kannada"   : ("kn-IN-GaganNeural",     "kn-IN",  "ಕನ್ನಡ"),
    "malayalam" : ("ml-IN-SobhanaNeural",   "ml-IN",  "മലയാളം"),
    "bengali"   : ("bn-IN-TanishaaNeural",  "bn-IN",  "বাংলা"),
    "marathi"   : ("mr-IN-AarohiNeural",    "mr-IN",  "मराठी"),
    "english"   : ("en-IN-NeerjaNeural",    "en-IN",  "English"),
    "gujarati"  : ("gu-IN-DhwaniNeural",    "gu-IN",  "ગુજરાતી"),
    "urdu"      : ("ur-PK-UzmaNeural",      "ur-PK",  "اردو"),
}

# Phonetic/spoken forms that Google STT returns for each language name
LANG_SPOKEN_FORMS = {
    "telugu"    : ["telugu","Telugu","తెలుగు","తెలుగ్","తెలుగులో",
                   "तेलुगु","तेलुगू","தெலுங்கு","தெலுகு","ತೆಲುಗು","തെലുഗു"],
    "hindi"     : ["hindi","Hindi","హిందీ","హింది","హిందీలో",
                   "हिंदी","हिन्दी","இந்தி","ಹಿಂದಿ","ഹിന്ദി"],
    "tamil"     : ["tamil","Tamil","తమిళం","తమిళ్","తమిళ",
                   "तमिल","தமிழ்","தமிழ","ತಮಿಳು","തമിഴ്"],
    "kannada"   : ["kannada","Kannada","కన్నడ","కన్నడం","కన్నడ్",
                   "कन्नड","கன்னடம்","ಕನ್ನಡ","കന്നഡ"],
    "malayalam" : ["malayalam","Malayalam","మలయాళం","మలయాళ్","మలయాళ",
                   "मलयालम","மலையாளம்","ಮಲಯಾಳಂ","മലയാളം"],
    "bengali"   : ["bengali","Bengali","బెంగాలీ","బెంగాలి","బెంగాల్",
                   "बंगाली","बांग्ला","வங்காளம்","ಬೆಂಗಾಲಿ","ബംഗാളി","বাংলা"],
    "marathi"   : ["marathi","Marathi","మరాఠీ","మరాఠి",
                   "मराठी","மராத்தி","ಮರಾಠಿ","മറാഠി"],
    "english"   : ["english","English","ఇంగ్లీష్","ఇంగ్లిష్","ఇంగ్లీష","అంగ్రేజీ",
                   "अंग्रेजी","इंग्लिश","ஆங்கிலம்","ಇಂಗ್ಲಿಷ್","ഇംഗ്ലീഷ്"],
    "gujarati"  : ["gujarati","Gujarati","గుజరాతీ","గుజరాతి","గుజరాత్",
                   "गुजराती","குஜராத்தி","ಗುಜರಾತಿ","ഗുജറാത്തി","ગુજરાતી"],
    "urdu"      : ["urdu","Urdu","ఉర్దూ","ఉర్దు",
                   "उर्दू","उर्दु","உருது","ಉರ್ದು","ഉർദു","اردو"],
}

# Active language — start in Telugu
_current_lang     = "telugu"
_current_lang_lock = threading.Lock()

def get_tts_voice() -> str:
    with _current_lang_lock:
        return SUPPORTED_LANGUAGES[_current_lang][0]

def get_stt_lang() -> str:
    with _current_lang_lock:
        return SUPPORTED_LANGUAGES[_current_lang][1]

def get_lang_display() -> str:
    with _current_lang_lock:
        return SUPPORTED_LANGUAGES[_current_lang][2]

def set_language(lang_key: str) -> bool:
    global _current_lang
    if lang_key in SUPPORTED_LANGUAGES:
        with _current_lang_lock:
            _current_lang = lang_key
        return True
    return False

def detect_language_from_text(text: str) -> str:
    """Return language key if any spoken form found in text, else empty string."""
    text_l = text.lower()
    for lang_key, forms in LANG_SPOKEN_FORMS.items():
        if any(f.lower() in text_l for f in forms):
            return lang_key
    return ""


def get_lang_changed_msg(lang_key: str) -> str:
    """Return a confirmation message in the NEW language."""
    msgs = {
        "telugu"    : "భాష తెలుగుకు మార్చబడింది",
        "hindi"     : "भाषा हिंदी में बदल दी गई है",
        "tamil"     : "மொழி தமிழுக்கு மாற்றப்பட்டது",
        "kannada"   : "ಭಾಷೆ ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ",
        "malayalam" : "ഭാഷ മലയാളത്തിലേക്ക് മാറ്റി",
        "bengali"   : "ভাষা বাংলায় পরিবর্তিত হয়েছে",
        "marathi"   : "भाषा मराठीत बदलली आहे",
        "english"   : "Language changed to English",
        "gujarati"  : "ભાષા ગુજરાતીમાં બદલવામાં આવી",
        "urdu"      : "زبان اردو میں تبدیل کر دی گئی",
    }
    return msgs.get(lang_key, "Language changed")


def get_menu_in_current_lang() -> str:
    """Return the command menu spoken in the current active language."""
    menus = {
        "telugu"  : ("మీకు ఆరు ఆదేశాలు అందుబాటులో ఉన్నాయి. "
                     "ఒకటి, స్థలం సేవ్ చేయి. "
                     "రెండు, వ్యక్తిని సేవ్ చేయి. "
                     "మూడు, స్థలానికి వెళ్ళు. "
                     "నాలుగు, నావిగేషన్ ఆపు. "
                     "అయిదు, భాష మార్చు. "
                     "ఆరు, టెక్స్ట్ చదువు."),
        "hindi"   : ("आपके पास छह आदेश हैं। "
                     "एक, जगह सेव करो। "
                     "दो, व्यक्ति सेव करो। "
                     "तीन, जगह पर जाओ। "
                     "चार, नेविगेशन बंद करो। "
                     "पाँच, भाषा बदलो। "
                     "छह, टेक्स्ट पढ़ो।"),
        "tamil"   : ("உங்களுக்கு ஆறு கட்டளைகள் உள்ளன. "
                     "ஒன்று, இடம் சேமி. "
                     "இரண்டு, நபரை சேமி. "
                     "மூன்று, இடத்திற்கு செல். "
                     "நான்கு, வழிசெலுத்தலை நிறுத்து. "
                     "ஐந்து, மொழி மாற்று. "
                     "ஆறு, உரை படி."),
        "english" : ("You have six commands. "
                     "One, save place. "
                     "Two, save person. "
                     "Three, navigate to place. "
                     "Four, stop navigation. "
                     "Five, change language. "
                     "Six, read text."),
    }
    with _current_lang_lock:
        key = _current_lang
    # fallback to English for languages without a menu defined
    return menus.get(key, menus["english"])


def get_mode() -> str:
    with APP_MODE_LOCK:
        return APP_MODE

def set_mode(m: str) -> None:
    global APP_MODE
    with APP_MODE_LOCK:
        APP_MODE = m

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

PLACES_DIR        = "saved_places"
SAVE_DIR          = "saved_faces"
DB_FILE           = "face_db.pkl"

# CLIP / nav
CLIP_CHECK_EVERY  = 1.5
ORB_MIN_MATCHES   = 8
ARRIVAL_THRESHOLD = 0.82
DEPTH_SCALE       = 5.0
CONF_THRESHOLD_NAV= 0.30

# YOLO / face
YOLO_MODEL        = "yolov8n.pt"
FRAME_SKIP        = 4
YOLO_COOLDOWN     = 4.0
FACE_COOLDOWN     = 4.0
CONF_THRESHOLD    = 0.45
FACE_SCALE        = 0.5
TOLERANCE         = 0.50
CAPTURE_PHOTOS    = 5
CAPTURE_GAP       = 0.5
FOCAL_PX          = 600
SPEAK_COOLDOWN    = 3.5

os.makedirs(PLACES_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,   exist_ok=True)

# ── Voice command Telugu strings ──────────────────────────────────

VC_MENU = (
    "మీకు ఆరు ఆదేశాలు అందుబాటులో ఉన్నాయి. "
    "ఒకటి, స్థలం సేవ్ చేయి. "
    "రెండు, వ్యక్తిని సేవ్ చేయి. "
    "మూడు, స్థలానికి వెళ్ళు. "
    "నాలుగు, నావిగేషన్ ఆపు. "
    "అయిదు, భాష మార్చు. "
    "ఆరు, టెక్స్ట్ చదువు."
)

VC_LISTEN_START = "వింటున్నాను చెప్పండి"
VC_NOT_HEARD    = "అర్థం కాలేదు మళ్ళీ చెప్పండి"
VC_NO_PLACES    = "స్థలాలు ఏమీ సేవ్ కాలేదు ముందు స్థలం సేవ్ చేయండి"
VC_ASK_PLACE_NAME  = "స్థలం పేరు చెప్పండి"
VC_ASK_PERSON_NAME = "వ్యక్తి పేరు చెప్పండి"
VC_ASK_DEST     = "మీరు ఎక్కడికి వెళ్ళాలో చెప్పండి"
VC_NAV_STOPPED  = "నావిగేషన్ ఆపబడింది"
VC_GOODBYE      = "దృష్టి ముగిసింది వీడ్కోలు"

# Keywords to detect each command (partial Telugu match)
# Each list covers: real Telugu + English + phonetic Telugu (what Google STT returns)
VC_CMD_SAVE_PLACE  = [
    # Telugu
    "స్థలం సేవ్", "స్థలం", "సేవ్ ప్లేస్",
    # English
    "save place", "place save", "save location",
    # Phonetic Telugu (Google STT output for English words)
    "సేవ్ ప్లేస్", "సేవ్ place",
]
VC_CMD_SAVE_PERSON = [
    # Telugu
    "వ్యక్తి", "వ్యక్తిని సేవ్", "సేవ్ వ్యక్తి", "పర్సన్",
    # English
    "save person", "save face", "person save", "face save", "add person",
    # Phonetic Telugu
    "సేవ్ పర్సన్", "సేవ్ ఫేస్",
]
VC_CMD_NAVIGATE    = [
    # Telugu
    "స్థలానికి", "నావిగేషన్", "వెళ్ళు", "ఎక్కడ", "వేర్ ఇస్",
    # English — specific phrases only, no single generic words
    "where is", "navigate", "go to", "take me",
    "where is place", "paris place", "navigate to",
    # Phonetic Telugu
    "వేర్", "గో టు", "నావిగేట్",
]
VC_CMD_STOP_NAV    = [
    # Telugu
    "ఆపు", "నిలిపివేయి", "స్టాప్",
    # English
    "stop", "cancel", "back", "stop navigation", "exit navigation",
    # Phonetic Telugu
    "స్టాప్ నావిగేషన్",
]
VC_CMD_QUIT        = [
    "quit drishti", "exit drishti", "app quit", "close app",
    "బయటకు వెళ్ళు", "మూసివేయి", "క్లోజ్",
]
VC_CMD_CHANGE_LANG = [
    # Telugu
    "భాష మార్చు", "భాష", "లాంగ్వేజ్",
    # English
    "change language", "language", "switch language", "change lang",
    # Phonetic Telugu
    "చేంజ్ లాంగ్వేజ్", "లాంగ్వేజ్ చేంజ్",
]
VC_CMD_READ_TEXT = [
    # English
    "read text", "read", "text", "scan text", "what does it say",
    "read this", "ocr", "scan",
    # Telugu
    "చదువు", "టెక్స్ట్ చదువు", "రీడ్ టెక్స్ట్",
    # Phonetic
    "రీడ్", "స్కాన్",
]
VC_CMD_CANCEL_OCR = [
    "cancel", "stop", "stop reading", "cancel reading",
    "ఆపు", "స్టాప్", "చదవడం ఆపు",
]

# ── Telugu strings ────────────────────────────────────────────────

NAV_MSG = {
    "left":      "ఎడమవైపు తిరగండి",
    "right":     "కుడివైపు తిరగండి",
    "forward":   "ముందుకు వెళ్ళండి",
    "arrived":   "మీరు గమ్యానికి చేరుకున్నారు",
    "searching": "స్థలం వెతుకుతున్నాం",
    "obstacle":  "ముందు అడ్డంకి ఉంది జాగ్రత్త",
}

DEFAULT_PLACES = [
    "kitchen","bedroom","bathroom","living room","dining room",
    "office","classroom","road","corridor","staircase",
    "entrance","garden","garage",
]

TELUGU_NAMES = {
    "kitchen":"వంటగది","bedroom":"పడక గది","bathroom":"బాత్రూమ్",
    "living room":"హాలు","dining room":"భోజన గది","office":"కార్యాలయం",
    "classroom":"తరగతి గది","road":"రోడ్డు","corridor":"వరండా",
    "staircase":"మెట్లు","entrance":"ముఖద్వారం","garden":"తోట","garage":"గరాజ్",
}

OBJECT_REF_HEIGHT_M = {
    "person":1.7,"car":1.5,"truck":2.5,"bus":3.0,"motorcycle":1.1,
    "bicycle":1.0,"chair":0.9,"dog":0.5,"cat":0.3,"tv":0.7,
    "laptop":0.3,"bottle":0.25,"cup":0.12,"cell phone":0.15,
    "book":0.22,"backpack":0.5,
}
DEFAULT_REF_H = 0.5

TELUGU_OBJ = {
    "person":"వ్యక్తి","chair":"కుర్చీ","car":"కారు","dog":"కుక్క",
    "cat":"పిల్లి","bicycle":"సైకిల్","motorcycle":"బైక్","bus":"బస్సు",
    "truck":"ట్రక్","tv":"టీవీ","laptop":"లాప్టాప్","bottle":"బాటిల్",
    "cup":"కప్పు","cell phone":"ఫోన్","book":"పుస్తకం","backpack":"బ్యాగ్",
}

# ══════════════════════════════════════════════════════════════════
# SHARED AUDIO  — single slot, latest wins, zero pileup
# ══════════════════════════════════════════════════════════════════

# Windows WASAPI fix — try different drivers until one works
_pygame_init_ok = False
for _freq in [44100, 22050, 16000]:
    for _driver in [None, "directsound", "winmm"]:
        try:
            if _driver:
                os.environ.setdefault("SDL_AUDIODRIVER", _driver)
            pygame.mixer.pre_init(_freq, -16, 1, 512)
            pygame.mixer.init()
            _pygame_init_ok = True
            print(f"[Audio] pygame mixer OK (freq={_freq})")
            break
        except Exception as _e:
            print(f"[Audio] pygame driver failed ({_driver}, {_freq}Hz): {_e}")
            try:
                pygame.mixer.quit()
            except Exception:
                pass
    if _pygame_init_ok:
        break

if not _pygame_init_ok:
    print("[Audio] ⚠️  pygame mixer unavailable — voice output disabled")
_pending_text : str | None = None
_audio_lock   = threading.Lock()
_audio_busy   = threading.Event()   # set while actually playing


def speak(text: str) -> None:
    """Non-blocking: silenced during save ops and voice command mode."""
    if get_mode() in ("save_face", "save_place", "listening"):
        return
    if _vc_running.is_set():   # voice command mic is active — stay silent
        return
    global _pending_text
    with _audio_lock:
        _pending_text = text


def speak_blocking(text: str) -> None:
    """Blocking speak: silenced during save ops and voice command mode."""
    if get_mode() in ("save_face", "save_place", "listening"):
        return
    if _vc_running.is_set():
        return
    asyncio.run(_play_once(text))


async def _play_once(text: str) -> None:
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        import edge_tts
        await edge_tts.Communicate(text, voice=get_tts_voice()).save(path)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.04)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[TTS] {e}")
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def _tts_runner() -> None:
    global _pending_text
    while True:
        text = None
        with _audio_lock:
            if _pending_text:
                text          = _pending_text
                _pending_text = None
        if text:
            _audio_busy.set()
            asyncio.run(_play_once(text))
            _audio_busy.clear()
        else:
            time.sleep(0.08)


threading.Thread(target=_tts_runner, daemon=True, name="TTS").start()


def speak_force(text: str) -> None:
    """Always speaks — used ONLY for voice command feedback.
    Stops TTS runner output and drains pending queue first to avoid mixer conflicts."""
    # drain any pending TTS so runner doesn't race us
    global _pending_text
    with _audio_lock:
        _pending_text = None
    # stop whatever is currently playing
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass
    time.sleep(0.05)   # tiny gap so mixer releases
    asyncio.run(_play_once(text))

# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def bbox_dist(bbox_h: int, ref_h: float) -> float:
    if bbox_h < 5:
        return 99.0
    return round(min((ref_h * FOCAL_PX) / bbox_h, 20.0), 1)

def face_dist(top: int, bottom: int) -> float:
    h = max(bottom - top, 1)
    return round(min((1.7 * FOCAL_PX) / h, 20.0), 1)

def direction_te(cx: int, frame_w: int) -> str:
    r = cx / float(frame_w)
    if r < 0.33:   return "Left"
    elif r > 0.66: return "Right"
    else:          return "Center"

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════
# YOLO
# ══════════════════════════════════════════════════════════════════

print("[YOLO] Loading model...")
from ultralytics import YOLO as _YOLO
yolo = _YOLO(YOLO_MODEL)
yolo.fuse()
print("[YOLO] Ready ✅")

# ══════════════════════════════════════════════════════════════════
# FACE DB
# ══════════════════════════════════════════════════════════════════

import face_recognition as _fr

face_db: dict[str, list] = {}

def load_face_db() -> None:
    global face_db
    if Path(DB_FILE).exists():
        with open(DB_FILE, "rb") as f:
            face_db = pickle.load(f)
        total = sum(len(v) for v in face_db.values())
        print(f"[FaceDB] {len(face_db)} people, {total} encodings")
    else:
        print("[FaceDB] Fresh start.")

def save_face_db() -> None:
    with open(DB_FILE, "wb") as f:
        pickle.dump(face_db, f)

def all_encodings():
    encs, names = [], []
    for name, enc_list in face_db.items():
        for enc in enc_list:
            encs.append(enc); names.append(name)
    return encs, names

load_face_db()

# ── Face recognition worker ───────────────────────────────────────
_rec_lock     = threading.Lock()
_rec_results  : list = []
_latest_small = [None]
_face_stop    = threading.Event()
_face_pause   = threading.Event()   # set → worker sleeps (face save / nav mode)


def _face_worker() -> None:
    known_encs, known_names = all_encodings()
    while not _face_stop.is_set():
        if _face_pause.is_set() or get_mode() != "default":
            time.sleep(0.1)
            continue
        small = _latest_small[0]
        if small is None:
            time.sleep(0.05)
            continue
        rgb = to_rgb(small)
        try:
            locs = _fr.face_locations(rgb, model="hog")
        except Exception:
            time.sleep(0.05)
            continue
        if not locs:
            with _rec_lock:
                _rec_results.clear()
            time.sleep(0.05)
            continue
        try:
            encs = _fr.face_encodings(rgb, locs)
        except Exception:
            time.sleep(0.05)
            continue
        results = []
        for enc, loc in zip(encs, locs):
            name, conf = "Unknown", 0.0
            if known_encs:
                dists = _fr.face_distance(known_encs, enc)
                idx   = int(np.argmin(dists))
                d     = float(dists[idx])
                if d <= TOLERANCE:
                    name = known_names[idx]
                    conf = round((1.0 - d) * 100, 1)
            results.append((name, conf) + loc)
        with _rec_lock:
            _rec_results.clear()
            _rec_results.extend(results)
        known_encs, known_names = all_encodings()
        time.sleep(0.04)


threading.Thread(target=_face_worker, daemon=True, name="FaceWorker").start()

# ══════════════════════════════════════════════════════════════════
# SAVE FACE  (S key in default mode)
# ══════════════════════════════════════════════════════════════════

_saving_face = False


def _save_face_thread_named(cap: cv2.VideoCapture, person_name: str) -> None:
    """Voice-triggered face save — name already known, skips input() prompt."""
    global _saving_face, face_db
    _saving_face = True
    set_mode("save_face")
    _face_pause.set()
    global _pending_text
    with _audio_lock:
        _pending_text = None
    try:
        print(f"\n[SaveFace-Voice] Capturing '{person_name}'...")
        speak_force(f"{person_name} ముఖం కెమెరా ముందు పెట్టండి")
        time.sleep(0.5)

        cap_encs: list  = []
        cap_frames: list = []
        attempt = 0
        while len(cap_encs) < CAPTURE_PHOTOS and attempt < 60:
            attempt += 1
            ret, frm = cap.read()
            if not ret:
                time.sleep(0.1); continue
            rgb  = to_rgb(frm)
            locs = _fr.face_locations(rgb, model="hog")
            if not locs:
                ov = frm.copy()
                cv2.putText(ov, "No face — move closer",
                            (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow("DRISHTI", ov); cv2.waitKey(1)
                time.sleep(0.25); continue
            encs = _fr.face_encodings(rgb, locs)
            if not encs: continue
            cap_encs.append(encs[0]); cap_frames.append(frm.copy())
            n = len(cap_encs)
            t2,r2,b2,l2 = locs[0]
            ov = frm.copy()
            cv2.rectangle(ov,(l2,t2),(r2,b2),(0,255,0),3)
            cv2.putText(ov,f"Capturing {n}/{CAPTURE_PHOTOS}",
                        (l2,max(t2-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            cv2.imshow("DRISHTI",ov); cv2.waitKey(1)
            time.sleep(CAPTURE_GAP)

        if not cap_encs:
            speak_force("ముఖం కనుగొనబడలేదు మళ్ళీ ప్రయత్నించండి")
            return

        pdir = os.path.join(SAVE_DIR, person_name)
        os.makedirs(pdir, exist_ok=True)
        ts = int(time.time())
        for i, frm in enumerate(cap_frames):
            cv2.imwrite(os.path.join(pdir, f"{person_name}_{ts}_{i+1}.jpg"), frm)
        if person_name not in face_db:
            face_db[person_name] = []
        face_db[person_name].extend(cap_encs)
        save_face_db()
        print(f"[SaveFace-Voice] '{person_name}' saved ✅")
        speak_force(f"{person_name} సేవ్ అయింది")
    finally:
        _face_pause.clear()
        _saving_face = False
        set_mode("default")
        print("[SaveFace-Voice] Back to default mode.\n")


def _save_face_thread(cap: cv2.VideoCapture) -> None:
    global _saving_face, face_db
    _saving_face = True
    set_mode("save_face")
    _face_pause.set()
    # drain pending audio immediately
    global _pending_text
    with _audio_lock:
        _pending_text = None

    try:
        print("\n[SaveFace] కెమెరా ముందు ఉండండి...")
        speak_blocking("కెమెరా ముందు ఉండండి")
        time.sleep(0.3)

        cap_encs: list  = []
        cap_frames: list = []
        attempt = 0

        while len(cap_encs) < CAPTURE_PHOTOS and attempt < 60:
            attempt += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            rgb  = to_rgb(frame)
            locs = _fr.face_locations(rgb, model="hog")
            if not locs:
                ov = frame.copy()
                cv2.putText(ov, "No face — move closer",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                cv2.imshow("DRISHTI", ov)
                cv2.waitKey(1)
                time.sleep(0.25)
                continue
            encs = _fr.face_encodings(rgb, locs)
            if not encs:
                continue
            cap_encs.append(encs[0])
            cap_frames.append(frame.copy())
            n = len(cap_encs)
            t, r, b, l = locs[0]
            ov = frame.copy()
            cv2.rectangle(ov, (l, t), (r, b), (0,255,0), 3)
            cv2.putText(ov, f"Capturing {n}/{CAPTURE_PHOTOS}",
                        (l, max(t-10,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("DRISHTI", ov)
            cv2.waitKey(1)
            print(f"[SaveFace] Photo {n}/{CAPTURE_PHOTOS}")
            time.sleep(CAPTURE_GAP)

        if not cap_encs:
            speak_blocking("ముఖం కనుగొనబడలేదు మళ్ళీ ప్రయత్నించండి")
            return

        print("[SaveFace] Type name + Enter: ", end="", flush=True)
        person_name = input().strip().lower()
        if not person_name:
            print("[SaveFace] No name — discarded.")
            return

        pdir = os.path.join(SAVE_DIR, person_name)
        os.makedirs(pdir, exist_ok=True)
        ts = int(time.time())
        for i, frm in enumerate(cap_frames):
            cv2.imwrite(os.path.join(pdir, f"{person_name}_{ts}_{i+1}.jpg"), frm)

        if person_name not in face_db:
            face_db[person_name] = []
        face_db[person_name].extend(cap_encs)
        save_face_db()
        print(f"[SaveFace] '{person_name}' saved ✅")
        speak_blocking(f"{person_name} సేవ్ అయింది")

    finally:
        _face_pause.clear()
        _saving_face = False
        set_mode("default")
        print("[SaveFace] Back to default mode.\n")

# ══════════════════════════════════════════════════════════════════
# CLIP
# ══════════════════════════════════════════════════════════════════

print("[CLIP] Loading model... (first time 1-2 min)")
import torch
from transformers import CLIPProcessor, CLIPModel

_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model.eval()
print("[CLIP] Ready ✅")

# ══════════════════════════════════════════════════════════════════
# MiDaS
# ══════════════════════════════════════════════════════════════════

print("[MiDaS] Loading depth model...")
MIDAS_OK = False
try:
    midas_dir     = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
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
                exportable=True, non_negative=True, blocks={"expand": True}
            )
            cp = os.path.expanduser("~/.cache/torch/hub/checkpoints/midas_v21_small_256.pt")
            if os.path.exists(cp):
                _midas.load_state_dict(torch.load(cp, map_location="cpu"))
                _midas.eval()
                _midas_xfm = Compose([
                    lambda img: {"image": img / 255.0},
                    Resize(256,256,resize_target=None,keep_aspect_ratio=True,
                           ensure_multiple_of=32,resize_method="upper_bound",
                           image_interpolation_method=cv2.INTER_CUBIC),
                    NormalizeImage(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                    PrepareForNet(),
                    lambda s: torch.from_numpy(s["image"]).unsqueeze(0),
                ])
                MIDAS_OK = True
                loaded_locally = True
                print("[MiDaS] Loaded from local cache ✅")
        except Exception as ex:
            print(f"[MiDaS] Local load failed: {ex}")
    if not loaded_locally:
        _midas     = torch.hub.load("intel-isl/MiDaS","MiDaS_small",trust_repo=True)
        _midas_xfm = torch.hub.load("intel-isl/MiDaS","transforms",trust_repo=True).small_transform
        _midas.eval()
        MIDAS_OK = True
        print("[MiDaS] Ready ✅")
except Exception as e:
    print(f"[MiDaS] Not available ({e}) — depth disabled")

# ══════════════════════════════════════════════════════════════════
# ORB
# ══════════════════════════════════════════════════════════════════

_orb     = cv2.ORB_create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ══════════════════════════════════════════════════════════════════
# PLACE DATABASE
# ══════════════════════════════════════════════════════════════════

class PlaceDB:
    def __init__(self, root: str) -> None:
        self.root   = Path(root)
        self.places: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        self.places.clear()
        if not self.root.exists():
            return
        for place_dir in sorted(self.root.iterdir()):
            if not place_dir.is_dir():
                continue
            name = place_dir.name
            imgs=[]; pils=[]; kps=[]; descs=[]; embs=[]
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
            self.places[name] = {"images":imgs,"pils":pils,"kps":kps,
                                 "descs":descs,"avg_emb":avg_emb,"all_embs":embs}
            print(f"[PlaceDB] '{name}' — {len(imgs)} images")
        print(f"[PlaceDB] Total: {len(self.places)} places ✅")

    def reload(self) -> None:
        self._load()

    def names(self) -> list[str]:
        return list(self.places.keys())


# ══════════════════════════════════════════════════════════════════
# VOICE COMMAND ENGINE
# Wake word: "listen" → mic opens → user speaks command
# All feedback in Telugu. After command finishes → back to idle.
# ══════════════════════════════════════════════════════════════════

_vc_queue     = queue.Queue()   # posts (tag, value) to main loop
_vc_running   = threading.Event()   # set while mic is active
_vc_stop      = threading.Event()


def _match_cmd(text: str, keywords: list) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _recognize_audio(recognizer, audio) -> str:
    """Try current language then English on the SAME audio clip — for commands/names."""
    stt = get_stt_lang()
    langs = [stt] if stt == "en-IN" else [stt, "en-IN"]
    for lang in langs:
        try:
            result = recognizer.recognize_google(audio, language=lang)
            if result:
                return result.lower()
        except Exception:
            continue
    return ""


def _recognize_audio_wake(recognizer, audio) -> str:
    """ALWAYS uses English for wake word — language setting never affects this.
    Tries en-IN first (Indian accent), then en-US, then te-IN as last resort."""
    for lang in ("en-IN", "en-US", "te-IN"):
        try:
            result = recognizer.recognize_google(audio, language=lang)
            if result:
                return result.lower()
        except Exception:
            continue
    return ""


def _listen_once(recognizer, src, timeout=7) -> str:
    """Record ONE utterance from already-open mic source and recognise it."""
    try:
        audio = recognizer.listen(src, timeout=timeout, phrase_time_limit=6)
        return _recognize_audio(recognizer, audio)
    except Exception:
        return ""


def _voice_command_thread() -> None:
    """
    Background thread — persistent mic, waits for wake word 'listen'.
    Uses English recognition for wake word (more reliable cross-language).
    Full silence enforced on all other audio during command processing.
    """
    if not SR_OK:
        return

    import speech_recognition as sr
    recognizer = sr.Recognizer()
    recognizer.energy_threshold    = 400
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold     = 0.6   # shorter pause = faster response
    mic = sr.Microphone()

    print("[VoiceCmd] Starting — energy threshold fixed at 3000 (ignores background noise)")
    print("[VoiceCmd] Say 'listen' / 'command' / 'activate' to start.")

    # ── Wake words: any of these trigger command mode ────────────
    # Broad net — Telugu + English + common misrecognitions
    WAKE_WORDS = [
        # English
        "listen", "command", "activate", "drishti", "assistant",
        # Telugu
        "లిసెన్", "వినండి", "దృష్టి", "కమాండ్", "ఆక్టివేట్",
        # Common misrecognitions of "listen" in Telugu ASR
        "విను", "లెజెండ్", "లీజన్", "లిజన్",
    ]

    # ── Open mic ONCE — stays open for entire session ─────────────
    with mic as src:
        # Calibrate on ambient noise, then set a MINIMUM floor of 300
        # This prevents threshold dropping too low (58 = picks up everything)
        print("[VoiceCmd] Calibrating for 2 seconds — stay quiet...")
        recognizer.adjust_for_ambient_noise(src, duration=2)
        ambient = recognizer.energy_threshold
        # Floor at 300, ceiling at 800, with 3x margin above ambient
        recognizer.energy_threshold        = max(300, min(800, ambient * 3.0))
        recognizer.dynamic_energy_threshold = False   # lock — no drift
        recognizer.pause_threshold          = 0.5
        recognizer.non_speaking_duration    = 0.4
        print(f"[VoiceCmd] Mic ready. Ambient={ambient:.0f} Threshold={recognizer.energy_threshold:.0f}")
        print("[VoiceCmd] Say 'listen' / 'command' / 'activate' to start.")

        while not _vc_stop.is_set():
            # ── STEP 1: Wait for wake word ────────────────────────
            try:
                audio = recognizer.listen(src, timeout=None, phrase_time_limit=3)
            except Exception:
                time.sleep(0.1)
                continue

            # Always print raw recognition so we can see what Google hears
            # Wake word ALWAYS in English — unaffected by language setting
            wake_text = _recognize_audio_wake(recognizer, audio)
            if not wake_text:
                continue

            print(f"[VoiceCmd] Heard: '{wake_text}'")

            # ── FIX 3: Direct cancel in OCR / nav modes (no wake word needed) ──
            if get_mode() == "ocr":
                if any(w in wake_text for w in ["cancel", "stop", "ఆపు", "స్టాప్"]):
                    print("[VoiceCmd] Direct cancel in OCR mode")
                    _vc_queue.put(("vc_stop_ocr", ""))
                    continue   # don't need full wake-word flow

            if get_mode() == "nav":
                if any(w in wake_text for w in ["cancel", "stop", "ఆపు", "స్టాప్"]):
                    print("[VoiceCmd] Direct cancel in NAV mode")
                    _vc_queue.put(("vc_stop_nav", ""))
                    continue

            # Match any wake word — also check substrings of multi-word phrases
            words_heard = wake_text.lower().split()
            triggered   = any(w in wake_text for w in WAKE_WORDS) or \
                          any(w in words_heard for w in ["command","activate","mode","listen"])
            if not triggered:
                continue

            # ── STEP 2: Wake confirmed — silence everything ────────
            _vc_running.set()
            set_mode("listening")
            global _pending_text
            with _audio_lock:
                _pending_text = None
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            time.sleep(0.1)  # let mixer fully stop
            print("[VoiceCmd] ✅ Wake word! Command mode active.")

            try:
                # Inline command? e.g. "listen navigate" / "listen save place"
                inline = wake_text
                for w in WAKE_WORDS:
                    inline = inline.replace(w, "")
                inline = inline.strip()

                # Only treat as inline command if it contains a real command keyword
                INLINE_KEYWORDS = [
                    "save", "navigate", "where", "stop", "change",
                    "language", "place", "person", "face", "quit"
                ]
                is_real_cmd = any(kw in inline for kw in INLINE_KEYWORDS)
                if inline and len(inline) > 4 and is_real_cmd:
                    print(f"[VoiceCmd] Inline: '{inline}'")
                    _process_vc_command(inline, src, recognizer)
                else:
                    # Just say "listening" — menu only if user asks for it
                    speak_force(VC_LISTEN_START)

                    try:
                        audio2 = recognizer.listen(src, timeout=7, phrase_time_limit=6)
                        cmd_text = _recognize_audio_wake(recognizer, audio2)
                        if not cmd_text:
                            cmd_text = _recognize_audio(recognizer, audio2)
                    except Exception:
                        cmd_text = ""

                    print(f"[VoiceCmd] Command: '{cmd_text}'")
                    if cmd_text:
                        if any(w in cmd_text for w in ["menu","help","commands","list","మెను","మెనూ","సహాయం","హెల్ప్","హెల్","మెన్యూ"]):
                            speak_force(get_menu_in_current_lang())
                        else:
                            _process_vc_command(cmd_text, src, recognizer)
                    else:
                        speak_force(VC_NOT_HEARD)

            except Exception as e:
                print(f"[VoiceCmd] Error: {e}")
                speak_force(VC_NOT_HEARD)
            finally:
                _vc_running.clear()
                if get_mode() == "listening":
                    set_mode("default")
                print("[VoiceCmd] Done — back to idle\n")


def _process_vc_command(text: str, src, recognizer) -> None:
    """Parse the command text and post appropriate action to main loop."""
    print(f"[VoiceCmd] Processing: '{text}'")

    # ── QUIT ──────────────────────────────────────────────────────
    if _match_cmd(text, VC_CMD_QUIT):
        print("[VoiceCmd] → QUIT")
        speak_force(VC_GOODBYE)
        _vc_queue.put(("quit", ""))
        return

    # ── CANCEL OCR — check BEFORE stop nav ("cancel" is in both) ──
    if get_mode() == "ocr" and _match_cmd(text, VC_CMD_CANCEL_OCR):
        print("[VoiceCmd] → CANCEL OCR")
        _vc_queue.put(("vc_stop_ocr", ""))
        return

    # ── STOP NAV ──────────────────────────────────────────────────
    if _match_cmd(text, VC_CMD_STOP_NAV):
        print("[VoiceCmd] → STOP NAV")
        speak_force(VC_NAV_STOPPED)
        _vc_queue.put(("vc_stop_nav", ""))
        return

    # ── SAVE PLACE (check BEFORE navigate — "save place" contains "place") ──
    if _match_cmd(text, VC_CMD_SAVE_PLACE):
        print("[VoiceCmd] → SAVE PLACE")
        speak_force(VC_ASK_PLACE_NAME)
        name_text = _listen_once(recognizer, src, timeout=8)
        print(f"[VoiceCmd] Place name heard: {name_text!r}")
        if name_text.strip():
            place_name = name_text.strip().lower()
            speak_force(f"{place_name} place saving started")
            _vc_queue.put(("vc_save_place", place_name))
        else:
            speak_force(VC_NOT_HEARD)
        return

    # ── SAVE PERSON (check BEFORE navigate) ───────────────────────
    if _match_cmd(text, VC_CMD_SAVE_PERSON):
        print("[VoiceCmd] → SAVE PERSON")
        speak_force(VC_ASK_PERSON_NAME)
        name_text = _listen_once(recognizer, src, timeout=8)
        print(f"[VoiceCmd] Person name heard: {name_text!r}")
        if name_text.strip():
            person_name = name_text.strip().lower()
            speak_force(f"{person_name} face saving started")
            _vc_queue.put(("vc_save_face", person_name))
        else:
            speak_force(VC_NOT_HEARD)
        return

    # ── NAVIGATE ──────────────────────────────────────────────────
    if _match_cmd(text, VC_CMD_NAVIGATE):
        if not DB.places:
            speak_force(VC_NO_PLACES)
            return
        # Read available places aloud
        names_str = ", ".join(DB.names())
        speak_force(f"అందుబాటులో ఉన్న స్థలాలు: {names_str}. {VC_ASK_DEST}")

        matched = ""
        attempts = 0
        while not matched and attempts < 3:
            attempts += 1
            try:
                audio_dest = recognizer.listen(src, timeout=8, phrase_time_limit=6)
                # Try English ONLY — place names are saved in English
                # Do NOT also try Telugu — it overwrites "hall" with "హాల్"
                dest_text = ""
                # Try en-IN → en-US → te-IN on SAME audio clip
                for _lng in ("en-IN", "en-US", "te-IN"):
                    try:
                        dest_text = recognizer.recognize_google(audio_dest, language=_lng).lower()
                        print(f"[VoiceCmd] STT({_lng}): {dest_text!r}")
                        break   # stop at first success
                    except Exception:
                        continue
            except Exception:
                dest_text = ""

            print(f"[VoiceCmd] Destination heard: {dest_text!r}")

            if not dest_text:
                speak_force("వినబడలేదు మళ్ళీ చెప్పండి")
                continue

            matched = _fuzzy_match_place(dest_text)
            if matched:
                speak_force(f"{matched} వైపు నావిగేషన్ మొదలవుతుంది")
                _vc_queue.put(("vc_nav_dest", matched))
            else:
                remaining = 3 - attempts
                if remaining > 0:
                    avail = ", ".join(DB.names())
                    speak_force(f"స్థలం అర్థం కాలేదు. మళ్ళీ చెప్పండి. అందుబాటులో: {avail}")
                else:
                    speak_force("స్థలం అర్థం కాలేదు. మళ్ళీ కమాండ్ ఇవ్వండి")
        return

    # ── SAVE PLACE ────────────────────────────────────────────────
    if _match_cmd(text, VC_CMD_SAVE_PLACE):
        print("[VoiceCmd] → SAVE PLACE")
        speak_force(VC_ASK_PLACE_NAME)
        name_text = _listen_once(recognizer, src, timeout=8)
        print(f"[VoiceCmd] Place name heard: {name_text}")
        if name_text.strip():
            place_name = name_text.strip().lower()
            speak_force(f"{place_name} స్థలం నమోదు మొదలవుతుంది")
            _vc_queue.put(("vc_save_place", place_name))
        else:
            speak_force(VC_NOT_HEARD)
        return

    # ── SAVE PERSON ───────────────────────────────────────────────
    if _match_cmd(text, VC_CMD_SAVE_PERSON):
        print("[VoiceCmd] → SAVE PERSON")
        speak_force(VC_ASK_PERSON_NAME)
        name_text = _listen_once(recognizer, src, timeout=8)
        print(f"[VoiceCmd] Person name heard: {name_text}")
        if name_text.strip():
            person_name = name_text.strip().lower()
            speak_force(f"{person_name} ముఖం నమోదు మొదలవుతుంది")
            _vc_queue.put(("vc_save_face", person_name))
        else:
            speak_force(VC_NOT_HEARD)
        return

    # ── CHANGE LANGUAGE ───────────────────────────────────────────
    if _match_cmd(text, VC_CMD_CHANGE_LANG):
        print("[VoiceCmd] → CHANGE LANGUAGE")
        # Ask which language in current language
        lang_list = ", ".join(SUPPORTED_LANGUAGES.keys())
        speak_force(f"ఏ భాషకు మార్చాలి? అందుబాటులో ఉన్న భాషలు: {lang_list}")
        try:
            audio_lang = recognizer.listen(src, timeout=7, phrase_time_limit=5)
            # Try multiple STT languages on SAME clip to catch any script
            lang_text = ""
            for _lng in ("en-IN", "te-IN", "hi-IN", "ta-IN", "kn-IN", "ml-IN"):
                try:
                    lang_text = recognizer.recognize_google(audio_lang, language=_lng).lower()
                    if lang_text:
                        break
                except Exception:
                    continue
        except Exception:
            lang_text = ""
        lang_key = ""
        attempts = 0
        while not lang_key and attempts < 3:
            attempts += 1
            if attempts > 1:
                # Re-listen for subsequent attempts
                try:
                    audio_lang2 = recognizer.listen(src, timeout=7, phrase_time_limit=5)
                    lang_text = ""
                    for _lng in ("en-IN", "te-IN", "hi-IN", "ta-IN", "kn-IN", "ml-IN"):
                        try:
                            lang_text = recognizer.recognize_google(audio_lang2, language=_lng).lower()
                            if lang_text: break
                        except Exception: continue
                except Exception:
                    lang_text = ""

            print(f"[VoiceCmd] Language heard: '{lang_text}'")
            if not lang_text:
                speak_force("Not heard. Say a language name: hindi, tamil, kannada, english")
                continue

            lang_key = detect_language_from_text(lang_text)
            if lang_key:
                set_language(lang_key)
                confirm_msg = get_lang_changed_msg(lang_key)
                print(f"[VoiceCmd] Language -> {lang_key}")
                speak_force(confirm_msg)
            else:
                print(f"[VoiceCmd] No language match for: '{lang_text}'")
                remaining = 3 - attempts
                supported = "telugu, hindi, tamil, kannada, malayalam, english"
                if remaining > 0:
                    speak_force(f"Not understood. Say again. Available: {supported}")
                else:
                    speak_force("Language not recognized. Try activate command again")
        return

    # ── READ TEXT ─────────────────────────────────────────────────
    if _match_cmd(text, VC_CMD_READ_TEXT):
        print("[VoiceCmd] → READ TEXT")
        _vc_queue.put(("vc_start_ocr", ""))
        return



    # ── Unknown ────────────────────────────────────────────────────
    print(f"[VoiceCmd] → NO MATCH for: '{text}'")
    speak_force(VC_NOT_HEARD)


def _fuzzy_match_place(text: str) -> str:
    """Find best matching saved place name from spoken text.
    Handles Telugu phonetic versions of English place names."""
    if not text or not DB.places:
        return ""
    text_l = text.lower().strip()

    # Direct match first
    for name in DB.names():
        if name.lower() in text_l or text_l in name.lower():
            return name

    # Partial word match
    words = text_l.split()
    for name in DB.names():
        name_words = name.lower().split()
        if any(w in name_words for w in words):
            return name

    # Phonetic Telugu → English mapping for common place names
    PHONETIC_MAP = {
        "హాల్":"hall", "హాలు":"hall", "హాల":"hall",
        "బెడ్రూమ్":"bedroom", "పడకగది":"bedroom", "బెడ్":"bedroom",
        "కిచెన్":"kitchen", "వంటగది":"kitchen",
        "బాత్రూమ్":"bathroom", "బాత్":"bathroom",
        "స్టడీ రూమ్":"study room", "స్టడీ":"study room", "చదువు గది":"study room",
        "గ్యారేజ్":"garage", "ఆఫీస్":"office",
        "క్లాస్రూమ్":"classroom", "తరగతి":"classroom",
        "గార్డెన్":"garden", "తోట":"garden",
        "కారిడార్":"corridor", "వరండా":"corridor",
        "ఎంట్రన్స్":"entrance", "ముఖద్వారం":"entrance",
        "రోడ్డు":"road", "రోడ్":"road",
        "స్టెయిర్కేస్":"staircase", "మెట్లు":"staircase",
        "లివింగ్ రూమ్":"living room", "హాలు":"living room",
        "డైనింగ్":"dining room", "భోజన గది":"dining room",
        "రూమ్":"room",
    }
    for telugu_word, english_name in PHONETIC_MAP.items():
        if telugu_word in text_l:
            # Now find this English name in DB
            for name in DB.names():
                if english_name in name.lower() or name.lower() in english_name:
                    return name

    # Last resort: character-level overlap
    best_name = ""
    best_score = 0
    for name in DB.names():
        overlap = sum(1 for c in text_l if c in name.lower())
        if overlap > best_score and overlap >= 2:
            best_score = overlap
            best_name  = name
    return best_name


# ══════════════════════════════════════════════════════════════════
# OCR ENGINE — EasyOCR + voice readout
# Triggered by voice command "read text", cancelled by "cancel"
# ══════════════════════════════════════════════════════════════════

_ocr_reader       = None
_ocr_reader_lock  = threading.Lock()
_ocr_active       = threading.Event()   # set when OCR mode is running
_ocr_last_text    = ""
_ocr_last_time    = 0.0
_ocr_frame        = [None]
_ocr_stop_flag    = threading.Event()


def _init_ocr_reader():
    """Load EasyOCR lazily — only when first needed."""
    global _ocr_reader
    with _ocr_reader_lock:
        if _ocr_reader is None:
            try:
                import easyocr
                print("[OCR] Loading EasyOCR model (first time ~30s)...")
                # FIX 1: Added "te" for Telugu support
                _ocr_reader = easyocr.Reader(["en", "te"], gpu=False)
                print("[OCR] EasyOCR ready (en+Telugu) ✅")
            except ImportError:
                print("[OCR] ⚠️  easyocr not installed. Run: pip install easyocr")
                _ocr_reader = None
    return _ocr_reader


# FIX 2: Replaced _detect_text_accurate with multi-pass preprocessing version
def _detect_text_accurate(frame_bgr) -> list:
    """Run EasyOCR on frame with multiple preprocessing passes for accuracy."""
    reader = _ocr_reader
    if reader is None:
        return []
    try:
        h, w = frame_bgr.shape[:2]
        # Scale up small frames — EasyOCR reads better on larger images
        scale = 1.0
        if w < 960:
            scale = 960 / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)
        elif w > 1280:
            scale = 1280 / w
            frame_bgr = cv2.resize(frame_bgr, (1280, int(h * scale)))

        all_results = []

        # --- Pass 1: Original colour image ---
        rgb_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            res1 = reader.readtext(
                rgb_orig, detail=1, paragraph=False,
                width_ths=0.5, height_ths=0.3,
                contrast_ths=0.05, adjust_contrast=0.7,
                batch_size=4, min_size=10,
            )
            all_results.extend(res1)
        except Exception as e:
            print(f"[OCR] Pass1 error: {e}")

        # --- Pass 2: CLAHE-enhanced grayscale (helps faded/low-contrast text) ---
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        rgb_enh = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        try:
            res2 = reader.readtext(
                rgb_enh, detail=1, paragraph=False,
                width_ths=0.5, height_ths=0.3,
                contrast_ths=0.05, adjust_contrast=0.5,
                batch_size=4, min_size=10,
            )
            all_results.extend(res2)
        except Exception as e:
            print(f"[OCR] Pass2 error: {e}")

        # --- Pass 3: Bilateral filter (keeps edges, removes noise — good for curved labels) ---
        bilateral = cv2.bilateralFilter(frame_bgr, 9, 75, 75)
        rgb_bi = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
        try:
            res3 = reader.readtext(
                rgb_bi, detail=1, paragraph=False,
                width_ths=0.5, height_ths=0.3,
                contrast_ths=0.05, adjust_contrast=0.6,
                batch_size=4, min_size=10,
            )
            all_results.extend(res3)
        except Exception as e:
            print(f"[OCR] Pass3 error: {e}")

        # --- Deduplicate: merge results with same/similar text ---
        seen_texts = {}
        for item in all_results:
            if len(item) == 3:
                box, text, conf = item
            elif len(item) == 2:
                box, text = item
                conf = 1.0
            else:
                continue
            text = text.strip()
            if not text or len(text) < 2:
                continue
            # Use lower-case as dedup key
            key = text.lower().replace(" ", "")
            # Keep highest confidence reading for duplicate text
            if key not in seen_texts or conf > seen_texts[key][1]:
                seen_texts[key] = (text, conf, box)

        # Filter by confidence
        out = [(t, c, b) for t, c, b in seen_texts.values() if c > 0.25]
        return out

    except Exception as e:
        print(f"[OCR] Error: {e}")
        return []


def _ocr_worker_thread(cap_ref) -> None:
    """Background thread: runs OCR continuously while OCR mode is active."""
    global _ocr_last_text, _ocr_last_time

    print("[OCR] Worker started")
    _init_ocr_reader()

    while not _ocr_stop_flag.is_set():
        if not _ocr_active.is_set():
            time.sleep(0.2)
            continue

        frame = _ocr_frame[0]
        if frame is None:
            time.sleep(0.2)
            continue

        results = _detect_text_accurate(frame)

        if not results:
            time.sleep(2.0)   # no text — wait longer before retry
            continue

        # Sort by vertical position (top to bottom reading order)
        results.sort(key=lambda r: r[2][0][1])

        # Group into one readable sentence
        all_text = " ".join([t for t, c, b in results if c > 0.4])
        all_text = all_text.strip()

        if not all_text:
            time.sleep(2.0)
            continue

        now = time.time()
        # Speak only when text actually changes
        if all_text != _ocr_last_text:
            _ocr_last_text = all_text
            _ocr_last_time = now
            print(f"[OCR] Detected: {all_text}")
            # FIX 5: Language-aware speak
            lang = _current_lang
            if lang == "telugu":
                speak(f"టెక్స్ట్: {all_text}")
            elif lang == "hindi":
                speak(f"टेक्स्ट: {all_text}")
            else:
                speak(f"Text: {all_text}")

        time.sleep(3.0)   # wait 3s between OCR runs — reduces CPU load

    print("[OCR] Worker stopped")


def start_ocr_mode(cap) -> None:
    """Start OCR mode — loads reader, starts worker, sets mode."""
    set_mode("ocr")
    _ocr_active.set()
    _ocr_last_text = ""

    # Start worker thread if not already running
    t = threading.Thread(target=_ocr_worker_thread, args=(cap,), daemon=True, name="OCRWorker")
    t.start()

    lang = _current_lang
    if lang == "english":
        speak_force("Text reading mode started. Point camera at text. Say cancel to stop.")
    elif lang == "hindi":
        speak_force("टेक्स्ट पढ़ना शुरू हुआ। कैमरा टेक्स्ट पर लगाएं। रोकने के लिए cancel कहें।")
    else:
        speak_force("టెక్స్ట్ చదవడం మొదలైంది. కెమెరాను టెక్స్ట్ వైపు తిప్పండి. ఆపడానికి cancel చెప్పండి.")


def stop_ocr_mode() -> None:
    """Stop OCR mode and return to default."""
    _ocr_active.clear()
    _ocr_last_text = ""
    set_mode("default")
    lang = _current_lang
    if lang == "english":
        speak_force("Text reading stopped.")
    elif lang == "hindi":
        speak_force("टेक्स्ट पढ़ना बंद हो गया।")
    else:
        speak_force("టెక్స్ట్ చదవడం ఆపబడింది.")
    print("[OCR] Mode stopped")


print("\n[PlaceDB] Loading...")
DB = PlaceDB(PLACES_DIR)

# ══════════════════════════════════════════════════════════════════
# SAVE PLACE  (W→save in nav mode uses this)
# ══════════════════════════════════════════════════════════════════

def _do_save_place(cap: cv2.VideoCapture, place_name: str) -> None:
    """Capture 5 photos and store to saved_places/<place_name>/"""
    place_dir = os.path.join(PLACES_DIR, place_name)
    os.makedirs(place_dir, exist_ok=True)
    existing  = len([f for f in os.listdir(place_dir) if f.endswith(".jpg")])

    print(f"\n[SavePlace] Capturing 5 photos of '{place_name}'")
    speak_blocking(f"{place_name} స్థలం నమోదు మొదలవుతుంది కెమెరా చుట్టూ తిప్పండి")
    saved = 0
    for i in range(5):
        print(f"[SavePlace] Photo {i+1}/5 in 2 seconds...")
        time.sleep(2)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(place_dir, f"{existing + i}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
        print(f"[SavePlace] ✅ {i+1}/5 saved")
        flash = frame.copy()
        cv2.putText(flash, f"Saving Place: {place_name}  {i+1}/5",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.imshow("DRISHTI", flash)
        cv2.waitKey(200)

    print(f"[SavePlace] Done — {saved}/5 photos saved for '{place_name}'")
    DB.reload()
    speak_blocking(f"{place_name} స్థలం సేవ్ అయింది")


def _save_place_thread(cap: cv2.VideoCapture, place_name: str) -> None:
    """Wrapper: sets mode to save_place, runs capture, returns to default."""
    set_mode("save_place")
    _face_pause.set()
    try:
        _do_save_place(cap, place_name)
    finally:
        _face_pause.clear()
        set_mode("default")
        print("[SavePlace] Back to default mode.\n")

# ══════════════════════════════════════════════════════════════════
# CLIP HELPERS
# ══════════════════════════════════════════════════════════════════

def clip_similarity(frame_bgr: np.ndarray, dest_name: str) -> float:
    """Compare frame against ALL saved images individually, return top-3 average."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = _clip_proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)      # shape [1, 512]

    place  = DB.places[dest_name]
    scores = []

    # Score against EACH saved image embedding individually
    for i, ref_emb in enumerate(place["all_embs"]):
        # ref_emb shape is [1, 512] — flatten both to [512] for clean dot product
        e1  = emb.squeeze(0)          # [512]
        e2  = ref_emb.squeeze(0)      # [512]
        sim = float(torch.dot(e1, e2))
        scores.append(sim)

    # Count how many individual images score above 0.80
    hits = sum(1 for s in scores if s >= 0.80)

    scores.sort(reverse=True)
    top3 = scores[:min(3, len(scores))]
    final = sum(top3) / len(top3)

    # Smart arrival: if 2 or more images score >= 0.80 → strong arrival signal
    # Boost final score so arrival triggers faster
    if hits >= 2:
        final = max(final, 0.83)   # guarantee arrival threshold crossed

    score_str = "  ".join([f"img{i}:{s:.2f}" for i, s in enumerate(scores)])
    print(f"[CLIP] {score_str}  top3={final:.2f}  hits={hits}/5  dest={dest_name}")

    return final


def clip_match_labels(pil_image: Image.Image, labels: list[str]) -> tuple[str, float]:
    prompts = [f"a photo of a {lbl}" for lbl in labels]
    inputs  = _clip_proc(text=prompts, images=pil_image,
                         return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _clip_model(**inputs).logits_per_image
        probs  = logits.softmax(dim=1)[0]
    idx = probs.argmax().item()
    return labels[idx], float(probs[idx])


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

# ══════════════════════════════════════════════════════════════════
# ORB DIRECTION
# ══════════════════════════════════════════════════════════════════

def orb_direction(frame_bgr: np.ndarray, dest_name: str) -> dict:
    result = {"direction":"forward","offset_x":0,"offset_y":0,
              "arrow_dst":None,"n_matches":0,"confidence":0.0}
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp_cur, desc_cur = _orb.detectAndCompute(gray, None)
    if desc_cur is None or len(desc_cur) < 5:
        return result
    place = DB.places[dest_name]
    h, w  = frame_bgr.shape[:2]
    cx, cy = w//2, h//2
    best_matches=[]; best_ref_kps=None
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
    result["confidence"] = min(1.0, len(best_matches)/40.0)
    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    pts_ref = np.float32([best_ref_kps[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 5.0)
    if H is None:
        offset_x = float(np.mean(pts_cur[:,0,0])) - float(np.mean(pts_ref[:,0,0]))
        result["offset_x"]  = offset_x
        result["direction"] = "left" if offset_x < -30 else ("right" if offset_x > 30 else "forward")
        result["arrow_dst"] = (int(np.mean(pts_cur[:,0,0])), cy)
        return result
    ref_h2, ref_w2 = DB.places[dest_name]["images"][0].shape[:2]
    ref_centre = np.float32([[ref_w2/2, ref_h2/2]]).reshape(-1,1,2)
    try:
        proj = cv2.perspectiveTransform(ref_centre, H)
        px, py = int(proj[0,0,0]), int(proj[0,0,1])
    except Exception:
        px, py = cx, cy
    offset_x = px - cx; offset_y = py - cy
    result["offset_x"]  = offset_x
    result["offset_y"]  = offset_y
    result["arrow_dst"] = (max(30,min(w-30,px)), max(30,min(h-30,py)))
    THRESH = w * 0.12
    if abs(offset_x) < THRESH and abs(offset_y) < THRESH*0.8:
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
                mode="bicubic", align_corners=False).squeeze()
        dm  = pred.cpu().numpy()
        raw = float(dm[max(0,min(dm.shape[0]-1,y)), max(0,min(dm.shape[1]-1,x))])
        return round(DEPTH_SCALE / (raw / dm.max() + 1e-6), 1)
    except Exception:
        return -1.0

# ══════════════════════════════════════════════════════════════════
# NAV WORKER
# ══════════════════════════════════════════════════════════════════

_nav_lock  = threading.Lock()
_nav_state = {
    "sim":0.0,"direction":"forward","offset_x":0,
    "arrow_dst":None,"n_matches":0,"confidence":0.0,
    "depth":-1.0,"arrived":False,
}
_nav_frame = [None]
_nav_dest  = [None]
_nav_stop  = threading.Event()


def nav_worker() -> None:
    last_speak_time = 0.0
    last_direction  = ""
    while not _nav_stop.is_set():
        time.sleep(CLIP_CHECK_EVERY)
        if get_mode() != "nav":
            continue
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
                    lang = _current_lang
                    if lang == "english":
                        speak(f"You have arrived at {dest}")
                    elif lang == "hindi":
                        speak(f"{dest} पहुंच गए")
                    elif lang == "tamil":
                        speak(f"{dest} வந்துவிட்டீர்கள்")
                    elif lang == "kannada":
                        speak(f"{dest} ತಲುಪಿದ್ದೀರಿ")
                    elif lang == "malayalam":
                        speak(f"{dest} എത്തിയിരിക്കുന്നു")
                    else:
                        speak(f"{dest} చేరుకున్నారు")
                    last_speak_time = now
                elif direction != last_direction or now - last_speak_time > 6.0:
                    lang = _current_lang
                    if lang == "english":
                        dir_en = {"left":"turn left","right":"turn right","forward":"go forward"}.get(direction,"go forward")
                        depth_str = f" {depth:.0f} meters" if depth > 0 else ""
                        speak(f"{dir_en}{depth_str}")
                    elif lang == "hindi":
                        dir_hi = {"left":"बाईं तरफ मुड़ें","right":"दाईं तरफ मुड़ें","forward":"आगे जाएं"}.get(direction,"आगे जाएं")
                        depth_str = f" {depth:.0f} मीटर" if depth > 0 else ""
                        speak(f"{dir_hi}{depth_str}")
                    elif lang == "tamil":
                        dir_ta = {"left":"இடது திரும்பவும்","right":"வலது திரும்பவும்","forward":"முன்னே செல்லவும்"}.get(direction,"முன்னே செல்லவும்")
                        speak(dir_ta)
                    elif lang == "kannada":
                        dir_kn = {"left":"ಎಡಕ್ಕೆ ತಿರುಗಿ","right":"ಬಲಕ್ಕೆ ತಿರುಗಿ","forward":"ಮುಂದೆ ಹೋಗಿ"}.get(direction,"ಮುಂದೆ ಹೋಗಿ")
                        speak(dir_kn)
                    elif lang == "malayalam":
                        dir_ml = {"left":"ഇടത്തോട്ട് തിരിയുക","right":"വലത്തോട്ട് തിരിയുക","forward":"മുന്നോട്ട് പോകുക"}.get(direction,"മുന്നോട്ട് പോകുക")
                        speak(dir_ml)
                    else:
                        te_dir    = NAV_MSG.get(direction, "ముందుకు వెళ్ళండి")
                        depth_str = f" {depth:.0f} మీటర్లు" if depth > 0 else ""
                        speak(f"{te_dir}{depth_str}")
                    last_speak_time = now
                    last_direction  = direction
        except Exception as e:
            print(f"[NavWorker] {e}")

# ══════════════════════════════════════════════════════════════════
# NON-BLOCKING INPUT QUEUE
# ══════════════════════════════════════════════════════════════════

_input_queue = queue.Queue()
_input_busy  = threading.Event()


def _ask(tag: str, prompt: str) -> None:
    def _worker():
        _input_busy.set()
        val = input(prompt).strip()
        _input_queue.put((tag, val))
        _input_busy.clear()
    threading.Thread(target=_worker, daemon=True).start()

# ══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════

def draw_nav_arrow(frame, direction, arrow_dst, confidence):
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2
    src = (cx, h-80)
    if arrow_dst and confidence > 0.3:
        tip = arrow_dst
    elif direction == "left":
        tip = (cx-180, cy)
    elif direction == "right":
        tip = (cx+180, cy)
    else:
        tip = (cx, cy-120)
    cv2.arrowedLine(frame, src, tip, (0,0,0), 12, tipLength=0.3)
    cv2.arrowedLine(frame, src, tip, (0,255,80), 6, tipLength=0.3)
    pulse = int(18 + 6*abs(np.sin(time.time()*3)))
    cv2.circle(frame, tip, pulse, (0,255,80), 3)
    cv2.circle(frame, tip, 6, (255,255,255), -1)


def draw_nav_hud(frame, dest, sim, direction, n_matches, depth, confidence):
    h, w = frame.shape[:2]; F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (0,0), (w,50), (10,10,10), -1)
    color = (0,255,80) if sim > 0.75 else (0,180,255)
    cv2.putText(frame, f"Navigating to: {dest.upper()}", (10,32), F, 0.8, color, 2)
    bar_x = w-220
    cv2.putText(frame, "Match:", (bar_x,22), F, 0.5, (180,180,180), 1)
    cv2.rectangle(frame, (bar_x+55,10), (bar_x+160,26), (50,50,50), -1)
    filled = int((bar_x+55) + sim*105)
    bc = (0,255,80) if sim > ARRIVAL_THRESHOLD else (0,180,255)
    cv2.rectangle(frame, (bar_x+55,10), (filled,26), bc, -1)
    cv2.putText(frame, f"{sim*100:.0f}%", (bar_x+165,22), F, 0.5, (220,220,220), 1)
    cv2.rectangle(frame, (0,h-48), (w,h), (10,10,10), -1)
    dir_en = {"left":"<- Turn LEFT","right":"-> Turn RIGHT",
              "forward":"^ Go FORWARD","arrived":"** ARRIVED **"}.get(direction,"^ Go FORWARD")
    cv2.putText(frame, dir_en, (10,h-18), F, 0.85, (0,255,150), 2)
    cv2.putText(frame, f"ORB:{n_matches}pts", (w-300,h-18), F, 0.55, (180,180,180), 1)
    if depth > 0:
        cv2.putText(frame, f"Depth:{depth}m", (w-180,h-18), F, 0.55, (150,220,255), 1)
    cv2.putText(frame, f"Q=Quit  R=Reset  ESC=Back", (w-220,h-36), F, 0.40, (120,120,120), 1)
    if direction == "arrived" or sim >= ARRIVAL_THRESHOLD:
        ov = frame.copy()
        cv2.rectangle(ov, (0,h//2-50), (w,h//2+50), (0,100,0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"  ARRIVED at {dest.upper()}!",
                    (w//2-200,h//2+15), F, 1.1, (0,255,100), 3)


def draw_default_hud(frame, yolo_cache, face_results):
    h, w = frame.shape[:2]; F = cv2.FONT_HERSHEY_SIMPLEX
    # YOLO boxes
    for (name, dist, side, x1, y1, x2, y2) in yolo_cache:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,220,0), 2)
        cv2.putText(frame, f"{name} {dist:.1f}m", (x1, max(y1-10,14)), F, 0.65, (0,220,0), 2)
        cv2.putText(frame, side, (x1,y2+20), F, 0.5, (0,200,255), 2)
    # Face boxes
    with _rec_lock:
        fr_copy = list(face_results)
    for item in fr_copy:
        name, conf, top, right, bottom, left = item
        t = int(top/FACE_SCALE); r2 = int(right/FACE_SCALE)
        b = int(bottom/FACE_SCALE); l = int(left/FACE_SCALE)
        dist  = face_dist(t, b)
        side  = direction_te((l+r2)//2, w)
        color = (0,220,0) if name != "Unknown" else (0,60,220)
        cv2.rectangle(frame, (l,t), (r2,b), color, 2)
        label = f"{name} {conf:.0f}% {dist:.1f}m" if name != "Unknown" else f"Unknown {dist:.1f}m"
        (tw,th),_ = cv2.getTextSize(label, F, 0.72, 2)
        cv2.rectangle(frame, (l,t-th-14), (l+tw+10,t), color, -1)
        cv2.putText(frame, label, (l+5,t-6), F, 0.72, (255,255,255), 2)
        cv2.putText(frame, side, (l,b+22), F, 0.52, (0,200,255), 2)
    # Status bar
    cv2.rectangle(frame, (0,h-32), (w,h), (10,10,10), -1)
    objs = f"Obj:{len(yolo_cache)}"
    faces= f"Faces:{len(fr_copy)}"
    cv2.putText(frame, f"{objs}  {faces}  |  Lang:{_current_lang.upper()}  |  S=Save  W=Nav  Q=Quit",
                (10,h-10), F, 0.47, (160,160,160), 1)


def draw_save_face_banner(frame):
    h, w = frame.shape[:2]
    mode = get_mode()
    if mode == "save_place":
        label = "SAVING PLACE — YOLO & FACE ON HOLD"
    elif mode == "listening":
        label = "VOICE COMMAND ACTIVE — LISTENING..."
    else:
        label = "SAVING FACE — YOLO & NAV ON HOLD"
    color = (0,180,0) if mode == "listening" else (0,50,100)
    cv2.rectangle(frame, (0,0), (w,50), color, -1)
    cv2.putText(frame, label, (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,220,255), 2)

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    if not cap.isOpened():
        print("❌ Camera failed!")
        return

    print("\n✅ Camera started")
    print("   Default: YOLO + Face Recognition")
    print("   S = Save face  |  W = Navigate to place  |  Q = Quit\n")

    # start nav worker thread
    nav_thread = threading.Thread(target=nav_worker, daemon=True, name="NavWorker")
    nav_thread.start()

    # start voice command thread
    if SR_OK:
        vc_thread = threading.Thread(target=_voice_command_thread, daemon=True, name="VoiceCmd")
        vc_thread.start()
        speak_force("దృష్టి సిద్ధంగా ఉంది. listen అని చెప్పి ఆదేశాలు ఇవ్వండి")
    else:
        print("[VoiceCmd] Disabled — install SpeechRecognition and pyaudio")

    # State
    nav_dest          = ""
    ref_thumb         = None
    arrived_time      = 0.0
    last_yolo_speak   = 0.0
    last_face_speak   : dict = {}
    yolo_cache        : list = []
    frame_count       = 0
    fps               = 0.0
    fps_timer         = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now  = time.time()
        mode = get_mode()

        if frame_count % 30 == 0:
            fps       = 30.0 / max(now - fps_timer, 1e-6)
            fps_timer = now

        # ── Feed face worker (only in default mode) ───────────────
        if mode == "default":
            small = cv2.resize(frame, (0,0), fx=FACE_SCALE, fy=FACE_SCALE)
            _latest_small[0] = small

        # ── Feed nav worker (only in nav mode) ────────────────────
        if mode == "nav":
            _nav_frame[0] = frame.copy()

        # ── Feed OCR worker (only in ocr mode) ───────────────────
        if mode == "ocr":
            _ocr_frame[0] = frame.copy()

        # ── Process queued input answers ──────────────────────────
        while not _input_queue.empty():
            tag, val = _input_queue.get_nowait()

            if tag == "save_choice":
                choice = val.lower()
                if choice == "f":
                    threading.Thread(
                        target=_save_face_thread, args=(cap,), daemon=True
                    ).start()
                elif choice == "p":
                    _ask("save_place_name", "[SavePlace] Place name (e.g. kitchen, my room): ")
                else:
                    print(f"[Save] Unknown choice '{val}' — press S again.")

            elif tag == "save_place_name":
                if val:
                    threading.Thread(
                        target=_save_place_thread, args=(cap, val), daemon=True
                    ).start()
                else:
                    print("[SavePlace] No name — skipped.")
                    set_mode("default")

            elif tag == "nav_dest":
                if val in DB.places:
                    nav_dest = val
                    ref_thumb= DB.places[val]["images"][0]
                    _nav_dest[0] = val
                    _nav_stop.clear()
                    with _nav_lock:
                        _nav_state.update({"arrived":False,"direction":"forward",
                                           "sim":0.0,"n_matches":0,"confidence":0.0,
                                           "depth":-1.0,"arrow_dst":None})
                    arrived_time = 0.0
                    set_mode("nav")
                    speak(f"{val} వైపు నావిగేషన్ మొదలవుతుంది")
                    print(f"\n✅ Navigating to: {val}\n")
                else:
                    print(f"❌ '{val}' not found. Available: {', '.join(DB.names())}")
                    set_mode("default")

            elif tag == "reset_dest":
                if val in DB.places:
                    nav_dest     = val
                    _nav_dest[0] = val
                    ref_thumb    = DB.places[val]["images"][0]
                    arrived_time = 0.0
                    speak(f"{val} వైపు నావిగేషన్ మొదలవుతుంది")
                    print(f"✅ Destination → {val}")
                else:
                    print(f"❌ '{val}' not in places.")

        # ── Process voice command queue ────────────────────────
        while not _vc_queue.empty():
            vc_tag, vc_val = _vc_queue.get_nowait()

            if vc_tag == "quit":
                _face_stop.set()
                _nav_stop.set()
                _vc_stop.set()
                cap.release()
                cv2.destroyAllWindows()
                pygame.mixer.quit()
                print("\nGoodbye.")
                return

            elif vc_tag == "vc_stop_nav":
                if get_mode() == "nav":
                    set_mode("default")
                    _nav_dest[0]  = None
                    nav_dest      = ""
                    ref_thumb     = None
                    arrived_time  = 0.0
                    with _nav_lock:
                        _nav_state.update({"arrived":False,"direction":"forward","sim":0.0})
                    print("[VoiceCmd] Nav stopped → default mode")

            elif vc_tag == "vc_nav_dest":
                dest_name = vc_val
                if dest_name in DB.places:
                    nav_dest     = dest_name
                    ref_thumb    = DB.places[dest_name]["images"][0]
                    _nav_dest[0] = dest_name
                    _nav_stop.clear()
                    with _nav_lock:
                        _nav_state.update({"arrived":False,"direction":"forward",
                                           "sim":0.0,"n_matches":0,"confidence":0.0,
                                           "depth":-1.0,"arrow_dst":None})
                    arrived_time = 0.0
                    set_mode("nav")
                    print(f"[VoiceCmd] Navigating to: {dest_name}")

            elif vc_tag == "vc_save_place":
                if get_mode() == "default":
                    threading.Thread(
                        target=_save_place_thread, args=(cap, vc_val), daemon=True
                    ).start()

            elif vc_tag == "vc_save_face":
                if get_mode() == "default":
                    threading.Thread(
                        target=_save_face_thread_named, args=(cap, vc_val), daemon=True
                    ).start()

            elif vc_tag == "vc_start_ocr":
                if get_mode() == "default":
                    start_ocr_mode(cap)

            elif vc_tag == "vc_stop_ocr":
                stop_ocr_mode()

        # ══════════════════════════════════════════════════════════
        # RENDER based on mode
        # ══════════════════════════════════════════════════════════

        display = frame.copy()
        mode    = get_mode()   # refresh after input processing

        # ── SAVE FACE / SAVE PLACE mode ────────────────────────────
        if mode in ("save_face", "save_place", "listening"):
            draw_save_face_banner(display)
            if mode == "save_place":
                action = "saving place..."
            elif mode == "listening":
                action = "voice command active — say your command..."
            else:
                action = "saving face..." 
            cv2.putText(display, f"Please wait — {action}",
                        (10, display.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)
            cv2.imshow("DRISHTI", display)
            cv2.waitKey(1)
            continue   # skip all key processing while saving

        # ── OCR mode ───────────────────────────────────────────────
        elif mode == "ocr":
            # Draw detected text boxes on screen
            # Show last OCR results on screen (from worker thread — no lag)
            if _ocr_last_text:
                F_ocr = cv2.FONT_HERSHEY_SIMPLEX
                # Display detected text in a readable box at bottom
                h_d2, w_d2 = display.shape[:2]
                cv2.rectangle(display, (0, h_d2-80), (w_d2, h_d2-45), (0,40,0), -1)
                # Truncate long text for display
                disp_text = _ocr_last_text[:80] + "..." if len(_ocr_last_text) > 80 else _ocr_last_text
                cv2.putText(display, disp_text, (10, h_d2-55),
                            F_ocr, 0.55, (0,255,100), 1)
            # Status bar
            h_d, w_d = display.shape[:2]
            cv2.rectangle(display, (0,0), (w_d,45), (0,60,0), -1)
            # FIX 4: English-only banner (OpenCV can't render Telugu Unicode)
            cv2.putText(display, "OCR MODE | Point camera at text | Say CANCEL to stop",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,100), 2)
            cv2.imshow("DRISHTI", display)
            cv2.waitKey(1)
            # ESC or Q exits OCR mode
            key_ocr = cv2.waitKey(1) & 0xFF
            if key_ocr == 27 or key_ocr == ord("q"):
                stop_ocr_mode()
            continue

        # ── NAV mode ───────────────────────────────────────────────
        elif mode == "nav":
            with _nav_lock:
                state = dict(_nav_state)

            # feature dots
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            kps_d, _ = _orb.detectAndCompute(gray, None)
            if kps_d:
                for kp in kps_d[:30]:
                    cv2.circle(display, (int(kp.pt[0]),int(kp.pt[1])), 3, (0,200,255), -1)

            if not state["arrived"]:
                draw_nav_arrow(display, state["direction"],
                               state["arrow_dst"], state["confidence"])

            # thumbnail
            if ref_thumb is not None:
                tw,th = 160,100
                thumb = cv2.resize(ref_thumb,(tw,th))
                h_d, w_d = display.shape[:2]
                display[15:15+th, w_d-tw-5:w_d-5] = thumb
                cv2.putText(display, f"Target: {nav_dest}",
                            (w_d-tw-5,15+th+18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(200,200,200),1)

            draw_nav_hud(display, nav_dest, state["sim"], state["direction"],
                         state["n_matches"], state["depth"], state["confidence"])

            # auto-return to default after arrival
            if state["arrived"]:
                if arrived_time == 0.0:
                    arrived_time = now
                    print(f"\n✅ Arrived at '{nav_dest}'! Returning to default in 3 s...")
                if now - arrived_time >= 3.0:
                    set_mode("default")
                    _nav_dest[0]  = None
                    arrived_time  = 0.0
                    nav_dest      = ""
                    ref_thumb     = None
                    with _nav_lock:
                        _nav_state.update({"arrived":False,"direction":"forward","sim":0.0})
                    print("[NAV] Back to default mode.\n")
            else:
                arrived_time = 0.0

        # ── DEFAULT mode ────────────────────────────────────────────
        else:
            # YOLO
            if frame_count % FRAME_SKIP == 0:
                results    = yolo(frame, verbose=False, conf=CONF_THRESHOLD)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls  = int(box.cls[0])
                        name = yolo.names[cls]
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cx   = (x1+x2)//2
                        bh   = y2-y1
                        ref  = OBJECT_REF_HEIGHT_M.get(name, DEFAULT_REF_H)
                        dist = bbox_dist(bh, ref)
                        side = direction_te(cx, frame.shape[1])
                        detections.append((name, dist, side, x1, y1, x2, y2))
                detections.sort(key=lambda d: d[1])
                yolo_cache = detections

                if yolo_cache and now - last_yolo_speak > YOLO_COOLDOWN:
                    n0,d0,s0 = yolo_cache[0][0], yolo_cache[0][1], yolo_cache[0][2]
                    tel = TELUGU_OBJ.get(n0, n0)
                    lang = _current_lang
                    if lang == "english":
                        sv0 = {"Left":"on your left","Right":"on your right","Center":"in front of you"}.get(s0,s0)
                        tel_en = n0  # use English object name
                        speak(f"{sv0} {tel_en} {d0} meters away watch out")
                    elif lang == "hindi":
                        sv0 = {"Left":"బాईं తరఫ","Right":"దాईం తరఫ","Center":"సామ్నే"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} మీటర్ దూర్ హై సావధాన్")
                    elif lang == "tamil":
                        sv0 = {"Left":"இடது பக்கம்","Right":"வலது பக்கம்","Center":"முன்னால்"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} மீட்டர் தொலைவில் உள்ளது கவனமாக இருங்கள்")
                    elif lang == "kannada":
                        sv0 = {"Left":"ಎಡಭಾಗ","Right":"ಬಲಭಾಗ","Center":"ಮುಂದೆ"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} ಮೀಟರ್ ದೂರದಲ್ಲಿದೆ ಎಚ್ಚರಿಕೆ")
                    elif lang == "malayalam":
                        sv0 = {"Left":"ഇടതുവശം","Right":"വലതുവശം","Center":"മുന്നിൽ"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} മീറ്റർ അകലെ ഉണ്ട് ശ്രദ്ധിക്കുക")
                    else:
                        sv0 = {"Left":"ఎడమ వైపు","Right":"కుడి వైపు","Center":"మీ ముందు"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} మీటర్ల దూరంలో వున్నారు చూసుకోండి")
                    last_yolo_speak = now

            # Face voice
            with _rec_lock:
                face_results_snap = list(_rec_results)

            seen = set()
            sorted_faces = sorted(face_results_snap,
                key=lambda d: -(int(d[4]/FACE_SCALE)-int(d[2]/FACE_SCALE)))
            for item in sorted_faces:
                name, conf, top, right, bottom, left = item
                seen.add(name)
                t = int(top/FACE_SCALE); r2 = int(right/FACE_SCALE)
                b = int(bottom/FACE_SCALE); l2 = int(left/FACE_SCALE)
                dist = face_dist(t, b)
                side = direction_te((l2+r2)//2, frame.shape[1])
                last_t = last_face_speak.get(name, 0.0)
                if now - last_t >= FACE_COOLDOWN:
                    last_face_speak[name] = now
                    # Build message in current active language
                    lang = _current_lang
                    if lang == "english":
                        side_voice = {"Left":"on your left","Right":"on your right","Center":"in front of you"}.get(side, side)
                        if name == "Unknown":
                            msg = f"{side_voice} unknown person {dist:.1f} meters away please be careful"
                        else:
                            msg = f"{side_voice} {name} {dist:.1f} meters away please be careful"
                    elif lang == "hindi":
                        side_voice = {"Left":"बाईं तरफ","Right":"दाईं तरफ","Center":"सामने"}.get(side, side)
                        if name == "Unknown":
                            msg = f"{side_voice} अजनबी व्यक्ति {dist:.1f} मीटर दूर है सावधान रहें"
                        else:
                            msg = f"{side_voice} {name} {dist:.1f} मीटर दूर है सावधान रहें"
                    elif lang == "tamil":
                        side_voice = {"Left":"இடது பக்கம்","Right":"வலது பக்கம்","Center":"முன்னால்"}.get(side, side)
                        if name == "Unknown":
                            msg = f"{side_voice} தெரியாத நபர் {dist:.1f} மீட்டர் தொலைவில் உள்ளார் கவனமாக இருங்கள்"
                        else:
                            msg = f"{side_voice} {name} {dist:.1f} மீட்டர் தொலைவில் உள்ளார் கவனமாக இருங்கள்"
                    elif lang == "kannada":
                        side_voice = {"Left":"ಎಡಭಾಗ","Right":"ಬಲಭಾಗ","Center":"ಮುಂದೆ"}.get(side, side)
                        if name == "Unknown":
                            msg = f"{side_voice} ಅಪರಿಚಿತ ವ್ಯಕ್ತಿ {dist:.1f} ಮೀಟರ್ ದೂರದಲ್ಲಿದ್ದಾರೆ ಎಚ್ಚರಿಕೆ"
                        else:
                            msg = f"{side_voice} {name} {dist:.1f} ಮೀಟರ್ ದೂರದಲ್ಲಿದ್ದಾರೆ ಎಚ್ಚರಿಕೆ"
                    elif lang == "malayalam":
                        side_voice = {"Left":"ഇടതുവശം","Right":"വലതുവശം","Center":"മുന്നിൽ"}.get(side, side)
                        if name == "Unknown":
                            msg = f"{side_voice} അപരിചിതൻ {dist:.1f} മീറ്റർ അകലെ ഉണ്ട് ശ്രദ്ധിക്കുക"
                        else:
                            msg = f"{side_voice} {name} {dist:.1f} മീറ്റർ അകലെ ഉണ്ട് ശ്രദ്ധിക്കുക"
                    else:
                        # Default Telugu
                        side_voice = {"Left":"ఎడమ వైపు","Right":"కుడి వైపు","Center":"మీ ముందు"}.get(side, side)
                        if name == "Unknown":
                            msg = f"{side_voice} గుర్తు తెలియని వ్యక్తి {dist:.1f} మీటర్ల దూరంలో వున్నారు చూసుకొని వెళ్ళండి"
                        else:
                            msg = f"{side_voice} {name} {dist:.1f} మీటర్ల దూరంలో వున్నారు చూసుకొని వెళ్ళండి"
                    speak(msg)
                    break
            for name in list(last_face_speak.keys()):
                if name not in seen:
                    del last_face_speak[name]

            draw_default_hud(display, yolo_cache, _rec_results)

        # FPS overlay
        cv2.putText(display, f"{fps:.0f}fps",
                    (display.shape[1]-75, display.shape[0]-52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

        cv2.imshow("DRISHTI", display)
        key = cv2.waitKey(1) & 0xFF

        # ── Q: Quit ────────────────────────────────────────────────
        if key == ord("q"):
            break

        # ── S: context-sensitive save ──────────────────────────────
        if key == ord("s") and not _input_busy.is_set():
            if mode == "default":
                # Ask: save face or save place?
                print("\n" + "═"*50)
                print("  SAVE — What do you want to save?")
                print("  F = Face  |  P = Place")
                print("═"*50)
                _ask("save_choice", "Enter F (face) or P (place): ")
            elif mode == "nav":
                # Save a new place directly while in nav
                _ask("save_place_name", "[SavePlace] Place name: ")

        # ── W: enter nav (default mode only) ──────────────────────
        if key == ord("w") and mode == "default" and not _input_busy.is_set():
            if not DB.places:
                print("\n⚠️  No saved places! Press S in nav mode to save one first.\n")
            else:
                print("\n" + "═"*55)
                print("  NAVIGATION — Available places:")
                for i, nm in enumerate(DB.names(), 1):
                    print(f"  {i}. {nm}")
                print("═"*55)
                _ask("nav_dest", "Enter destination name: ")
                set_mode("nav")   # will revert if name invalid (handled in input processing)

        # ── R: reset destination (nav mode only) ──────────────────
        if key == ord("r") and mode == "nav" and not _input_busy.is_set():
            print("\nAvailable places:")
            for i, nm in enumerate(DB.names(), 1):
                print(f"  {i}. {nm}")
            _ask("reset_dest", "New destination: ")

        # ── ESC: exit nav → default ────────────────────────────────
        if key == 27 and mode == "nav":
            set_mode("default")
            nav_dest      = ""
            ref_thumb     = None
            arrived_time  = 0.0
            _nav_dest[0]  = None
            with _nav_lock:
                _nav_state.update({"arrived":False,"direction":"forward","sim":0.0})
            print("\n[NAV] Exited → back to default mode.\n")

    # ── Cleanup ────────────────────────────────────────────────────
    _face_stop.set()
    _nav_stop.set()
    _vc_stop.set()
    _ocr_stop_flag.set()
    _ocr_active.clear()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("\nGoodbye.")


if __name__ == "__main__":
    main()
"""
╔══════════════════════════════════════════════════════════════════════╗
║   DRISHTI COMBINED — VOICE AUTH + YOLO + FACE + PLACE NAVIGATION    ║
║                                                                      ║
║   STARTUP SEQUENCE:                                                  ║
║     1. OpenWakeWord listens silently for "Hey Jarvis"                ║
║        → App is INVISIBLE / silent until wake word fires             ║
║     2. Wake word detected → mic released → Voice Auth runs           ║
║     3. Owner verified → models load in background → DRISHTI starts  ║
║                                                                      ║
║   DEFAULT MODE  : YOLO object detection + Face recognition           ║
║   VOICE (always): say "listen" → menu in Telugu                      ║
║   S key         : Save a face or place (keyboard fallback)           ║
║   W key         : Enter place-navigation mode (keyboard fallback)    ║
║   R key         : Reset nav destination (nav mode only)              ║
║   ESC key       : Exit nav → back to default mode                    ║
║   Q key         : Quit                                               ║
║                                                                      ║
║   NEW VOICE COMMANDS:                                                 ║
║   "send message" → asks number, confirms, asks msg, sends SMS        ║
║   "s.o.s"        → emergency call via Twilio                         ║
║                                                                      ║
║   INSTALL:                                                            ║
║   pip install openwakeword                                           ║
║   pip install torch torchvision transformers                         ║
║   pip install opencv-contrib-python numpy pillow                     ║
║   pip install edge-tts pygame timm                                   ║
║   pip install ultralytics face_recognition                           ║
║   pip install SpeechRecognition pyaudio                              ║
║   pip install sounddevice scipy speechbrain                          ║
║   pip install twilio easyocr                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import re, os, asyncio, numpy as np, sounddevice as sd, pygame, edge_tts
from scipy.io.wavfile import write as wav_write

# ══════════════════════════════════════════════════════════════════
# SECTION -1 — OPENWAKEWORD GATE  ("Hey Jarvis")
#
#  • Uses pyaudio to read raw mic chunks — completely independent of
#    pygame / SpeechRecognition so there are ZERO audio conflicts.
#  • Once the wake word fires we stop the pyaudio stream and give the
#    OS 0.5 s to release the device before the rest of the app opens it.
#  • If openwakeword or pyaudio is missing we skip silently and
#    proceed straight to auth (backward-compatible).
# ══════════════════════════════════════════════════════════════════

_OWW_CHUNK        = 1280          # 80 ms @ 16 kHz — OWW's expected frame size
_OWW_SAMPLE_RATE  = 16000
_OWW_THRESHOLD    = 0.5           # confidence threshold (0–1); lower = more sensitive
_OWW_MODEL_NAME   = "hey_jarvis"  # built-in OWW model name

def _wait_for_wake_word() -> None:
    """Block until 'Hey Jarvis' is detected, then return.
    If openwakeword or pyaudio is unavailable, returns immediately."""

    # ── Try importing dependencies ────────────────────────────────
    try:
        import openwakeword
        from openwakeword.model import Model as OWWModel
        import pyaudio
    except ImportError as _ie:
        print(f"[WakeWord] openwakeword / pyaudio not found ({_ie}) — skipping wake-word gate.")
        print("[WakeWord] Run:  pip install openwakeword pyaudio")
        return

    print("\n" + "═"*60)
    print("  DRISHTI — WAKE WORD LISTENER")
    print("  Say  >>>  Hey Jarvis  <<<  to start")
    print("═"*60)

    # ── Download pre-trained models if not already cached ─────────
    try:
        print("[WakeWord] Checking pre-trained models...")
        openwakeword.utils.download_models()
        print("[WakeWord] Models ready ✅")
    except Exception as e:
        print(f"[WakeWord] Model download warning: {e} — continuing anyway")

    # ── Load the wake-word model ──────────────────────────────────
    try:
        oww = OWWModel(wakeword_models=[_OWW_MODEL_NAME], inference_framework="onnx")
        print(f"[WakeWord] Model '{_OWW_MODEL_NAME}' loaded ✅")
    except Exception as e:
        print(f"[WakeWord] Model load failed ({e}) — skipping wake-word gate.")
        return

    # ── Open a pyaudio input stream ───────────────────────────────
    pa = pyaudio.PyAudio()
    stream = None
    detected = False
    try:
        stream = pa.open(
            rate=_OWW_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=_OWW_CHUNK,
        )
        stream.start_stream()
        print("[WakeWord] Listening... (mic is active)\n")

        # ── Listen loop ───────────────────────────────────────────
        while True:
            try:
                raw = stream.read(_OWW_CHUNK, exception_on_overflow=False)
            except OSError:
                continue

            # OWW expects a 1-D numpy int16 array
            audio_chunk = np.frombuffer(raw, dtype=np.int16)

            # Run inference — returns dict {model_name: numpy_scalar}
            # e.g. {"hey_jarvis": 0.032}  — value is a plain float/numpy scalar
            predictions = oww.predict(audio_chunk)

            # Safely extract score regardless of whether value is
            # a scalar, a list, or a numpy array
            raw_val = predictions.get(_OWW_MODEL_NAME, 0.0)
            if hasattr(raw_val, '__len__'):
                # It's a list/array — take the last element
                score = float(raw_val[-1]) if len(raw_val) > 0 else 0.0
            else:
                # It's already a plain scalar
                score = float(raw_val)

            if score > 0.1:   # print any non-trivial activity for debugging
                print(f"[WakeWord] score={score:.3f}", end="\r", flush=True)

            if score >= _OWW_THRESHOLD:
                print(f"\n[WakeWord] ✅ 'Hey Jarvis' detected!  (score={score:.2f})")
                print("[WakeWord] Handing off to DRISHTI auth...\n")
                detected = True
                break

    except Exception as e:
        print(f"\n[WakeWord] Stream error: {e}")
        print("[WakeWord] Skipping wake-word gate and continuing...")
    finally:
        # ── CRITICAL: fully release the mic before anything else opens it ──
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        try:
            pa.terminate()
        except Exception:
            pass

    import time as _t
    _t.sleep(0.6)   # give the OS time to fully release the audio device


# ── Run the wake-word gate NOW (blocks until "Hey Jarvis") ────────
_wait_for_wake_word()


# ══════════════════════════════════════════════════════════════════
# SECTION 0 — VOICE AUTHENTICATION
# ══════════════════════════════════════════════════════════════════

print("[Auth] Loading SpeechBrain speaker model...")
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
_auth_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec",
    local_strategy=LocalStrategy.COPY,
)
print("[Auth] Speaker model loaded ✅")

_AUTH_SAMPLE_RATE  = 16000
_AUTH_DURATION     = 5
_OWNER_EMBED_PATH  = "owner_voice.npy"
_OWNER_FOLDER      = "owner_recordings"
_AUTH_THRESHOLD    = 0.75
_AUTH_MAX_ATTEMPTS = 3
os.makedirs(_OWNER_FOLDER, exist_ok=True)

_pygame_auth_ok = False
for _freq in [44100, 22050, 16000]:
    try:
        pygame.mixer.pre_init(_freq, -16, 1, 512)
        pygame.mixer.init()
        _pygame_auth_ok = True
        print(f"[Auth] pygame mixer OK (freq={_freq})")
        break
    except Exception as _pe:
        print(f"[Auth] pygame {_freq}Hz failed: {_pe}")
        try: pygame.mixer.quit()
        except Exception: pass

async def _auth_tts_async(text: str, filename: str = "auth_temp.mp3") -> None:
    comm = edge_tts.Communicate(text, voice="te-IN-ShrutiNeural")
    await comm.save(filename)
    if _pygame_auth_ok:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.05)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    try: os.remove(filename)
    except Exception: pass

def _auth_speak(text: str) -> None:
    print(f"[Auth TTS] {text}")
    try: asyncio.run(_auth_tts_async(text))
    except Exception as e: print(f"[Auth TTS] Error: {e}")

def _auth_record(filename: str, duration: int = _AUTH_DURATION) -> str:
    print(f"[Auth] Recording {duration}s → {filename}")
    audio = sd.rec(int(duration * _AUTH_SAMPLE_RATE), samplerate=_AUTH_SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    wav_write(filename, _AUTH_SAMPLE_RATE, audio)
    return filename

def _auth_extract_embedding(wav_path: str) -> np.ndarray:
    signal = _auth_model.load_audio(wav_path)
    emb    = _auth_model.encode_batch(signal)
    return emb.squeeze().detach().cpu().numpy()

def _auth_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def _enroll_owner() -> None:
    print("\n[Auth] ══ OWNER ENROLLMENT ══")
    _auth_speak("దృష్టిలో మీ గొంతు నమోదు మొదలవుతుంది. మూడు సార్లు మాట్లాడండి")
    embeddings = []
    for i in range(3):
        print(f"[Auth] Sample {i+1}/3")
        _auth_speak(f"నమూనా {i+1}. ఇప్పుడు మాట్లాడండి")
        wav_path = os.path.join(_OWNER_FOLDER, f"owner_{i}.wav")
        _auth_record(wav_path, duration=_AUTH_DURATION)
        emb = _auth_extract_embedding(wav_path)
        embeddings.append(emb)
        print(f"[Auth] Sample {i+1} captured ✅")
    owner_emb = np.mean(embeddings, axis=0)
    np.save(_OWNER_EMBED_PATH, owner_emb)
    print("[Auth] Owner voice saved ✅")
    _auth_speak("మీ గొంతు నమోదు అయింది. దృష్టి మొదలవుతుంది")

def _verify_owner() -> bool:
    if not os.path.exists(_OWNER_EMBED_PATH):
        print("[Auth] No owner profile found — running first-time enrollment")
        _auth_speak("మొదటిసారి వాడుతున్నారు. మీ గొంతు నమోదు చేయాలి.")
        _enroll_owner()
        _auth_speak("నమోదు పూర్తయింది. ఇప్పుడు ధృవీకరణ చేయండి.")
    owner_emb = np.load(_OWNER_EMBED_PATH)
    print("\n[Auth] ══ VOICE VERIFICATION ══")
    _auth_speak("దృష్టి తెరవడానికి మీ గొంతు ధృవీకరించండి. మాట్లాడండి")
    for attempt in range(1, _AUTH_MAX_ATTEMPTS + 1):
        print(f"[Auth] Attempt {attempt}/{_AUTH_MAX_ATTEMPTS}")
        wav_tmp = "auth_verify_tmp.wav"
        _auth_record(wav_tmp, duration=_AUTH_DURATION)
        test_emb = _auth_extract_embedding(wav_tmp)
        try: os.remove(wav_tmp)
        except Exception: pass
        score = _auth_cosine(owner_emb, test_emb)
        print(f"[Auth] Similarity score: {score:.4f}  (threshold={_AUTH_THRESHOLD})")
        if score >= _AUTH_THRESHOLD:
            print("[Auth] ✅ OWNER VERIFIED — launching DRISHTI")
            _auth_speak("యజమాని ధృవీకరించబడ్డారు. దృష్టి మొదలవుతుంది")
            return True
        else:
            remaining = _AUTH_MAX_ATTEMPTS - attempt
            print(f"[Auth] ❌ Not verified (score={score:.2f}). Remaining: {remaining}")
            if remaining > 0:
                _auth_speak(f"గుర్తుపట్టలేదు. మళ్ళీ ప్రయత్నించండి. {remaining} అవకాశాలు మిగిలాయి")
            else:
                _auth_speak("గుర్తుపట్టలేదు. దృష్టి నిలిపివేయబడింది")
    return False

# ── FIX: Start loading heavy models in background DURING auth ─────
import threading as _bg_thread
import importlib, time

_model_ready = {
    "yolo": False, "clip": False, "midas": False,
    "face_rec": False,
}
_model_lock = _bg_thread.Lock()

def _preload_models_bg():
    """Load all heavy models in background while auth is happening."""
    global yolo, _clip_model, _clip_proc, MIDAS_OK, _midas, _midas_xfm
    import cv2, torch
    from pathlib import Path
    from PIL import Image

    # YOLO
    try:
        from ultralytics import YOLO as _YOLO
        yolo = _YOLO("yolov8n.pt")
        yolo.fuse()
        with _model_lock: _model_ready["yolo"] = True
        print("[YOLO] Ready ✅  (preloaded during auth)")
    except Exception as e:
        print(f"[YOLO] Failed: {e}")

    # CLIP
    try:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        with _model_lock: _model_ready["clip"] = True
        print("[CLIP] Ready ✅  (preloaded during auth)")
    except Exception as e:
        print(f"[CLIP] Failed: {e}")

    # MiDaS
    try:
        MIDAS_OK = False
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
                _midas = MidasNet_small(path=None, features=64, backbone="efficientnet_lite3",
                                        exportable=True, non_negative=True, blocks={"expand": True})
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
                    print("[MiDaS] Loaded from local cache ✅  (preloaded during auth)")
            except Exception as ex:
                print(f"[MiDaS] Local load failed: {ex}")
        if not loaded_locally:
            _midas     = torch.hub.load("intel-isl/MiDaS","MiDaS_small",trust_repo=True)
            _midas_xfm = torch.hub.load("intel-isl/MiDaS","transforms",trust_repo=True).small_transform
            _midas.eval()
            MIDAS_OK = True
            print("[MiDaS] Ready ✅  (preloaded during auth)")
        with _model_lock: _model_ready["midas"] = True
    except Exception as e:
        print(f"[MiDaS] Not available ({e}) — depth disabled")
        MIDAS_OK = False
        with _model_lock: _model_ready["midas"] = True  # mark done even if failed

    # face_recognition (import is slow)
    try:
        import face_recognition as _fr_preload
        with _model_lock: _model_ready["face_rec"] = True
        print("[FaceRec] Preloaded ✅")
    except Exception as e:
        print(f"[FaceRec] Preload failed: {e}")
        with _model_lock: _model_ready["face_rec"] = True

# Stub globals so preload thread can assign them
yolo = None
_clip_model = None
_clip_proc  = None
MIDAS_OK    = False
_midas      = None
_midas_xfm  = None

# Start preloading immediately
_preload_thread = _bg_thread.Thread(target=_preload_models_bg, daemon=True, name="ModelPreload")
_preload_thread.start()
print("[Preload] Models loading in background during authentication...")

# ── RUN AUTH ──────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  DRISHTI — VOICE AUTHENTICATION")
print("═"*60)

if not _verify_owner():
    print("\n🚫 Authentication failed. Exiting.")
    pygame.mixer.quit()
    raise SystemExit(1)

print("\n✅ Authenticated. Waiting for models...")

# Wait for models that haven't finished yet
_preload_thread.join(timeout=120)
print("✅ All models ready. Starting DRISHTI...\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS (models already loaded above)
# ══════════════════════════════════════════════════════════════════

import cv2, threading, tempfile, pickle, queue
from pathlib import Path
from PIL import Image

try:
    import speech_recognition as _sr
    SR_OK = True
except ImportError:
    SR_OK = False
    print('[VoiceCmd] speech_recognition not installed — voice commands disabled')

import torch
from transformers import CLIPProcessor, CLIPModel

try:
    from twilio.rest import Client as _TwilioClient
    TWILIO_OK = True
    print("[Twilio] Available ✅")
except ImportError:
    TWILIO_OK = False
    print("[Twilio] Not installed — SMS/SOS disabled. Run: pip install twilio")

import face_recognition as _fr

TWILIO_ACCOUNT_SID = "ACe4dc585d5e791912e0661c1b4477ac3c"
TWILIO_AUTH_TOKEN  = "abafdf8620f76808868e3af8600edb2e"
TWILIO_FROM_NUMBER = "+19015935722"
SOS_NUMBERS        = ["+917842174988"]

# ══════════════════════════════════════════════════════════════════
# GLOBAL MODE FLAGS
# ══════════════════════════════════════════════════════════════════

APP_MODE      = "default"
APP_MODE_LOCK = threading.Lock()

def get_mode() -> str:
    with APP_MODE_LOCK: return APP_MODE

def set_mode(m: str) -> None:
    global APP_MODE
    with APP_MODE_LOCK: APP_MODE = m

# ══════════════════════════════════════════════════════════════════
# LANGUAGE CONFIG
# ══════════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES = {
    "telugu"    : ("te-IN-ShrutiNeural",  "te-IN",  "తెలుగు"),
    "hindi"     : ("hi-IN-SwaraNeural",   "hi-IN",  "हिंदी"),
    "tamil"     : ("ta-IN-PallaviNeural", "ta-IN",  "தமிழ்"),
    "kannada"   : ("kn-IN-GaganNeural",   "kn-IN",  "ಕನ್ನಡ"),
    "malayalam" : ("ml-IN-SobhanaNeural", "ml-IN",  "മലയാളം"),
    "bengali"   : ("bn-IN-TanishaaNeural","bn-IN",  "বাংলা"),
    "marathi"   : ("mr-IN-AarohiNeural",  "mr-IN",  "मराठी"),
    "english"   : ("en-IN-NeerjaNeural",  "en-IN",  "English"),
    "gujarati"  : ("gu-IN-DhwaniNeural",  "gu-IN",  "ગુજરાતી"),
    "urdu"      : ("ur-PK-UzmaNeural",    "ur-PK",  "اردو"),
}
LANG_SPOKEN_FORMS = {
    "telugu"   :["telugu","Telugu","తెలుగు","తెలుగ్","तेलुगु","தெலுங்கு","ತೆಲುಗು","തെലുഗു"],
    "hindi"    :["hindi","Hindi","హిందీ","हिंदी","இந்தி","ಹಿಂದಿ","ഹിന്ദി"],
    "tamil"    :["tamil","Tamil","తమిళం","తమిళ్","तमिल","தமிழ்","ತಮಿಳು","തമിഴ്"],
    "kannada"  :["kannada","Kannada","కన్నడ","कन्नड","கன்னடம்","ಕನ್ನಡ","കന്നഡ"],
    "malayalam":["malayalam","Malayalam","మలయాళం","मलयालम","மலையாளம்","ಮಲಯಾಳಂ","മലയാളം"],
    "bengali"  :["bengali","Bengali","బెంగాలీ","बंगाली","வங்காளம்","ಬೆಂಗಾಲಿ","ബംഗാളി","বাংলা"],
    "marathi"  :["marathi","Marathi","మరాఠీ","मराठी","மராத்தி","ಮರಾಠಿ","മറാഠി"],
    "english"  :["english","English","ఇంగ్లీష్","अंग्रेजी","ஆங்கிலம்","ಇಂಗ್ಲಿಷ್","ഇംഗ്ലീഷ്"],
    "gujarati" :["gujarati","Gujarati","గుజరాతీ","गुजराती","குஜராத்தி","ಗುಜರಾತಿ","ഗുജറാത്തി","ગુજرাதী"],
    "urdu"     :["urdu","Urdu","ఉర్దూ","उर्दू","உருது","ಉರ್ದು","ഉർദു","اردو"],
}
_current_lang      = "telugu"
_current_lang_lock = threading.Lock()

def get_tts_voice() -> str:
    with _current_lang_lock: return SUPPORTED_LANGUAGES[_current_lang][0]
def get_stt_lang() -> str:
    with _current_lang_lock: return SUPPORTED_LANGUAGES[_current_lang][1]
def set_language(lang_key: str) -> bool:
    global _current_lang
    if lang_key in SUPPORTED_LANGUAGES:
        with _current_lang_lock: _current_lang = lang_key
        return True
    return False
def detect_language_from_text(text: str) -> str:
    t = text.lower()
    for k, forms in LANG_SPOKEN_FORMS.items():
        if any(f.lower() in t for f in forms): return k
    return ""
def get_lang_changed_msg(lang_key: str) -> str:
    return {"telugu":"భాష తెలుగుకు మార్చబడింది","hindi":"भाषा हिंदी में बदल दी गई है",
            "tamil":"மொழி தமிழுக்கு மாற்றப்பட்டது","kannada":"ಭಾಷೆ ಕನ್ನಡಕ್ಕೆ ಬದಲಾಯಿಸಲಾಗಿದೆ",
            "malayalam":"ഭാഷ മലയാളത്തിലേക്ക് മാറ്റി","english":"Language changed to English",
            "bengali":"ভাষা বাংলায় পরিবর্তিত হয়েছে","marathi":"भाषा मराठीत बदलली आहे",
            "gujarati":"ભાષા ગુજरাதીમાં બدলвामां આвी","urdu":"زبان اردو میں تبدیل کر دی گئی",
            }.get(lang_key, "Language changed")
def get_menu_in_current_lang() -> str:
    menus = {
        "telugu" : ("మీకు ఎనిమిది ఆదేశాలు అందుబాటులో ఉన్నాయి. "
                    "ఒకటి స్థలం సేవ్. రెండు వ్యక్తిని సేవ్. మూడు స్థలానికి వెళ్ళు. "
                    "నాలుగు నావిగేషన్ ఆపు. అయిదు భాష మార్చు. ఆరు టెక్స్ట్ చదువు. "
                    "ఏడు మెసేజ్ పంపు. ఎనిమిది ఎస్ ఓ ఎస్."),
        "hindi"  : ("आपके पास आठ आदेश हैं। एक जगह सेव। दो व्यक्ति सेव। "
                    "तीन जगह जाओ। चार नेविगेशन बंद। पाँच भाषा बदलो। "
                    "छह टेक्स्ट पढ़ो। सात संदेश भेजो। आठ एस ओ एस।"),
        "english": ("You have eight commands. One save place. Two save person. "
                    "Three navigate. Four stop navigation. Five change language. "
                    "Six read text. Seven send message. Eight S O S."),
    }
    with _current_lang_lock: key = _current_lang
    return menus.get(key, menus["english"])

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

PLACES_DIR = "saved_places"; SAVE_DIR = "saved_faces"; DB_FILE = "face_db.pkl"
CLIP_CHECK_EVERY=1.5; ORB_MIN_MATCHES=8; ARRIVAL_THRESHOLD=0.82; DEPTH_SCALE=5.0
YOLO_MODEL="yolov8n.pt"; FRAME_SKIP=4; YOLO_COOLDOWN=4.0; FACE_COOLDOWN=4.0
CONF_THRESHOLD=0.45; FACE_SCALE=0.5; TOLERANCE=0.50; CAPTURE_PHOTOS=5
CAPTURE_GAP=0.5; FOCAL_PX=600; SPEAK_COOLDOWN=3.5
os.makedirs(PLACES_DIR, exist_ok=True); os.makedirs(SAVE_DIR, exist_ok=True)

VC_LISTEN_START="వింటున్నాను చెప్పండి"; VC_NOT_HEARD="అర్థం కాలేదు మళ్ళీ చెప్పండి"
VC_NO_PLACES="స్థలాలు ఏమీ సేవ్ కాలేదు ముందు స్థలం సేవ్ చేయండి"
VC_ASK_PLACE_NAME="స్థలం పేరు చెప్పండి"; VC_ASK_PERSON_NAME="వ్యక్తి పేరు చెప్పండి"
VC_ASK_DEST="మీరు ఎక్కడికి వెళ్ళాలో చెప్పండి"; VC_NAV_STOPPED="నావిగేషన్ ఆపబడింది"
VC_GOODBYE="దృష్టి ముగిసింది వీడ్కోలు"

VC_CMD_SAVE_PLACE =["స్థలం సేవ్","స్థలం","సేవ్ ప్లేస్","save place","place save","save location"]
VC_CMD_SAVE_PERSON=["వ్యక్తి","వ్యక్తిని సేవ్","పర్సన్","save person","save face","person save","face save","add person","సేవ్ పర్సన్"]
VC_CMD_NAVIGATE   =["స్థలానికి","నావిగేషన్","వెళ్ళు","ఎక్కడ","వేర్ ఇస్","where is","navigate","go to","take me","navigate to","వేర్","నావిగేట్"]
VC_CMD_STOP_NAV   =["ఆపు","నిలిపివేయి","స్టాప్","stop","cancel","back","stop navigation","exit navigation"]
VC_CMD_QUIT       =["quit drishti","exit drishti","app quit","close app","బయటకు వెళ్ళు","మూసివేయి"]
VC_CMD_CHANGE_LANG=["భాష మార్చు","భాష","లాంగ్వేజ్","change language","language","switch language","చేంజ్ లాంగ్వేజ్"]
VC_CMD_READ_TEXT  =["read text","read","text","scan","ocr","చదువు","టెక్స్ట్ చదువు","రీడ్"]
VC_CMD_CANCEL_OCR =["cancel","stop","stop reading","ఆపు","స్టాప్","చదవడం ఆపు"]
VC_CMD_SEND_MSG   =["send message","send sms","message","sms","send a message","text message","మెసేజ్ పంపు","మెసేజ్","సందేశం పంపు","send msg","msg"]
VC_CMD_SOS        =["sos","s.o.s","s o s","emergency","help me","అత్యవసరం","ఎస్ ఓ ఎస్","emergency help","call for help","అత్యవసర కాల్"]

NAV_MSG={"left":"ఎడమవైపు తిరగండి","right":"కుడివైపు తిరగండి","forward":"ముందుకు వెళ్ళండి",
         "arrived":"మీరు గమ్యానికి చేరుకున్నారు","obstacle":"ముందు అడ్డంకి ఉంది జాగ్రత్త"}
OBJECT_REF_HEIGHT_M={"person":1.7,"car":1.5,"truck":2.5,"bus":3.0,"motorcycle":1.1,
    "bicycle":1.0,"chair":0.9,"dog":0.5,"cat":0.3,"tv":0.7,"laptop":0.3,
    "bottle":0.25,"cup":0.12,"cell phone":0.15,"book":0.22,"backpack":0.5}
DEFAULT_REF_H=0.5
TELUGU_OBJ={"person":"వ్యక్తి","chair":"కుర్చీ","car":"కారు","dog":"కుక్క","cat":"పిల్లి",
    "bicycle":"సైకిల్","motorcycle":"బైక్","bus":"బస్సు","truck":"ట్రక్","tv":"టీవీ",
    "laptop":"లాప్టాప్","bottle":"బాటిల్","cup":"కప్పు","cell phone":"ఫోన్",
    "book":"పుస్తకం","backpack":"బ్యాగ్"}

# ══════════════════════════════════════════════════════════════════
# SHARED AUDIO
# ══════════════════════════════════════════════════════════════════

_pending_text: str | None = None
_audio_lock   = threading.Lock()
_audio_busy   = threading.Event()
_vc_running   = threading.Event()  # defined early — speak() needs it

def speak(text: str) -> None:
    if get_mode() in ("save_face","save_place","listening","send_msg","sos"): return
    if _vc_running.is_set(): return
    global _pending_text
    with _audio_lock: _pending_text = text

def speak_blocking(text: str) -> None:
    if get_mode() in ("save_face","save_place","listening","send_msg","sos"): return
    if _vc_running.is_set(): return
    asyncio.run(_play_once(text))

async def _play_once(text: str) -> None:
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        import edge_tts as _et
        await _et.Communicate(text, voice=get_tts_voice()).save(path)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.04)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[TTS] {e}")
    finally:
        try: os.remove(path)
        except Exception: pass

def _tts_runner() -> None:
    global _pending_text
    while True:
        text = None
        with _audio_lock:
            if _pending_text:
                text = _pending_text; _pending_text = None
        if text:
            _audio_busy.set()
            asyncio.run(_play_once(text))
            _audio_busy.clear()
        else:
            time.sleep(0.08)

threading.Thread(target=_tts_runner, daemon=True, name="TTS").start()

def speak_force(text: str) -> None:
    """Stop any audio immediately and speak blocking. Used only in voice command handlers."""
    global _pending_text
    with _audio_lock: _pending_text = None
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception: pass
    time.sleep(0.05)
    asyncio.run(_play_once(text))

def speak_force_free(text: str) -> None:
    """FIX: Speak confirmation AFTER _vc_running is cleared — for SMS/SOS confirm."""
    global _pending_text
    with _audio_lock: _pending_text = None
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception: pass
    time.sleep(0.05)
    asyncio.run(_play_once(text))

# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def bbox_dist(bbox_h,ref_h):
    if bbox_h<5: return 99.0
    return round(min((ref_h*FOCAL_PX)/bbox_h,20.0),1)
def face_dist(top,bottom):
    h=max(bottom-top,1); return round(min((1.7*FOCAL_PX)/h,20.0),1)
def direction_te(cx,frame_w):
    r=cx/float(frame_w)
    if r<0.33: return "Left"
    elif r>0.66: return "Right"
    else: return "Center"
def to_rgb(bgr):
    return np.ascontiguousarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════
# YOLO (already loaded in preload thread)
# ══════════════════════════════════════════════════════════════════

print("[YOLO] Loading model...")
from ultralytics import YOLO as _YOLO
if yolo is None:
    yolo = _YOLO(YOLO_MODEL); yolo.fuse()
print("[YOLO] Ready ✅")

# ══════════════════════════════════════════════════════════════════
# FACE DB
# ══════════════════════════════════════════════════════════════════

face_db: dict[str, list] = {}

def load_face_db() -> None:
    global face_db
    if Path(DB_FILE).exists():
        with open(DB_FILE,"rb") as f: face_db=pickle.load(f)
        total=sum(len(v) for v in face_db.values())
        print(f"[FaceDB] {len(face_db)} people, {total} encodings")
    else: print("[FaceDB] Fresh start.")
def save_face_db() -> None:
    with open(DB_FILE,"wb") as f: pickle.dump(face_db,f)
def all_encodings():
    encs,names=[],[]
    for name,enc_list in face_db.items():
        for enc in enc_list: encs.append(enc); names.append(name)
    return encs,names
load_face_db()

_rec_lock=threading.Lock(); _rec_results:list=[]; _latest_small=[None]
_face_stop=threading.Event(); _face_pause=threading.Event()

def _face_worker() -> None:
    known_encs,known_names=all_encodings()
    while not _face_stop.is_set():
        if _face_pause.is_set() or get_mode()!="default":
            time.sleep(0.1); continue
        small=_latest_small[0]
        if small is None: time.sleep(0.05); continue
        rgb=to_rgb(small)
        try: locs=_fr.face_locations(rgb,model="hog")
        except Exception: time.sleep(0.05); continue
        if not locs:
            with _rec_lock: _rec_results.clear()
            time.sleep(0.05); continue
        try: encs=_fr.face_encodings(rgb,locs)
        except Exception: time.sleep(0.05); continue
        results=[]
        for enc,loc in zip(encs,locs):
            name,conf="Unknown",0.0
            if known_encs:
                dists=_fr.face_distance(known_encs,enc)
                idx=int(np.argmin(dists)); d=float(dists[idx])
                if d<=TOLERANCE: name=known_names[idx]; conf=round((1.0-d)*100,1)
            results.append((name,conf)+loc)
        with _rec_lock: _rec_results.clear(); _rec_results.extend(results)
        known_encs,known_names=all_encodings()
        time.sleep(0.04)

threading.Thread(target=_face_worker,daemon=True,name="FaceWorker").start()

# ══════════════════════════════════════════════════════════════════
# SAVE FACE
# ══════════════════════════════════════════════════════════════════

_saving_face=False

def _save_face_thread_named(cap,person_name):
    global _saving_face,face_db
    _saving_face=True; set_mode("save_face"); _face_pause.set()
    global _pending_text
    with _audio_lock: _pending_text=None
    try:
        print(f"\n[SaveFace-Voice] Capturing '{person_name}'...")
        speak_force(f"{person_name} ముఖం కెమెరా ముందు పెట్టండి")
        time.sleep(0.5)
        cap_encs=[]; cap_frames=[]; attempt=0
        while len(cap_encs)<CAPTURE_PHOTOS and attempt<60:
            attempt+=1
            ret,frm=cap.read()
            if not ret: time.sleep(0.1); continue
            rgb=to_rgb(frm); locs=_fr.face_locations(rgb,model="hog")
            if not locs:
                ov=frm.copy()
                cv2.putText(ov,"No face — move closer",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                cv2.imshow("DRISHTI",ov); cv2.waitKey(1); time.sleep(0.25); continue
            encs=_fr.face_encodings(rgb,locs)
            if not encs: continue
            cap_encs.append(encs[0]); cap_frames.append(frm.copy()); n=len(cap_encs)
            t2,r2,b2,l2=locs[0]; ov=frm.copy()
            cv2.rectangle(ov,(l2,t2),(r2,b2),(0,255,0),3)
            cv2.putText(ov,f"Capturing {n}/{CAPTURE_PHOTOS}",(l2,max(t2-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            cv2.imshow("DRISHTI",ov); cv2.waitKey(1); time.sleep(CAPTURE_GAP)
        if not cap_encs: speak_force("ముఖం కనుగొనబడలేదు మళ్ళీ ప్రయత్నించండి"); return
        pdir=os.path.join(SAVE_DIR,person_name); os.makedirs(pdir,exist_ok=True)
        ts=int(time.time())
        for i,frm in enumerate(cap_frames):
            cv2.imwrite(os.path.join(pdir,f"{person_name}_{ts}_{i+1}.jpg"),frm)
        if person_name not in face_db: face_db[person_name]=[]
        face_db[person_name].extend(cap_encs); save_face_db()
        print(f"[SaveFace-Voice] '{person_name}' saved ✅")
        speak_force(f"{person_name} సేవ్ అయింది")
    finally:
        _face_pause.clear(); _saving_face=False; set_mode("default")
        print("[SaveFace-Voice] Back to default mode.\n")

def _save_face_thread(cap):
    global _saving_face,face_db
    _saving_face=True; set_mode("save_face"); _face_pause.set()
    global _pending_text
    with _audio_lock: _pending_text=None
    try:
        print("\n[SaveFace] కెమెరా ముందు ఉండండి..."); speak_blocking("కెమెరా ముందు ఉండండి"); time.sleep(0.3)
        cap_encs=[]; cap_frames=[]; attempt=0
        while len(cap_encs)<CAPTURE_PHOTOS and attempt<60:
            attempt+=1; ret,frame=cap.read()
            if not ret: time.sleep(0.1); continue
            rgb=to_rgb(frame); locs=_fr.face_locations(rgb,model="hog")
            if not locs:
                ov=frame.copy()
                cv2.putText(ov,"No face — move closer",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
                cv2.imshow("DRISHTI",ov); cv2.waitKey(1); time.sleep(0.25); continue
            encs=_fr.face_encodings(rgb,locs)
            if not encs: continue
            cap_encs.append(encs[0]); cap_frames.append(frame.copy()); n=len(cap_encs)
            t,r,b,l=locs[0]; ov=frame.copy()
            cv2.rectangle(ov,(l,t),(r,b),(0,255,0),3)
            cv2.putText(ov,f"Capturing {n}/{CAPTURE_PHOTOS}",(l,max(t-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            cv2.imshow("DRISHTI",ov); cv2.waitKey(1); print(f"[SaveFace] Photo {n}/{CAPTURE_PHOTOS}"); time.sleep(CAPTURE_GAP)
        if not cap_encs: speak_blocking("ముఖం కనుగొనబడలేదు మళ్ళీ ప్రయత్నించండి"); return
        print("[SaveFace] Type name + Enter: ",end="",flush=True)
        person_name=input().strip().lower()
        if not person_name: print("[SaveFace] No name — discarded."); return
        pdir=os.path.join(SAVE_DIR,person_name); os.makedirs(pdir,exist_ok=True)
        ts=int(time.time())
        for i,frm in enumerate(cap_frames):
            cv2.imwrite(os.path.join(pdir,f"{person_name}_{ts}_{i+1}.jpg"),frm)
        if person_name not in face_db: face_db[person_name]=[]
        face_db[person_name].extend(cap_encs); save_face_db()
        print(f"[SaveFace] '{person_name}' saved ✅"); speak_blocking(f"{person_name} సేవ్ అయింది")
    finally:
        _face_pause.clear(); _saving_face=False; set_mode("default")
        print("[SaveFace] Back to default mode.\n")

# ══════════════════════════════════════════════════════════════════
# CLIP (already loaded in preload thread)
# ══════════════════════════════════════════════════════════════════

print("[CLIP] Loading model...")
if _clip_model is None:
    _clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
print("[CLIP] Ready ✅")

# ══════════════════════════════════════════════════════════════════
# MiDaS (already loaded in preload thread)
# ══════════════════════════════════════════════════════════════════

print("[MiDaS] Loading depth model...")
if not MIDAS_OK:
    try:
        _midas=torch.hub.load("intel-isl/MiDaS","MiDaS_small",trust_repo=True)
        _midas_xfm=torch.hub.load("intel-isl/MiDaS","transforms",trust_repo=True).small_transform
        _midas.eval(); MIDAS_OK=True; print("[MiDaS] Ready ✅")
    except Exception as e:
        print(f"[MiDaS] Not available ({e}) — depth disabled")
else:
    print("[MiDaS] Ready ✅ (preloaded)")

# ══════════════════════════════════════════════════════════════════
# ORB
# ══════════════════════════════════════════════════════════════════

_orb=cv2.ORB_create(nfeatures=1000)
_matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

# ══════════════════════════════════════════════════════════════════
# PLACE DATABASE
# ══════════════════════════════════════════════════════════════════

class PlaceDB:
    def __init__(self,root):
        self.root=Path(root); self.places={}; self._load()
    def _load(self):
        self.places.clear()
        if not self.root.exists(): return
        for place_dir in sorted(self.root.iterdir()):
            if not place_dir.is_dir(): continue
            name=place_dir.name; imgs=[]; pils=[]; kps=[]; descs=[]; embs=[]
            for img_path in sorted(place_dir.glob("*.jpg")):
                bgr=cv2.imread(str(img_path))
                if bgr is None: continue
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                kp,desc=_orb.detectAndCompute(gray,None)
                if desc is not None: kps.append(kp); descs.append(desc)
                inp=_clip_proc(images=pil,return_tensors="pt")
                with torch.no_grad():
                    emb=_clip_model.get_image_features(**inp)
                    emb=emb/emb.norm(dim=-1,keepdim=True)
                embs.append(emb); imgs.append(bgr); pils.append(pil)
            if not imgs: continue
            avg_emb=torch.stack(embs).mean(dim=0); avg_emb=avg_emb/avg_emb.norm()
            self.places[name]={"images":imgs,"pils":pils,"kps":kps,"descs":descs,"avg_emb":avg_emb,"all_embs":embs}
            print(f"[PlaceDB] '{name}' — {len(imgs)} images")
        print(f"[PlaceDB] Total: {len(self.places)} places ✅")
    def reload(self): self._load()
    def names(self): return list(self.places.keys())

# ══════════════════════════════════════════════════════════════════
# TWILIO — SMS and SOS CALL
# ══════════════════════════════════════════════════════════════════

def _send_sms(to_number: str, body: str) -> bool:
    if not TWILIO_OK: print("[SMS] Twilio not installed"); return False
    try:
        client=_TwilioClient(TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN)
        msg=client.messages.create(body=body,from_=TWILIO_FROM_NUMBER,to=to_number)
        print(f"[SMS] Sent ✅  SID={msg.sid}  to={to_number}")
        return True
    except Exception as e:
        print(f"[SMS] Failed ❌  {e}"); return False

def _sos_call(to_numbers: list) -> None:
    if not TWILIO_OK: print("[SOS] Twilio not installed"); return
    twiml=("<Response><Say language='en-IN'>"
           "This is an automated safety alert from the blind assistance system. "
           "The user may require immediate assistance. Please contact them right away."
           "</Say></Response>")
    try:
        client=_TwilioClient(TWILIO_ACCOUNT_SID,TWILIO_AUTH_TOKEN)
        for number in to_numbers:
            try:
                call=client.calls.create(to=number,from_=TWILIO_FROM_NUMBER,twiml=twiml)
                print(f"[SOS] Call placed ✅  SID={call.sid}  to={number}")
            except Exception as e:
                print(f"[SOS] Call failed for {number}: {e}")
    except Exception as e:
        print(f"[SOS] Client error: {e}")

# ══════════════════════════════════════════════════════════════════
# DIGIT-BY-DIGIT PHONE NUMBER COLLECTION
# ══════════════════════════════════════════════════════════════════

_DIGIT_WORD_MAP={
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "సున్న":"0","సున్నా":"0","జీరో":"0","ఒకటి":"1","వన్":"1",
    "రెండు":"2","టూ":"2","మూడు":"3","త్రీ":"3","నాలుగు":"4","ఫోర్":"4",
    "అయిదు":"5","ఫైవ్":"5","ఆరు":"6","సిక్స్":"6","ఏడు":"7","సెవెన్":"7",
    "ఎనిమిది":"8","ఎయిట్":"8","తొమ్మిది":"9","నైన్":"9",
    "शून्य":"0","एक":"1","दो":"2","तीन":"3","चार":"4",
    "पाँच":"5","छह":"6","सात":"7","आठ":"8","नौ":"9",
}

def _extract_10_digits(text: str) -> str:
    text=text.strip().lower()
    digits_direct=re.sub(r'\D','',text)
    if len(digits_direct)==10: return digits_direct
    if len(digits_direct)>10: return digits_direct[-10:]
    tokens=re.split(r'[\s,./\-]+',text)
    built=[]
    for tok in tokens:
        tok=tok.strip()
        if not tok: continue
        if len(tok)==1 and tok.isdigit(): built.append(tok); continue
        matched=_DIGIT_WORD_MAP.get(tok,"")
        if matched: built.append(matched); continue
        for word,digit in _DIGIT_WORD_MAP.items():
            if word in tok: built.append(digit); break
    if len(built)==10: return "".join(built)
    m=re.search(r'\b(\d{10})\b',text)
    if m: return m.group(1)
    all_digits=re.findall(r'\d',text)
    combined=re.sub(r'\D','',"".join(all_digits))
    if len(combined)>=10: return combined[-10:]
    return ""

def _listen_stt(recognizer, src, timeout=10, phrase_limit=15) -> str:
    try:
        audio=recognizer.listen(src,timeout=timeout,phrase_time_limit=phrase_limit)
    except Exception: return ""
    for lng in ("en-IN","en-US","te-IN","hi-IN"):
        try:
            result=recognizer.recognize_google(audio,language=lng)
            if result:
                print(f"[STT] ({lng}): {result!r}")
                return result
        except Exception: continue
    return ""

YES_WORDS=["yes","yeah","correct","ok","okay","sure","right","confirm","avunu","avun",
           "అవును","అవు","సరే","కరెక్ట్","హా","ha","haa","yep","han","haan","aaa","aa"]

def _confirm_yn(recognizer, src, prompt: str) -> bool:
    speak_force(prompt)
    raw=_listen_stt(recognizer, src, timeout=8, phrase_limit=5)
    print(f"[Confirm] heard: {raw!r}")
    return any(w.lower() in raw.lower() for w in YES_WORDS)

def _collect_phone_number_by_voice(recognizer, src) -> str:
    lang=_current_lang
    for attempt in range(1,3):
        if attempt==1:
            prompt=("పది అంకెల మొబైల్ నంబర్ ఒకేసారి చెప్పండి" if lang=="telugu"
                    else "Say all ten digits of the mobile number.")
        else:
            prompt=("మళ్ళీ ప్రయత్నించండి. పది అంకెలు చెప్పండి" if lang=="telugu"
                    else "Try again. Say all ten digits.")
        speak_force(prompt)
        raw=_listen_stt(recognizer, src, timeout=12, phrase_limit=15)
        if not raw: continue
        digits=_extract_10_digits(raw)
        print(f"[Phone] Parsed: {digits!r} from: {raw!r}")
        if len(digits)==10:
            number_e164="+91"+digits
            spaced=" ".join(digits)
            confirm_prompt=(f"నంబర్ ప్లస్ తొంభై ఒకటి {spaced}. సరైనదా? అవును లేదా కాదు" if lang=="telugu"
                           else f"Number plus 9 1 {spaced}. Correct? Say yes or no.")
            if _confirm_yn(recognizer, src, confirm_prompt):
                return number_e164
            else:
                speak_force("మళ్ళీ చెప్పండి" if lang=="telugu" else "Say the number again.")
        else:
            speak_force("పది అంకెలు వినబడలేదు మళ్ళీ చెప్పండి" if lang=="telugu"
                       else "Could not get ten digits. Please try again.")
    return ""

def _collect_message_by_voice(recognizer, src) -> str:
    lang=_current_lang
    for _attempt in range(3):
        speak_force("మీ మెసేజ్ చెప్పండి" if lang=="telugu" else "Please say your message now.")
        raw=_listen_stt(recognizer, src, timeout=12, phrase_limit=15)
        if not raw:
            speak_force("వినబడలేదు మళ్ళీ చెప్పండి" if lang=="telugu" else "Not heard, try again."); continue
        confirm_prompt=(f"మీరు చెప్పింది: {raw}. సరైనదా? అవును లేదా కాదు" if lang=="telugu"
                       else f"You said: {raw}. Correct? Say yes or no.")
        if _confirm_yn(recognizer, src, confirm_prompt):
            return raw.strip()
        speak_force("సరే మళ్ళీ చెప్పండి" if lang=="telugu" else "Okay, say your message again.")
    return ""

def _handle_send_message(recognizer, src) -> None:
    lang=_current_lang
    set_mode("send_msg")
    global _pending_text
    with _audio_lock: _pending_text=None
    try:
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
    except Exception: pass
    time.sleep(0.1)
    print("\n[SendMsg] ══ SEND MESSAGE MODE ══")
    speak_force("మెసేజ్ పంపే మోడ్ మొదలైంది. పది అంకెల నంబర్ చెప్పండి" if lang=="telugu"
               else "Send message mode. Say the 10 digit phone number.")
    try:
        phone=_collect_phone_number_by_voice(recognizer,src)
        if not phone:
            speak_force("నంబర్ అర్థం కాలేదు. రద్దు చేయబడింది" if lang=="telugu" else "Number not understood. Cancelled."); return
        message=_collect_message_by_voice(recognizer,src)
        if not message:
            speak_force("మెసేజ్ అర్థం కాలేదు. రద్దు చేయబడింది" if lang=="telugu" else "Message not understood. Cancelled."); return
        speak_force(f"{phone} కి మెసేజ్ పంపుతున్నాం" if lang=="telugu" else f"Sending message to {phone}")
        _sms_result = [None]
        def _send():
            _sms_result[0] = _send_sms(phone, message)
        t=threading.Thread(target=_send, daemon=True)
        t.start()
        t.join(timeout=15)
        _sms_ok = _sms_result[0]
        return
    except Exception as e:
        print(f"[SendMsg] Error: {e}"); speak_force("మెసేజ్ లో సమస్య వచ్చింది")
        _sms_ok = False
    finally:
        _vc_running.clear()
        set_mode("default")
        print("[SendMsg] Back to default mode.\n")
    if '_sms_ok' in dir() and _sms_ok is not None:
        if _sms_ok:
            speak_force_free("మెసేజ్ పంపబడింది" if lang=="telugu" else "Message sent successfully")
        else:
            speak_force_free("మెసేజ్ పంపడం విఫలమైంది" if lang=="telugu" else "Message sending failed")

def _handle_sos(recognizer, src) -> None:
    lang=_current_lang
    set_mode("sos")
    global _pending_text
    with _audio_lock: _pending_text=None
    try:
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
    except Exception: pass
    time.sleep(0.1)
    print("\n[SOS] ══ SOS MODE ══")
    speak_force("అత్యవసర కాల్ చేస్తారా? అవును అంటే కాల్ చేస్తాం" if lang=="telugu"
               else "Emergency call mode. Say yes to call, or no to cancel.")
    confirmed=_confirm_yn(recognizer,src,"")
    print(f"[SOS] Confirmed: {confirmed}")
    _do_call=confirmed
    try:
        if _do_call:
            speak_force("అత్యవసర కాల్ చేస్తున్నాం జాగ్రత్తగా ఉండండి" if lang=="telugu"
                       else "Placing emergency call. Please stay safe.")
    except Exception as e:
        print(f"[SOS] Error: {e}")
    finally:
        _vc_running.clear()
        set_mode("default")
        print("[SOS] Back to default mode.\n")
    if _do_call:
        def _do():
            _sos_call(SOS_NUMBERS)
            speak_force_free("అత్యవసర కాల్ పంపబడింది" if lang=="telugu" else "Emergency call placed.")
        threading.Thread(target=_do, daemon=True, name="SOSCall").start()
    else:
        speak_force_free("అత్యవసర కాల్ రద్దు చేయబడింది" if lang=="telugu" else "Emergency call cancelled.")

# ══════════════════════════════════════════════════════════════════
# VOICE COMMAND ENGINE
# ══════════════════════════════════════════════════════════════════

_vc_queue=queue.Queue(); _vc_stop=threading.Event()

def _match_cmd(text,keywords):
    t=text.lower(); return any(k.lower() in t for k in keywords)

def _recognize_audio(recognizer,audio):
    stt=get_stt_lang(); langs=[stt] if stt=="en-IN" else [stt,"en-IN"]
    for lang in langs:
        try:
            result=recognizer.recognize_google(audio,language=lang)
            if result: return result.lower()
        except Exception: continue
    return ""

def _recognize_audio_wake(recognizer,audio):
    for lang in ("en-IN","en-US","te-IN"):
        try:
            result=recognizer.recognize_google(audio,language=lang)
            if result: return result.lower()
        except Exception: continue
    return ""

def _listen_once(recognizer,src,timeout=7):
    try:
        audio=recognizer.listen(src,timeout=timeout,phrase_time_limit=6)
        return _recognize_audio(recognizer,audio)
    except Exception: return ""

def _voice_command_thread() -> None:
    if not SR_OK: return
    import speech_recognition as sr
    recognizer=sr.Recognizer()
    recognizer.energy_threshold=400; recognizer.dynamic_energy_threshold=True
    recognizer.pause_threshold=0.6; mic=sr.Microphone()
    WAKE_WORDS=["listen","command","activate","drishti","assistant",
                "లిసెన్","వినండి","దృష్టి","కమాండ్","ఆక్టివేట్","విను","లెజెండ్","లీజన్","లిజన్"]
    with mic as src:
        print("[VoiceCmd] Calibrating for 2 seconds — stay quiet...")
        recognizer.adjust_for_ambient_noise(src,duration=2)
        ambient=recognizer.energy_threshold
        recognizer.energy_threshold=max(300,min(800,ambient*3.0))
        recognizer.dynamic_energy_threshold=False
        recognizer.pause_threshold=0.5; recognizer.non_speaking_duration=0.4
        print(f"[VoiceCmd] Mic ready. Threshold={recognizer.energy_threshold:.0f}")
        print("[VoiceCmd] Say 'listen' to activate commands.")
        while not _vc_stop.is_set():
            try: audio=recognizer.listen(src,timeout=None,phrase_time_limit=3)
            except Exception: time.sleep(0.1); continue
            wake_text=_recognize_audio_wake(recognizer,audio)
            if not wake_text: continue
            print(f"[VoiceCmd] Heard: '{wake_text}'")
            if _match_cmd(wake_text,VC_CMD_SOS):
                _vc_running.set(); set_mode("listening")
                global _pending_text
                with _audio_lock: _pending_text=None
                try:
                    if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
                except Exception: pass
                try: _handle_sos(recognizer,src)
                except Exception as e: print(f"[VoiceCmd-SOS] {e}")
                finally:
                    if _vc_running.is_set(): _vc_running.clear()
                    if get_mode()=="listening": set_mode("default")
                    print("[VoiceCmd] SOS handled\n")
                continue
            if get_mode()=="ocr" and any(w in wake_text for w in ["cancel","stop","ఆపు","స్టాప్"]):
                _vc_queue.put(("vc_stop_ocr","")); continue
            if get_mode()=="nav" and any(w in wake_text for w in ["cancel","stop","ఆపు","స్టాప్"]):
                _vc_queue.put(("vc_stop_nav","")); continue
            words_heard=wake_text.lower().split()
            triggered=any(w in wake_text for w in WAKE_WORDS) or \
                      any(w in words_heard for w in ["command","activate","mode","listen"])
            if not triggered: continue
            _vc_running.set(); set_mode("listening")
            with _audio_lock: _pending_text=None
            try:
                if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
            except Exception: pass
            time.sleep(0.1)
            print("[VoiceCmd] ✅ Wake word! Command mode active.")
            try:
                inline=wake_text
                for w in WAKE_WORDS: inline=inline.replace(w,"")
                inline=inline.strip()
                INLINE_KEYWORDS=["save","navigate","where","stop","change","language",
                                  "place","person","face","quit","message","sms","sos","emergency"]
                is_real_cmd=any(kw in inline for kw in INLINE_KEYWORDS)
                if inline and len(inline)>4 and is_real_cmd:
                    print(f"[VoiceCmd] Inline: '{inline}'")
                    _process_vc_command(inline,src,recognizer)
                else:
                    speak_force(VC_LISTEN_START)
                    try:
                        audio2=recognizer.listen(src,timeout=7,phrase_time_limit=6)
                        cmd_text=_recognize_audio_wake(recognizer,audio2)
                        if not cmd_text: cmd_text=_recognize_audio(recognizer,audio2)
                    except Exception: cmd_text=""
                    print(f"[VoiceCmd] Command: '{cmd_text}'")
                    if cmd_text:
                        if any(w in cmd_text for w in ["menu","help","commands","list","మెను","మెనూ","సహాయం","హెల్ప్"]):
                            speak_force(get_menu_in_current_lang())
                        else:
                            _process_vc_command(cmd_text,src,recognizer)
                    else:
                        speak_force(VC_NOT_HEARD)
            except Exception as e:
                print(f"[VoiceCmd] Error: {e}"); speak_force(VC_NOT_HEARD)
            finally:
                if _vc_running.is_set(): _vc_running.clear()
                if get_mode()=="listening": set_mode("default")
                print("[VoiceCmd] Done — back to idle\n")

def _process_vc_command(text,src,recognizer):
    print(f"[VoiceCmd] Processing: '{text}'")
    if _match_cmd(text,VC_CMD_QUIT):
        speak_force(VC_GOODBYE); _vc_queue.put(("quit","")); return
    if get_mode()=="ocr" and _match_cmd(text,VC_CMD_CANCEL_OCR):
        _vc_queue.put(("vc_stop_ocr","")); return
    if _match_cmd(text,VC_CMD_SOS):
        _handle_sos(recognizer,src); return
    if _match_cmd(text,VC_CMD_SEND_MSG):
        _handle_send_message(recognizer,src); return
    if _match_cmd(text,VC_CMD_STOP_NAV):
        speak_force(VC_NAV_STOPPED); _vc_queue.put(("vc_stop_nav","")); return
    if _match_cmd(text,VC_CMD_SAVE_PLACE):
        speak_force(VC_ASK_PLACE_NAME)
        name_text=_listen_once(recognizer,src,timeout=8)
        print(f"[VoiceCmd] Place name heard: {name_text!r}")
        if name_text.strip():
            speak_force(f"{name_text.strip().lower()} place saving started")
            _vc_queue.put(("vc_save_place",name_text.strip().lower()))
        else: speak_force(VC_NOT_HEARD)
        return
    if _match_cmd(text,VC_CMD_SAVE_PERSON):
        speak_force(VC_ASK_PERSON_NAME)
        name_text=_listen_once(recognizer,src,timeout=8)
        print(f"[VoiceCmd] Person name heard: {name_text!r}")
        if name_text.strip():
            speak_force(f"{name_text.strip().lower()} face saving started")
            _vc_queue.put(("vc_save_face",name_text.strip().lower()))
        else: speak_force(VC_NOT_HEARD)
        return
    if _match_cmd(text,VC_CMD_NAVIGATE):
        if not DB.places: speak_force(VC_NO_PLACES); return
        names_str=", ".join(DB.names()); speak_force(f"అందుబాటులో ఉన్న స్థలాలు: {names_str}. {VC_ASK_DEST}")
        matched=""; attempts=0
        while not matched and attempts<3:
            attempts+=1
            try:
                audio_dest=recognizer.listen(src,timeout=8,phrase_time_limit=6)
                dest_text=""
                for _lng in ("en-IN","en-US","te-IN"):
                    try:
                        dest_text=recognizer.recognize_google(audio_dest,language=_lng).lower()
                        print(f"[VoiceCmd] STT({_lng}): {dest_text!r}"); break
                    except Exception: continue
            except Exception: dest_text=""
            if not dest_text: speak_force("వినబడలేదు మళ్ళీ చెప్పండి"); continue
            matched=_fuzzy_match_place(dest_text)
            if matched:
                speak_force(f"{matched} వైపు నావిగేషన్ మొదలవుతుంది"); _vc_queue.put(("vc_nav_dest",matched))
            else:
                remaining=3-attempts
                avail=", ".join(DB.names())
                if remaining>0: speak_force(f"స్థలం అర్థం కాలేదు. అందుబాటులో: {avail}. మళ్ళీ చెప్పండి")
                else: speak_force("స్థలం అర్థం కాలేదు. మళ్ళీ కమాండ్ ఇవ్వండి")
        return
    if _match_cmd(text,VC_CMD_CHANGE_LANG):
        lang_list=", ".join(SUPPORTED_LANGUAGES.keys())
        speak_force(f"ఏ భాషకు మార్చాలి? అందుబాటులో: {lang_list}")
        lang_key=""; attempts=0
        try:
            audio_lang=recognizer.listen(src,timeout=7,phrase_time_limit=5)
            lang_text=""
            for _lng in ("en-IN","te-IN","hi-IN","ta-IN","kn-IN","ml-IN"):
                try:
                    lang_text=recognizer.recognize_google(audio_lang,language=_lng).lower()
                    if lang_text: break
                except Exception: continue
        except Exception: lang_text=""
        while not lang_key and attempts<3:
            attempts+=1
            if attempts>1:
                try:
                    al2=recognizer.listen(src,timeout=7,phrase_time_limit=5); lang_text=""
                    for _lng in ("en-IN","te-IN","hi-IN","ta-IN","kn-IN","ml-IN"):
                        try:
                            lang_text=recognizer.recognize_google(al2,language=_lng).lower()
                            if lang_text: break
                        except Exception: continue
                except Exception: lang_text=""
            print(f"[VoiceCmd] Language heard: '{lang_text}'")
            if not lang_text: speak_force("Not heard. Say: hindi, tamil, kannada, english"); continue
            lang_key=detect_language_from_text(lang_text)
            if lang_key: set_language(lang_key); speak_force(get_lang_changed_msg(lang_key))
            else:
                remaining=3-attempts
                supported="telugu, hindi, tamil, kannada, malayalam, english"
                if remaining>0: speak_force(f"Not understood. Available: {supported}. Say again.")
                else: speak_force("Language not recognized. Try activate command again")
        return
    if _match_cmd(text,VC_CMD_READ_TEXT):
        _vc_queue.put(("vc_start_ocr","")); return
    print(f"[VoiceCmd] → NO MATCH for: '{text}'")
    speak_force(VC_NOT_HEARD)

def _fuzzy_match_place(text):
    if not text or not DB.places: return ""
    text_l=text.lower().strip()
    for name in DB.names():
        if name.lower() in text_l or text_l in name.lower(): return name
    words=text_l.split()
    for name in DB.names():
        name_words=name.lower().split()
        if any(w in name_words for w in words): return name
    PHONETIC_MAP={"హాల్":"hall","హాలు":"hall","బెడ్రూమ్":"bedroom","పడకగది":"bedroom",
                  "కిచెన్":"kitchen","వంటగది":"kitchen","బాత్రూమ్":"bathroom",
                  "స్టడీ రూమ్":"study room","స్టడీ":"study room","చదువు గది":"study room",
                  "గ్యారేజ్":"garage","ఆఫీస్":"office","క్లాస్రూమ్":"classroom",
                  "తరగతి":"classroom","గార్డెన్":"garden","తోట":"garden",
                  "కారిడార్":"corridor","వరండా":"corridor","ఎంట్రన్స్":"entrance",
                  "ముఖద్వారం":"entrance","రోడ్డు":"road","స్టెయిర్కేస్":"staircase",
                  "మెట్లు":"staircase","లివింగ్ రూమ్":"living room","డైనింగ్":"dining room",
                  "భోజన గది":"dining room","రూమ్":"room"}
    for telugu_word,english_name in PHONETIC_MAP.items():
        if telugu_word in text_l:
            for name in DB.names():
                if english_name in name.lower() or name.lower() in english_name: return name
    best_name=""; best_score=0
    for name in DB.names():
        overlap=sum(1 for c in text_l if c in name.lower())
        if overlap>best_score and overlap>=2: best_score=overlap; best_name=name
    return best_name

# ══════════════════════════════════════════════════════════════════
# OCR ENGINE
# ══════════════════════════════════════════════════════════════════

_ocr_reader=None; _ocr_reader_lock=threading.Lock(); _ocr_active=threading.Event()
_ocr_last_text=""; _ocr_last_time=0.0; _ocr_frame=[None]; _ocr_stop_flag=threading.Event()

def _init_ocr_reader():
    global _ocr_reader
    with _ocr_reader_lock:
        if _ocr_reader is None:
            try:
                import easyocr
                print("[OCR] Loading EasyOCR model (first time ~30s)...")
                _ocr_reader=easyocr.Reader(["en","te"],gpu=False)
                print("[OCR] EasyOCR ready (en+Telugu) ✅")
            except ImportError:
                print("[OCR] easyocr not installed. Run: pip install easyocr")
                _ocr_reader=None
    return _ocr_reader

def _detect_text_accurate(frame_bgr):
    reader=_ocr_reader
    if reader is None: return []
    try:
        h,w=frame_bgr.shape[:2]
        if w<960: frame_bgr=cv2.resize(frame_bgr,(int(w*960/w),int(h*960/w)),interpolation=cv2.INTER_CUBIC)
        elif w>1280: frame_bgr=cv2.resize(frame_bgr,(1280,int(h*1280/w)))
        all_results=[]
        rgb_orig=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        try:
            res1=reader.readtext(rgb_orig,detail=1,paragraph=False,width_ths=0.5,height_ths=0.3,
                                 contrast_ths=0.05,adjust_contrast=0.7,batch_size=4,min_size=10)
            all_results.extend(res1)
        except Exception as e: print(f"[OCR] Pass1: {e}")
        gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        enhanced=clahe.apply(gray)
        kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
        sharpened=cv2.filter2D(enhanced,-1,kernel)
        rgb_enh=cv2.cvtColor(sharpened,cv2.COLOR_GRAY2RGB)
        try:
            res2=reader.readtext(rgb_enh,detail=1,paragraph=False,width_ths=0.5,height_ths=0.3,
                                 contrast_ths=0.05,adjust_contrast=0.5,batch_size=4,min_size=10)
            all_results.extend(res2)
        except Exception as e: print(f"[OCR] Pass2: {e}")
        seen_texts={}
        for item in all_results:
            if len(item)==3: box,text,conf=item
            elif len(item)==2: box,text=item; conf=1.0
            else: continue
            text=text.strip()
            if not text or len(text)<2: continue
            key=text.lower().replace(" ","")
            if key not in seen_texts or conf>seen_texts[key][1]:
                seen_texts[key]=(text,conf,box)
        return [(t,c,b) for t,c,b in seen_texts.values() if c>0.25]
    except Exception as e:
        print(f"[OCR] Error: {e}"); return []

def _ocr_worker_thread(cap_ref):
    global _ocr_last_text,_ocr_last_time
    print("[OCR] Worker started"); _init_ocr_reader()
    while not _ocr_stop_flag.is_set():
        if not _ocr_active.is_set(): time.sleep(0.2); continue
        frame=_ocr_frame[0]
        if frame is None: time.sleep(0.2); continue
        results=_detect_text_accurate(frame)
        if not results: time.sleep(2.0); continue
        results.sort(key=lambda r:r[2][0][1])
        all_text=" ".join([t for t,c,b in results if c>0.4]).strip()
        if not all_text: time.sleep(2.0); continue
        now=time.time()
        if all_text!=_ocr_last_text:
            _ocr_last_text=all_text; _ocr_last_time=now
            print(f"[OCR] Detected: {all_text}")
            lang=_current_lang
            if lang=="telugu": speak(f"టెక్స్ట్: {all_text}")
            elif lang=="hindi": speak(f"टेक्स्ट: {all_text}")
            else: speak(f"Text: {all_text}")
        time.sleep(3.0)
    print("[OCR] Worker stopped")

def start_ocr_mode(cap):
    global _ocr_last_text
    set_mode("ocr"); _ocr_active.set(); _ocr_last_text=""
    threading.Thread(target=_ocr_worker_thread,args=(cap,),daemon=True,name="OCRWorker").start()
    lang=_current_lang
    if lang=="english": speak_force("Text reading mode started. Point camera at text. Say cancel to stop.")
    elif lang=="hindi": speak_force("टेक्स्ट पढ़ना शुरू हुआ। cancel कहें।")
    else: speak_force("టెక్స్ట్ చదవడం మొదలైంది. కెమెరాను టెక్స్ట్ వైపు తిప్పండి. ఆపడానికి cancel చెప్పండి.")

def stop_ocr_mode():
    global _ocr_last_text
    _ocr_active.clear(); _ocr_last_text=""; set_mode("default")
    lang=_current_lang
    if lang=="english": speak_force("Text reading stopped.")
    elif lang=="hindi": speak_force("टेक्स्ट पढ़ना बंद हो गया।")
    else: speak_force("టెక్స్ట్ చదవడం ఆపబడింది.")
    print("[OCR] Mode stopped")

print("\n[PlaceDB] Loading...")
DB = PlaceDB(PLACES_DIR)

# ══════════════════════════════════════════════════════════════════
# SAVE PLACE
# ══════════════════════════════════════════════════════════════════

def _do_save_place(cap,place_name):
    place_dir=os.path.join(PLACES_DIR,place_name); os.makedirs(place_dir,exist_ok=True)
    existing=len([f for f in os.listdir(place_dir) if f.endswith(".jpg")])
    print(f"\n[SavePlace] Capturing 5 photos of '{place_name}'")
    speak_blocking(f"{place_name} స్థలం నమోదు మొదలవుతుంది కెమెరా చుట్టూ తిప్పండి")
    saved=0
    for i in range(5):
        print(f"[SavePlace] Photo {i+1}/5 in 2 seconds..."); time.sleep(2)
        ret,frame=cap.read()
        if not ret: continue
        path=os.path.join(place_dir,f"{existing+i}.jpg"); cv2.imwrite(path,frame); saved+=1
        flash=frame.copy()
        cv2.putText(flash,f"Saving Place: {place_name}  {i+1}/5",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
        cv2.imshow("DRISHTI",flash); cv2.waitKey(200)
    print(f"[SavePlace] Done — {saved}/5 photos saved for '{place_name}'")
    DB.reload(); speak_blocking(f"{place_name} స్థలం సేవ్ అయింది")

def _save_place_thread(cap,place_name):
    set_mode("save_place"); _face_pause.set()
    try: _do_save_place(cap,place_name)
    finally: _face_pause.clear(); set_mode("default"); print("[SavePlace] Back to default mode.\n")

# ══════════════════════════════════════════════════════════════════
# CLIP HELPERS
# ══════════════════════════════════════════════════════════════════

def clip_similarity(frame_bgr,dest_name):
    pil=Image.fromarray(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
    inp=_clip_proc(images=pil,return_tensors="pt")
    with torch.no_grad():
        emb=_clip_model.get_image_features(**inp); emb=emb/emb.norm(dim=-1,keepdim=True)
    place=DB.places[dest_name]; scores=[]
    for i,ref_emb in enumerate(place["all_embs"]):
        e1=emb.squeeze(0); e2=ref_emb.squeeze(0); scores.append(float(torch.dot(e1,e2)))
    hits=sum(1 for s in scores if s>=0.80)
    scores.sort(reverse=True); top3=scores[:min(3,len(scores))]; final=sum(top3)/len(top3)
    if hits>=2: final=max(final,0.83)
    score_str="  ".join([f"img{i}:{s:.2f}" for i,s in enumerate(scores)])
    print(f"[CLIP] {score_str}  top3={final:.2f}  hits={hits}/5  dest={dest_name}")
    return final

# ══════════════════════════════════════════════════════════════════
# ORB DIRECTION
# ══════════════════════════════════════════════════════════════════

def orb_direction(frame_bgr,dest_name):
    result={"direction":"forward","offset_x":0,"offset_y":0,"arrow_dst":None,"n_matches":0,"confidence":0.0}
    gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
    kp_cur,desc_cur=_orb.detectAndCompute(gray,None)
    if desc_cur is None or len(desc_cur)<5: return result
    place=DB.places[dest_name]; h,w=frame_bgr.shape[:2]; cx,cy=w//2,h//2
    best_matches=[]; best_ref_kps=None
    for ref_desc,ref_kps in zip(place["descs"],place["kps"]):
        if ref_desc is None or len(ref_desc)<5: continue
        try:
            raw=_matcher.match(desc_cur,ref_desc)
            good=sorted(raw,key=lambda m:m.distance)[:50]
            good=[m for m in good if m.distance<60]
            if len(good)>len(best_matches): best_matches=good; best_ref_kps=ref_kps
        except Exception: continue
    result["n_matches"]=len(best_matches)
    if len(best_matches)<ORB_MIN_MATCHES or best_ref_kps is None: return result
    result["confidence"]=min(1.0,len(best_matches)/40.0)
    pts_cur=np.float32([kp_cur[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    pts_ref=np.float32([best_ref_kps[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)
    H,_=cv2.findHomography(pts_ref,pts_cur,cv2.RANSAC,5.0)
    if H is None:
        offset_x=float(np.mean(pts_cur[:,0,0]))-float(np.mean(pts_ref[:,0,0]))
        result["offset_x"]=offset_x
        result["direction"]="left" if offset_x<-30 else ("right" if offset_x>30 else "forward")
        result["arrow_dst"]=(int(np.mean(pts_cur[:,0,0])),cy); return result
    ref_h2,ref_w2=DB.places[dest_name]["images"][0].shape[:2]
    ref_centre=np.float32([[ref_w2/2,ref_h2/2]]).reshape(-1,1,2)
    try: proj=cv2.perspectiveTransform(ref_centre,H); px,py=int(proj[0,0,0]),int(proj[0,0,1])
    except Exception: px,py=cx,cy
    offset_x=px-cx; offset_y=py-cy
    result["offset_x"]=offset_x; result["offset_y"]=offset_y
    result["arrow_dst"]=(max(30,min(w-30,px)),max(30,min(h-30,py)))
    THRESH=w*0.12
    if abs(offset_x)<THRESH and abs(offset_y)<THRESH*0.8: result["direction"]="forward"
    elif offset_x<-THRESH: result["direction"]="left"
    elif offset_x>THRESH: result["direction"]="right"
    else: result["direction"]="forward"
    return result

def estimate_depth_at(frame_bgr,x,y):
    if not MIDAS_OK: return -1.0
    try:
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB); inp=_midas_xfm(rgb)
        with torch.no_grad():
            pred=_midas(inp)
            pred=torch.nn.functional.interpolate(pred.unsqueeze(1),size=frame_bgr.shape[:2],
                                                 mode="bicubic",align_corners=False).squeeze()
        dm=pred.cpu().numpy()
        raw=float(dm[max(0,min(dm.shape[0]-1,y)),max(0,min(dm.shape[1]-1,x))])
        return round(DEPTH_SCALE/(raw/dm.max()+1e-6),1)
    except Exception: return -1.0

# ══════════════════════════════════════════════════════════════════
# NAV WORKER
# ══════════════════════════════════════════════════════════════════

_nav_lock=threading.Lock()
_nav_state={"sim":0.0,"direction":"forward","offset_x":0,"arrow_dst":None,
            "n_matches":0,"confidence":0.0,"depth":-1.0,"arrived":False}
_nav_frame=[None]; _nav_dest=[None]; _nav_stop=threading.Event()

def nav_worker():
    last_speak_time=0.0; last_direction=""
    while not _nav_stop.is_set():
        time.sleep(CLIP_CHECK_EVERY)
        if get_mode()!="nav": continue
        frame=_nav_frame[0]; dest=_nav_dest[0]
        if frame is None or dest is None or dest not in DB.places: continue
        try:
            sim=clip_similarity(frame,dest); orb_res=orb_direction(frame,dest)
            direction=orb_res["direction"]; arrow_dst=orb_res["arrow_dst"]
            n_matches=orb_res["n_matches"]; confidence=orb_res["confidence"]; offset_x=orb_res["offset_x"]
            arrived=sim>=ARRIVAL_THRESHOLD
            if arrived: direction="arrived"
            depth=-1.0
            if arrow_dst and MIDAS_OK: depth=estimate_depth_at(frame,arrow_dst[0],arrow_dst[1])
            with _nav_lock:
                _nav_state.update({"sim":sim,"direction":direction,"offset_x":offset_x,
                                   "arrow_dst":arrow_dst,"n_matches":n_matches,
                                   "confidence":confidence,"depth":depth,"arrived":arrived})
            now=time.time()
            if now-last_speak_time>SPEAK_COOLDOWN:
                lang=_current_lang
                if arrived:
                    if lang=="english": speak(f"You have arrived at {dest}")
                    elif lang=="hindi": speak(f"{dest} पहुंच गए")
                    elif lang=="tamil": speak(f"{dest} வந்துவிட்டீர்கள்")
                    elif lang=="kannada": speak(f"{dest} ತಲುಪಿದ್ದೀರಿ")
                    elif lang=="malayalam": speak(f"{dest} എത്തിയിരിക്കുന്നു")
                    else: speak(f"{dest} చేరుకున్నారు")
                    last_speak_time=now
                elif direction!=last_direction or now-last_speak_time>6.0:
                    if lang=="english":
                        dir_en={"left":"turn left","right":"turn right","forward":"go forward"}.get(direction,"go forward")
                        depth_str=f" {depth:.0f} meters" if depth>0 else ""; speak(f"{dir_en}{depth_str}")
                    elif lang=="hindi":
                        dir_hi={"left":"बाईं तरफ मुड़ें","right":"दाईं तरफ मुड़ें","forward":"आगे जाएं"}.get(direction,"आगे जाएं")
                        depth_str=f" {depth:.0f} मीटर" if depth>0 else ""; speak(f"{dir_hi}{depth_str}")
                    elif lang=="tamil":
                        speak({"left":"இடது திரும்பவும்","right":"வலது திரும்பவும்","forward":"முன்னே செல்லவும்"}.get(direction,"முன்னே செல்லவும்"))
                    elif lang=="kannada":
                        speak({"left":"ಎಡಕ್ಕೆ ತಿರುಗಿ","right":"ಬಲಕ್ಕೆ ತಿರುಗಿ","forward":"ಮುಂದೆ ಹೋಗಿ"}.get(direction,"ಮುಂದೆ ಹೋಗಿ"))
                    elif lang=="malayalam":
                        speak({"left":"ഇടത്തോട്ട് തിരിയുക","right":"വലത്തോട്ട് തിരിയുക","forward":"മുന്നോട്ട് പോകുക"}.get(direction,"മുന്നോട്ട് പോകുക"))
                    else:
                        te_dir=NAV_MSG.get(direction,"ముందుకు వెళ్ళండి")
                        depth_str=f" {depth:.0f} మీటర్లు" if depth>0 else ""; speak(f"{te_dir}{depth_str}")
                    last_speak_time=now; last_direction=direction
        except Exception as e: print(f"[NavWorker] {e}")

# ══════════════════════════════════════════════════════════════════
# NON-BLOCKING INPUT QUEUE
# ══════════════════════════════════════════════════════════════════

_input_queue=queue.Queue(); _input_busy=threading.Event()

def _ask(tag,prompt):
    def _worker():
        _input_busy.set(); val=input(prompt).strip(); _input_queue.put((tag,val)); _input_busy.clear()
    threading.Thread(target=_worker,daemon=True).start()

# ══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════

def draw_nav_arrow(frame,direction,arrow_dst,confidence):
    h,w=frame.shape[:2]; cx,cy=w//2,h//2; src=(cx,h-80)
    if arrow_dst and confidence>0.3: tip=arrow_dst
    elif direction=="left": tip=(cx-180,cy)
    elif direction=="right": tip=(cx+180,cy)
    else: tip=(cx,cy-120)
    cv2.arrowedLine(frame,src,tip,(0,0,0),12,tipLength=0.3)
    cv2.arrowedLine(frame,src,tip,(0,255,80),6,tipLength=0.3)
    pulse=int(18+6*abs(np.sin(time.time()*3)))
    cv2.circle(frame,tip,pulse,(0,255,80),3); cv2.circle(frame,tip,6,(255,255,255),-1)

def draw_nav_hud(frame,dest,sim,direction,n_matches,depth,confidence):
    h,w=frame.shape[:2]; F=cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame,(0,0),(w,50),(10,10,10),-1)
    color=(0,255,80) if sim>0.75 else (0,180,255)
    cv2.putText(frame,f"Navigating to: {dest.upper()}",(10,32),F,0.8,color,2)
    bar_x=w-220
    cv2.putText(frame,"Match:",(bar_x,22),F,0.5,(180,180,180),1)
    cv2.rectangle(frame,(bar_x+55,10),(bar_x+160,26),(50,50,50),-1)
    filled=int((bar_x+55)+sim*105)
    bc=(0,255,80) if sim>ARRIVAL_THRESHOLD else (0,180,255)
    cv2.rectangle(frame,(bar_x+55,10),(filled,26),bc,-1)
    cv2.putText(frame,f"{sim*100:.0f}%",(bar_x+165,22),F,0.5,(220,220,220),1)
    cv2.rectangle(frame,(0,h-48),(w,h),(10,10,10),-1)
    dir_en={"left":"<- Turn LEFT","right":"-> Turn RIGHT","forward":"^ Go FORWARD","arrived":"** ARRIVED **"}.get(direction,"^ Go FORWARD")
    cv2.putText(frame,dir_en,(10,h-18),F,0.85,(0,255,150),2)
    cv2.putText(frame,f"ORB:{n_matches}pts",(w-300,h-18),F,0.55,(180,180,180),1)
    if depth>0: cv2.putText(frame,f"Depth:{depth}m",(w-180,h-18),F,0.55,(150,220,255),1)
    cv2.putText(frame,"Q=Quit  R=Reset  ESC=Back",(w-220,h-36),F,0.40,(120,120,120),1)
    if direction=="arrived" or sim>=ARRIVAL_THRESHOLD:
        ov=frame.copy(); cv2.rectangle(ov,(0,h//2-50),(w,h//2+50),(0,100,0),-1)
        cv2.addWeighted(ov,0.5,frame,0.5,0,frame)
        cv2.putText(frame,f"  ARRIVED at {dest.upper()}!",(w//2-200,h//2+15),F,1.1,(0,255,100),3)

def draw_default_hud(frame,yolo_cache,face_results):
    h,w=frame.shape[:2]; F=cv2.FONT_HERSHEY_SIMPLEX
    for(name,dist,side,x1,y1,x2,y2) in yolo_cache:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(frame,f"{name} {dist:.1f}m",(x1,max(y1-10,14)),F,0.65,(0,220,0),2)
        cv2.putText(frame,side,(x1,y2+20),F,0.5,(0,200,255),2)
    with _rec_lock: fr_copy=list(face_results)
    for item in fr_copy:
        name,conf,top,right,bottom,left=item
        t=int(top/FACE_SCALE); r2=int(right/FACE_SCALE); b=int(bottom/FACE_SCALE); l=int(left/FACE_SCALE)
        dist=face_dist(t,b); side=direction_te((l+r2)//2,w)
        color=(0,220,0) if name!="Unknown" else (0,60,220)
        cv2.rectangle(frame,(l,t),(r2,b),color,2)
        label=f"{name} {conf:.0f}% {dist:.1f}m" if name!="Unknown" else f"Unknown {dist:.1f}m"
        (tw,th),_=cv2.getTextSize(label,F,0.72,2)
        cv2.rectangle(frame,(l,t-th-14),(l+tw+10,t),color,-1)
        cv2.putText(frame,label,(l+5,t-6),F,0.72,(255,255,255),2)
        cv2.putText(frame,side,(l,b+22),F,0.52,(0,200,255),2)
    cv2.rectangle(frame,(0,h-32),(w,h),(10,10,10),-1)
    cv2.putText(frame,f"Obj:{len(yolo_cache)}  Faces:{len(fr_copy)}  |  Lang:{_current_lang.upper()}  |  S=Save  W=Nav  Q=Quit",
                (10,h-10),F,0.47,(160,160,160),1)

def draw_save_face_banner(frame):
    h,w=frame.shape[:2]; mode=get_mode()
    label_map={"save_place":"SAVING PLACE — YOLO & FACE ON HOLD",
               "listening":"VOICE COMMAND ACTIVE — LISTENING...",
               "send_msg":"SEND MESSAGE MODE — PLEASE SPEAK",
               "sos":"** SOS / EMERGENCY MODE **"}
    label=label_map.get(mode,"SAVING FACE — YOLO & NAV ON HOLD")
    color_map={"listening":(0,180,0),"send_msg":(100,50,0),"sos":(0,0,180)}
    color=color_map.get(mode,(0,50,100))
    cv2.rectangle(frame,(0,0),(w,50),color,-1)
    cv2.putText(frame,label,(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,220,255),2)

# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    if not cap.isOpened(): print("❌ Camera failed!"); return

    print("\n✅ Camera started")
    print("   Default: YOLO + Face Recognition")
    print("   S=Save  W=Navigate  Q=Quit")
    print("   Voice: 'listen'→menu  'sos'→emergency  'send message'→SMS\n")

    threading.Thread(target=nav_worker,daemon=True,name="NavWorker").start()

    if SR_OK:
        threading.Thread(target=_voice_command_thread,daemon=True,name="VoiceCmd").start()
        speak_force("దృష్టి సిద్ధంగా ఉంది. listen అని చెప్పి ఆదేశాలు ఇవ్వండి")
    else:
        print("[VoiceCmd] Disabled — install SpeechRecognition and pyaudio")

    nav_dest=""; ref_thumb=None; arrived_time=0.0
    last_yolo_speak=0.0; last_face_speak:dict={}
    yolo_cache:list=[]; frame_count=0; fps=0.0; fps_timer=time.time()

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_count+=1; now=time.time(); mode=get_mode()
        if frame_count%30==0:
            fps=30.0/max(now-fps_timer,1e-6); fps_timer=now
        if mode=="default":
            small=cv2.resize(frame,(0,0),fx=FACE_SCALE,fy=FACE_SCALE); _latest_small[0]=small
        if mode=="nav": _nav_frame[0]=frame.copy()
        if mode=="ocr": _ocr_frame[0]=frame.copy()

        # Input queue
        while not _input_queue.empty():
            tag,val=_input_queue.get_nowait()
            if tag=="save_choice":
                if val.lower()=="f": threading.Thread(target=_save_face_thread,args=(cap,),daemon=True).start()
                elif val.lower()=="p": _ask("save_place_name","[SavePlace] Place name: ")
                else: print(f"[Save] Unknown choice '{val}'")
            elif tag=="save_place_name":
                if val: threading.Thread(target=_save_place_thread,args=(cap,val),daemon=True).start()
                else: print("[SavePlace] No name — skipped."); set_mode("default")
            elif tag=="nav_dest":
                if val in DB.places:
                    nav_dest=val; ref_thumb=DB.places[val]["images"][0]
                    _nav_dest[0]=val; _nav_stop.clear()
                    with _nav_lock: _nav_state.update({"arrived":False,"direction":"forward","sim":0.0,"n_matches":0,"confidence":0.0,"depth":-1.0,"arrow_dst":None})
                    arrived_time=0.0; set_mode("nav"); speak(f"{val} వైపు నావిగేషన్ మొదలవుతుంది"); print(f"\n✅ Navigating to: {val}\n")
                else: print(f"❌ '{val}' not found."); set_mode("default")
            elif tag=="reset_dest":
                if val in DB.places:
                    nav_dest=val; _nav_dest[0]=val; ref_thumb=DB.places[val]["images"][0]; arrived_time=0.0
                    speak(f"{val} వైపు నావిగేషన్ మొదలవుతుంది"); print(f"✅ Destination → {val}")
                else: print(f"❌ '{val}' not in places.")

        # Voice command queue
        while not _vc_queue.empty():
            vc_tag,vc_val=_vc_queue.get_nowait()
            if vc_tag=="quit":
                _face_stop.set(); _nav_stop.set(); _vc_stop.set()
                _ocr_stop_flag.set(); _ocr_active.clear()
                cap.release(); cv2.destroyAllWindows(); pygame.mixer.quit()
                print("\nGoodbye."); return
            elif vc_tag=="vc_stop_nav":
                if get_mode()=="nav":
                    set_mode("default"); _nav_dest[0]=None; nav_dest=""; ref_thumb=None; arrived_time=0.0
                    with _nav_lock: _nav_state.update({"arrived":False,"direction":"forward","sim":0.0})
            elif vc_tag=="vc_nav_dest":
                dest_name=vc_val
                if dest_name in DB.places:
                    nav_dest=dest_name; ref_thumb=DB.places[dest_name]["images"][0]
                    _nav_dest[0]=dest_name; _nav_stop.clear()
                    with _nav_lock: _nav_state.update({"arrived":False,"direction":"forward","sim":0.0,"n_matches":0,"confidence":0.0,"depth":-1.0,"arrow_dst":None})
                    arrived_time=0.0; set_mode("nav"); print(f"[VoiceCmd] Navigating to: {dest_name}")
            elif vc_tag=="vc_save_place":
                if get_mode()=="default": threading.Thread(target=_save_place_thread,args=(cap,vc_val),daemon=True).start()
            elif vc_tag=="vc_save_face":
                if get_mode()=="default": threading.Thread(target=_save_face_thread_named,args=(cap,vc_val),daemon=True).start()
            elif vc_tag=="vc_start_ocr":
                if get_mode()=="default": start_ocr_mode(cap)
            elif vc_tag=="vc_stop_ocr":
                stop_ocr_mode()

        # Render
        display=frame.copy(); mode=get_mode()

        if mode in ("save_face","save_place","listening","send_msg","sos"):
            draw_save_face_banner(display)
            action_map={"save_place":"saving place...","listening":"voice command — say your command...",
                        "send_msg":"send message mode — speak the number and message...",
                        "sos":"SOS emergency mode..."}
            action=action_map.get(mode,"saving face...")
            cv2.putText(display,f"Please wait — {action}",(10,display.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,200,255),2)
            cv2.imshow("DRISHTI",display); cv2.waitKey(1); continue

        elif mode=="ocr":
            if _ocr_last_text:
                h_d2,w_d2=display.shape[:2]
                cv2.rectangle(display,(0,h_d2-80),(w_d2,h_d2-45),(0,40,0),-1)
                disp_text=_ocr_last_text[:80]+"..." if len(_ocr_last_text)>80 else _ocr_last_text
                cv2.putText(display,disp_text,(10,h_d2-55),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,100),1)
            h_d,w_d=display.shape[:2]
            cv2.rectangle(display,(0,0),(w_d,45),(0,60,0),-1)
            cv2.putText(display,"OCR MODE | Point camera at text | Say CANCEL to stop",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,100),2)
            cv2.imshow("DRISHTI",display)
            key_ocr=cv2.waitKey(1)&0xFF
            if key_ocr==27 or key_ocr==ord("q"): stop_ocr_mode()
            continue

        elif mode=="nav":
            with _nav_lock: state=dict(_nav_state)
            gray=cv2.cvtColor(display,cv2.COLOR_BGR2GRAY); kps_d,_=_orb.detectAndCompute(gray,None)
            if kps_d:
                for kp in kps_d[:30]: cv2.circle(display,(int(kp.pt[0]),int(kp.pt[1])),3,(0,200,255),-1)
            if not state["arrived"]: draw_nav_arrow(display,state["direction"],state["arrow_dst"],state["confidence"])
            if ref_thumb is not None:
                tw,th=160,100; thumb=cv2.resize(ref_thumb,(tw,th)); h_d,w_d=display.shape[:2]
                display[15:15+th,w_d-tw-5:w_d-5]=thumb
                cv2.putText(display,f"Target: {nav_dest}",(w_d-tw-5,15+th+18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            draw_nav_hud(display,nav_dest,state["sim"],state["direction"],state["n_matches"],state["depth"],state["confidence"])
            if state["arrived"]:
                if arrived_time==0.0: arrived_time=now; print(f"\n✅ Arrived at '{nav_dest}'! Back to default in 3s...")
                if now-arrived_time>=3.0:
                    set_mode("default"); _nav_dest[0]=None; arrived_time=0.0; nav_dest=""; ref_thumb=None
                    with _nav_lock: _nav_state.update({"arrived":False,"direction":"forward","sim":0.0})
                    print("[NAV] Back to default mode.\n")
            else: arrived_time=0.0

        else:  # default
            if frame_count%FRAME_SKIP==0:
                results=yolo(frame,verbose=False,conf=CONF_THRESHOLD); detections=[]
                for r in results:
                    for box in r.boxes:
                        cls=int(box.cls[0]); name=yolo.names[cls]
                        x1,y1,x2,y2=map(int,box.xyxy[0]); cx=(x1+x2)//2; bh=y2-y1
                        ref=OBJECT_REF_HEIGHT_M.get(name,DEFAULT_REF_H)
                        dist=bbox_dist(bh,ref); side=direction_te(cx,frame.shape[1])
                        detections.append((name,dist,side,x1,y1,x2,y2))
                detections.sort(key=lambda d:d[1]); yolo_cache=detections
                if yolo_cache and now-last_yolo_speak>YOLO_COOLDOWN:
                    n0,d0,s0=yolo_cache[0][0],yolo_cache[0][1],yolo_cache[0][2]
                    tel=TELUGU_OBJ.get(n0,n0); lang=_current_lang
                    if lang=="english":
                        sv0={"Left":"on your left","Right":"on your right","Center":"in front of you"}.get(s0,s0)
                        speak(f"{sv0} {n0} {d0} meters away watch out")
                    elif lang=="hindi":
                        sv0={"Left":"బాईం తరఫ","Right":"దాईం తరఫ","Center":"సామ్నే"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} మీటర్ దూర్ హై సావధాన్")
                    elif lang=="tamil":
                        sv0={"Left":"இடது பக்கம்","Right":"வலது பக்கம்","Center":"முன்னால்"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} மீட்டர் தொலைவில் உள்ளது கவனமாக இருங்கள்")
                    elif lang=="kannada":
                        sv0={"Left":"ಎಡಭಾಗ","Right":"ಬಲಭಾಗ","Center":"ಮುಂದೆ"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} ಮೀಟರ್ ದೂರದಲ್ಲಿದೆ ಎಚ್ಚರಿಕೆ")
                    elif lang=="malayalam":
                        sv0={"Left":"ഇടതുവശം","Right":"വലതുവശം","Center":"മുന്നിൽ"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} മീറ്റർ അകലെ ഉണ്ട് ശ്രദ്ധിക്കുക")
                    else:
                        sv0={"Left":"ఎడమ వైపు","Right":"కుడి వైపు","Center":"మీ ముందు"}.get(s0,s0)
                        speak(f"{sv0} {tel} {d0} మీటర్ల దూరంలో వున్నారు చూసుకోండి")
                    last_yolo_speak=now
            with _rec_lock: face_results_snap=list(_rec_results)
            seen=set()
            sorted_faces=sorted(face_results_snap,key=lambda d:-(int(d[4]/FACE_SCALE)-int(d[2]/FACE_SCALE)))
            for item in sorted_faces:
                name,conf,top,right,bottom,left=item; seen.add(name)
                t=int(top/FACE_SCALE); r2=int(right/FACE_SCALE); b=int(bottom/FACE_SCALE); l2=int(left/FACE_SCALE)
                dist=face_dist(t,b); side=direction_te((l2+r2)//2,frame.shape[1])
                last_t=last_face_speak.get(name,0.0)
                if now-last_t>=FACE_COOLDOWN:
                    last_face_speak[name]=now; lang=_current_lang
                    if lang=="english":
                        sv={"Left":"on your left","Right":"on your right","Center":"in front of you"}.get(side,side)
                        msg=f"{sv} {'unknown person' if name=='Unknown' else name} {dist:.1f} meters away please be careful"
                    elif lang=="hindi":
                        sv={"Left":"బాईం తరఫ","Right":"దాయీం తరఫ","Center":"సామ్నే"}.get(side,side)
                        msg=f"{sv} {'అజ్ఞాత వ్యక్తి' if name=='Unknown' else name} {dist:.1f} మీటర్ దూర్ హై సావధాన్ రహేం"
                    elif lang=="tamil":
                        sv={"Left":"இடது பக்கம்","Right":"வலது பக்கம்","Center":"முன்னால்"}.get(side,side)
                        msg=f"{sv} {'தெரியாத நபர்' if name=='Unknown' else name} {dist:.1f} மீட்டர் தொலைவில் கவனமாக இருங்கள்"
                    elif lang=="kannada":
                        sv={"Left":"ಎಡಭಾಗ","Right":"ಬಲಭಾಗ","Center":"ಮುಂದೆ"}.get(side,side)
                        msg=f"{sv} {'ಅಪರಿಚಿತ ವ್ಯಕ್ತಿ' if name=='Unknown' else name} {dist:.1f} ಮೀಟರ್ ದೂರದಲ್ಲಿದ್ದಾರೆ ಎಚ್ಚರಿಕೆ"
                    elif lang=="malayalam":
                        sv={"Left":"ഇടതുവശം","Right":"വലതുവശം","Center":"മുന്നിൽ"}.get(side,side)
                        msg=f"{sv} {'അപരിചിതൻ' if name=='Unknown' else name} {dist:.1f} മീറ്റർ അകലെ ഉണ്ട് ശ്രദ്ധിക്കുക"
                    else:
                        sv={"Left":"ఎడమ వైపు","Right":"కుడి వైపు","Center":"మీ ముందు"}.get(side,side)
                        msg=(f"{sv} గుర్తు తెలియని వ్యక్తి {dist:.1f} మీటర్ల దూరంలో వున్నారు చూసుకొని వెళ్ళండి"
                             if name=="Unknown" else f"{sv} {name} {dist:.1f} మీటర్ల దూరంలో వున్నారు చూసుకొని వెళ్ళండి")
                    speak(msg); break
            for name in list(last_face_speak.keys()):
                if name not in seen: del last_face_speak[name]
            draw_default_hud(display,yolo_cache,_rec_results)

        cv2.putText(display,f"{fps:.0f}fps",(display.shape[1]-75,display.shape[0]-52),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,100),1)
        cv2.imshow("DRISHTI",display); key=cv2.waitKey(1)&0xFF

        if key==ord("q"): break
        if key==ord("s") and not _input_busy.is_set():
            if mode=="default":
                print("\n"+"═"*50+"\n  SAVE — F=Face  |  P=Place\n"+"═"*50)
                _ask("save_choice","Enter F (face) or P (place): ")
            elif mode=="nav": _ask("save_place_name","[SavePlace] Place name: ")
        if key==ord("w") and mode=="default" and not _input_busy.is_set():
            if not DB.places: print("\n⚠️  No saved places!\n")
            else:
                print("\n"+"═"*55+"\n  NAVIGATION — Available places:")
                for i,nm in enumerate(DB.names(),1): print(f"  {i}. {nm}")
                print("═"*55); _ask("nav_dest","Enter destination name: "); set_mode("nav")
        if key==ord("r") and mode=="nav" and not _input_busy.is_set():
            for i,nm in enumerate(DB.names(),1): print(f"  {i}. {nm}")
            _ask("reset_dest","New destination: ")
        if key==27 and mode=="nav":
            set_mode("default"); nav_dest=""; ref_thumb=None; arrived_time=0.0; _nav_dest[0]=None
            with _nav_lock: _nav_state.update({"arrived":False,"direction":"forward","sim":0.0})
            print("\n[NAV] Exited → back to default mode.\n")

    _face_stop.set(); _nav_stop.set(); _vc_stop.set()
    _ocr_stop_flag.set(); _ocr_active.clear()
    cap.release(); cv2.destroyAllWindows(); pygame.mixer.quit()
    print("\nGoodbye.")

if __name__=="__main__":
    main()
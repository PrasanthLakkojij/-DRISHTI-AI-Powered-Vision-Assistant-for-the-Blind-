# 👁️ DRISHTI — AI-Powered Vision Assistant for the Blind

> *"Acting as eyes for those who cannot see"*

DRISHTI is an AI-powered voice assistant that helps blind and low-vision individuals navigate the world independently. It combines real-time **object detection**, **face recognition**, **voice authentication**, **AI scene description**, **text reading**, **place navigation**, a **general AI assistant**, and **emergency SOS** — all activated by saying *"Hey Jarvis"* and controlled entirely by voice.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Wake Word | Say **"Hey Jarvis"** to activate — silent until then |
| 🔐 Voice Authentication | Only the registered owner's voice can use DRISHTI |
| 👁️ Object Detection | Real-time YOLOv8-based detection with distance estimation |
| 👤 Face Recognition | Identifies known people and announces their name and distance |
| 🧠 Describe Mode | AI (Groq vision model) describes the full scene aloud on request |
| 🤖 AI Mode | Ask DRISHTI anything — general questions answered aloud by AI |
| 📖 Text Reading | Reads signs, labels, and documents aloud (Groq vision model) |
| 🧭 Place Navigation | Learns and guides the user back to saved locations |
| 🌐 Multilingual Voice | Responds in Telugu, Hindi, Tamil, Kannada, Malayalam, English, and more |
| 🆘 SOS Emergency | Say **"S.O.S."** to trigger an emergency call via Twilio |
| 💬 Send Message | Say **"send message"** to dictate and send an SMS to any number |

---

## 🚀 Quick Setup (Run in 5 Steps)

### Step 1 — Clone the repository

```bash
git clone https://github.com/PrasanthLakkojij/-DRISHTI-AI-Powered-Vision-Assistant-for-the-Blind-.git
cd '.\-DRISHTI-AI-Powered-Vision-Assistant-for-the-Blind-'
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv drishti_env

# Windows:
drishti_env\Scripts\activate

# Linux / Mac:
source drishti_env/bin/activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** `face_recognition` requires `cmake` and `dlib`. If it fails, run:
> ```bash
> pip install cmake
> pip install dlib
> pip install face_recognition
> ```

### Step 4 — Add your API keys

DRISHTI's AI features (Describe, AI Mode, Text Reading) and emergency features (SOS, SMS) need API keys. Create a file named `.env` in the project folder:

```bash
GROQ_API_KEY=your_groq_key_here
TWILIO_ACCOUNT_SID=your_twilio_sid_here
TWILIO_AUTH_TOKEN=your_twilio_token_here
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
```

> Get a free Groq key at [console.groq.com](https://console.groq.com) — required for Describe, AI Mode, and Text Reading.
> Twilio is only needed for SOS and Send Message — the app runs fine without it, with those two features disabled.

### Step 5 — Run DRISHTI

```bash
python main.py
```

Then say **"Hey Jarvis"** to start!

> On the very first run, DRISHTI introduces itself and asks you to enroll your voice (3 short voice samples) before it can be used.

---

## 🎮 Voice Commands

All commands below are spoken **after saying "listen"** to wake DRISHTI into command mode.

| Say This | Action |
|---|---|
| say **"Hey Jarvis"** | Wakes DRISHTI from sleep |
| Say **"listen"** | Opens the voice command menu |
| say **Menu** | Lists all available commands aloud |
| say **"Save person"** | Learns and saves a new face |
| say **"save place"** | Learns and saves a new location |
| say **"where is place"** / "navigate" | Starts guided navigation to a saved place |
| say **"stop navigation"** | Exits navigation mode |
| say **"read text"** | Reads any visible text aloud |
| say **"describe"** | Describes the full surrounding scene aloud |
| say **"ai mode"** | Starts a general AI Q&A conversation |
| Press **"Change language"** | Switches the spoken language |
| say **"send message"** | Dictates and sends an SMS |
| Say **"s.o.s"** | Places an emergency call via Twilio |
| say **"cancel"** | Stops the current mode (describe, AI mode, navigation, reading) |

---

## 📋 Requirements

- Python 3.9 (built and tested on 3.9; newer versions may need adjusted package versions)
- Webcam / Camera
- Microphone
- Windows / Linux / Mac
- A free [Groq](https://console.groq.com) API key
- A [Twilio](https://www.twilio.com) account (optional, for SOS/SMS)

---

## 🛠️ Technologies Used

- **YOLOv8** — Real-time object detection
- **OpenCV** — Computer vision and camera handling
- **SpeechBrain (ECAPA-TDNN)** — Voice authentication
- **OpenWakeWord** — Wake word detection ("Hey Jarvis")
- **CLIP + ORB** — Place recognition and navigation
- **Groq (Llama 4 Scout, GPT-OSS-120B, Whisper)** — Scene description, AI Q&A, text reading, and speech-to-text
- **Edge TTS** — Natural multilingual voice output
- **Twilio** — SOS emergency calls and SMS
- **PyTorch** — Deep learning backend
- **face_recognition** — Face identification

---

## 📁 File Structure

```
DRISHTI/
├── main.py                  ← Main entry point (run this)
├── requirements.txt         ← All dependencies
├── .env                     ← Your API keys (you create this, not included)
└── .gitignore               ← Excludes secrets and personal data from git
```

> **Note:** Model weight files (`yolov8n.pt`) and SpeechBrain's cached models auto-download on first run. Personal/biometric files generated on your own device — `owner_voice.npy`, `owner_recordings/`, `saved_faces/`, `saved_places/`, `face_db.pkl` — are intentionally excluded from this repository and are not shared between devices.

---

## 👨‍💻 Developer

**Prasanth Lakkoji**
Project: DRISHTI — Vision for the Visually Impaired

---

## 📄 License

This project is open source and available for educational and research purposes.

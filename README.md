# 👁️ DRISHTI — AI-Powered Vision Assistant for the Blind

> *"Acting as eyes for those who cannot see"*

DRISHTI is an AI-powered visual assistant that helps visually impaired individuals navigate the world. It combines **object detection**, **face recognition**, **voice authentication**, **text reading**, **place navigation**, and **emergency SOS** — all activated by saying *"Hey Jarvis"*.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Wake Word | Say **"Hey Jarvis"** to activate — silent until then |
| 🔐 Voice Authentication | Only the registered owner can use DRISHTI |
| 👁️ Object Detection | Real-time YOLO-based detection with distance estimation |
| 👤 Face Recognition | Identifies known people and announces their name |
| 🗣️ change Language  | changes language from one to one |
| 📖 Text Reading (OCR) | Reads signs, labels, and documents aloud |
| 🧭 Place Navigation | Guides user to saved locations |
| 🔊 Telugu Voice Support | All responses in Telugu language |
| 🆘 SOS Emergency | Say **"s.o.s"** to trigger emergency call via Twilio |
| 💬 Send Message | Say **"send message"** to send SMS to any number |

---

## 🚀 Quick Setup (Run in 4 Steps)

### Step 1 — Clone the repository

```bash
git clone https://github.com/PrasanthLakkojij/-DRISHTI-AI-Powered-Vision-Assistant-for-the-Blind-.git
cd -DRISHTI-AI-Powered-Vision-Assistant-for-the-Blind-
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

### Step 4 — Run DRISHTI

```bash
python main.py
```

Then say **"Hey Jarvis"** to start!

---

## 🎮 Controls

| Input | Action |
|---|---|
| say **Menu** | says available commands  |
| Say **"Hey Jarvis"** | Wake up DRISHTI |
| Say **"listen"** | Open voice command menu (in Telugu) |
| Say **"s.o.s"** | Emergency call via Twilio |
| Say **"send message"** | Send SMS to a number |
| say **"Save person"** | Save a face of a person |
| say **"save place"** | save a particulare place  |
| say **"where is place"** | Enters into navigation mode |
| say **"cancel"** | cancel current command |
| Press **"Change language"** | changes to another language |


---

## 📋 Requirements

- Python 3.8 or higher
- Webcam / Camera
- Microphone
- Windows / Linux / Mac

---

## 🛠️ Technologies Used

- **YOLOv8** — Object Detection
- **OpenCV** — Computer Vision
- **SpeechBrain** — Voice Authentication
- **OpenWakeWord** — Wake Word Detection ("Hey Jarvis")
- **EasyOCR** — Text Reading
- **Edge TTS** — Telugu Voice Output
- **Twilio** — SOS Emergency Call & SMS
- **PyTorch** — Deep Learning Backend
- **face_recognition** — Face Recognition

---

## 📁 File Structure

```
DRISHTI/
├── main.py                  ← Main entry point (run this)
├── wakeup.py                ← Wake word detection
├── voice_verication.py      ← Voice authentication
├── face_recognize.py        ← Face recognition module
├── place_recognize.py       ← Place recognition module
├── place+navigation.py      ← Navigation logic
├── yolo+face_recogniz.py    ← YOLO + face combined
├── distance_measure.py      ← Distance estimation
├── text_read.py             ← OCR text reading
├── navigation.py            ← Navigation helpers
├── sos.py                   ← Emergency SOS
├── test.py                  ← Testing module
├── yolov8n.pt               ← YOLOv8 model weights
├── owner_voice.npy          ← Owner voice profile
└── requirements.txt         ← All dependencies
```

---

## 👨‍💻 Developer

**Prasanth Lakkoji**
Project: DRISHTI — Vision for the Visually Impaired

---

## 📄 License

This project is open source and available for educational and research purposes.

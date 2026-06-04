import os
import asyncio
import numpy as np
import sounddevice as sd
import pygame
import edge_tts

from googletrans import Translator
from scipy.io.wavfile import write

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ==========================================
# 🚀 LOAD SPEECHBRAIN MODEL
# ==========================================
print("[VV] Loading SpeechBrain model...")

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec",
    local_strategy=LocalStrategy.COPY
)

print("[VV] MODEL LOADED SUCCESSFULLY")

# ==========================================
# 📁 CONFIG
# ==========================================
SAMPLE_RATE = 16000
DURATION = 5
OWNER_EMBED_PATH = "owner_voice.npy"

# ==========================================
# 📁 OWNER RECORDINGS FOLDER
# ==========================================
OWNER_FOLDER = "owner_recordings"

os.makedirs(OWNER_FOLDER, exist_ok=True)

# ==========================================
# 🔊 INIT AUDIO
# ==========================================
pygame.mixer.init()

# ==========================================
# 🌍 TRANSLATOR
# ==========================================
translator = Translator()

# ==========================================
# 🌍 TRANSLATE ENGLISH → TELUGU
# ==========================================
def translate_to_telugu(text):

    translated = translator.translate(
        text,
        dest='te'
    )

    return translated.text

# ==========================================
# 🔊 TELUGU TTS
# ==========================================
async def telugu_tts(text, filename="temp_voice.mp3"):

    communicate = edge_tts.Communicate(
        text,
        voice="te-IN-ShrutiNeural"
    )

    await communicate.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

    pygame.mixer.music.unload()

    # 🔥 DELETE AUDIO AFTER PLAYING
    if os.path.exists(filename):
        os.remove(filename)

# ==========================================
# 🗣 SPEAK FUNCTION
# ==========================================
def speak(text):

    telugu_text = translate_to_telugu(text)

    print(f"🗣 Telugu: {telugu_text}")

    asyncio.run(
        telugu_tts(telugu_text)
    )

# ==========================================
# 🎤 RECORD AUDIO
# ==========================================
def record_audio(filename="temp.wav", duration=DURATION):

    print(f"\n🎙 Recording for {duration} seconds...")

    speak("Please speak now")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    write(filename, SAMPLE_RATE, audio)

    print("✅ Recording saved")

    return filename

# ==========================================
# 🧠 EXTRACT SPEAKER EMBEDDING
# ==========================================
def extract_embedding(wav_path):

    signal = model.load_audio(wav_path)

    embedding = model.encode_batch(signal)

    return embedding.squeeze().detach().cpu().numpy()

# ==========================================
# 📊 COSINE SIMILARITY
# ==========================================
def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

# ==========================================
# 🔐 ENROLL OWNER VOICE
# ==========================================
def enroll_voice():

    print("\n🔐 OWNER ENROLLMENT STARTED")

    speak("Owner enrollment started")

    embeddings = []

    for i in range(3):

        print(f"\n📢 Sample {i+1}/3")

        speak(f"Sample {i+1}")

        wav_file = record_audio(
            os.path.join(
                OWNER_FOLDER,
                f"owner_{i}.wav"
            )
        )

        emb = extract_embedding(wav_file)

        embeddings.append(emb)

    # 🔥 Average embeddings
    owner_embedding = np.mean(
        embeddings,
        axis=0
    )

    np.save(
        OWNER_EMBED_PATH,
        owner_embedding
    )

    print("\n✅ OWNER VOICE SAVED")

    speak("Owner voice saved successfully")

# ==========================================
# 🔍 VERIFY OWNER VOICE
# ==========================================
def verify_voice(threshold=0.75):

    if not os.path.exists(OWNER_EMBED_PATH):

        print("❌ No owner enrolled")

        speak(
            "No owner enrolled. Please enroll first"
        )

        return

    print("\n🎤 SPEAK FOR VERIFICATION")

    speak("Speak for verification")

    owner_embedding = np.load(
        OWNER_EMBED_PATH
    )

    wav_file = record_audio("test.wav")

    test_embedding = extract_embedding(
        wav_file
    )

    # 🔥 DELETE TEST AUDIO
    if os.path.exists(wav_file):
        os.remove(wav_file)

    score = cosine_similarity(
        owner_embedding,
        test_embedding
    )

    print(f"\n📊 Similarity Score: {score:.4f}")

    # ======================================
    # OWNER VERIFIED
    # ======================================
    if score >= threshold:

        print("\n🟢 OWNER VERIFIED")
        print("🔓 ACCESS GRANTED")

        speak(
            "Owner verified. Access granted"
        )

    # ======================================
    # UNKNOWN SPEAKER
    # ======================================
    else:

        print("\n🔴 UNKNOWN SPEAKER")
        print("🚫 ACCESS DENIED")

        speak(
            "Unknown speaker. Access denied"
        )

# ==========================================
# 🚀 MAIN MENU
# ==========================================
if __name__ == "__main__":

    speak(
        "Voice authentication system started"
    )

    while True:

        print("""
====================================
🔐 OWNER VOICE AUTHENTICATION
====================================
1. Enroll Owner Voice
2. Verify Voice
3. Exit
====================================
""")

        choice = input("👉 Enter choice: ")

        # ==================================
        # ENROLL
        # ==================================
        if choice == "1":

            enroll_voice()

        # ==================================
        # VERIFY
        # ==================================
        elif choice == "2":

            verify_voice()

        # ==================================
        # EXIT
        # ==================================
        elif choice == "3":

            speak("System shutting down")

            print("\n👋 Exiting System...")

            break

        # ==================================
        # INVALID CHOICE
        # ==================================
        else:

            print("\n❌ Invalid Choice")

            speak("Invalid choice")
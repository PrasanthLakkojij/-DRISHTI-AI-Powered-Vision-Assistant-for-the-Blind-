import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

# Auto-download pre-trained models on first run
openwakeword.utils.download_models()

# Load wake word model — "hey jarvis" is a built-in pre-trained one
model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

# Audio settings
CHUNK = 1280        # 80ms at 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("🎤 Listening... Say 'Hey Jarvis' to trigger!")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        audio = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio, dtype=np.int16)

        prediction = model.predict(audio_np)

        for wake_word, score in prediction.items():
            if score > 0.5:
                print(f"✅ Wake word detected: '{wake_word}' (confidence: {score:.2f})")

except KeyboardInterrupt:
    print("\nStopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
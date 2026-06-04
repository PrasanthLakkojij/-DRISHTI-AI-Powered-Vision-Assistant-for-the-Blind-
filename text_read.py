import cv2
import easyocr
import asyncio
import edge_tts
import pygame
import threading
import time
import uuid
from paddleocr import PaddleOCR

# ==========================
# INIT
# ==========================

print("Loading EasyOCR...")
easy_reader = easyocr.Reader(['en'])

print("Loading PaddleOCR...")
paddle_reader = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

pygame.mixer.init()

last_spoken = ""
last_time = 0

# ==========================
# TTS
# ==========================

def speak(text):
    threading.Thread(
        target=lambda: asyncio.run(tts(text)),
        daemon=True
    ).start()

async def tts(text):
    try:
        filename = f"tts_{uuid.uuid4().hex}.mp3"

        communicate = edge_tts.Communicate(
            text,
            voice="en-US-AriaNeural"
        )

        await communicate.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

    except Exception as e:
        print(e)

# ==========================
# OCR FUNCTION
# ==========================

def detect_text(frame):

    results = []

    # EasyOCR
    try:
        easy = easy_reader.readtext(frame)

        for item in easy:
            box = item[0]
            text = item[1]
            conf = item[2]

            if conf > 0.4:
                results.append(
                    (
                        text,
                        conf,
                        box
                    )
                )
    except:
        pass

    # PaddleOCR
    try:
        paddle = paddle_reader.ocr(frame)

        if paddle and paddle[0]:
            for line in paddle[0]:

                box = line[0]
                text = line[1][0]
                conf = line[1][1]

                if conf > 0.4:
                    results.append(
                        (
                            text,
                            conf,
                            box
                        )
                    )
    except:
        pass

    return results

# ==========================
# CAMERA
# ==========================

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        exit()

    print("\nPress Q to quit\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = detect_text(frame)

        detected_texts = []

        for text, conf, box in results:

            detected_texts.append(text)

            pts = []

            for p in box:
                pts.append(
                    (
                        int(p[0]),
                        int(p[1])
                    )
                )

            for i in range(len(pts)):
                cv2.line(
                    frame,
                    pts[i],
                    pts[(i + 1) % len(pts)],
                    (0,255,0),
                    2
                )

            x = pts[0][0]
            y = pts[0][1]

            cv2.putText(
                frame,
                text,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        unique_text = " ".join(
            list(dict.fromkeys(detected_texts))
        )

        current = time.time()

        if (
            unique_text
            and unique_text != last_spoken
            and current - last_time > 5
        ):

            print("\nDetected:")
            print(unique_text)

            speak(unique_text)

            last_spoken = unique_text
            last_time = current

        cv2.imshow(
            "GOD LEVEL OCR",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
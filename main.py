import json
import wave
from collections import deque

import sounddevice as sd
import webrtcvad
from vosk import Model, KaldiRecognizer

import requests


SERVER_URL = "http://192.168.1.100:8000/audio"


def upload_wav(filename):
    try:
        with open(filename, "rb") as f:
            response = requests.post(
                SERVER_URL,
                files={
                    "file": (
                        filename,
                        f,
                        "audio/wav"
                    )
                },
                timeout=30
            )

        print(response.json())

    except Exception as e:
        print("Ошибка отправки:", e)

# --------------------
# CONFIG
# --------------------

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

WAKE_WORD = "ася"

LED_PIN = 17

SILENCE_TIMEOUT_MS = 1500

# --------------------
# LED
# --------------------

try:
    from gpiozero import LED

    led = LED(LED_PIN)
except Exception:
    class DummyLed:
        def on(self):
            print("[LED ON]")

        def off(self):
            print("[LED OFF]")

    led = DummyLed()

# --------------------
# VAD
# --------------------

vad = webrtcvad.Vad(2)

# --------------------
# VOSK
# --------------------

model = Model("vosk-model-small-ru-0.22")

grammar = json.dumps(
    [
        WAKE_WORD,
        "[unk]"
    ],
    ensure_ascii=False
)

recognizer = KaldiRecognizer(
    model,
    SAMPLE_RATE,
    grammar
)

# --------------------
# STATE
# --------------------

state = "WAIT_WAKE"

prebuffer = deque(maxlen=20)

recorded_frames = []
silence_frames = 0

print("Готов. Жду слово:", WAKE_WORD)

# --------------------
# AUDIO LOOP
# --------------------

with sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_SIZE,
    dtype="int16",
    channels=1
) as stream:

    while True:

        frame, _ = stream.read(FRAME_SIZE)
        frame = bytes(frame)  # Конвертируем memoryview в bytes

        if state == "WAIT_WAKE":

            prebuffer.append(frame)

            if recognizer.AcceptWaveform(frame):
                result = json.loads(
                    recognizer.Result()
                )

                text = result.get("text", "").strip()

                if WAKE_WORD in text:
                    print("Активировано!")

                    led.on()

                    recorded_frames = list(prebuffer)

                    silence_frames = 0

                    state = "RECORD"

        elif state == "RECORD":

            recorded_frames.append(frame)

            is_speech = vad.is_speech(
                frame,
                SAMPLE_RATE
            )

            if is_speech:
                silence_frames = 0
            else:
                silence_frames += 1

            silence_ms = silence_frames * FRAME_MS

            if silence_ms >= SILENCE_TIMEOUT_MS:

                filename = "command.wav"

                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)

                    for chunk in recorded_frames:
                        wf.writeframes(chunk)

                print(f"Сохранено: {filename}")

                upload_wav(filename)

                led.off()

                recognizer.Reset()

                state = "WAIT_WAKE"

                print("Жду слово:", WAKE_WORD)

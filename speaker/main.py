import array
import base64
import json
import math
import uuid
import wave
from collections import deque
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

import websocket  # pip install websocket-client


BASE_DIR = Path(__file__).resolve().parent

SERVER_WS_URL = "ws://127.0.0.1:8000/ws"
DEVICE_ID = "speaker_01"


def send_wav(filename):
    """Отправляет WAV на сервер по WebSocket и показывает распознанный текст."""
    try:
        with open(filename, "rb") as f:
            wav_bytes = f.read()

        ws = websocket.create_connection(SERVER_WS_URL, timeout=30)
        try:
            ws.send(json.dumps({
                "type": "audio_request",
                "request_id": str(uuid.uuid4()),
                "device_id": DEVICE_ID,
                "audio_data": base64.b64encode(wav_bytes).decode("ascii"),
                "format": "wav",
                "sample_rate": SAMPLE_RATE,
            }))

            response = json.loads(ws.recv())
        finally:
            ws.close()

        if response.get("type") == "audio_response":
            print("Распознано:", response.get("text_response", ""))
        else:
            print("Ошибка сервера:", response.get("message", response))

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

# Порог энергии (RMS) фрейма: ниже = тишина. Подбирается под микрофон/шум.
SILENCE_RMS_THRESHOLD = 500

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
# VAD (энергетический, без нативных зависимостей)
# --------------------

def is_speech(frame_bytes):
    """True, если громкость фрейма выше порога тишины."""
    samples = array.array("h")
    samples.frombytes(frame_bytes)
    if not samples:
        return False
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    return rms > SILENCE_RMS_THRESHOLD

# --------------------
# VOSK
# --------------------

model = Model(str(BASE_DIR / "models" / "vosk-model-small-ru-0.22"))

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

            if is_speech(frame):
                silence_frames = 0
            else:
                silence_frames += 1

            silence_ms = silence_frames * FRAME_MS

            if silence_ms >= SILENCE_TIMEOUT_MS:

                filename = str(BASE_DIR / "command.wav")

                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)

                    for chunk in recorded_frames:
                        wf.writeframes(chunk)

                print(f"Сохранено: {filename}")

                send_wav(filename)

                led.off()

                recognizer.Reset()

                state = "WAIT_WAKE"

                print("Жду слово:", WAKE_WORD)

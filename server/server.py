"""
Asya — сервер транскрипции (MVP).

WebSocket-сервер: принимает аудио от колонки, прогоняет через
faster-whisper, отправляет распознанный текст в локальную LLM (Ollama),
озвучивает ответ через TTS (pyttsx3/SAPI5) и возвращает текст + аудио.

Протокол (см. AGENTS.MD):
  Запрос  -> {"type": "audio_request",  "request_id": ..., "audio_data": <base64 WAV>, ...}
  Ответ   -> {"type": "audio_response", "request_id": ..., "status": "success",
              "text_response": ..., "audio_data": <base64 WAV ответа LLM>}
  Ошибка  -> {"type": "error", "request_id": ..., "error_code": ..., "message": ...}

Запуск (из папки server/, модель должна быть скачана заранее через download_model.py):
    python download_model.py --model small
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
import tempfile
import time
from pathlib import Path

import pyttsx3
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# --------------------
# CONFIG
# --------------------

BASE_DIR = Path(__file__).resolve().parent

# Размер модели: tiny / base / small / medium / large-v3
# Для MVP на CPU "small" — хороший баланс качество/скорость.
MODEL_SIZE = os.environ.get("ASYA_MODEL", "small")
DEVICE = os.environ.get("ASYA_DEVICE", "cpu")          # "cuda" если есть GPU
COMPUTE_TYPE = os.environ.get("ASYA_COMPUTE", "int8")  # на CPU int8 быстрее всего
LANGUAGE = os.environ.get("ASYA_LANG", "ru")

MODEL_DIR = Path(os.environ.get(
    "ASYA_MODEL_DIR",
    str(BASE_DIR / "models" / f"whisper-{MODEL_SIZE}"),
))

OLLAMA_URL = os.environ.get("ASYA_OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("ASYA_OLLAMA_MODEL", "qwen3.5:0.8b")
LLM_SYSTEM_PROMPT = os.environ.get(
    "ASYA_LLM_SYSTEM_PROMPT",
    "Ты голосовой ассистент Ася. Отвечай кратко и по-русски, в одном-двух предложениях.",
)

TTS_VOICE_HINT = os.environ.get("ASYA_TTS_VOICE", "ru")  # подстрока в id/языке голоса SAPI5

# --------------------
# MODEL
# --------------------

if not (MODEL_DIR / "model.bin").exists():
    raise RuntimeError(
        f"Модель не найдена в {MODEL_DIR}. "
        f"Сначала запустите: python download_model.py --model {MODEL_SIZE}"
    )

print(f"Загружаю модель faster-whisper из {MODEL_DIR} ({DEVICE}, {COMPUTE_TYPE})...")
model = WhisperModel(str(MODEL_DIR), device=DEVICE, compute_type=COMPUTE_TYPE, local_files_only=True)
print("Модель готова.")

# --------------------
# TTS (pyttsx3 / SAPI5)
# --------------------

# pyttsx3 на Windows (SAPI5) виснет на втором runAndWait() при переиспользовании
# одного engine в процессе — поэтому здесь только резолвим id голоса один раз,
# а сам engine создаём заново на каждый вызов synthesize_speech().
TTS_VOICE_ID = None

_probe_engine = pyttsx3.init()
for voice in _probe_engine.getProperty("voices"):
    if TTS_VOICE_HINT.lower() in voice.id.lower() or any(
        TTS_VOICE_HINT.lower() in lang.lower() for lang in (voice.languages or [])
    ):
        TTS_VOICE_ID = voice.id
        print(f"[TTS] Голос: {voice.name}")
        break
else:
    print(f"[TTS] Голос с подстрокой '{TTS_VOICE_HINT}' не найден, использую системный по умолчанию")
_probe_engine.stop()
del _probe_engine

app = FastAPI(title="Asya STT")


def transcribe(wav_bytes: bytes) -> dict:
    started = time.time()

    segments, info = model.transcribe(
        io.BytesIO(wav_bytes),
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_ms = int((time.time() - started) * 1000)

    print(f"[STT] ({elapsed_ms} ms) {text!r}")

    return {
        "text": text,
        "language": info.language,
        "duration": round(info.duration, 2),
        "processing_time_ms": elapsed_ms,
    }


def ask_llm(text: str) -> str:
    started = time.time()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "think": False,
        },
        timeout=60,
    )
    response.raise_for_status()

    answer = response.json()["message"]["content"].strip()
    elapsed_ms = int((time.time() - started) * 1000)

    print(f"[LLM] ({elapsed_ms} ms) {answer!r}")

    return answer


def synthesize_speech(text: str) -> bytes:
    started = time.time()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine = pyttsx3.init()
        if TTS_VOICE_ID:
            engine.setProperty("voice", TTS_VOICE_ID)

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        engine.stop()
        del engine

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    elapsed_ms = int((time.time() - started) * 1000)
    print(f"[TTS] ({elapsed_ms} ms) синтезировано {len(wav_bytes)} байт")

    return wav_bytes


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Колонка подключена")

    try:
        while True:
            msg = await websocket.receive_json()
            request_id = msg.get("request_id")

            if msg.get("type") != "audio_request":
                continue

            try:
                wav_bytes = base64.b64decode(msg["audio_data"])
                result = transcribe(wav_bytes)
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error_code": "STT_ERROR",
                    "message": str(e),
                })
                continue

            try:
                llm_answer = ask_llm(result["text"])
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error_code": "LLM_ERROR",
                    "message": str(e),
                })
                continue

            try:
                answer_wav = synthesize_speech(llm_answer)
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "request_id": request_id,
                    "error_code": "TTS_ERROR",
                    "message": str(e),
                })
                continue

            await websocket.send_json({
                "type": "audio_response",
                "request_id": request_id,
                "status": "success",
                "text_response": llm_answer,
                "audio_data": base64.b64encode(answer_wav).decode("ascii"),
                "metadata": {
                    "transcribed_text": result["text"],
                    "language": result["language"],
                    "duration": result["duration"],
                    "processing_time_ms": result["processing_time_ms"],
                },
            })

    except WebSocketDisconnect:
        print("[WS] Колонка отключена")

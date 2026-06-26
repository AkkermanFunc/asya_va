"""
Asya — сервер транскрипции (MVP).

WebSocket-сервер: принимает аудио от колонки, прогоняет через
faster-whisper и возвращает распознанный текст.

Протокол (см. AGENTS.MD):
  Запрос  -> {"type": "audio_request",  "request_id": ..., "audio_data": <base64 WAV>, ...}
  Ответ   -> {"type": "audio_response", "request_id": ..., "status": "success", "text_response": ...}
  Ошибка  -> {"type": "error", "request_id": ..., "error_code": ..., "message": ...}

Запуск (из папки server/, модель должна быть скачана заранее через download_model.py):
    python download_model.py --model small
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
import time
from pathlib import Path

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

            await websocket.send_json({
                "type": "audio_response",
                "request_id": request_id,
                "status": "success",
                "text_response": result["text"],
                "metadata": {
                    "language": result["language"],
                    "duration": result["duration"],
                    "processing_time_ms": result["processing_time_ms"],
                },
            })

    except WebSocketDisconnect:
        print("[WS] Колонка отключена")

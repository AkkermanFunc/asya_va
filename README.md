# Asya Voice Assistant

Локальный голосовой ассистент. MVP: колонка слушает слово активации «ася», записывает фразу и отправляет на сервер для распознавания речи (faster-whisper). См. [AGENTS.MD](AGENTS.MD) для полной архитектуры.

## Структура

- `speaker/` — скрипт колонки (Vosk wake-word + запись + отправка по WebSocket)
- `server/` — сервер транскрипции (FastAPI + faster-whisper)

## Быстрый старт

### Сервер

```bash
cd server
pip install -r requirements.txt
python download_model.py --model small   # один раз, скачивает модель локально
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Колонка

```bash
cd speaker
pip install -r requirements.txt
python main.py
```

При первом запуске `speaker/main.py` ожидает модель Vosk в `speaker/models/vosk-model-small-ru-0.22/` — скачать можно с https://alphacephei.com/vosk/models (распаковать в эту папку).

Скажите «ася …» — колонка запишет фразу и выведет в консоль распознанный текст.

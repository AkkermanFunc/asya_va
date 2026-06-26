"""
Одноразовая загрузка модели faster-whisper на диск.

После запуска server.py грузит модель локально, без обращений к Hugging Face.

Запуск:
    python download_model.py --model small
"""

import argparse
from pathlib import Path

from faster_whisper.utils import download_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true", help="скачать заново, даже если уже есть")
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
        or Path(__file__).resolve().parent / "models" / f"whisper-{args.model}"
    )

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"Модель уже скачана: {output_dir}")
        return

    print(f"Скачиваю модель '{args.model}' в {output_dir}...")
    path = download_model(args.model, output_dir=str(output_dir))
    print(f"Готово: {path}")


if __name__ == "__main__":
    main()

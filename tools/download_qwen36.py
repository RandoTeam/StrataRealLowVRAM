#!/usr/bin/env python3
"""
Download Qwen3.6-35B-A3B GGUF model from Hugging Face:
Repo: HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
"""

import os
import sys
from huggingface_hub import hf_hub_download

REPO_ID = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
DEFAULT_FILENAME = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Strata-data", "models", "qwen3.6-35b"))

def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILENAME
    os.makedirs(TARGET_DIR, exist_ok=True)
    target_path = os.path.join(TARGET_DIR, filename)

    print("=" * 65)
    print(f"Загрузка модели: {filename}")
    print(f"Репозиторий HF : {REPO_ID}")
    print(f"Целевая папка  : {TARGET_DIR}")
    print("=" * 65)

    if os.path.exists(target_path):
        size_gb = os.path.getsize(target_path) / (1024 ** 3)
        print(f"[OK] Файл уже скачан: {target_path} ({size_gb:.2f} ГБ)")
        return

    print("Начинаю загрузку (поддерживается докачка)...")
    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False
    )
    print(f"\n[УСПЕХ] Модель успешно загружена в:\n{downloaded_file}")

if __name__ == "__main__":
    main()

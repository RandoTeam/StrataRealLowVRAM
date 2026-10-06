"""Resumable Downloader for Full BF16 Models from Hugging Face."""

import argparse
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model(repo_id: str, filename: str, target_dir: str) -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"  STRATA BF16 MODEL DOWNLOADER")
    print("=" * 68)
    print(f"  Repository:  {repo_id}")
    print(f"  Filename:    {filename}")
    print(f"  Target Dir:  {target_path}")
    print("=" * 68)

    t0 = time.time()
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    elapsed = time.time() - t0
    size_gib = Path(file_path).stat().st_size / (1024**3)

    print("=" * 68)
    print(f"  DOWNLOAD COMPLETED: {Path(file_path).name}")
    print(f"  Size: {size_gib:.2f} GiB")
    print(f"  Elapsed: {elapsed / 60:.2f} min (average {size_gib * 1024 / max(1, elapsed):.2f} MB/s)")
    print("=" * 68)
    return Path(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BF16 models from Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID")
    parser.add_argument("--filename", required=True, help="Filename to download")
    parser.add_argument("--target-dir", default="models/temp_bf16", help="Target local directory")

    args = parser.parse_args()
    download_model(args.repo_id, args.filename, args.target_dir)

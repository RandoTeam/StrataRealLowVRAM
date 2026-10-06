"""Resumable and Ultra-Fast Downloader for Full BF16 Models from Hugging Face.

Uses native Windows curl.exe with HTTP Range requests (-C -) and auto-retry
to guarantee 100% resilient downloads without Python file locking bugs.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_url


def download_model(repo_id: str, filename: str, target_dir: str) -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    target_file = target_path / filename

    url = hf_hub_url(repo_id=repo_id, filename=filename)

    print("=" * 68)
    print("  STRATA BF16 RESUMABLE MODEL DOWNLOADER (NATIVE CURL)")
    print("=" * 68)
    print(f"  Repository:   {repo_id}")
    print(f"  Filename:     {filename}")
    print(f"  Download URL: {url}")
    print(f"  Target File:  {target_file}")
    if target_file.exists():
        existing_gib = target_file.stat().st_size / (1024**3)
        print(f"  Resuming:     Existing {existing_gib:.2f} GiB on disk")
    print("=" * 68, flush=True)

    t0 = time.time()
    curl_cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--retry", "20",
        "--retry-delay", "3",
        "--connect-timeout", "30",
        "-o", str(target_file),
        url,
    ]

    ret = subprocess.run(curl_cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"curl.exe failed with exit code {ret.returncode}")

    elapsed = time.time() - t0
    final_size_gib = target_file.stat().st_size / (1024**3)

    print("\n" + "=" * 68)
    print(f"  DOWNLOAD COMPLETED: {target_file.name}")
    print(f"  Final Size: {final_size_gib:.2f} GiB ({target_file.stat().st_size:,} bytes)")
    print(f"  Elapsed:    {elapsed / 60:.2f} min")
    print("=" * 68, flush=True)
    return target_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BF16 models from Hugging Face with native curl resume")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID")
    parser.add_argument("--filename", required=True, help="Filename to download")
    parser.add_argument("--target-dir", default="models/temp_bf16", help="Target local directory")

    args = parser.parse_args()
    download_model(args.repo_id, args.filename, args.target_dir)

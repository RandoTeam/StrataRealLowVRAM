"""High-Performance Multi-Stream Downloader for Full BF16 Models.

Uses native Rust hf_transfer with 32 parallel streams, exponential backoff,
authenticated token passing, and live progress reporting to saturate the network link.
Bypasses Windows NTFS file-locking bugs by writing directly to target destination.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import requests

try:
    import hf_transfer
except ImportError:
    hf_transfer = None


def get_hf_token() -> str:
    """Retrieve Hugging Face token from environment or cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return ""


def get_file_metadata(repo_id: str, filename: str, token: str) -> tuple[str, int]:
    """Resolve direct Hugging Face URL and total file size."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.head(url, headers=headers, allow_redirects=True, timeout=15)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))
    return url, total_size


def download_bf16(
    repo_id: str,
    filename: str,
    target_dir: str,
    max_files: int = 32,
    chunk_size_mb: int = 16,
) -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    out_file = target_path / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)

    token = get_hf_token()
    url, total_size = get_file_metadata(repo_id, filename, token)
    total_gib = total_size / (1024**3)

    print("=" * 72, flush=True)
    print("  STRATA ACCELERATED BF16 DOWNLOADER (MULTI-STREAM RUST ENGINE)", flush=True)
    print("=" * 72, flush=True)
    print(f"  Repository:      {repo_id}", flush=True)
    print(f"  Filename:        {filename}", flush=True)
    print(f"  Target File:     {out_file}", flush=True)
    print(f"  Total Size:      {total_gib:.2f} GiB ({total_size:,} bytes)", flush=True)
    print(f"  Parallel Streams: {max_files}", flush=True)
    print(f"  Chunk Size:      {chunk_size_mb} MB", flush=True)
    print(f"  Auth Token:      {'CONFIGURED' if token else 'ANONYMOUS'}", flush=True)
    print("=" * 72, flush=True)

    if out_file.exists() and out_file.stat().st_size == total_size:
        print(f"[strata-download] File already exists and matches total size ({total_gib:.2f} GiB). Skipping download.", flush=True)
        return out_file

    t0 = time.time()
    last_print = t0
    downloaded = 0
    chunk_bytes = chunk_size_mb * 1024 * 1024

    def progress_callback(chunk_len: int):
        nonlocal downloaded, last_print
        downloaded += chunk_len
        now = time.time()
        if now - last_print >= 2.0 or downloaded >= total_size:
            elapsed = max(0.001, now - t0)
            cur_gib = downloaded / (1024**3)
            pct = (downloaded / max(1, total_size)) * 100.0
            rate_mbs = (downloaded / elapsed) / (1024 * 1024)
            remaining_bytes = max(0, total_size - downloaded)
            eta_sec = remaining_bytes / max(1, rate_mbs * 1024 * 1024)
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            eta_s = int(eta_sec % 60)
            print(
                f"[download] {cur_gib:6.2f} / {total_gib:6.2f} GiB ({pct:5.1f}%) | "
                f"Rate: {rate_mbs:5.2f} MB/s | ETA: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s",
                flush=True,
            )
            last_print = now

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if hf_transfer is not None:
        print("[strata-download] Launching Rust-accelerated hf_transfer pipeline...", flush=True)
        hf_transfer.download(
            url=url,
            filename=str(out_file),
            max_files=max_files,
            chunk_size=chunk_bytes,
            parallel_failures=min(16, max_files),
            max_retries=10,
            headers=headers,
            callback=progress_callback,
        )
    else:
        print("[strata-download] WARNING: hf_transfer not found. Falling back to single-stream requests...", flush=True)
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_bytes):
                    if chunk:
                        f.write(chunk)
                        progress_callback(len(chunk))

    total_elapsed = time.time() - t0
    final_size_gib = out_file.stat().st_size / (1024**3)
    avg_speed_mbs = (out_file.stat().st_size / max(1.0, total_elapsed)) / (1024 * 1024)

    print("=" * 72, flush=True)
    print(f"  DOWNLOAD SUCCESSFUL: {out_file.name}", flush=True)
    print(f"  Final Size:   {final_size_gib:.2f} GiB", flush=True)
    print(f"  Total Time:   {total_elapsed / 60:.2f} min", flush=True)
    print(f"  Average Rate: {avg_speed_mbs:.2f} MB/s", flush=True)
    print("=" * 72, flush=True)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stream accelerated BF16 downloader")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID")
    parser.add_argument("--filename", required=True, help="Target filename")
    parser.add_argument("--target-dir", default="models/temp_bf16", help="Target output folder")
    parser.add_argument("--max-files", type=int, default=32, help="Number of parallel streams")
    parser.add_argument("--chunk-size", type=int, default=16, help="Chunk size in MB")
    args = parser.parse_args()

    download_bf16(
        repo_id=args.repo_id,
        filename=args.filename,
        target_dir=args.target_dir,
        max_files=args.max_files,
        chunk_size_mb=args.chunk_size,
    )

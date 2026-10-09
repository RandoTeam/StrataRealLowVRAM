import os
import sys
import shutil
import zipfile

def package_release():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    release_name = "StrataRealLowVRAM-v0.1.41-win-x64-q2_0"
    dist_dir = os.path.join(root, "dist")
    stage_dir = os.path.join(dist_dir, release_name)
    zip_path = os.path.join(dist_dir, f"{release_name}.zip")

    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)

    print(f"Staging release in: {stage_dir}")

    # Root files
    root_files = [
        "START-HERE.bat",
        "run-q2_0.bat",
        "strata-q2_0.json",
        "setup.py",
        "README.md",
        "LICENSE",
        ".gitignore",
    ]
    for rf in root_files:
        src = os.path.join(root, rf)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stage_dir, rf))
            print(f"  + Copied {rf}")
        else:
            print(f"  ! Warning: {rf} not found")

    # Engine directory (binaries + hook)
    engine_src = os.path.join(root, "engine")
    engine_dst = os.path.join(stage_dir, "engine")
    os.makedirs(engine_dst, exist_ok=True)
    for f in os.listdir(engine_src):
        src = os.path.join(engine_src, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(engine_dst, f))
            print(f"  + engine/{f} ({os.path.getsize(src) / (1024*1024):.1f} MB)")

    # Serve directory (Python server + Web UI)
    serve_src = os.path.join(root, "serve")
    serve_dst = os.path.join(stage_dir, "serve")
    shutil.copytree(serve_src, serve_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp"))
    print("  + Copied serve/ directory")

    # Data directory (static metadata tables, exclude learned cache)
    data_src = os.path.join(root, "data")
    data_dst = os.path.join(stage_dir, "data")
    os.makedirs(data_dst, exist_ok=True)
    for f in os.listdir(data_src):
        src = os.path.join(data_src, f)
        if os.path.isfile(src) and not f.endswith("learned.bin"):
            shutil.copy2(src, os.path.join(data_dst, f))
            print(f"  + data/{f}")

    # Tools directory
    tools_src = os.path.join(root, "tools")
    tools_dst = os.path.join(stage_dir, "tools")
    shutil.copytree(tools_src, tools_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.obj", "*.exp", "*.lib", "*.tmp"))
    print("  + Copied tools/ directory")

    print(f"\nCompressing to: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root_dir, dirs, files in os.walk(stage_dir):
            for file in files:
                abs_path = os.path.join(root_dir, file)
                rel_path = os.path.relpath(abs_path, dist_dir)
                zf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Release package ready: {zip_path} ({zip_size_mb:.2f} MB)")

if __name__ == "__main__":
    package_release()

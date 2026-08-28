#!/usr/bin/env python3
"""
00_fetch_models.py — bootstrap the on-disk model artifacts.

Run this ONCE on a fresh machine, before `docker compose build`:

    python scripts/00_fetch_models.py            # download + verify
    python scripts/00_fetch_models.py --check    # verify only (exit 1 if missing)
    python scripts/00_fetch_models.py --with-insightface   # also pre-warm buffalo_l

Stdlib only — deliberately runnable before `pip install -r requirements.txt`,
and before any container exists.

Model inventory
───────────────
  face_landmarker.task    3.6 MB   downloaded from Google's MediaPipe CDN
  open-closed-eye.onnx     46 KB   vendored in git (see NOTE below)
  InsightFace buffalo_l   ~600 MB  auto-downloaded on first use into
                                   ~/.insightface (the cm_insightface_cache
                                   volume in docker compose)

NOTE on open-closed-eye-0001: the Open Model Zoo mirror this file used to come
from (download.01.org) now 404s, and storage.openvinotoolkit.org does not serve
it at any of the documented paths. Because the file is only 46 KB it is checked
into git instead of downloaded, so a fresh clone always has it. This script
verifies it rather than fetching it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_PATH") or (REPO_ROOT / "models"))

# ── Downloadable artifacts ───────────────────────────────────────────────────
DOWNLOADS = [
    {
        "name": "MediaPipe FaceLandmarker",
        "path": MODELS_DIR / "face_landmarker.task",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "size": 3_758_596,
    },
]

# ── Artifacts that must already be on disk (vendored in git) ─────────────────
VENDORED = [
    {
        "name": "OMZ open-closed-eye-0001",
        "path": MODELS_DIR / "public" / "open-closed-eye-0001" / "open-closed-eye.onnx",
        "sha256": "4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593",
        "size": 46_164,
        "hint": (
            "This file ships in the repository. If it is missing you likely have "
            "a partial clone or an over-eager .gitignore — restore it with:\n"
            "    git checkout -- models/public/open-closed-eye-0001/open-closed-eye.onnx"
        ),
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(spec: dict) -> tuple[bool, str]:
    path: Path = spec["path"]
    if not path.exists():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != spec["size"]:
        return False, f"wrong size ({actual_size} != {spec['size']})"
    actual = sha256_of(path)
    if actual != spec["sha256"]:
        return False, f"checksum mismatch ({actual[:16]}… != {spec['sha256'][:16]}…)"
    return True, "ok"


def download(spec: dict) -> None:
    path: Path = spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    ↓ {spec['url']}")

    # Download to a temp file in the destination dir, then atomically rename,
    # so an interrupted run never leaves a half-written model behind.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "classroom-monitor/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r      {pct:3d}%  {done:,} / {total:,} bytes", end="", flush=True)
            print()
        shutil.move(str(tmp), str(path))
    finally:
        tmp.unlink(missing_ok=True)


def prewarm_insightface(model_name: str) -> bool:
    """Force InsightFace to pull buffalo_l into ~/.insightface before first run."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("    ! insightface is not installed in this interpreter — skipping.")
        print("      Inside the container it downloads automatically on first use.")
        return True
    print(f"    ↓ InsightFace '{model_name}' (~600 MB on first run)")
    FaceAnalysis(name=model_name)
    print("      cached under ~/.insightface")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and verify model artifacts.")
    ap.add_argument("--check", action="store_true", help="verify only; do not download")
    ap.add_argument("--with-insightface", action="store_true",
                    help="also pre-download the InsightFace buffalo_l pack")
    args = ap.parse_args()

    print(f"Models directory: {MODELS_DIR}")
    failures: list[str] = []

    print("\nDownloadable artifacts")
    for spec in DOWNLOADS:
        ok, why = verify(spec)
        if ok:
            print(f"  ✓ {spec['name']}  ({why})")
            continue
        if args.check:
            print(f"  ✗ {spec['name']}  — {why}")
            failures.append(spec["name"])
            continue
        print(f"  … {spec['name']}  — {why}, fetching")
        try:
            download(spec)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"  ✗ {spec['name']}  — download failed: {exc}")
            failures.append(spec["name"])
            continue
        ok, why = verify(spec)
        print(f"  {'✓' if ok else '✗'} {spec['name']}  ({why})")
        if not ok:
            failures.append(spec["name"])

    print("\nVendored artifacts (shipped in git)")
    for spec in VENDORED:
        ok, why = verify(spec)
        print(f"  {'✓' if ok else '✗'} {spec['name']}  ({why})")
        if not ok:
            failures.append(spec["name"])
            print("      " + spec["hint"].replace("\n", "\n      "))

    if args.with_insightface and not args.check:
        print("\nInsightFace recognition pack")
        prewarm_insightface(os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_l"))

    if failures:
        print(f"\nFAILED — {len(failures)} artifact(s) unusable: {', '.join(failures)}")
        print("The app will start but face-behaviour analysis will fail at first inference.")
        return 1

    print("\nAll model artifacts present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

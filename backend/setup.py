"""Download static assets once: swagger-ui files + mediapipe pose model."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
MODELS = ROOT / "models"

FILES = [
    # Swagger UI (pinned to match fastapi 0.115.x)
    (STATIC / "swagger-ui-bundle.js", "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"),
    (STATIC / "swagger-ui.css", "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css"),
    # MediaPipe pose landmarker model (GPU-compatible float16)
    (MODELS / "pose_landmarker.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
]


def download_all() -> None:
    for dest, url in FILES:
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            print(f"  [skip] {dest.name}")
            continue
        print(f"  [downloading] {dest.name} ...")
        urlretrieve(url, dest)
        print(f"  [done] {dest.name}")
    print("  All assets ready.")


if __name__ == "__main__":
    download_all()

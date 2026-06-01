from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from PIL import Image


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
        suffix = ".png"

    path = target_dir / f"{uuid4().hex}{suffix}"

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path


def read_bgr(path: Path):
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Image file does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty: {path}")

    # Method 1: Windows-safe OpenCV decode
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is not None:
        return img

    # Method 2: PIL fallback
    try:
        pil_img = Image.open(path).convert("RGB")
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise ValueError(f"Could not read image: {path}. PIL error: {e}")


def bgr_to_rgb(img):
    if img is None:
        raise ValueError("Cannot convert empty image from BGR to RGB")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path: Path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img is None:
        raise ValueError(f"Cannot write empty image: {path}")

    ext = path.suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
        ext = ".png"
        path = path.with_suffix(ext)

    ok, buf = cv2.imencode(ext, img)

    if not ok:
        raise ValueError(f"Could not encode image: {path}")

    buf.tofile(str(path))
    return path
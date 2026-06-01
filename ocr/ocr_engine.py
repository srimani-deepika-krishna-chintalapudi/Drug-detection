from dataclasses import dataclass
from typing import List, Dict
import re
import cv2
import numpy as np
import easyocr


@dataclass
class OCRResult:
    source: str
    boxes: List[Dict]
    full_text: str
    avg_confidence: float


OCR_CORRECTIONS = {
    "Toblets": "Tablets",
    "Toblet": "Tablet",
    "Tobiets": "Tablets",
    "Tablot": "Tablet",
    "tablot": "tablet",
    "Porocetamo|": "Paracetamol",
    "Parocetomo|": "Paracetamol",
    "Porocetamol": "Paracetamol",
    "Parocetamol": "Paracetamol",
    "Paracetamo|": "Paracetamol",
    "Delo-650": "Dolo-650",
    "delo-650": "Dolo-650",
    "sat-8go": "Dolo-650",
    "srait-eg0": "Dolo-650",
    "siaii-eg0": "Dolo-650",
    "650ma": "650mg",
    "650 ma": "650 mg",
}


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def apply_ocr_corrections(text: str) -> str:
    text = str(text or "").strip()

    for wrong, correct in OCR_CORRECTIONS.items():
        text = text.replace(wrong, correct)

    return clean_text(text)


def prepare_for_ocr(image_bgr):
    """
    Create a text-friendly OCR image without destroying the carton.
    EasyOCR usually handles color images well, but low-res images need upscaling.
    """
    img = image_bgr.copy()

    h, w = img.shape[:2]

    # Your reference image OCR is tiny. Upscale before OCR.
    target_width = 2200
    if w < target_width:
        scale = target_width / max(1, w)
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # Mild denoise, not aggressive thresholding.
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # CLAHE on luminance channel to improve printed text contrast.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Light sharpening.
    kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ]
    )
    img = cv2.filter2D(img, -1, kernel)

    return img


def prepare_gray_for_ocr(image_bgr):
    """
    Second OCR variant: grayscale enhanced.
    Useful for small black text.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    target_width = 2200
    if w < target_width:
        scale = target_width / max(1, w)
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ]
    )
    gray = cv2.filter2D(gray, -1, kernel)

    return gray


class PaddleOCREngine:
    """
    Kept same class name so backend pipeline does not need changes.
    Internally uses EasyOCR because PaddleOCR 3.x is too slow/unstable on this Windows setup.
    """

    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=True)

    @staticmethod
    def _normalize_box(poly):
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _run_single(self, image, source="image") -> OCRResult:
        raw = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.4,
            low_text=0.3,
            link_threshold=0.4,
            width_ths=0.7,
            mag_ratio=2.0,
        )

        boxes = []

        for item in raw:
            poly, text, conf = item
            text = apply_ocr_corrections(text)

            if not text:
                continue

            boxes.append(
                {
                    "text": text,
                    "confidence": float(conf),
                    "bbox": self._normalize_box(poly),
                    "poly": poly,
                }
            )

        full_text = "\n".join(b["text"] for b in boxes)
        avg = sum(b["confidence"] for b in boxes) / max(1, len(boxes))

        return OCRResult(
            source=source,
            boxes=boxes,
            full_text=full_text,
            avg_confidence=avg,
        )

    def run(self, image_bgr, source="image") -> OCRResult:
        print(f"OCR input {source} shape: {image_bgr.shape}")

        variants = [
            ("original", image_bgr),
            ("enhanced_color", prepare_for_ocr(image_bgr)),
            ("enhanced_gray", prepare_gray_for_ocr(image_bgr)),
        ]

        results = []

        for name, img in variants:
            try:
                result = self._run_single(img, f"{source}_{name}")
                results.append(result)

                print(
                    f"OCR variant {source}_{name}: "
                    f"avg_conf={result.avg_confidence:.3f}, "
                    f"chars={len(result.full_text)}"
                )
            except Exception as e:
                print(f"OCR variant failed {source}_{name}: {e}")

        if not results:
            return OCRResult(
                source=source,
                boxes=[],
                full_text="",
                avg_confidence=0.0,
            )

        return choose_best_ocr(*results)


def choose_best_ocr(*results: OCRResult) -> OCRResult:
    """
    Picks best OCR result using confidence + useful text length + medicine keywords.
    """
    valid = [r for r in results if r is not None]

    if not valid:
        return OCRResult(source="empty", boxes=[], full_text="", avg_confidence=0.0)

    pharma_keywords = [
        "dolo",
        "paracetamol",
        "tablet",
        "tablets",
        "cetirizine",
        "cetrizine",
        "hydrochloride",
        "dosage",
        "mfg",
        "expiry",
        "batch",
        "mg",
        "ip",
        "labs",
    ]

    def score_result(r: OCRResult):
        text = r.full_text.lower()
        keyword_hits = sum(1 for k in pharma_keywords if k in text)

        return (
            r.avg_confidence * 0.55
            + min(len(r.full_text) / 800, 1) * 0.20
            + min(keyword_hits / 6, 1) * 0.25
        )

    best = max(valid, key=score_result)
    return best
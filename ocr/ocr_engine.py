from dataclasses import dataclass
from typing import List, Dict, Tuple
import hashlib
import re

import cv2
import numpy as np
from paddleocr import PaddleOCR


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
    "Porocetamo|": "Paracetamol",
    "Parocetomo|": "Paracetamol",
    "Porocetamol": "Paracetamol",
    "Parocetamol": "Paracetamol",
    "Paracetamo|": "Paracetamol",
    "Delo-650": "Dolo-650",
    "delo-650": "Dolo-650",
    "650ma": "650mg",
    "650 ma": "650 mg",
    "B N0": "B.NO",
    "B.N0": "B.NO",
    "BNO": "B.NO",
    "B NQ": "B.NO",
    "Batch N0": "Batch No",
    "Botch": "Batch",
    "8atch": "Batch",
    "Mfg": "MFG",
    "MFG.": "MFG",
}


PHARMA_KEYWORDS = [
    "dolo", "pan", "paracetamol", "pantoprazole", "domperidone",
    "tablet", "tablets", "capsule", "capsules", "mg", "ip",
    "batch", "b.no", "mfg", "mfd", "exp", "expiry", "mrp",
    "physician", "children", "labs",
]


CRITICAL_KEYWORDS = [
    "mrp", "batch", "b.no", "exp", "expiry", "mfg", "mfd",
    "manufactured", "lic", "license", "paracetamol", "pantoprazole",
    "domperidone", "dolo", "pan", "barcode",
]


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("₹", "rs")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s.strip()


def apply_ocr_corrections(text: str) -> str:
    text = str(text or "").strip()
    for wrong, correct in OCR_CORRECTIONS.items():
        text = text.replace(wrong, correct)
    return clean_text(text)


def image_hash(image_bgr) -> str:
    if image_bgr is None:
        return "none"
    return hashlib.md5(image_bgr.tobytes()).hexdigest()


def resize_for_ocr(image_bgr, target_width=1500, max_width=2300):
    img = image_bgr.copy()
    h, w = img.shape[:2]

    if w < target_width:
        scale = target_width / max(1, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif w > max_width:
        scale = max_width / max(1, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img


def prepare_for_ocr(image_bgr):
    img = resize_for_ocr(image_bgr)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)

    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)

    return img


def bbox_from_poly(poly):
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def scale_box_back(bbox, original_shape, processed_shape):
    oh, ow = original_shape[:2]
    ph, pw = processed_shape[:2]

    sx = ow / max(1, pw)
    sy = oh / max(1, ph)

    x1, y1, x2, y2 = bbox
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]


def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_size(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1), max(1, y2 - y1)


def detect_orientation(bbox):
    bw, bh = box_size(bbox)
    if bh / bw > 2.0:
        return "vertical"
    if bw / bh > 1.6:
        return "horizontal"
    return "square"


def assign_zone(bbox, image_shape):
    h, w = image_shape[:2]
    cx, cy = box_center(bbox)
    nx = cx / max(1, w)
    ny = cy / max(1, h)

    if nx < 0.25:
        col = "left"
    elif nx > 0.75:
        col = "right"
    else:
        col = "center"

    if ny < 0.25:
        row = "top"
    elif ny > 0.75:
        row = "bottom"
    else:
        row = "middle"

    return f"{row}_{col}"


def crop_with_pad(img, bbox, pad=12):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return img[y1:y2, x1:x2]


def rotate_for_vertical_ocr(crop):
    return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)


def split_columns(boxes, min_gap=200):
    if not boxes:
        return []

    centers = sorted([(b["bbox"][0] + b["bbox"][2]) / 2 for b in boxes])

    if len(centers) < 2:
        return [boxes]

    gaps = []
    for i in range(len(centers) - 1):
        gaps.append((centers[i + 1] - centers[i], centers[i], centers[i + 1]))

    biggest_gap, left_x, right_x = max(gaps, key=lambda x: x[0])

    if biggest_gap < min_gap:
        return [boxes]

    split_x = (left_x + right_x) / 2
    left_col, right_col = [], []

    for b in boxes:
        cx = (b["bbox"][0] + b["bbox"][2]) / 2
        if cx < split_x:
            left_col.append(b)
        else:
            right_col.append(b)

    return [c for c in [left_col, right_col] if c]


def make_full_text(boxes, image_shape=None):
    if not boxes:
        return ""

    horizontal = [b for b in boxes if b.get("orientation") != "vertical"]
    vertical = [b for b in boxes if b.get("orientation") == "vertical"]

    output = []

    columns = split_columns(horizontal)
    for idx, col in enumerate(columns, 1):
        col = sorted(col, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        output.append(f"=== COLUMN {idx} ===")
        for b in col:
            output.append(b["text"])
        output.append("")

    if vertical:
        output.append("=== VERTICAL TEXT ===")
        for b in sorted(vertical, key=lambda x: (x["bbox"][0], x["bbox"][1])):
            output.append(b["text"])

    return "\n".join(output).strip()


class PaddleOCREngine:
    def __init__(self, run_enhanced_variant=True, enable_vertical_retry=True):
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            det_model_dir=r"C:\Users\Admin\.paddleocr\whl\det\en\en_PP-OCRv3_det_infer",
            rec_model_dir=r"C:\Users\Admin\.paddleocr\whl\rec\en\en_PP-OCRv4_rec_infer",
            cls_model_dir=r"C:\Users\Admin\.paddleocr\whl\cls\ch_ppocr_mobile_v2.0_cls_infer",
            show_log=False,
        )

        self.run_enhanced_variant = run_enhanced_variant
        self.enable_vertical_retry = enable_vertical_retry
        self._single_cache = {}
        self._final_cache = {}

    def _call_ocr(self, image, cls=True):
        if hasattr(self.ocr, "ocr"):
            return self.ocr.ocr(image, cls=cls)
        return self.ocr.predict(image)

    def _parse_ocr_output(self, raw):
        """
        Supports PaddleOCR 2.x output:
        [
            [
                [poly, (text, conf)],
                ...
            ]
        ]

        Also supports PaddleOCR 3.x dict-style output:
        [
            {
                "rec_texts": [...],
                "rec_scores": [...],
                "rec_polys": [...]
            }
        ]
        """
        items = []

        if not raw:
            return items

        for page in raw:
            if isinstance(page, dict):
                texts = page.get("rec_texts", [])
                scores = page.get("rec_scores", [])
                polys = page.get("rec_polys", [])

                for text, conf, poly in zip(texts, scores, polys):
                    poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                    items.append((str(text), float(conf), poly_list))

            elif isinstance(page, list):
                for line in page:
                    try:
                        poly = line[0]
                        text = line[1][0]
                        conf = line[1][1]
                        poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                        items.append((str(text), float(conf), poly_list))
                    except Exception:
                        continue

        return items

    def _vertical_retry(self, processed_image, raw_bbox, current_text, current_conf):
        if not self.enable_vertical_retry:
            return current_text, current_conf

        try:
            crop = crop_with_pad(processed_image, raw_bbox, pad=10)
            if crop.size == 0:
                return current_text, current_conf

            rotated = rotate_for_vertical_ocr(crop)
            v_raw = self._call_ocr(rotated, cls=True)
            v_items = self._parse_ocr_output(v_raw)

            best_text = current_text
            best_conf = float(current_conf)

            for vt, vc, _ in v_items:
                vt = apply_ocr_corrections(vt)
                vc = float(vc)
                if vt and vc > best_conf:
                    best_text = vt
                    best_conf = vc

            return best_text, best_conf

        except Exception as e:
            print("Vertical OCR retry failed:", e)
            return current_text, current_conf

    def _run_single(self, image, source, original_shape) -> OCRResult:
        cache_key = f"{source}:{image_hash(image)}:{original_shape}"

        if cache_key in self._single_cache:
            return self._single_cache[cache_key]

        raw = self._call_ocr(image, cls=True)
        parsed_items = self._parse_ocr_output(raw)

        boxes = []

        for text, conf, poly_list in parsed_items:
            text = apply_ocr_corrections(text)
            if not text:
                continue

            raw_bbox = bbox_from_poly(poly_list)
            bbox = scale_box_back(raw_bbox, original_shape, image.shape)

            orientation = detect_orientation(bbox)
            is_vertical = orientation == "vertical"

            if is_vertical:
                text, conf = self._vertical_retry(image, raw_bbox, text, float(conf))

            bw, bh = box_size(bbox)
            cx, cy = box_center(bbox)
            oh, ow = original_shape[:2]
            norm_text = normalize_text(text)

            boxes.append({
                "text": text,
                "normalized_text": norm_text,
                "confidence": float(conf),
                "bbox": bbox,
                "poly": poly_list,
                "is_vertical": is_vertical,
                "orientation": orientation,
                "zone": assign_zone(bbox, original_shape),
                "center": [cx, cy],
                "norm_center": [cx / max(1, ow), cy / max(1, oh)],
                "size": [bw, bh],
                "norm_size": [bw / max(1, ow), bh / max(1, oh)],
                "is_critical": any(k in norm_text for k in CRITICAL_KEYWORDS),
            })

        full_text = make_full_text(boxes, original_shape)
        avg = sum(b.get("confidence", 0.0) for b in boxes) / max(1, len(boxes))

        result = OCRResult(
            source=source,
            boxes=boxes,
            full_text=full_text,
            avg_confidence=avg,
        )

        self._single_cache[cache_key] = result
        return result

    def run(self, image_bgr, source="image") -> OCRResult:
        if image_bgr is None:
            return OCRResult(source=source, boxes=[], full_text="", avg_confidence=0.0)

        original_shape = image_bgr.shape
        final_key = f"{source}:{image_hash(image_bgr)}:{original_shape}:enh={self.run_enhanced_variant}"

        if final_key in self._final_cache:
            return self._final_cache[final_key]

        variants: List[Tuple[str, np.ndarray]] = [
            ("original_resized", resize_for_ocr(image_bgr))
        ]

        if self.run_enhanced_variant:
            enhanced = prepare_for_ocr(image_bgr)
            if image_hash(enhanced) != image_hash(variants[0][1]):
                variants.append(("enhanced_color", enhanced))

        results = []
        seen_variant_hashes = set()

        for name, img in variants:
            hsh = image_hash(img)
            if hsh in seen_variant_hashes:
                continue
            seen_variant_hashes.add(hsh)

            try:
                result = self._run_single(img, f"{source}_{name}", original_shape)
                results.append(result)

                print(
                    f"OCR Paddle {source}_{name}: "
                    f"avg_conf={result.avg_confidence:.3f}, "
                    f"boxes={len(result.boxes)}, "
                    f"chars={len(result.full_text)}"
                )

            except Exception as e:
                print(f"OCR Paddle failed {source}_{name}: {e}")

        if not results:
            final = OCRResult(source=source, boxes=[], full_text="", avg_confidence=0.0)
        else:
            final = choose_best_ocr(*results)

        if len(final.boxes) == 0:
            print(f"WARNING: OCR extracted 0 boxes for {source}. Comparison result is unreliable.")

        self._final_cache[final_key] = final
        return final


def choose_best_ocr(*results: OCRResult) -> OCRResult:
    valid = [r for r in results if r is not None]

    if not valid:
        return OCRResult(source="empty", boxes=[], full_text="", avg_confidence=0.0)

    def score_result(r):
        text = r.full_text.lower()
        keyword_hits = sum(1 for k in PHARMA_KEYWORDS if k in text)
        useful_boxes = sum(1 for b in r.boxes if len(str(b.get("text", "")).strip()) >= 2)
        critical_boxes = sum(1 for b in r.boxes if b.get("is_critical"))

        return (
            r.avg_confidence * 0.42
            + min(len(r.full_text) / 900, 1) * 0.22
            + min(keyword_hits / 8, 1) * 0.20
            + min(useful_boxes / 25, 1) * 0.10
            + min(critical_boxes / 5, 1) * 0.06
        )

    return max(valid, key=score_result)

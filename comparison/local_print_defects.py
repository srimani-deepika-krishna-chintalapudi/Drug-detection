import cv2
import numpy as np


def _crop(img, bbox, pad=8):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def _ink_density(crop):
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    return np.count_nonzero(th) / max(1, th.size)


def _blur_score(crop):
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _text_clean(t):
    return str(t or "").strip().lower()


def _same_text(a, b):
    a = _text_clean(a)
    b = _text_clean(b)

    if not a or not b:
        return False

    return a == b


def detect_local_print_defects(ref_boxes, sus_boxes, ref_img, sus_img):
    issues = []
    used = set()

    for rb in ref_boxes:
        rt = rb.get("text", "")
        rbbox = rb.get("bbox")

        if not rt or not rbbox:
            continue

        best_idx = None

        for i, sb in enumerate(sus_boxes):
            if i in used:
                continue

            st = sb.get("text", "")
            sbbox = sb.get("bbox")

            if not st or not sbbox:
                continue

            if _same_text(rt, st):
                best_idx = i
                break

        if best_idx is None:
            continue

        used.add(best_idx)
        sb = sus_boxes[best_idx]

        ref_crop = _crop(ref_img, rbbox)
        sus_crop = _crop(sus_img, sb["bbox"])

        if ref_crop is None or sus_crop is None:
            continue

        ref_density = _ink_density(ref_crop)
        sus_density = _ink_density(sus_crop)

        ref_blur = _blur_score(ref_crop)
        sus_blur = _blur_score(sus_crop)

        density_diff = abs(ref_density - sus_density)
        blur_ratio = min(ref_blur, sus_blur) / max(ref_blur, sus_blur, 1.0)

        # detects missing dots, broken letters, ink spread, weak print, damaged local characters
        if density_diff > 0.075 or blur_ratio < 0.45:
            direction = "heavier/darker" if sus_density > ref_density else "lighter/weaker"

            issues.append({
                "issue_type": "local_print_defect",
                "reference": rt,
                "uploaded": sb.get("text", ""),
                "difference": (
                    f"Local print defect detected. Uploaded print appears {direction}. "
                    f"Ink density change={density_diff:.3f}, sharpness ratio={blur_ratio:.2f}."
                ),
                "severity": "Medium",
                "confidence": 86,
                "ref_bbox": rbbox,
                "suspect_bbox": sb.get("bbox"),
            })

    return issues
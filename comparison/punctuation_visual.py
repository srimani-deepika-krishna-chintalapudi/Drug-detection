import cv2
import numpy as np
import re
from difflib import SequenceMatcher


def _clean_base(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _similar(a, b):
    a = _clean_base(a)
    b = _clean_base(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _position_ok(ref_bbox, sus_bbox):
    rcx, rcy = _bbox_center(ref_bbox)
    scx, scy = _bbox_center(sus_bbox)

    return abs(rcx - scx) <= 450 and abs(rcy - scy) <= 250


def _bbox_big_enough(bbox):
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    return w >= 20 and h >= 10 and (w * h) >= 160


def _has_final_dot_visual(image_bgr, bbox):
    if image_bgr is None or bbox is None:
        return False

    if not _bbox_big_enough(bbox):
        return False

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = image_bgr.shape[:2]

    pad = 6
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ch, cw = gray.shape[:2]

    if ch < 10 or cw < 20:
        return False

    # Final full stop usually appears near right-bottom.
    roi = gray[int(ch * 0.45):ch, int(cw * 0.70):cw]

    if roi.size == 0:
        return False

    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        9,
    )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if 2 <= ww <= 12 and 2 <= hh <= 12 and 3 <= area <= 80:
            ratio = ww / max(1, hh)

            if 0.55 <= ratio <= 1.8:
                return True

    return False


def _best_match(ref_box, sus_boxes):
    ref_text = str(ref_box.get("text", "")).strip()
    ref_bbox = ref_box.get("bbox")

    best = None
    best_score = 0.0

    for sb in sus_boxes:
        sus_text = str(sb.get("text", "")).strip()
        sus_bbox = sb.get("bbox")

        if not sus_text or not sus_bbox:
            continue

        if not _bbox_big_enough(sus_bbox):
            continue

        text_score = _similar(ref_text, sus_text)

        if text_score < 0.90:
            continue

        if not _position_ok(ref_bbox, sus_bbox):
            continue

        rcx, rcy = _bbox_center(ref_bbox)
        scx, scy = _bbox_center(sus_bbox)

        pos_dist = abs(rcx - scx) + abs(rcy - scy)
        pos_score = max(0.0, 1.0 - pos_dist / 700.0)

        final_score = text_score * 0.80 + pos_score * 0.20

        if final_score > best_score:
            best_score = final_score
            best = sb

    return best


def detect_visual_punctuation(ref_boxes, sus_boxes, ref_img, sus_img):
    issues = []

    for rb in ref_boxes:
        ref_text = str(rb.get("text", "")).strip()
        ref_bbox = rb.get("bbox")

        if not ref_text or not ref_bbox:
            continue

        if not _bbox_big_enough(ref_bbox):
            continue

        sb = _best_match(rb, sus_boxes)

        if sb is None:
            continue

        sus_text = str(sb.get("text", "")).strip()
        sus_bbox = sb.get("bbox")

        ref_dot = ref_text.endswith(".") or _has_final_dot_visual(ref_img, ref_bbox)
        sus_dot = sus_text.endswith(".") or _has_final_dot_visual(sus_img, sus_bbox)

        if ref_dot != sus_dot:
            issues.append({
                "issue_type": "punctuation",
                "reference": ref_text + ("." if ref_dot and not ref_text.endswith(".") else ""),
                "uploaded": sus_text + ("." if sus_dot and not sus_text.endswith(".") else ""),
                "difference": "Final full stop punctuation differs visually.",
                "severity": "Medium",
                "confidence": 88,
                "ref_bbox": ref_bbox,
                "suspect_bbox": sus_bbox,
            })

    return issues
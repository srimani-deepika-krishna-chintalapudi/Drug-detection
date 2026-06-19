import re
import cv2
import numpy as np
from rapidfuzz import fuzz


def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _bbox(box):
    b = box.get("bbox")
    if not b or len(b) != 4:
        return None
    return [int(v) for v in b]


def _crop(img, bbox, pad=4):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def _is_address(text):
    t = str(text or "").lower()
    address_words = [
        "baddi", "solan", "h.p", "hp", "dist", "distt", "district",
        "village", "road", "plot", "industrial", "area", "pradesh",
        "india", "khurd", "bhatauli", "bhatouli", "mumbai", "sikkim"
    ]
    return any(w in t for w in address_words)


def _is_critical(text, medicine_name=None):
    t = str(text or "").lower()
    c = _clean(t)

    critical = [
        "mfg", "exp", "mrp", "bno", "batch", "lic", "license",
        "mg", "ml", "tablet", "capsule", "contains", "schedule",
        "dosage", "physician"
    ]

    if medicine_name and _clean(medicine_name) in c:
        return True

    return any(k in c for k in critical)


def _match_boxes(ref_boxes, sus_boxes):
    matches = []
    used = set()

    for rb in ref_boxes:
        rt = str(rb.get("text", "")).strip()
        rbbox = _bbox(rb)

        if not rt or not rbbox:
            continue

        rc = _clean(rt)
        if len(rc) < 2:
            continue

        best = None
        best_score = -1

        for si, sb in enumerate(sus_boxes):
            if si in used:
                continue

            st = str(sb.get("text", "")).strip()
            sbbox = _bbox(sb)

            if not st or not sbbox:
                continue

            sc = _clean(st)
            if len(sc) < 2:
                continue

            length_ratio = min(len(rc), len(sc)) / max(len(rc), len(sc))
            if length_ratio < 0.75:
                continue

            score = fuzz.ratio(rc, sc)

            if rc == sc:
                score = 100

            if score > best_score:
                best_score = score
                best = (si, sb, score)

        if best and best_score >= 85:
            si, sb, score = best
            used.add(si)
            matches.append((rb, sb, score))

    return matches


def _component_gaps(crop):
    if crop is None:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    h, w = th.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw < 2 or bh < 5:
            continue

        if bh < h * 0.20:
            continue

        boxes.append((x, y, x + bw, y + bh))

    boxes = sorted(boxes, key=lambda b: b[0])

    if len(boxes) < 2:
        return []

    gaps = []

    for i in range(len(boxes) - 1):
        gap = boxes[i + 1][0] - boxes[i][2]

        if gap >= 0:
            gaps.append(gap)

    return gaps


def _gap_stats(gaps):
    if not gaps:
        return {
            "mean": 0,
            "max": 0,
            "count": 0,
        }

    return {
        "mean": float(np.mean(gaps)),
        "max": float(np.max(gaps)),
        "count": len(gaps),
    }


def _gap_difference(ref_gaps, sus_gaps):
    if not ref_gaps or not sus_gaps:
        return 0, ""

    ref_stats = _gap_stats(ref_gaps)
    sus_stats = _gap_stats(sus_gaps)

    ref_mean = ref_stats["mean"]
    sus_mean = sus_stats["mean"]

    if ref_mean <= 0:
        return 0, ""

    mean_change = abs(sus_mean - ref_mean) / max(ref_mean, 1e-6) * 100

    ref_max = ref_stats["max"]
    sus_max = sus_stats["max"]

    max_change = abs(sus_max - ref_max) / max(ref_max, 1e-6) * 100 if ref_max else 0

    reason = (
        f"Character/symbol gap changed. "
        f"Reference mean gap={ref_mean:.2f}px, uploaded mean gap={sus_mean:.2f}px, "
        f"Reference max gap={ref_max:.2f}px, uploaded max gap={sus_max:.2f}px"
    )

    return max(mean_change, max_change), reason


def detect_char_symbol_gap_differences(
    ref_ocr_boxes,
    sus_ocr_boxes,
    authentic_bgr,
    suspect_bgr,
    medicine_name=None,
    threshold_percent=35.0,
):
    print("CHAR SYMBOL GAP DETECTOR RUNNING")

    issues = []

    matches = _match_boxes(ref_ocr_boxes, sus_ocr_boxes)

    for ref_box, sus_box, score in matches:
        ref_text = str(ref_box.get("text", "")).strip()
        sus_text = str(sus_box.get("text", "")).strip()

        if _is_address(ref_text) or _is_address(sus_text):
            continue

        is_critical = (
            _is_critical(ref_text, medicine_name)
            or _is_critical(sus_text, medicine_name)
        )

        # Avoid too much noise on random text
        if not is_critical and _clean(ref_text) != _clean(sus_text):
            continue

        ref_bbox = _bbox(ref_box)
        sus_bbox = _bbox(sus_box)

        if not ref_bbox or not sus_bbox:
            continue

        ref_crop = _crop(authentic_bgr, ref_bbox)
        sus_crop = _crop(suspect_bgr, sus_bbox)

        ref_gaps = _component_gaps(ref_crop)
        sus_gaps = _component_gaps(sus_crop)

        gap_change, reason = _gap_difference(ref_gaps, sus_gaps)

        if gap_change < threshold_percent:
            continue

        x1, y1, x2, y2 = sus_bbox

        issues.append({
            "issue_type": "char_symbol_spacing_mismatch",
            "category": "spacing_verification",
            "severity": "high" if is_critical else "medium",
            "title": "Character/symbol spacing mismatch detected",
            "description": (
                f"{reason}. Reference='{ref_text}', Uploaded='{sus_text}', "
                f"gap change={gap_change:.2f}%"
            ),
            "difference": (
                f"{reason}. Reference='{ref_text}', Uploaded='{sus_text}', "
                f"gap change={gap_change:.2f}%"
            ),
            "reference": ref_text,
            "uploaded": sus_text,
            "reference_text": ref_text,
            "suspect_text": sus_text,
            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
            "bbox": sus_bbox,
            "fault_bbox_yxyx": [y1, x1, y2, x2],
            "confidence": round(min(0.95, score / 100), 2),
            "gap_change_percent": round(gap_change, 2),
        })

        print(
            "Char-symbol spacing:",
            ref_text,
            "=>",
            sus_text,
            "gap change:",
            round(gap_change, 2),
        )

    print("Char Symbol Gap Detector:", len(issues), "issues")
    return issues
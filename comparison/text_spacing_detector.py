import re
import cv2
import numpy as np
from rapidfuzz import fuzz


def _clean_for_match(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _visible(text):
    return str(text or "").replace(" ", "•")


def _bbox(box):
    b = box.get("bbox")
    if not b or len(b) != 4:
        return None
    return [int(v) for v in b]


def _crop(img, bbox, pad=5):
    if img is None or bbox is None:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def _norm_center(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1))


def _is_vertical(box):
    bbox = _bbox(box)
    if not bbox:
        return False

    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    return (
        box.get("is_vertical") is True
        or str(box.get("orientation", "")).lower() == "vertical"
        or bh / bw > 2.2
    )


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
    c = _clean_for_match(t)

    critical = [
        "mfg", "exp", "mrp", "bno", "batch", "lic", "license",
        "mg", "ml", "tablet", "capsule", "contains", "schedule",
        "dosage", "physician", "drug", "caution", "rx"
    ]

    if medicine_name and _clean_for_match(medicine_name) in c:
        return True

    return any(k in c for k in critical)


def _match_boxes(ref_boxes, sus_boxes, ref_shape, sus_shape):
    matches = []
    used_sus = set()

    for rb in ref_boxes:
        ref_text = str(rb.get("text", "")).strip()
        ref_bbox = _bbox(rb)

        if not ref_text or not ref_bbox:
            continue

        #if _is_vertical(rb):
         #   continue

        ref_clean = _clean_for_match(ref_text)
        if len(ref_clean) < 2:
            continue

        ref_cx, ref_cy = _norm_center(ref_bbox, ref_shape)

        best = None
        best_score = -1

        for si, sb in enumerate(sus_boxes):
            if si in used_sus:
                continue

            sus_text = str(sb.get("text", "")).strip()
            sus_bbox = _bbox(sb)

            if not sus_text or not sus_bbox:
                continue

            #if _is_vertical(sb):
             #   continue

            sus_clean = _clean_for_match(sus_text)
            if len(sus_clean) < 2:
                continue

            length_ratio = min(len(ref_clean), len(sus_clean)) / max(len(ref_clean), len(sus_clean))

            # Prevent wrong matches like Excipients -> Protect from light
            if length_ratio < 0.55:
                continue

            similarity = fuzz.ratio(ref_clean, sus_clean)

            # If cleaned text is same, raw spacing may still differ
            if ref_clean == sus_clean:
                similarity = 100

            if similarity < 70:
                continue

            sus_cx, sus_cy = _norm_center(sus_bbox, sus_shape)
            dx = abs(ref_cx - sus_cx)
            dy = abs(ref_cy - sus_cy)

            # Avoid matching same words from totally different carton panels
            if dx > 0.45 or dy > 0.45:
                continue

            score = similarity - ((dx + dy) * 100)

            if score > best_score:
                best_score = score
                best = (si, sb, similarity, dx, dy)

        if best:
            si, sb, similarity, dx, dy = best
            used_sus.add(si)
            matches.append((rb, sb, similarity, dx, dy))

    return matches


def _binary_text_image(crop):
    if crop is None:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    return th


def _component_boxes(crop):
    th = _binary_text_image(crop)

    if th is None:
        return []

    h, w = th.shape[:2]

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw < 2 or bh < 5:
            continue

        if bh < h * 0.18:
            continue

        if bw > w * 0.80:
            continue

        boxes.append([x, y, x + bw, y + bh])

    boxes = sorted(boxes, key=lambda b: b[0])

    return boxes


def _gaps_from_components(boxes):
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
        return None

    return {
        "count": len(gaps),
        "mean": float(np.mean(gaps)),
        "median": float(np.median(gaps)),
        "max": float(np.max(gaps)),
    }


def _raw_spacing_changed(ref_text, sus_text):
    if ref_text == sus_text:
        return False, ""

    ref_no_space = ref_text.replace(" ", "")
    sus_no_space = sus_text.replace(" ", "")

    if ref_no_space == sus_no_space:
        return True, "Raw OCR text has inserted/removed space characters"

    return False, ""


def _punctuation_spacing_changed(ref_text, sus_text):
    punct = [",", ".", ":", ";", "/", "-", ")", "(", "%"]

    for p in punct:
        ref_before = f" {p}" in ref_text
        sus_before = f" {p}" in sus_text

        ref_after = f"{p} " in ref_text
        sus_after = f"{p} " in sus_text

        if ref_before != sus_before:
            return True, f"Space before symbol '{p}' changed"

        if ref_after != sus_after:
            return True, f"Space after symbol '{p}' changed"

    return False, ""


def _compare_component_gaps(ref_crop, sus_crop):
    ref_boxes = _component_boxes(ref_crop)
    sus_boxes = _component_boxes(sus_crop)

    ref_gaps = _gaps_from_components(ref_boxes)
    sus_gaps = _gaps_from_components(sus_boxes)

    ref_stats = _gap_stats(ref_gaps)
    sus_stats = _gap_stats(sus_gaps)

    if not ref_stats or not sus_stats:
        return 0, ""

    ref_median = ref_stats["median"]
    sus_median = sus_stats["median"]

    ref_max = ref_stats["max"]
    sus_max = sus_stats["max"]

    if ref_median <= 0:
        return 0, ""

    median_change = abs(sus_median - ref_median) / max(ref_median, 1e-6) * 100
    max_change = abs(sus_max - ref_max) / max(ref_max, 1e-6) * 100 if ref_max > 0 else 0

    change = max(median_change, max_change)

    reason = (
        f"Character/symbol spacing changed. "
        f"Reference gap sequence={ref_median:.2f}px, uploaded gap sequence={sus_median:.2f}px, "
        f"Reference max gap={ref_max:.2f}px, uploaded max gap={sus_max:.2f}px"
    )

    return change, reason


def _make_issue(ref_text, sus_text, ref_bbox, sus_bbox, reason, confidence, gap_change=None):
    x1, y1, x2, y2 = sus_bbox

    difference = (
        f"{reason}. Expected=[{_visible(ref_text)}], "
        f"Actual=[{_visible(sus_text)}]"
    )

    if gap_change is not None:
        difference += f", gap_change={gap_change:.2f}%"

    return {
        "issue_type": "text_spacing_mismatch",
        "category": "spacing_verification",
        "severity": "high",
        "title": "Text spacing mismatch detected",
        "description": difference,
        "difference": difference,
        "reference": ref_text,
        "uploaded": sus_text,
        "reference_text": ref_text,
        "suspect_text": sus_text,
        "ref_bbox": ref_bbox,
        "suspect_bbox": sus_bbox,
        "bbox": sus_bbox,
        "fault_bbox_yxyx": [y1, x1, y2, x2],
        "confidence": confidence,
    }


def detect_text_spacing_differences(
    ref_ocr_boxes,
    sus_ocr_boxes,
    authentic_bgr,
    suspect_bgr,
    medicine_name=None,
    gap_threshold_percent=28.0,
):
    print("TEXT SPACING DETECTOR RUNNING")

    if authentic_bgr is None or suspect_bgr is None:
        return []

    issues = []

    matches = _match_boxes(
        ref_ocr_boxes,
        sus_ocr_boxes,
        authentic_bgr.shape,
        suspect_bgr.shape,
    )

    for ref_box, sus_box, similarity, dx, dy in matches:
        print("SPACING MATCH:", ref_box.get("text"), "=>", sus_box.get("text"), "sim=", similarity, "dx=", dx, "dy=", dy)
        ref_text = str(ref_box.get("text", "")).strip()
        sus_text = str(sus_box.get("text", "")).strip()

        ref_bbox = _bbox(ref_box)
        sus_bbox = _bbox(sus_box)

        if not ref_bbox or not sus_bbox:
            continue
        ref_vertical = _is_vertical(ref_box)
        sus_vertical = _is_vertical(sus_box)

        # Detect orientation mismatch
        if ref_vertical != sus_vertical:
            issues.append(
                _make_issue(
                    ref_text,
                    sus_text,
                    ref_bbox,
                    sus_bbox,
                    "Vertical orientation changed",
                    0.95,
                )
            )
            continue

        # Detect vertical alignment shift
        if ref_vertical and sus_vertical:

            ref_cx, ref_cy = _norm_center(ref_bbox, authentic_bgr.shape)
            sus_cx, sus_cy = _norm_center(sus_bbox, suspect_bgr.shape)

            dx_shift = abs(ref_cx - sus_cx)
            dy_shift = abs(ref_cy - sus_cy)

            if dx_shift > 0.02 or dy_shift > 0.03:
                issues.append(
                    _make_issue(
                        ref_text,
                        sus_text,
                        ref_bbox,
                        sus_bbox,
                        f"Vertical text shifted (x={dx_shift:.3f}, y={dy_shift:.3f})",
                        0.93,
                    )
                )
                continue

        if _is_address(ref_text) or _is_address(sus_text):
            continue

        ref_clean = _clean_for_match(ref_text)
        sus_clean = _clean_for_match(sus_text)

        is_critical = (
            _is_critical(ref_text, medicine_name)
            or _is_critical(sus_text, medicine_name)
            or ref_clean == _clean_for_match(medicine_name)
            or sus_clean == _clean_for_match(medicine_name)
        )

        # Raw OCR spacing: A, vs A ,
        raw_changed, raw_reason = _raw_spacing_changed(ref_text, sus_text)
        punct_changed, punct_reason = _punctuation_spacing_changed(ref_text, sus_text)

        if raw_changed or punct_changed:
            reason = punct_reason or raw_reason

            issues.append(
                _make_issue(
                    ref_text,
                    sus_text,
                    ref_bbox,
                    sus_bbox,
                    reason,
                    confidence=round(min(0.96, similarity / 100), 2),
                )
            )

            print("Text spacing raw:", ref_text, "=>", sus_text, reason)
            continue

        # Image-level spacing check
        if not is_critical:
            continue

        ref_crop = _crop(authentic_bgr, ref_bbox)
        sus_crop = _crop(suspect_bgr, sus_bbox)

        gap_change, gap_reason = _compare_component_gaps(ref_crop, sus_crop)
        print("GAP CHECK:", ref_text, "=>", sus_text, "gap_change=", gap_change)

        if gap_change >= gap_threshold_percent:
            issues.append(
                _make_issue(
                    ref_text,
                    sus_text,
                    ref_bbox,
                    sus_bbox,
                    gap_reason,
                    confidence=round(min(0.94, similarity / 100), 2),
                    gap_change=gap_change,
                )
            )

            print(
                "Text spacing visual:",
                ref_text,
                "=>",
                sus_text,
                "gap:",
                round(gap_change, 2),
            )

    print("Text Spacing Detector:", len(issues), "issues")
    return issues
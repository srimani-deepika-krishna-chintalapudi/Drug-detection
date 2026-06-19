import re
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


def _norm_center(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1))


def _is_address(text):
    t = str(text or "").lower()
    words = [
        "baddi", "solan", "h.p", "hp", "dist", "distt", "district",
        "village", "road", "plot", "industrial", "area", "pradesh",
        "india", "khurd", "bhatauli", "bhatouli", "mumbai", "sikkim"
    ]
    return any(w in t for w in words)


def _is_critical(text, medicine_name=None):
    t = str(text or "").lower()
    c = _clean_for_match(t)

    critical = [
        "mfg", "exp", "mrp", "bno", "batch", "lic", "license",
        "mg", "ml", "tablet", "capsule", "contains", "schedule",
        "dosage", "physician", "drug", "caution", "rx",
        "pantoprazole", "domperidone", "betahistine", "paracetamol"
    ]

    if medicine_name and _clean_for_match(medicine_name) in c:
        return True

    return any(k in c for k in critical)


def _char_diff_count(a, b):
    a = str(a or "")
    b = str(b or "")

    n = max(len(a), len(b))
    diff = 0

    for i in range(n):
        ca = a[i] if i < len(a) else ""
        cb = b[i] if i < len(b) else ""

        if ca != cb:
            diff += 1

    return diff


def _match_boxes(ref_boxes, sus_boxes, ref_shape, sus_shape, medicine_name=None):
    matches = []
    used_sus = set()

    for rb in ref_boxes:
        ref_text = str(rb.get("text", "")).strip()
        ref_bbox = _bbox(rb)

        if not ref_text or not ref_bbox:
            continue

        ref_clean = _clean_for_match(ref_text)

        if len(ref_clean) < 3:
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

            sus_clean = _clean_for_match(sus_text)

            if len(sus_clean) < 3:
                continue

            length_ratio = min(len(ref_clean), len(sus_clean)) / max(len(ref_clean), len(sus_clean))

            if length_ratio < 0.70:
                continue

            similarity = fuzz.ratio(ref_clean, sus_clean)

            # Real spelling mistakes are usually high similarity but not exact.
            if similarity < 72:
                continue

            sus_cx, sus_cy = _norm_center(sus_bbox, sus_shape)

            dx = abs(ref_cx - sus_cx)
            dy = abs(ref_cy - sus_cy)

            # Same physical region only
            if dx > 0.25 or dy > 0.25:
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


def detect_spelling_mismatches(
    ref_ocr_boxes,
    sus_ocr_boxes,
    authentic_bgr,
    suspect_bgr,
    medicine_name=None,
):
    print("SPELLING MISMATCH DETECTOR RUNNING")

    if authentic_bgr is None or suspect_bgr is None:
        return []

    issues = []

    matches = _match_boxes(
        ref_ocr_boxes,
        sus_ocr_boxes,
        authentic_bgr.shape,
        suspect_bgr.shape,
        medicine_name=medicine_name,
    )

    for ref_box, sus_box, similarity, dx, dy in matches:
        ref_text = str(ref_box.get("text", "")).strip()
        sus_text = str(sus_box.get("text", "")).strip()

        ref_bbox = _bbox(ref_box)
        sus_bbox = _bbox(sus_box)

        if not ref_bbox or not sus_bbox:
            continue

        ref_clean = _clean_for_match(ref_text)
        sus_clean = _clean_for_match(sus_text)

        if ref_clean == sus_clean:
            continue

        # Ignore pure spacing differences here. Spacing detector handles those.
        if ref_text.replace(" ", "") == sus_text.replace(" ", ""):
            continue

        # Avoid address OCR noise unless you really want address spelling.
        if _is_address(ref_text) or _is_address(sus_text):
            continue

        is_critical = (
            _is_critical(ref_text, medicine_name)
            or _is_critical(sus_text, medicine_name)
        )

        # Ignore random low-value text unless it is a close spelling mismatch
        if not is_critical and similarity < 85:
            continue

        diff_count = _char_diff_count(ref_text, sus_text)

        x1, y1, x2, y2 = sus_bbox

        reason = (
            f"Spelling/text mismatch detected. "
            f"Expected=[{_visible(ref_text)}], Actual=[{_visible(sus_text)}], "
            f"Similarity={similarity:.2f}, char differences={diff_count}"
        )

        issues.append({
            "issue_type": "spelling_mismatch",
            "category": "text_verification",
            "severity": "high" if is_critical else "medium",
            "title": "Spelling mismatch detected",
            "description": reason,
            "difference": reason,
            "reference": ref_text,
            "uploaded": sus_text,
            "reference_text": ref_text,
            "suspect_text": sus_text,
            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
            "bbox": sus_bbox,
            "fault_bbox_yxyx": [y1, x1, y2, x2],
            "confidence": round(min(0.95, similarity / 100), 2),
        })

        print("Spelling mismatch:", ref_text, "=>", sus_text, "sim=", similarity)

    print("Spelling Mismatch Detector:", len(issues), "issues")
    return issues
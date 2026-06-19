import re
from rapidfuzz import fuzz


CRITICAL_KEYWORDS = [
    "mfg", "exp", "mrp", "batch", "bno", "b.no", "license", "lic",
    "composition", "contains", "schedule", "mg", "ml"
]

ADDRESS_KEYWORDS = [
    "baddi", "solan", "h.p", "hp", "distt", "district", "village",
    "road", "plot", "industrial", "area", "pradesh", "india",
    "khurd", "bhatauli", "bhatouli"
]


def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _bbox(box):
    b = box.get("bbox")
    if not b or len(b) != 4:
        return None
    return [int(v) for v in b]


def _width(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1)


def _center_y(bbox):
    x1, y1, x2, y2 = bbox
    return (y1 + y2) / 2


def _is_address_text(text):
    t = str(text or "").lower()
    return any(w in t for w in ADDRESS_KEYWORDS)


def _is_critical_text(text, medicine_name=None):
    t = str(text or "").lower()
    c = _clean(t)

    if medicine_name and _clean(medicine_name) in c:
        return True

    return any(k.replace(".", "") in c for k in CRITICAL_KEYWORDS)


def _colon_spacing_signature(text):
    text = str(text or "")

    return {
        "has_colon": ":" in text,
        "has_dot": "." in text,
        "space_before_colon": bool(re.search(r"\s+:", text)),
        "space_after_colon": bool(re.search(r":\s+", text)),
        "colon_no_space_after": bool(re.search(r":[^\s]", text)),
        "label_colon": bool(re.search(r"(mfg|exp|mrp|batch|b\.?no|lic)\s*:", text, re.I)),
        "label_dot": bool(re.search(r"(mfg|exp|mrp|batch|b\.?no|lic)\s*\.", text, re.I)),
    }


def _colon_spacing_changed(ref_text, sus_text):
    r = _colon_spacing_signature(ref_text)
    s = _colon_spacing_signature(sus_text)

    important_ref = bool(re.search(r"(mfg|exp|mrp|batch|b\.?no|lic)", ref_text, re.I))
    important_sus = bool(re.search(r"(mfg|exp|mrp|batch|b\.?no|lic)", sus_text, re.I))

    if not (important_ref or important_sus):
        return False, ""

    checks = [
        ("has_colon", "Colon presence changed"),
        ("space_before_colon", "Space before colon changed"),
        ("space_after_colon", "Space after colon changed"),
        ("colon_no_space_after", "Space after colon changed"),
        ("label_colon", "Label colon pattern changed"),
        ("label_dot", "Dot/colon pattern changed"),
    ]

    for key, reason in checks:
        if r[key] != s[key]:
            return True, reason

    return False, ""


def _norm_center(bbox, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1))


def _match_boxes(ref_boxes, sus_boxes, ref_img_shape, sus_img_shape):
    matches = []
    used_sus = set()

    for rb in ref_boxes:
        rt = str(rb.get("text", "")).strip()
        rbbox = _bbox(rb)

        if not rt or not rbbox:
            continue

        rc = _clean(rt)
        if len(rc) < 3:
            continue

        ref_cx, ref_cy = _norm_center(rbbox, ref_img_shape)

        best = None
        best_score = -1

        for si, sb in enumerate(sus_boxes):
            if si in used_sus:
                continue

            st = str(sb.get("text", "")).strip()
            sbbox = _bbox(sb)

            if not st or not sbbox:
                continue

            sc = _clean(st)
            if len(sc) < 3:
                continue

            length_ratio = min(len(rc), len(sc)) / max(len(rc), len(sc))
            if length_ratio < 0.75:
                continue

            score = fuzz.ratio(rc, sc)

            # For spacing detector, only compare nearly same text.
            if score < 88:
                continue

            sus_cx, sus_cy = _norm_center(sbbox, sus_img_shape)
            dx = abs(ref_cx - sus_cx)
            dy = abs(ref_cy - sus_cy)

            # Prevent matching same text from different carton panels.
            if dx > 0.18 or dy > 0.18:
                continue

            # Penalize distant matches.
            score = score - ((dx + dy) * 100)

            if score > best_score:
                best_score = score
                best = (si, sb, score)

        if best and best_score >= 70:
            si, sb, score = best
            used_sus.add(si)
            matches.append((rb, sb, score))

    return matches

def detect_spacing_mismatches(
    ref_ocr_boxes,
    sus_ocr_boxes,
    authentic_bgr,
    suspect_bgr,
    medicine_name=None,
    threshold_percent=5.0,
):
    print("SPACING MISMATCH DETECTOR RUNNING")

    if authentic_bgr is None or suspect_bgr is None:
        return []

    ref_img_w = authentic_bgr.shape[1]
    sus_img_w = suspect_bgr.shape[1]

    issues = []
    matches = _match_boxes(ref_ocr_boxes, sus_ocr_boxes,authentic_bgr.shape, suspect_bgr.shape,)

    for ref_box, sus_box, match_score in matches:
        ref_text = str(ref_box.get("text", "")).strip()
        sus_text = str(sus_box.get("text", "")).strip()

        ref_bbox = _bbox(ref_box)
        sus_bbox = _bbox(sus_box)

        if not ref_bbox or not sus_bbox:
            continue

        # Avoid address/location spacing false positives
        if _is_address_text(ref_text) or _is_address_text(sus_text):
            # This detector should not flag address spacing.
            continue

        ref_clean = _clean(ref_text)
        sus_clean = _clean(sus_text)

        if medicine_name and _clean(ref_text) == _clean(medicine_name):
            continue

        is_critical = (
            _is_critical_text(ref_text, medicine_name)
            or _is_critical_text(sus_text, medicine_name)
        )

        colon_changed, colon_reason = _colon_spacing_changed(ref_text, sus_text)

        ref_norm_width = _width(ref_bbox) / max(ref_img_w, 1)
        sus_norm_width = _width(sus_bbox) / max(sus_img_w, 1)

        variance_percent = (
            abs(ref_norm_width - sus_norm_width)
            / max(ref_norm_width, 1e-6)
            * 100
        )

        text_same_after_cleaning = ref_clean == sus_clean

        should_report = False
        reason = ""

        if colon_changed:
            should_report = True
            reason = colon_reason
            severity = "high"

        elif is_critical and text_same_after_cleaning and variance_percent > threshold_percent:
            should_report = True
            reason = "Critical text width/spacing changed"
            severity = "high"

        else:
            continue

        x1, y1, x2, y2 = sus_bbox

        issue = {
            "issue_type": "spacing_mismatch",
            "category": "spacing_verification",
            "severity": severity,
            "title": "Spacing mismatch detected",

            "description": (
                f"{reason}. Reference: '{ref_text}' | Uploaded: '{sus_text}' | "
                f"Width variance: {variance_percent:.2f}%"
            ),

            "difference": (
                f"{reason}. Reference: '{ref_text}' | Uploaded: '{sus_text}' | "
                f"Width variance: {variance_percent:.2f}%"
            ),

            "reference": ref_text,
            "uploaded": sus_text,
            "reference_text": ref_text,
            "suspect_text": sus_text,

            "reference_span": round(ref_norm_width, 4),
            "suspect_span": round(sus_norm_width, 4),
            "variance_percent": round(variance_percent, 2),

            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
            "bbox": sus_bbox,
            "fault_bbox_yxyx": [y1, x1, y2, x2],

            "confidence": round(min(0.95, match_score / 100), 2),
        }

        print(
            "Spacing mismatch:",
            ref_text,
            "=>",
            sus_text,
            "variance:",
            round(variance_percent, 2),
        )

        issues.append(issue)

    print("Dedicated Spacing Mismatch:", len(issues), "issues")
    return issues
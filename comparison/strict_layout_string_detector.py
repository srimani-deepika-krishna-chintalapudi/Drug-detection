import re
from rapidfuzz import fuzz


CRITICAL_KEYWORDS = [
    "mfg", "exp", "mrp", "bno", "b.no", "batch", "lic", "license",
    "mg", "ml", "tablet", "capsule", "contains", "schedule"
]

ADDRESS_WORDS = [
    "baddi", "solan", "h.p", "hp", "dist", "distt", "district",
    "village", "road", "plot", "industrial", "area", "pradesh",
    "india", "khurd", "bhatauli", "bhatouli", "mumbai", "sikkim"
]


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
        or bh / bw > 2.3
    )


def _is_address(text):
    t = str(text or "").lower()
    return any(w in t for w in ADDRESS_WORDS)


def _is_critical(text, medicine_name=None):
    t = str(text or "").lower()
    c = _clean_for_match(t)

    if medicine_name and _clean_for_match(medicine_name) in c:
        return True

    return any(k.replace(".", "") in c for k in CRITICAL_KEYWORDS)


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


def _classify_string_mismatch(ref_text, sus_text):
    if ref_text == sus_text:
        return None

    if ref_text.replace(" ", "") == sus_text.replace(" ", ""):
        return "strict_spacing_mismatch"

    return "strict_string_mismatch"


def _make_issue(
    issue_type,
    title,
    reason,
    ref_text,
    sus_text,
    ref_bbox,
    sus_bbox,
    confidence=0.9,
):
    x1, y1, x2, y2 = sus_bbox

    return {
        "issue_type": issue_type,
        "category": "strict_layout_string_verification",
        "severity": "high",
        "title": title,
        "description": (
            f"{reason}. Expected=[{_visible(ref_text)}], "
            f"Actual=[{_visible(sus_text)}], "
            f"Exact char variance={_char_diff_count(ref_text, sus_text)}"
        ),
        "difference": (
            f"{reason}. Expected=[{_visible(ref_text)}], "
            f"Actual=[{_visible(sus_text)}], "
            f"Exact char variance={_char_diff_count(ref_text, sus_text)}"
        ),
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


def _match_boxes(ref_boxes, sus_boxes, ref_shape, sus_shape):
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

            # HARD RULE: do not compare unrelated strings
            length_ratio = min(len(ref_clean), len(sus_clean)) / max(len(ref_clean), len(sus_clean))
            if length_ratio < 0.80:
                continue

            similarity = fuzz.ratio(ref_clean, sus_clean)
            if ref_clean == sus_clean:
                similarity = 100

            if similarity < 88:
                continue

            sus_cx, sus_cy = _norm_center(sus_bbox, sus_shape)
            dx = abs(ref_cx - sus_cx)
            dy = abs(ref_cy - sus_cy)

            # HARD RULE: avoid matching same/near text from different carton panels
            if dx > 0.22 or dy > 0.22:
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

def detect_strict_layout_string_differences(
    ref_ocr_boxes,
    sus_ocr_boxes,
    authentic_bgr,
    suspect_bgr,
    medicine_name=None,
):
    print("STRICT LAYOUT STRING DETECTOR RUNNING")

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
        ref_text = str(ref_box.get("text", "")).strip()
        sus_text = str(sus_box.get("text", "")).strip()

        ref_bbox = _bbox(ref_box)
        sus_bbox = _bbox(sus_box)

        if not ref_bbox or not sus_bbox:
            continue

        ref_vertical = _is_vertical(ref_box)
        sus_vertical = _is_vertical(sus_box)

        ref_critical = _is_critical(ref_text, medicine_name)
        sus_critical = _is_critical(sus_text, medicine_name)
        is_critical = ref_critical or sus_critical

        is_address = _is_address(ref_text) or _is_address(sus_text)

        # 1. Orientation mismatch
        if ref_vertical != sus_vertical:
            issues.append(
                _make_issue(
                    "vertical_orientation_mismatch",
                    "Vertical orientation mismatch detected",
                    "Source and uploaded text orientation differ",
                    ref_text,
                    sus_text,
                    ref_bbox,
                    sus_bbox,
                    confidence=0.95,
                )
            )
            continue

        # 2. Vertical layout shift
        if ref_vertical and sus_vertical:
            if dx > 0.015 or dy > 0.025:
                issues.append(
                    _make_issue(
                        "vertical_layout_shift",
                        "Vertical text layout shift detected",
                        f"Vertical text shifted. x shift={dx:.3f}, y shift={dy:.3f}",
                        ref_text,
                        sus_text,
                        ref_bbox,
                        sus_bbox,
                        confidence=0.90,
                    )
                )

        mismatch_type = _classify_string_mismatch(ref_text, sus_text)
        if mismatch_type is None:
            continue

# Ignore address spacing-only changes
        if is_address and mismatch_type == "strict_spacing_mismatch":
            continue

# Ignore generic spacing noise
        if not is_critical and mismatch_type == "strict_spacing_mismatch":
            continue

        if mismatch_type == "strict_string_mismatch":

    # Keep spelling mismatch only if boxes are
    # basically in the same physical location
                if dx <= 0.08 and dy <= 0.08:
                    title = "Strict spelling/text mismatch detected"
                    reason = "Raw character/string mismatch detected at same layout position"
                else:
                    continue

        elif mismatch_type == "strict_spacing_mismatch":

                title = "Strict spacing mismatch detected"
                reason = "Space character mismatch detected"

        else:
            continue
        '''if mismatch_type == "strict_string_mismatch":
            continue

        if mismatch_type == "strict_spacing_mismatch":
            title = "Strict spacing mismatch detected"
            reason = "Space character mismatch detected"
        else:
            continue'''

        issues.append(
            _make_issue(
                mismatch_type,
                title,
                reason,
                ref_text,
                sus_text,
                ref_bbox,
                sus_bbox,
                confidence=round(min(0.98, similarity / 100), 2),
            )
        )

    print("Strict layout issues:", len(issues))
    return issues
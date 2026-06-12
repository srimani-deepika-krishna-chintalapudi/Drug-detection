import re
from rapidfuzz import fuzz


def _norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t or "").lower())


def _center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


def _w(b):
    return max(1, b[2] - b[0])


def _h(b):
    return max(1, b[3] - b[1])


def _is_vertical(box):
    b = box.get("bbox")
    if not b:
        return False

    text = str(box.get("text", "")).strip()
    if len(_norm(text)) < 2:
        return False

    return _h(b) / _w(b) >= 1.7


def _relative_box(bbox, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox

    return [
        x1 / max(w, 1),
        y1 / max(h, 1),
        x2 / max(w, 1),
        y2 / max(h, 1),
    ]


def _rel_center(bbox, img_shape):
    rb = _relative_box(bbox, img_shape)
    return (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2


def detect_vertical_text_differences(
    ref_boxes,
    sus_boxes,
    ref_img_shape,
    sus_img_shape,
    min_match_score=70,
    position_threshold=0.025,
    size_threshold=0.35,
):
    issues = []

    ref_vertical = [b for b in ref_boxes if _is_vertical(b)]
    sus_vertical = [b for b in sus_boxes if _is_vertical(b)]

    used = set()

    for ref in ref_vertical:
        ref_text = ref.get("text", "")
        ref_norm = _norm(ref_text)

        best_i = None
        best_score = 0

        for i, sus in enumerate(sus_vertical):
            if i in used:
                continue

            sus_text = sus.get("text", "")
            score = fuzz.ratio(ref_norm, _norm(sus_text))

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None or best_score < min_match_score:
            issues.append({
                "issue_type": "vertical_text_mismatch",
                "reference": ref_text,
                "uploaded": "Missing / not detected",
                "difference": f"Vertical text '{ref_text}' is missing or not matched in suspect image.",
                "severity": "High",
                "confidence": 88,
                "ref_bbox": ref["bbox"],
                "suspect_bbox": None,
            })
            continue

        sus = sus_vertical[best_i]
        used.add(best_i)

        sus_text = sus.get("text", "")

        ref_cx, ref_cy = _rel_center(ref["bbox"], ref_img_shape)
        sus_cx, sus_cy = _rel_center(sus["bbox"], sus_img_shape)

        dx = abs(ref_cx - sus_cx)
        dy = abs(ref_cy - sus_cy)

        ref_rw = _w(ref["bbox"]) / max(ref_img_shape[1], 1)
        ref_rh = _h(ref["bbox"]) / max(ref_img_shape[0], 1)

        sus_rw = _w(sus["bbox"]) / max(sus_img_shape[1], 1)
        sus_rh = _h(sus["bbox"]) / max(sus_img_shape[0], 1)

        width_diff = abs(ref_rw - sus_rw) / max(ref_rw, 0.001)
        height_diff = abs(ref_rh - sus_rh) / max(ref_rh, 0.001)

        if _norm(ref_text) != _norm(sus_text):
            issues.append({
                "issue_type": "vertical_text_mismatch",
                "reference": ref_text,
                "uploaded": sus_text,
                "difference": f"Vertical text content changed: '{ref_text}' vs '{sus_text}'.",
                "severity": "High",
                "confidence": max(85, int(best_score)),
                "ref_bbox": ref["bbox"],
                "suspect_bbox": sus["bbox"],
            })

        if dx >= position_threshold or dy >= position_threshold:
            issues.append({
                "issue_type": "vertical_text_alignment",
                "reference": ref_text,
                "uploaded": sus_text,
                "difference": (
                    f"Vertical text position changed. "
                    f"Horizontal shift={dx:.3f}, vertical shift={dy:.3f}."
                ),
                "severity": "Medium",
                "confidence": min(95, int(75 + (dx + dy) * 400)),
                "ref_bbox": ref["bbox"],
                "suspect_bbox": sus["bbox"],
            })

        if width_diff >= size_threshold or height_diff >= size_threshold:
            issues.append({
                "issue_type": "vertical_text_size",
                "reference": ref_text,
                "uploaded": sus_text,
                "difference": (
                    f"Vertical text size changed. "
                    f"Width difference={width_diff:.2f}, height difference={height_diff:.2f}."
                ),
                "severity": "Medium",
                "confidence": min(92, int(72 + max(width_diff, height_diff) * 30)),
                "ref_bbox": ref["bbox"],
                "suspect_bbox": sus["bbox"],
            })

    for i, sus in enumerate(sus_vertical):
        if i not in used:
            issues.append({
                "issue_type": "vertical_text_mismatch",
                "reference": "Missing in reference",
                "uploaded": sus.get("text", ""),
                "difference": f"Extra vertical text found in suspect image: '{sus.get('text', '')}'.",
                "severity": "Medium",
                "confidence": 82,
                "ref_bbox": None,
                "suspect_bbox": sus["bbox"],
            })

    return issues
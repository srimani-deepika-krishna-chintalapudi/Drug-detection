import re
from difflib import SequenceMatcher


def _clean(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text.strip()


def _sim(a, b):
    a = _clean(a)
    b = _clean(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _find_same_text_box(ref_box, sus_boxes):
    ref_text = ref_box.get("text", "")
    best = None
    best_score = 0

    for sb in sus_boxes:
        score = _sim(ref_text, sb.get("text", ""))

        if score > best_score:
            best_score = score
            best = sb

    if best_score >= 0.88:
        return best

    return None


def _nearest_left_or_above(target_box, all_boxes):
    tx, ty = _center(target_box["bbox"])

    best = None
    best_dist = 999999

    for b in all_boxes:
        if b is target_box:
            continue

        text = str(b.get("text", "")).strip()
        bbox = b.get("bbox")

        if not text or not bbox:
            continue

        bx, by = _center(bbox)

        # nearby left or same-line/above context
        if bx > tx and abs(by - ty) < 80:
            continue

        dx = tx - bx
        dy = ty - by

        dist = (dx * dx + dy * dy) ** 0.5

        if dist < best_dist:
            best_dist = dist
            best = b

    return best


def detect_local_alignment(ref_boxes, sus_boxes):
    issues = []

    for rb in ref_boxes:
        ref_text = str(rb.get("text", "")).strip()
        ref_bbox = rb.get("bbox")

        if not ref_text or not ref_bbox:
            continue

        # focus on numeric/license/date/batch-like fields
        clean = _clean(ref_text)

        if not re.search(r"\d", clean):
            continue

        if len(clean) < 3:
            continue

        sb = _find_same_text_box(rb, sus_boxes)
        if not sb:
            continue

        ref_anchor = _nearest_left_or_above(rb, ref_boxes)
        sus_anchor = _nearest_left_or_above(sb, sus_boxes)

        if not ref_anchor or not sus_anchor:
            continue

        # anchors should represent similar nearby text
        if _sim(ref_anchor.get("text", ""), sus_anchor.get("text", "")) < 0.55:
            continue

        rx, ry = _center(ref_bbox)
        sx, sy = _center(sb["bbox"])

        rax, ray = _center(ref_anchor["bbox"])
        sax, say = _center(sus_anchor["bbox"])

        ref_dx = rx - rax
        ref_dy = ry - ray

        sus_dx = sx - sax
        sus_dy = sy - say

        shift_x = abs(ref_dx - sus_dx)
        shift_y = abs(ref_dy - sus_dy)

        if shift_x >= 35 or shift_y >= 25:
            issues.append({
                "issue_type": "local_alignment",
                "reference": ref_text,
                "uploaded": sb.get("text", ""),
                "difference": (
                    f"Field alignment changed relative to nearby text. "
                    f"x shift={shift_x:.1f}px, y shift={shift_y:.1f}px."
                ),
                "severity": "Medium",
                "confidence": 88,
                "ref_bbox": ref_bbox,
                "suspect_bbox": sb.get("bbox"),
            })

    return issues
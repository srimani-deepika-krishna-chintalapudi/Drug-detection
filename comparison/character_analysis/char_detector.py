import re
from rapidfuzz import fuzz

from comparison.character_analysis.char_spacing import compare_character_spacing


def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _bbox(box):
    bbox = box.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return [int(v) for v in bbox]


def _crop(img, bbox, pad=6):
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


def _is_valid_text(text):
    raw = str(text or "").strip()
    clean = _clean(raw)

    if len(clean) < 5:
        return False

    if len(clean) > 25:
        return False

    if raw.count(" ") > 2:
        return False

    if clean.isdigit():
        return False

    digits = sum(ch.isdigit() for ch in clean)
    if digits / max(1, len(clean)) > 0.45:
        return False

    letters = sum(ch.isalpha() for ch in clean)
    if letters < 4:
        return False

    return True


def _position_close(ref_bbox, sus_bbox, ref_img, sus_img, threshold=0.06):
    rh, rw = ref_img.shape[:2]
    sh, sw = sus_img.shape[:2]

    rcx = ((ref_bbox[0] + ref_bbox[2]) / 2) / rw
    rcy = ((ref_bbox[1] + ref_bbox[3]) / 2) / rh

    scx = ((sus_bbox[0] + sus_bbox[2]) / 2) / sw
    scy = ((sus_bbox[1] + sus_bbox[3]) / 2) / sh

    return abs(rcx - scx) <= threshold and abs(rcy - scy) <= threshold


def _best_match(ref_box, sus_boxes, used_sus, ref_img, sus_img):
    ref_text = str(ref_box.get("text", "") or "")
    ref_clean = _clean(ref_text)
    ref_bbox = _bbox(ref_box)

    best_idx = None
    best_score = 0

    for i, sus_box in enumerate(sus_boxes):
        if i in used_sus:
            continue

        sus_text = str(sus_box.get("text", "") or "")
        sus_clean = _clean(sus_text)
        sus_bbox = _bbox(sus_box)

        if not sus_bbox:
            continue

        if not _is_valid_text(sus_text):
            continue

        text_score = fuzz.ratio(ref_clean, sus_clean)

        if text_score < 96:
            continue

        if not _position_close(ref_bbox, sus_bbox, ref_img, sus_img):
            continue

        if text_score > best_score:
            best_score = text_score
            best_idx = i

    return best_idx, best_score


def detect_character_spacing_differences(
    ref_boxes,
    sus_boxes,
    ref_img,
    sus_img,
):
    issues = []
    used_sus = set()

    print("\n########## CHARACTER SPACING DETECTOR RUNNING ##########")

    for rb in ref_boxes:
        ref_text = str(rb.get("text", "") or "")
        ref_bbox = _bbox(rb)

        if not ref_bbox:
            continue

        if not _is_valid_text(ref_text):
            continue

        best_idx, score = _best_match(
            rb,
            sus_boxes,
            used_sus,
            ref_img,
            sus_img,
        )

        if best_idx is None:
            continue

        sb = sus_boxes[best_idx]
        sus_text = str(sb.get("text", "") or "")
        sus_bbox = _bbox(sb)

        if not sus_bbox:
            continue

        ref_crop = _crop(ref_img, ref_bbox)
        sus_crop = _crop(sus_img, sus_bbox)

        if ref_crop is None or sus_crop is None:
            continue

        result = compare_character_spacing(
            ref_crop,
            sus_crop,
            threshold=0.35,
        )

        if not result or not result.get("different"):
            continue

        pattern_diff = float(result.get("pattern_diff", 0) or 0)
        median_diff = float(result.get("median_diff", 0) or 0)
        max_diff = float(result.get("max_diff", 0) or 0)

        if pattern_diff < 0.35 and median_diff < 0.35 and max_diff < 0.70:
            continue

        used_sus.add(best_idx)

        print("CHAR ISSUE:", ref_text, "VS", sus_text)

        issues.append({
            "issue_type": "character_spacing",
            "reference": ref_text,
            "uploaded": sus_text,
            "difference": (
                f"Character spacing differs. "
                f"pattern_diff={pattern_diff:.2f}, "
                f"median_diff={median_diff:.2f}, "
                f"max_diff={max_diff:.2f}"
            ),
            "severity": "Medium",
            "confidence": 85,
            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
        })

    return issues
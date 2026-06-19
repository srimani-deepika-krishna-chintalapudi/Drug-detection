import re
from rapidfuzz import fuzz
from comparison.character_analysis.char_spacing import compare_character_spacing
from backend.image_processing import crop_bbox
from comparison.pharma_text_utils import is_critical_pharma_text, is_address_or_location_text

def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _is_spacing_candidate(box):
    text = str(box.get("text", "")).strip()
    clean = _clean(text)
    bbox = box.get("bbox")

    if not bbox or len(bbox) != 4:
        return False

    if len(clean) < 3:
        return False

    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    # Skip tiny OCR text
    if h < 25 or w < 60:
        return False

    # Good candidates: brand/logo/short printed labels
    if len(clean) <= 18:
        return True

    # Also allow large OCR text lines
    if h >= 35:
        return True

    return False


def _crop(img, bbox, pad=8):
    if img is None or bbox is None:
        return None

    if isinstance(bbox, dict):
        bbox = bbox.get("bbox")

    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()

    if len(bbox) != 4:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]

def _bbox(box):
    b = box.get("bbox")

    if isinstance(b, dict):
        b = b.get("bbox")

    if hasattr(b, "tolist"):
        b = b.tolist()

    if not b or len(b) != 4:
        return None

    return [int(v) for v in b]

def detect_generic_text_spacing(ref_boxes, sus_boxes, ref_img, sus_img):
    issues = []
    used_sus = set()

    ref_candidates = [b for b in ref_boxes if _is_spacing_candidate(b)]
    sus_candidates = [b for b in sus_boxes if _is_spacing_candidate(b)]

    for ref_box in ref_candidates:
        ref_text = str(ref_box.get("text", "")).strip()
        ref_clean = _clean(ref_text)

        best_i = None
        best_score = 0

        for i, sus_box in enumerate(sus_candidates):
            if i in used_sus:
                continue

            sus_text = str(sus_box.get("text", "")).strip()
            sus_clean = _clean(sus_text)

            score = fuzz.ratio(ref_clean, sus_clean)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None or best_score < 90:
            continue

        sus_box = sus_candidates[best_i]
        used_sus.add(best_i)

        ref_bbox = _bbox(ref_box)
        sus_bbox = _bbox(sus_box)
        if not ref_bbox or not sus_bbox:
            continue
        ref_crop = crop_bbox(ref_img, ref_bbox)
        sus_crop = crop_bbox(sus_img, sus_bbox)

        if ref_crop is None or sus_crop is None:
            continue

        is_addr = is_address_or_location_text(ref_text)
        is_crit = is_critical_pharma_text(ref_text)
        
        thresh = 0.12
        if is_addr and not is_crit:
            thresh = 0.22
        elif not is_crit:
            thresh = 0.18

        result = compare_character_spacing(ref_crop, sus_crop, threshold=thresh)

        if result:
            issues.append({
                "issue_type": "text_character_spacing",
                "reference": ref_text,
                "uploaded": sus_box.get("text", ""),
                "difference": "Letter spacing differs in a matching printed text/logo region.",
                "severity": "Medium",
                "confidence": max(88, int(result.get("confidence", 88))),
                "ref_bbox": ref_bbox,
                "suspect_bbox": sus_bbox,
            })

    return issues
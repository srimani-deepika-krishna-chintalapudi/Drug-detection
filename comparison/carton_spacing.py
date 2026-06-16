import cv2
import numpy as np
from rapidfuzz import fuzz


def _clean(text):
    import re
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _bbox(box):
    b = box.get("bbox") or box.get("ref_bbox") or box.get("suspect_bbox")
    if not b or len(b) != 4:
        return None
    return [int(v) for v in b]


def _crop(img, bbox, pad=4):
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


def _ink_column_profile(crop):
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Remove tiny dots/noise but preserve letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    col = np.sum(binary > 0, axis=0).astype(np.float32)

    if col.max() <= 0:
        return None

    col = col / col.max()
    return col


def _gap_signature(crop):
    profile = _ink_column_profile(crop)
    if profile is None:
        return None

    threshold = 0.08
    ink = profile > threshold

    gaps = []
    in_gap = False
    start = 0

    for i, has_ink in enumerate(ink):
        if not has_ink and not in_gap:
            start = i
            in_gap = True
        elif has_ink and in_gap:
            length = i - start
            if length >= 2:
                gaps.append(length)
            in_gap = False

    if in_gap:
        length = len(ink) - start
        if length >= 2:
            gaps.append(length)

    if not gaps:
        return None

    width = max(1, crop.shape[1])

    gaps = np.array(gaps, dtype=np.float32) / width

    return {
        "gap_count": len(gaps),
        "mean_gap": float(np.mean(gaps)),
        "max_gap": float(np.max(gaps)),
        "std_gap": float(np.std(gaps)),
        "p75_gap": float(np.percentile(gaps, 75)),
    }


def _spacing_difference(ref_crop, sus_crop):
    r = _gap_signature(ref_crop)
    s = _gap_signature(sus_crop)

    if not r or not s:
        return None

    mean_diff = abs(r["mean_gap"] - s["mean_gap"]) / max(r["mean_gap"], s["mean_gap"], 1e-6)
    max_diff = abs(r["max_gap"] - s["max_gap"]) / max(r["max_gap"], s["max_gap"], 1e-6)
    std_diff = abs(r["std_gap"] - s["std_gap"]) / max(r["std_gap"], s["std_gap"], 1e-6)

    score = max(mean_diff, max_diff, std_diff)

    if score < 0.18:
        return None

    return {
        "score": score,
        "ref_mean": r["mean_gap"],
        "sus_mean": s["mean_gap"],
        "ref_max": r["max_gap"],
        "sus_max": s["max_gap"],
        "ref_gap_count": r["gap_count"],
        "sus_gap_count": s["gap_count"],
    }


def detect_carton_spacing_issues(ref_boxes, sus_boxes, ref_img, sus_img):
    issues = []

    used_sus = set()

    for ref_box in ref_boxes:
        ref_text = ref_box.get("text", "")
        ref_clean = _clean(ref_text)

        if len(ref_clean) < 5:
            continue

        ref_bbox = _bbox(ref_box)
        if not ref_bbox:
            continue

        best_idx = None
        best_score = 0

        for i, sus_box in enumerate(sus_boxes):
            if i in used_sus:
                continue

            sus_text = sus_box.get("text", "")
            sus_clean = _clean(sus_text)

            if len(sus_clean) < 5:
                continue

            sim = fuzz.ratio(ref_clean, sus_clean)

            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_idx is None or best_score < 75:
            continue

        sus_box = sus_boxes[best_idx]
        used_sus.add(best_idx)

        sus_bbox = _bbox(sus_box)

        ref_crop = _crop(ref_img, ref_bbox, pad=6)
        sus_crop = _crop(sus_img, sus_bbox, pad=6)

        diff = _spacing_difference(ref_crop, sus_crop)

        if not diff:
            continue

        confidence = int(min(94, 70 + diff["score"] * 80))

        issues.append({
            "issue_type": "carton_spacing_difference",
            "reference": ref_text,
            "uploaded": sus_box.get("text", ""),
            "difference": (
                "Carton-wide text spacing difference detected in this printed region. "
                f"Reference mean gap={diff['ref_mean']:.3f}, "
                f"uploaded mean gap={diff['sus_mean']:.3f}."
            ),
            "severity": "Medium",
            "confidence": confidence,
            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
        })

    return issues
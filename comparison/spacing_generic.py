import re
import cv2
import numpy as np
from rapidfuzz import fuzz


def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _text(box):
    return str(box.get("text") or box.get("label") or "")


def _bbox(box):
    b = box.get("bbox") or box.get("ref_bbox") or box.get("suspect_bbox")
    if not b or len(b) != 4:
        return None
    return [int(v) for v in b]


def _crop(img, bbox, pad=5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _binary(crop):
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]


def _ink_segments(crop):
    bw = _binary(crop)
    if bw is None:
        return []

    h, w = bw.shape
    col = np.sum(bw > 0, axis=0)

    threshold = max(1, h * 0.06)
    ink_cols = np.where(col > threshold)[0]

    if len(ink_cols) == 0:
        return []

    segments = []
    start = ink_cols[0]
    prev = ink_cols[0]

    for x in ink_cols[1:]:
        if x - prev > 1:
            if prev - start >= 1:
                segments.append((start, prev))
            start = x
        prev = x

    if prev - start >= 1:
        segments.append((start, prev))

    # remove tiny noise
    return [(a, b) for a, b in segments if (b - a + 1) >= 2]


def _gap_features(crop):
    segments = _ink_segments(crop)

    if len(segments) < 2:
        return None

    w = max(1, crop.shape[1])
    gaps = []

    for i in range(len(segments) - 1):
        gap = segments[i + 1][0] - segments[i][1]
        if gap >= 1:
            gaps.append(gap / w)

    if not gaps:
        return None

    gaps = np.array(gaps, dtype=np.float32)

    return {
        "gap_count": len(gaps),
        "mean": float(np.mean(gaps)),
        "max": float(np.max(gaps)),
        "std": float(np.std(gaps)),
        "p75": float(np.percentile(gaps, 75)),
    }


def _compare_gap_features(ref_crop, sus_crop):
    r = _gap_features(ref_crop)
    s = _gap_features(sus_crop)

    if not r or not s:
        return None

    mean_diff = abs(r["mean"] - s["mean"]) / max(r["mean"], s["mean"], 1e-6)
    max_diff = abs(r["max"] - s["max"]) / max(r["max"], s["max"], 1e-6)
    std_diff = abs(r["std"] - s["std"]) / max(r["std"], s["std"], 1e-6)

    score = max(mean_diff, max_diff, std_diff)

    return {
        "score": score,
        "ref_mean": r["mean"],
        "sus_mean": s["mean"],
        "ref_max": r["max"],
        "sus_max": s["max"],
        "ref_gap_count": r["gap_count"],
        "sus_gap_count": s["gap_count"],
    }


def _word_spacing_difference(ref_text, sus_text):
    ref_spaces = len(re.findall(r"\s+", ref_text))
    sus_spaces = len(re.findall(r"\s+", sus_text))

    ref_punct_space = len(re.findall(r"\s+[,.:;]\s*|[,.:;]\s+", ref_text))
    sus_punct_space = len(re.findall(r"\s+[,.:;]\s*|[,.:;]\s+", sus_text))

    if ref_spaces != sus_spaces:
        return True, f"Word spacing count changed: reference={ref_spaces}, uploaded={sus_spaces}."

    if ref_punct_space != sus_punct_space:
        return True, "Spacing around punctuation/symbols changed."

    return False, ""


def detect_generic_spacing_issues(ref_boxes, sus_boxes, ref_img, sus_img):
    issues = []
    used = set()

    for ref_box in ref_boxes:
        ref_text = _text(ref_box)
        ref_clean = _clean(ref_text)

        if len(ref_clean) < 5:
            continue

        ref_bbox = _bbox(ref_box)
        if not ref_bbox:
            continue

        best_i = None
        best_sim = 0

        for i, sus_box in enumerate(sus_boxes):
            if i in used:
                continue

            sus_text = _text(sus_box)
            sus_clean = _clean(sus_text)

            if len(sus_clean) < 5:
                continue

            sim = fuzz.ratio(ref_clean, sus_clean)

            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i is None or best_sim < 88:
            continue

        sus_box = sus_boxes[best_i]
        sus_text = _text(sus_box)
        sus_bbox = _bbox(sus_box)

        if not sus_bbox:
            continue

        used.add(best_i)

        ref_crop = _crop(ref_img, ref_bbox)
        sus_crop = _crop(sus_img, sus_bbox)

        word_changed, word_reason = _word_spacing_difference(ref_text, sus_text)
        visual = _compare_gap_features(ref_crop, sus_crop)

        if not word_changed and (not visual or visual["score"] < 0.22):
            continue

        if word_changed:
            issue_type = "word_or_symbol_spacing"
            reason = word_reason
            confidence = 88
        else:
            issue_type = "character_spacing"
            reason = (
                "Letter/character spacing visually changed. "
                f"Reference mean gap={visual['ref_mean']:.3f}, "
                f"uploaded mean gap={visual['sus_mean']:.3f}."
            )
            confidence = int(min(94, 72 + visual["score"] * 80))

        issues.append({
            "issue_type": issue_type,
            "reference": ref_text,
            "uploaded": sus_text,
            "difference": reason,
            "severity": "Medium",
            "confidence": confidence,
            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
        })

    return issues
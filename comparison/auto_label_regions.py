import cv2
import numpy as np
import re
from difflib import SequenceMatcher


LABEL_KEYWORDS = {
    "contains": ["contains"],
    "dosage": ["dosage"],
    "manufactured_by": ["manufactured", "manufactured by"],
    "batch_block": ["b.no", "batch", "mfg", "exp", "mrp"],
    "license": ["lic", "license"],
    "address": ["village", "baddi", "solan", "india", "khurd"],
}

AUTO_VISUAL_LABELS_ENABLED_FOR = {
    "vertin",
    "betahistine",
}


def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _sim(a, b):
    a = _clean(a)
    b = _clean(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def _crop(img, bbox, pad=10):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def _find_label_regions(boxes):
    regions = []

    for label, keywords in LABEL_KEYWORDS.items():
        for b in boxes:
            text = str(b.get("text", "") or "").strip()
            bbox = b.get("bbox")

            if not text or not bbox:
                continue

            clean_text = _clean(text)

            if any(_clean(kw) in clean_text for kw in keywords):
                regions.append({
                    "label": label,
                    "text": text,
                    "bbox": bbox,
                    "box": b,
                })

    return regions


def _medicine_allows_common_label_check(ref_boxes, sus_boxes):
    all_text = " ".join(
        str(b.get("text", "")).lower()
        for b in list(ref_boxes or []) + list(sus_boxes or [])
    )

    return any(med in all_text for med in AUTO_VISUAL_LABELS_ENABLED_FOR)


def _find_best_region(ref_region, sus_regions):
    best = None
    best_score = 0.0

    for s in sus_regions:
        if s["label"] != ref_region["label"]:
            continue

        score = _sim(ref_region["text"], s["text"])

        # Do not force score to 0.70. That was causing wrong matches.
        if score < 0.50:
            continue

        if score > best_score:
            best_score = score
            best = s

    return best, best_score


def _binary(crop_img):
    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    return th


def _ink_density(crop_img):
    th = _binary(crop_img)

    if th is None:
        return 0.0

    return np.count_nonzero(th) / max(1, th.size)


def _sharpness(crop_img):
    if crop_img is None or crop_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _colon_gap(crop_img):
    th = _binary(crop_img)

    if th is None:
        return None

    h, w = th.shape[:2]

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    small = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if 1 <= cw <= 12 and 1 <= ch <= 12 and 1 <= area <= 90:
            small.append((x, y, cw, ch))

    colon_xs = []

    for i in range(len(small)):
        x1, y1, w1, h1 = small[i]
        cx1 = x1 + w1 / 2
        cy1 = y1 + h1 / 2

        for j in range(i + 1, len(small)):
            x2, y2, w2, h2 = small[j]
            cx2 = x2 + w2 / 2
            cy2 = y2 + h2 / 2

            if abs(cx1 - cx2) <= 6 and 4 <= abs(cy1 - cy2) <= 24:
                colon_xs.append(int((cx1 + cx2) / 2))

    if not colon_xs:
        return None

    colon_x = max(colon_xs)

    col_ink = np.sum(th > 0, axis=0)
    has_ink = col_ink > max(1, h * 0.04)

    prev_ink = None

    for x in range(colon_x - 2, -1, -1):
        if has_ink[x]:
            prev_ink = x
            break

    if prev_ink is None:
        return None

    return max(0, colon_x - prev_ink)


def _visual_difference_score(ref_crop, sus_crop):
    if ref_crop is None or sus_crop is None:
        return 0.0

    rh, rw = ref_crop.shape[:2]
    sh, sw = sus_crop.shape[:2]

    if rh <= 0 or rw <= 0 or sh <= 0 or sw <= 0:
        return 0.0

    sus_crop = cv2.resize(sus_crop, (rw, rh), interpolation=cv2.INTER_AREA)

    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
    sus_gray = cv2.cvtColor(sus_crop, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, sus_gray)

    return float(np.mean(diff))


def detect_auto_label_region_differences(ref_boxes, sus_boxes, ref_img, sus_img):
    issues = []

    ref_regions = _find_label_regions(ref_boxes)
    sus_regions = _find_label_regions(sus_boxes)

    allow_common_labels = _medicine_allows_common_label_check(
        ref_boxes,
        sus_boxes,
    )

    for rr in ref_regions:
        sr, score = _find_best_region(rr, sus_regions)

        if sr is None:
            continue

        ref_crop = _crop(ref_img, rr["bbox"], pad=12)
        sus_crop = _crop(sus_img, sr["bbox"], pad=12)

        if ref_crop is None or sus_crop is None:
            continue

        label = rr["label"]

        if (
            label in {"contains", "dosage", "manufactured_by"}
            and not allow_common_labels
        ):
            continue

        if label in {"contains", "dosage", "manufactured_by"}:
            rg = _colon_gap(ref_crop)
            sg = _colon_gap(sus_crop)

            if rg is not None and sg is not None and abs(rg - sg) >= 2:
                issues.append({
                    "issue_type": "auto_label_region_difference",
                    "reference": f"Reference {label} label region (colon gap={rg}px)",
                    "uploaded": f"Uploaded {label} label region (colon gap={sg}px)",
                    "difference": (
                        f"Label punctuation/colon spacing differs for {label}. "
                        f"Reference gap={rg}px, uploaded gap={sg}px."
                    ),
                    "severity": "Medium",
                    "confidence": 88,
                    "ref_bbox": rr["bbox"],
                    "suspect_bbox": sr["bbox"],
                })

            continue

        visual_score = _visual_difference_score(ref_crop, sus_crop)
        ref_density = _ink_density(ref_crop)
        sus_density = _ink_density(sus_crop)
        density_diff = abs(ref_density - sus_density)

        ref_sharp = _sharpness(ref_crop)
        sus_sharp = _sharpness(sus_crop)
        sharp_ratio = min(ref_sharp, sus_sharp) / max(ref_sharp, sus_sharp, 1.0)

        if visual_score > 35 or density_diff > 0.08 or sharp_ratio < 0.45:
            issues.append({
                "issue_type": "auto_label_region_difference",
                "reference": (
                    f"Reference {label} label region "
                    f"(text='{rr['text']}', density={ref_density:.3f}, sharpness={ref_sharp:.1f})"
                ),
                "uploaded": (
                    f"Uploaded {label} label region "
                    f"(text='{sr['text']}', density={sus_density:.3f}, sharpness={sus_sharp:.1f})"
                ),
                "difference": (
                    f"Important label region differs for {label}. "
                    f"visual score={visual_score:.1f}, density diff={density_diff:.3f}, "
                    f"sharpness ratio={sharp_ratio:.2f}."
                ),
                "severity": "Medium",
                "confidence": 82,
                "ref_bbox": rr["bbox"],
                "suspect_bbox": sr["bbox"],
            })

    return issues
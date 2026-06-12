from typing import List, Dict
from dataclasses import dataclass, asdict
from rapidfuzz import fuzz
import cv2
import numpy as np
import re


@dataclass
class TypographyIssue:
    reference: str
    uploaded: str
    difference: str
    severity: str
    confidence: float
    ref_bbox: list | None
    suspect_bbox: list | None
    issue_type: str = "typography"


def safe_text(box):
    return str(box.get("text", "")).strip()

def ink_density(crop_img):
    if crop_img is None or crop_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    ink = np.count_nonzero(th)
    total = th.shape[0] * th.shape[1]

    return ink / max(1, total)

def safe_bbox(box):
    bbox = box.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return [int(x) for x in bbox]


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    return len(normalize_text(text).split())


def comparable_text(ref_text, sus_text):
    ref_norm = normalize_text(ref_text)
    sus_norm = normalize_text(sus_text)

    if not ref_norm or not sus_norm:
        return False

    if word_count(ref_norm) != word_count(sus_norm):
        return False

    score = max(
        fuzz.ratio(ref_norm, sus_norm),
        fuzz.token_sort_ratio(ref_norm, sus_norm),
    )

    return score >= 88


def crop(image, bbox, pad=4):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def normalize_crop(crop_img):
    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    target_h = 80
    h, w = gray.shape[:2]

    if h <= 0 or w <= 0:
        return None

    scale = target_h / float(h)
    new_w = max(20, int(w * scale))

    gray = cv2.resize(
        gray,
        (new_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th


def ink_gap_profile(binary_img):
    if binary_img is None:
        return None

    col_ink = np.sum(binary_img > 0, axis=0)

    ink_threshold = max(1, int(binary_img.shape[0] * 0.04))
    has_ink = col_ink > ink_threshold

    gaps = []
    current_gap = 0

    for has in has_ink:
        if not has:
            current_gap += 1
        else:
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0

    if current_gap > 0:
        gaps.append(current_gap)

    internal_gaps = gaps[1:-1] if len(gaps) > 2 else []

    ink_cols = int(np.sum(has_ink))
    total_cols = int(len(has_ink))

    if total_cols <= 0:
        return None

    return {
        "internal_gaps": internal_gaps,
        "avg_gap": float(np.mean(internal_gaps)) if internal_gaps else 0.0,
        "max_gap": float(np.max(internal_gaps)) if internal_gaps else 0.0,
        "gap_count": len(internal_gaps),
        "ink_ratio": ink_cols / total_cols,
        "width": total_cols,
    }


def match_boxes(ref_boxes: List[Dict], sus_boxes: List[Dict], min_score=88):
    matches = []
    used_sus = set()

    for rb in ref_boxes:
        rt = safe_text(rb)
        rbbox = safe_bbox(rb)

        if not rt or not rbbox:
            continue

        best_idx = None
        best_score = 0

        for i, sb in enumerate(sus_boxes):
            if i in used_sus:
                continue

            st = safe_text(sb)
            sbbox = safe_bbox(sb)

            if not st or not sbbox:
                continue

            if not comparable_text(rt, st):
                continue

            score = max(
                fuzz.ratio(normalize_text(rt), normalize_text(st)),
                fuzz.token_sort_ratio(normalize_text(rt), normalize_text(st)),
            )

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= min_score:
            used_sus.add(best_idx)
            matches.append((rb, sus_boxes[best_idx], best_score))

    return matches


def compare_letter_spacing(ref_img, sus_img, ref_box, sus_box):
    ref_text = safe_text(ref_box)
    sus_text = safe_text(sus_box)

    if not comparable_text(ref_text, sus_text):
        return None

    ref_bbox = safe_bbox(ref_box)
    sus_bbox = safe_bbox(sus_box)

    if not ref_bbox or not sus_bbox:
        return None

    ref_crop = crop(ref_img, ref_bbox)
    sus_crop = crop(sus_img, sus_bbox)

    ref_bin = normalize_crop(ref_crop)
    sus_bin = normalize_crop(sus_crop)

    ref_profile = ink_gap_profile(ref_bin)
    sus_profile = ink_gap_profile(sus_bin)

    if not ref_profile or not sus_profile:
        return None

    ref_avg = ref_profile["avg_gap"]
    sus_avg = sus_profile["avg_gap"]

    ref_max = ref_profile["max_gap"]
    sus_max = sus_profile["max_gap"]

    if max(ref_avg, sus_avg) < 3.0 and max(ref_max, sus_max) < 6.0:
        return None

    avg_gap_diff = abs(ref_avg - sus_avg)
    max_gap_diff = abs(ref_max - sus_max)

    avg_ratio = avg_gap_diff / max(ref_avg, sus_avg, 1.0)
    max_ratio = max_gap_diff / max(ref_max, sus_max, 1.0)

    if avg_ratio > 0.45 or max_ratio > 0.50:
        direction = "closer/tighter" if sus_avg < ref_avg else "wider/more spaced"

        return asdict(TypographyIssue(
            reference=ref_text,
            uploaded=sus_text,
            difference=(
                f"Visible internal letter spacing change: uploaded text appears {direction}. "
                f"Reference avg gap {ref_avg:.2f}px, uploaded avg gap {sus_avg:.2f}px; "
                f"reference max gap {ref_max:.2f}px, uploaded max gap {sus_max:.2f}px."
            ),
            severity="Medium",
            confidence=80,
            ref_bbox=ref_bbox,
            suspect_bbox=sus_bbox,
            issue_type="letter_spacing",
        ))

    return None


def compare_typography(
    ref_boxes: List[Dict],
    sus_boxes: List[Dict],
    ref_img=None,
    sus_img=None,
) -> List[Dict]:
    issues = []

    matches = match_boxes(ref_boxes, sus_boxes, min_score=88)

    for rb, sb, score in matches:
        ref_text = safe_text(rb)
        sus_text = safe_text(sb)

        if not comparable_text(ref_text, sus_text):
            continue

        ref_bbox = safe_bbox(rb)
        sus_bbox = safe_bbox(sb)

        if not ref_bbox or not sus_bbox:
            continue

        rh = max(1, ref_bbox[3] - ref_bbox[1])
        sh = max(1, sus_bbox[3] - sus_bbox[1])
        rw = max(1, ref_bbox[2] - ref_bbox[0])
        sw = max(1, sus_bbox[2] - sus_bbox[0])

        if ref_img is not None and sus_img is not None:
            ref_crop = crop(ref_img, ref_bbox)
            sus_crop = crop(sus_img, sus_bbox)

            ref_density = ink_density(ref_crop)
            sus_density = ink_density(sus_crop)

            density_change = abs(ref_density - sus_density)

            if density_change > 0.08:
                direction = "bolder/thicker" if sus_density > ref_density else "lighter/thinner"

                issues.append(asdict(TypographyIssue(
                    reference=ref_text,
                    uploaded=sus_text,
                    difference=(
                        f"Visible print boldness/thickness difference. "
                        f"Uploaded appears {direction}. "
                        f"Reference ink density {ref_density:.3f}, "
                        f"uploaded ink density {sus_density:.3f}."
                    ),
                    severity="Medium",
                    confidence=86,
                    ref_bbox=ref_bbox,
                    suspect_bbox=sus_bbox,
                    issue_type="boldness",
                )))

        height_change = abs(rh - sh) / max(rh, sh)
        width_change = abs(rw - sw) / max(rw, sw)

        if height_change > 0.30 and abs(rh - sh) >= 10:
            issues.append(asdict(TypographyIssue(
                reference=ref_text,
                uploaded=sus_text,
                difference=(
                    f"Visible font size change: reference height {rh}px, "
                    f"uploaded height {sh}px"
                ),
                severity="Medium",
                confidence=80,
                ref_bbox=ref_bbox,
                suspect_bbox=sus_bbox,
                issue_type="font_size",
            )))

        if width_change > 0.35 and abs(rw - sw) >= 18:
            issues.append(asdict(TypographyIssue(
                reference=ref_text,
                uploaded=sus_text,
                difference=(
                    f"Visible word width/spacing change: reference width {rw}px, "
                    f"uploaded width {sw}px"
                ),
                severity="Medium",
                confidence=78,
                ref_bbox=ref_bbox,
                suspect_bbox=sus_bbox,
                issue_type="word_width_spacing",
            )))

        if ref_img is not None and sus_img is not None:
            spacing_issue = compare_letter_spacing(
                ref_img,
                sus_img,
                rb,
                sb,
            )
            if spacing_issue:
                issues.append(spacing_issue)

    return issues
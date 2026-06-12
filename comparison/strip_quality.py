from typing import List, Dict
from dataclasses import dataclass, asdict
from rapidfuzz import fuzz
import cv2
import numpy as np
import re


@dataclass
class StripIssue:
    reference: str
    uploaded: str
    difference: str
    severity: str
    confidence: float
    ref_bbox: list | None
    suspect_bbox: list | None
    issue_type: str


IMPORTANT_FIELD_KEYWORDS = [
    "batch", "b.no", "b no", "mfg", "mfd", "exp", "expiry",
    "mrp", "rs", "₹", "price", "lic", "license", "physician",
    "children", "dosage", "composition", "contains"
]


def text(box):
    return str(box.get("text", "")).strip()


def bbox(box):
    b = box.get("bbox")
    if not b or len(b) != 4:
        return None
    return [int(x) for x in b]


def box_h(b):
    return max(1, b[3] - b[1])


def box_w(b):
    return max(1, b[2] - b[0])


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def crop(img, b, pad=5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def normalize_text(s):
    return re.sub(r"\s+", " ", str(s or "").strip())


def alnum_only(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def is_probable_brand(t):
    t2 = normalize_text(t)

    if len(t2) < 3:
        return False

    # Brand text is often uppercase, short, repeated, and not a sentence.
    alpha = re.sub(r"[^A-Za-z]", "", t2)
    if not alpha:
        return False

    uppercase_ratio = sum(1 for c in alpha if c.isupper()) / max(1, len(alpha))

    if uppercase_ratio >= 0.6 and len(t2) <= 25:
        return True

    if "-" in t2 and len(t2) <= 25:
        return True

    return False


def is_important_field(t):
    low = t.lower()
    return any(k in low for k in IMPORTANT_FIELD_KEYWORDS)


def match_boxes(ref_boxes: List[Dict], sus_boxes: List[Dict], min_score=82):
    matches = []
    used = set()

    for rb in ref_boxes:
        rt = text(rb)
        rbbox = bbox(rb)

        if not rt or not rbbox:
            continue

        best_idx = None
        best_score = 0

        for i, sb in enumerate(sus_boxes):
            if i in used:
                continue

            st = text(sb)
            sbbox = bbox(sb)

            if not st or not sbbox:
                continue

            score = max(
                fuzz.ratio(rt.lower(), st.lower()),
                fuzz.partial_ratio(rt.lower(), st.lower()),
                fuzz.token_sort_ratio(rt.lower(), st.lower()),
            )

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= min_score:
            used.add(best_idx)
            matches.append((rb, sus_boxes[best_idx], best_score))

    return matches


def crop_ink_profile(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None

    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return None

    target_h = 80
    scale = target_h / h
    new_w = max(20, int(w * scale))
    gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 11
    )

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    col_ink = np.sum(th > 0, axis=0)
    has_ink = col_ink > max(1, int(th.shape[0] * 0.04))

    gaps = []
    g = 0

    for v in has_ink:
        if not v:
            g += 1
        else:
            if g:
                gaps.append(g)
            g = 0

    if g:
        gaps.append(g)

    internal = gaps[1:-1] if len(gaps) > 2 else []

    ink_ratio = np.sum(has_ink) / max(1, len(has_ink))

    return {
        "avg_gap": float(np.mean(internal)) if internal else 0.0,
        "max_gap": float(np.max(internal)) if internal else 0.0,
        "gap_count": len(internal),
        "ink_ratio": float(ink_ratio),
        "width": len(has_ink),
    }


def detect_internal_brand_spacing(ref_img, sus_img, rb, sb):
    rt, st = text(rb), text(sb)
    rbbox, sbbox = bbox(rb), bbox(sb)

    if not rbbox or not sbbox:
        return None

    # Only for same/similar brand text.
    if not is_probable_brand(rt):
        return None

    if alnum_only(rt) != alnum_only(st):
        return None

    ref_crop = crop(ref_img, rbbox)
    sus_crop = crop(sus_img, sbbox)

    rp = crop_ink_profile(ref_crop)
    sp = crop_ink_profile(sus_crop)

    if not rp or not sp:
        return None

    ref_avg, sus_avg = rp["avg_gap"], sp["avg_gap"]
    ref_max, sus_max = rp["max_gap"], sp["max_gap"]

    if max(ref_avg, sus_avg) < 3 and max(ref_max, sus_max) < 6:
        return None

    avg_ratio = abs(ref_avg - sus_avg) / max(ref_avg, sus_avg, 1)
    max_ratio = abs(ref_max - sus_max) / max(ref_max, sus_max, 1)

    if avg_ratio > 0.40 or max_ratio > 0.45:
        direction = "less/closer" if sus_avg < ref_avg else "more/wider"

        return asdict(StripIssue(
            reference=rt,
            uploaded=st,
            difference=(
                f"Visible internal spacing difference in brand text. "
                f"Uploaded spacing appears {direction} than reference."
            ),
            severity="Medium",
            confidence=82,
            ref_bbox=rbbox,
            suspect_bbox=sbbox,
            issue_type="brand_internal_spacing",
        ))

    return None


def detect_font_size_or_width(rb, sb):
    rt, st = text(rb), text(sb)
    rbbox, sbbox = bbox(rb), bbox(sb)

    if not rbbox or not sbbox:
        return []

    issues = []

    rh, sh = box_h(rbbox), box_h(sbbox)
    rw, sw = box_w(rbbox), box_w(sbbox)

    h_change = abs(rh - sh) / max(rh, sh)
    w_change = abs(rw - sw) / max(rw, sw)

    if h_change > 0.30 and abs(rh - sh) >= 10:
        issues.append(asdict(StripIssue(
            reference=rt,
            uploaded=st,
            difference=f"Visible font size difference. Reference height {rh}px, uploaded height {sh}px.",
            severity="Medium",
            confidence=80,
            ref_bbox=rbbox,
            suspect_bbox=sbbox,
            issue_type="font_size",
        )))

    if w_change > 0.35 and abs(rw - sw) >= 18:
        issues.append(asdict(StripIssue(
            reference=rt,
            uploaded=st,
            difference=f"Visible text width/spacing difference. Reference width {rw}px, uploaded width {sw}px.",
            severity="Medium",
            confidence=78,
            ref_bbox=rbbox,
            suspect_bbox=sbbox,
            issue_type="text_width_spacing",
        )))

    return issues


def detect_alignment_for_important_fields(rb, sb):
    rt, st = text(rb), text(sb)
    rbbox, sbbox = bbox(rb), bbox(sb)

    if not rbbox or not sbbox:
        return None

    if not (is_important_field(rt) or is_probable_brand(rt)):
        return None

    rcx, rcy = center(rbbox)
    scx, scy = center(sbbox)

    x_shift = abs(rcx - scx)
    y_shift = abs(rcy - scy)

    if x_shift > 70 or y_shift > 50:
        return asdict(StripIssue(
            reference=rt,
            uploaded=st,
            difference=f"Visible alignment/position mismatch. x shift {x_shift:.1f}px, y shift {y_shift:.1f}px.",
            severity="Medium",
            confidence=72,
            ref_bbox=rbbox,
            suspect_bbox=sbbox,
            issue_type="alignment",
        ))

    return None


def detect_punctuation_changes(ref_boxes, sus_boxes):
    issues = []
    matches = match_boxes(ref_boxes, sus_boxes, min_score=75)

    punctuation = [".", ",", ":", ";", "-", "/", "(", ")"]

    for rb, sb, _ in matches:
        rt, st = text(rb), text(sb)

        if alnum_only(rt) != alnum_only(st):
            continue

        for p in punctuation:
            if p in rt and p not in st:
                issues.append(asdict(StripIssue(
                    reference=rt,
                    uploaded=st,
                    difference=f"Missing punctuation '{p}'",
                    severity="High",
                    confidence=90,
                    ref_bbox=bbox(rb),
                    suspect_bbox=bbox(sb),
                    issue_type="punctuation",
                )))
            elif p not in rt and p in st:
                issues.append(asdict(StripIssue(
                    reference=rt,
                    uploaded=st,
                    difference=f"Extra punctuation '{p}'",
                    severity="Medium",
                    confidence=85,
                    ref_bbox=bbox(rb),
                    suspect_bbox=bbox(sb),
                    issue_type="punctuation",
                )))

    return issues


def detect_overcoding_colour(ref_img, sus_img, rb, sb):
    rt, st = text(rb), text(sb)
    rbbox, sbbox = bbox(rb), bbox(sb)

    if not rbbox or not sbbox:
        return None

    if not (is_important_field(rt) or is_important_field(st)):
        return None

    ref_crop = crop(ref_img, rbbox, pad=6)
    sus_crop = crop(sus_img, sbbox, pad=6)

    if ref_crop is None or sus_crop is None:
        return None

    ref_hsv = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2HSV)
    sus_hsv = cv2.cvtColor(sus_crop, cv2.COLOR_BGR2HSV)

    ref_mean = np.mean(ref_hsv.reshape(-1, 3), axis=0)
    sus_mean = np.mean(sus_hsv.reshape(-1, 3), axis=0)

    hue_diff = abs(float(ref_mean[0]) - float(sus_mean[0]))
    sat_diff = abs(float(ref_mean[1]) - float(sus_mean[1]))

    if hue_diff > 15 and sat_diff > 25:
        return asdict(StripIssue(
            reference=rt,
            uploaded=st,
            difference="Visible overcoding/print colour difference in important field.",
            severity="Medium",
            confidence=75,
            ref_bbox=rbbox,
            suspect_bbox=sbbox,
            issue_type="overcoding_colour",
        ))

    return None


def detect_strip_quality_issues(
    ref_boxes: List[Dict],
    sus_boxes: List[Dict],
    ref_img,
    sus_img,
) -> List[Dict]:

    issues = []

    matches = match_boxes(ref_boxes, sus_boxes, min_score=82)

    for rb, sb, score in matches:
        spacing_issue = detect_internal_brand_spacing(ref_img, sus_img, rb, sb)
        if spacing_issue:
            issues.append(spacing_issue)

        issues.extend(detect_font_size_or_width(rb, sb))

        alignment_issue = detect_alignment_for_important_fields(rb, sb)
        if alignment_issue:
            issues.append(alignment_issue)

        colour_issue = detect_overcoding_colour(ref_img, sus_img, rb, sb)
        if colour_issue:
            issues.append(colour_issue)

    issues.extend(detect_punctuation_changes(ref_boxes, sus_boxes))

    return issues
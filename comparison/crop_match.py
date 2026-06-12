from rapidfuzz import fuzz
import cv2
import numpy as np


def crop_bbox(img, bbox, pad=6):
    h, w = img.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return img[y1:y2, x1:x2]


def box_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_size(b):
    x1, y1, x2, y2 = b
    return (
        max(1, x2 - x1),
        max(1, y2 - y1),
    )


def position_score(ref_bbox, sus_bbox, img_shape):
    h, w = img_shape[:2]

    rcx, rcy = box_center(ref_bbox)
    scx, scy = box_center(sus_bbox)

    dx = abs(rcx - scx) / max(1, w)
    dy = abs(rcy - scy) / max(1, h)

    dist = (dx * dx + dy * dy) ** 0.5

    return max(0, 100 * (1 - dist * 3))


def size_similarity(ref_bbox, sus_bbox):
    rw, rh = box_size(ref_bbox)
    sw, sh = box_size(sus_bbox)

    width_ratio = min(rw, sw) / max(rw, sw)
    height_ratio = min(rh, sh) / max(rh, sh)

    return 100 * ((width_ratio + height_ratio) / 2)


def match_ocr_boxes(
    ref_boxes,
    sus_boxes,
    image_shape,
    min_score=70,
):
    matches = []
    used = set()

    for rb in ref_boxes:

        ref_text = str(rb.get("text", "")).strip()
        ref_bbox = rb.get("bbox")

        if not ref_text or not ref_bbox:
            continue

        best_idx = None
        best_score = -1

        for i, sb in enumerate(sus_boxes):

            if i in used:
                continue

            sus_text = str(sb.get("text", "")).strip()
            sus_bbox = sb.get("bbox")

            if not sus_text or not sus_bbox:
                continue

            text_score = max(
                fuzz.ratio(
                    ref_text.lower(),
                    sus_text.lower(),
                ),
                fuzz.partial_ratio(
                    ref_text.lower(),
                    sus_text.lower(),
                ),
                fuzz.token_sort_ratio(
                    ref_text.lower(),
                    sus_text.lower(),
                ),
            )

            pos_score = position_score(
                ref_bbox,
                sus_bbox,
                image_shape,
            )

            sz_score = size_similarity(
                ref_bbox,
                sus_bbox,
            )

            final_score = (
                text_score * 0.55
                + pos_score * 0.25
                + sz_score * 0.20
            )

            if final_score > best_score:
                best_score = final_score
                best_idx = i

        if best_idx is not None and best_score >= min_score:
            used.add(best_idx)

            matches.append(
                (
                    rb,
                    sus_boxes[best_idx],
                    best_score,
                )
            )

    return matches


def compare_crop_visual(ref_crop, sus_crop):
    if ref_crop is None or sus_crop is None:
        return None

    if ref_crop.size == 0 or sus_crop.size == 0:
        return None

    sus_crop = cv2.resize(
        sus_crop,
        (
            ref_crop.shape[1],
            ref_crop.shape[0],
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    ref_gray = cv2.cvtColor(
        ref_crop,
        cv2.COLOR_BGR2GRAY,
    )

    sus_gray = cv2.cvtColor(
        sus_crop,
        cv2.COLOR_BGR2GRAY,
    )

    ref_bin = cv2.adaptiveThreshold(
        ref_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    sus_bin = cv2.adaptiveThreshold(
        sus_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    ref_ink = np.mean(ref_bin > 0)
    sus_ink = np.mean(sus_bin > 0)

    ink_diff = abs(ref_ink - sus_ink)

    return {
        "ref_ink": ref_ink,
        "sus_ink": sus_ink,
        "ink_diff": ink_diff,
    }


def detect_crop_based_differences(
    ref_img,
    sus_img,
    ref_boxes,
    sus_boxes,
):
    issues = []

    matches = match_ocr_boxes(
        ref_boxes,
        sus_boxes,
        ref_img.shape,
    )

    for rb, sb, score in matches:

        ref_crop = crop_bbox(
            ref_img,
            rb["bbox"],
        )

        sus_crop = crop_bbox(
            sus_img,
            sb["bbox"],
        )

        metrics = compare_crop_visual(
            ref_crop,
            sus_crop,
        )

        if not metrics:
            continue

        if metrics["ink_diff"] > 0.08:

            issues.append({
                "issue_type": "crop_typography",
                "reference": rb["text"],
                "uploaded": sb["text"],
                "difference": (
                    "Visible print thickness / "
                    "boldness difference detected"
                ),
                "severity": "Medium",
                "confidence": round(
                    min(
                        95,
                        score,
                    ),
                    2,
                ),
                "ref_bbox": rb["bbox"],
                "suspect_bbox": sb["bbox"],
            })

    return issues
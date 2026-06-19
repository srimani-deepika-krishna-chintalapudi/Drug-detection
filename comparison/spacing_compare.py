import re
import numpy as np
from rapidfuzz import fuzz
from comparison.pharma_text_utils import is_critical_pharma_text, is_address_or_location_text


def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_height(bbox):
    return max(1, bbox[3] - bbox[1])


def box_width(bbox):
    return max(1, bbox[2] - bbox[0])


def normalize_text(text):
    return " ".join(str(text or "").lower().strip().split())


def plain_text(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def group_lines(ocr_boxes, y_thresh_ratio=0.65):
    boxes = [
        b for b in ocr_boxes
        if b.get("text") and b.get("bbox") and len(b.get("bbox")) == 4
    ]

    boxes = sorted(boxes, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    lines = []

    for box in boxes:
        cy = box_center(box["bbox"])[1]
        h = box_height(box["bbox"])
        placed = False

        for line in lines:
            if abs(cy - line["cy"]) <= max(h, line["avg_h"]) * y_thresh_ratio:
                line["items"].append(box)
                line["cy"] = np.mean([box_center(i["bbox"])[1] for i in line["items"]])
                line["avg_h"] = np.mean([box_height(i["bbox"]) for i in line["items"]])
                placed = True
                break

        if not placed:
            lines.append({
                "items": [box],
                "cy": cy,
                "avg_h": h,
            })

    output = []

    for line in lines:
        items = sorted(line["items"], key=lambda x: x["bbox"][0])

        x1 = min(i["bbox"][0] for i in items)
        y1 = min(i["bbox"][1] for i in items)
        x2 = max(i["bbox"][2] for i in items)
        y2 = max(i["bbox"][3] for i in items)

        text = " ".join(str(i["text"]) for i in items)

        output.append({
            "text": text,
            "norm_text": normalize_text(text),
            "plain_text": plain_text(text),
            "items": items,
            "bbox": [x1, y1, x2, y2],
            "avg_h": float(np.mean([box_height(i["bbox"]) for i in items])),
        })

    return output


def match_lines(ref_lines, sus_lines, min_score=70):
    pairs = []
    used = set()

    for ref in ref_lines:
        best_idx = None
        best_score = -1

        for idx, sus in enumerate(sus_lines):
            if idx in used:
                continue

            score = fuzz.token_sort_ratio(ref["norm_text"], sus["norm_text"])

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= min_score:
            used.add(best_idx)
            pairs.append((ref, sus_lines[best_idx], best_score))

    return pairs


def word_gaps(line):
    items = line["items"]
    gaps = []

    for i in range(len(items) - 1):
        left = items[i]["bbox"]
        right = items[i + 1]["bbox"]

        gap = right[0] - left[2]
        avg_h = (box_height(left) + box_height(right)) / 2
        norm_gap = gap / max(avg_h, 1)

        gaps.append({
            "left_word": items[i]["text"],
            "right_word": items[i + 1]["text"],
            "norm_gap": norm_gap,
        })

    return gaps


def char_spacing_score(line):
    text = line["plain_text"]
    chars = max(len(text), 1)

    width = box_width(line["bbox"])
    height = max(line["avg_h"], 1)

    return (width / chars) / height


def make_issue(issue_type, reference, uploaded, difference, ref_bbox, sus_bbox, confidence, severity="Medium"):
    return {
        "issue_type": issue_type,
        "reference": reference,
        "uploaded": uploaded,
        "difference": difference,
        "severity": severity,
        "confidence": int(confidence),
        "ref_bbox": ref_bbox,
        "suspect_bbox": sus_bbox,
    }


def detect_spacing_differences(
    authentic_ocr_boxes,
    suspect_ocr_boxes,
    word_gap_threshold=0.45,
    char_spacing_threshold=0.18,
    min_text_similarity=72,
):
    issues = []

    ref_lines = group_lines(authentic_ocr_boxes)
    sus_lines = group_lines(suspect_ocr_boxes)

    matched = match_lines(ref_lines, sus_lines, min_score=min_text_similarity)

    for ref_line, sus_line, score in matched:
        ref_text = ref_line["text"]
        sus_text = sus_line["text"]

        ref_norm = ref_line["norm_text"]
        sus_norm = sus_line["norm_text"]

        ref_plain = ref_line["plain_text"]
        sus_plain = sus_line["plain_text"]

        if len(ref_plain) < 4 or len(sus_plain) < 4:
            continue

        is_addr = is_address_or_location_text(ref_text)
        is_crit = is_critical_pharma_text(ref_text)

        # 1. Same letters, but spaces or separators changed
        if ref_plain == sus_plain and ref_norm != sus_norm:
            if is_addr and not is_crit:
                pass # skip pure space change in address
            else:
                issues.append(make_issue(
                "space_difference",
                ref_text,
                sus_text,
                f"Same text detected but spacing/separator placement changed: '{ref_text}' vs '{sus_text}'.",
                ref_line["bbox"],
                sus_line["bbox"],
                90,
                "Medium",
            ))

        # 2. Word gap difference
        ref_gaps = word_gaps(ref_line)
        sus_gaps = word_gaps(sus_line)

        if len(ref_gaps) == len(sus_gaps) and len(ref_gaps) > 0:
            for ref_gap, sus_gap in zip(ref_gaps, sus_gaps):
                diff = abs(ref_gap["norm_gap"] - sus_gap["norm_gap"])

                word_thresh = word_gap_threshold
                if is_addr and not is_crit:
                    word_thresh = 0.8
                elif not is_crit:
                    word_thresh = 0.65

                if diff >= word_thresh:
                    issues.append(make_issue(
                        "word_width_spacing",
                        ref_text,
                        sus_text,
                        (
                            f"Word spacing changed near "
                            f"'{ref_gap['left_word']} {ref_gap['right_word']}'. "
                            f"Reference gap={ref_gap['norm_gap']:.2f}, "
                            f"suspect gap={sus_gap['norm_gap']:.2f}."
                        ),
                        ref_line["bbox"],
                        sus_line["bbox"],
                        min(95, 70 + diff * 20),
                        "Medium",
                    ))

        # 3. Letter spacing / stretched text
        if ref_plain == sus_plain and score >= 85:
            ref_score = char_spacing_score(ref_line)
            sus_score = char_spacing_score(sus_line)
            diff = abs(ref_score - sus_score)

            char_thresh = char_spacing_threshold
            if is_addr and not is_crit:
                char_thresh = 0.35
            elif not is_crit:
                char_thresh = 0.26

            if diff >= char_thresh:
                issues.append(make_issue(
                    "letter_spacing",
                    ref_text,
                    sus_text,
                    (
                        f"Letter spacing changed. "
                        f"Reference spacing={ref_score:.2f}, "
                        f"suspect spacing={sus_score:.2f}."
                    ),
                    ref_line["bbox"],
                    sus_line["bbox"],
                    min(94, 68 + diff * 35),
                    "Medium",
                ))

    return issues
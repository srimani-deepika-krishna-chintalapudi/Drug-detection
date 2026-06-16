import re
from rapidfuzz import fuzz
from comparison.character_analysis.char_spacing import compare_character_spacing

def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _center_y(box):
    x1, y1, x2, y2 = box["bbox"]
    return (y1 + y2) / 2


def _height(box):
    x1, y1, x2, y2 = box["bbox"]
    return max(1, y2 - y1)


def _group_same_line(boxes):
    valid = [b for b in boxes if b.get("bbox") and str(b.get("text", "")).strip()]
    valid = sorted(valid, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    lines = []

    for b in valid:
        cy = _center_y(b)
        h = _height(b)

        placed = False
        for line in lines:
            if abs(cy - line["cy"]) <= max(12, h * 0.7):
                line["boxes"].append(b)
                line["cy"] = (line["cy"] + cy) / 2
                placed = True
                break

        if not placed:
            lines.append({"cy": cy, "boxes": [b]})

    for line in lines:
        line["boxes"] = sorted(line["boxes"], key=lambda b: b["bbox"][0])

    return lines


def _line_text(line):
    return " ".join(str(b.get("text", "")).strip() for b in line["boxes"])


def _line_bbox(line):
    bs = line["boxes"]
    return [
        min(b["bbox"][0] for b in bs),
        min(b["bbox"][1] for b in bs),
        max(b["bbox"][2] for b in bs),
        max(b["bbox"][3] for b in bs),
    ]


def _gaps(line):
    bs = line["boxes"]
    gaps = []

    for i in range(len(bs) - 1):
        left = bs[i]["bbox"]
        right = bs[i + 1]["bbox"]

        gap = right[0] - left[2]
        h = max(_height(bs[i]), _height(bs[i + 1]))

        gaps.append(gap / max(1, h))

    return gaps


def _best_matching_line(ref_line, sus_lines):
    ref_norm = _clean(_line_text(ref_line))

    best = None
    best_score = 0

    for s in sus_lines:
        sus_norm = _clean(_line_text(s))

        if not ref_norm or not sus_norm:
            continue

        common = len(set(ref_norm) & set(sus_norm))
        total = max(1, len(set(ref_norm) | set(sus_norm)))
        score = common / total

        if score > best_score:
            best_score = score
            best = s

    if best_score >= 0.65:
        return best

    return None


def detect_space_differences(ref_boxes, sus_boxes, ref_img=None, sus_img=None):
    issues = []

    ref_lines = _group_same_line(ref_boxes)
    sus_lines = _group_same_line(sus_boxes)

    for ref_line in ref_lines:
        sus_line = _best_matching_line(ref_line, sus_lines)

        if not sus_line:
            continue

        ref_text = _line_text(ref_line)
        sus_text = _line_text(sus_line)
        if fuzz.ratio(_clean(ref_text), _clean(sus_text)) < 88:
            continue
        if len(ref_text) > 80 or len(sus_text) > 80:
            continue
        if len(_clean(ref_text)) < 6:
            continue

        ref_norm = _clean(ref_text)
        sus_norm = _clean(sus_text)

        # only compare spacing when text content is same after removing spaces/punctuation
        if ref_norm != sus_norm:
            continue

        ref_gaps = _gaps(ref_line)
        sus_gaps = _gaps(sus_line)

        if not ref_gaps or not sus_gaps:
            continue

        n = min(len(ref_gaps), len(sus_gaps))

        for i in range(n):
            diff = abs(ref_gaps[i] - sus_gaps[i])

            if diff >= 0.45:
                issues.append({
                    "issue_type": "space_difference",
                    "reference": ref_text,
                    "uploaded": sus_text,
                    "difference": (
                        f"Visible spacing difference detected between words/characters. "
                        f"Reference normalized gap={ref_gaps[i]:.2f}, "
                        f"uploaded normalized gap={sus_gaps[i]:.2f}."
                    ),
                    "severity": "Medium",
                    "confidence": 86,
                    "ref_bbox": _line_bbox(ref_line),
                    "suspect_bbox": _line_bbox(sus_line),
                })
                break

    return issues
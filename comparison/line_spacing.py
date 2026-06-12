import re
from rapidfuzz import fuzz


def _clean(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t or "").lower()).strip()


def _bbox(b):
    box = b.get("bbox")
    if not box or len(box) != 4:
        return None
    return [int(x) for x in box]


def _line_items(boxes):
    items = []

    for b in boxes:
        text = str(b.get("text", "") or "").strip()
        bbox = _bbox(b)

        if not text or not bbox:
            continue

        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)

        if len(_clean(text)) < 4:
            continue

        items.append({
            "text": text,
            "clean": _clean(text),
            "bbox": bbox,
            "cy": (y1 + y2) / 2,
            "height": h,
        })

    return sorted(items, key=lambda x: x["bbox"][1])


def _match_lines(ref_lines, sus_lines):
    matches = []
    used = set()

    for r in ref_lines:
        best_i = None
        best_score = 0

        for i, s in enumerate(sus_lines):
            if i in used:
                continue

            score = fuzz.token_sort_ratio(r["clean"], s["clean"])

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is not None and best_score >= 75:
            used.add(best_i)
            matches.append((r, sus_lines[best_i], best_score))

    return matches


def detect_line_spacing_differences(ref_boxes, sus_boxes):
    issues = []

    ref_lines = _line_items(ref_boxes)
    sus_lines = _line_items(sus_boxes)

    matches = _match_lines(ref_lines, sus_lines)

    for i in range(len(matches) - 1):
        r1, s1, _ = matches[i]
        r2, s2, _ = matches[i + 1]

        ref_gap = r2["bbox"][1] - r1["bbox"][3]
        sus_gap = s2["bbox"][1] - s1["bbox"][3]

        ref_norm = ref_gap / max(r1["height"], r2["height"], 1)
        sus_norm = sus_gap / max(s1["height"], s2["height"], 1)

        diff = abs(ref_norm - sus_norm)

        if diff >= 0.45 and abs(ref_gap - sus_gap) >= 8:
            issues.append({
                "issue_type": "line_spacing",
                "reference": f"{r1['text']} → {r2['text']} gap={ref_gap}px",
                "uploaded": f"{s1['text']} → {s2['text']} gap={sus_gap}px",
                "difference": (
                    "Vertical spacing between consecutive text lines differs. "
                    f"Reference normalized gap={ref_norm:.2f}, uploaded={sus_norm:.2f}."
                ),
                "severity": "Medium",
                "confidence": 82,
                "ref_bbox": [
                    min(r1["bbox"][0], r2["bbox"][0]),
                    r1["bbox"][1],
                    max(r1["bbox"][2], r2["bbox"][2]),
                    r2["bbox"][3],
                ],
                "suspect_bbox": [
                    min(s1["bbox"][0], s2["bbox"][0]),
                    s1["bbox"][1],
                    max(s1["bbox"][2], s2["bbox"][2]),
                    s2["bbox"][3],
                ],
            })

    return issues
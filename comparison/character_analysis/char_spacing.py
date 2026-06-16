import numpy as np

from comparison.character_analysis.char_segmenter import extract_character_boxes


def measure_character_spacing(crop):
    boxes, _ = extract_character_boxes(crop)

    if len(boxes) < 2:
        return None

    boxes = sorted(boxes, key=lambda b: b[0])

    heights = [max(1, b[3] - b[1]) for b in boxes]
    avg_height = float(np.mean(heights))

    gaps = []

    for i in range(len(boxes) - 1):
        gap = boxes[i + 1][0] - boxes[i][2]

        if gap >= 0:
            gaps.append(gap / max(1, avg_height))

    if len(gaps) < 2:
        return None

    return {
        "char_count": len(boxes),
        "gaps": gaps,
        "median_gap": float(np.median(gaps)),
        "avg_gap": float(np.mean(gaps)),
        "max_gap": float(np.max(gaps)),
    }


def _compare_gap_patterns(ref_gaps, sus_gaps):
    n = min(len(ref_gaps), len(sus_gaps))

    if n < 2:
        return 0.0

    ref = np.array(ref_gaps[:n], dtype=float)
    sus = np.array(sus_gaps[:n], dtype=float)

    return float(np.mean(np.abs(ref - sus)))


def compare_character_spacing(ref_crop, sus_crop, threshold=0.18):
    ref_gaps = measure_character_spacing(ref_crop)
    sus_gaps = measure_character_spacing(sus_crop)

    if not ref_gaps or not sus_gaps:
        return None

    n = min(len(ref_gaps), len(sus_gaps))
    ref_gaps = ref_gaps[:n]
    sus_gaps = sus_gaps[:n]

    diffs = [
        abs(r - s) / max(r, s, 1)
        for r, s in zip(ref_gaps, sus_gaps)
    ]

    max_diff = max(diffs)
    avg_diff = sum(diffs) / len(diffs)

    if max_diff < threshold and avg_diff < threshold * 0.65:
        return None

    return {
        "issue_type": "character_spacing",
        "difference": (
            f"Letter-to-letter spacing differs. "
            f"Max gap change={max_diff:.2f}, avg gap change={avg_diff:.2f}."
        ),
        "severity": "Medium",
        "confidence": int(min(95, 70 + max_diff * 100)),
    }
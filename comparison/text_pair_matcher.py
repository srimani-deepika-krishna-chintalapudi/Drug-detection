from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
import math


def _center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _width(b):
    return max(1, b[2] - b[0])


def _height(b):
    return max(1, b[3] - b[1])


def _clean(text):
    return "".join(str(text).lower().split())


def _line_id(box, img_h):
    """
    Very simple line grouping.
    Can be improved later.
    """
    y = (box["bbox"][1] + box["bbox"][3]) / 2
    return int(y / max(20, img_h * 0.02))


def get_matched_text_pairs(
    ref_boxes,
    sus_boxes,
    ref_shape,
    sus_shape,
):
    """
    Returns:

    [
        {
            "reference": ref_box,
            "suspect": sus_box,
            "score": 92.4
        }
    ]
    """

    ref_h, ref_w = ref_shape[:2]
    sus_h, sus_w = sus_shape[:2]

    cost = []

    for r in ref_boxes:

        rb = r["bbox"]
        rcx, rcy = _center(rb)

        row = []

        for s in sus_boxes:

            sb = s["bbox"]
            scx, scy = _center(sb)

            # --------------------
            # Text similarity
            # --------------------

            sim = fuzz.ratio(
                _clean(r["text"]),
                _clean(s["text"]),
            )

            # --------------------
            # Position
            # --------------------

            dx = abs(rcx / ref_w - scx / sus_w)
            dy = abs(rcy / ref_h - scy / sus_h)

            # --------------------
            # Size
            # --------------------

            w_ratio = min(
                _width(rb),
                _width(sb),
            ) / max(
                _width(rb),
                _width(sb),
            )

            h_ratio = min(
                _height(rb),
                _height(sb),
            ) / max(
                _height(rb),
                _height(sb),
            )

            # --------------------
            # Line
            # --------------------

            line_diff = abs(
                _line_id(r, ref_h)
                - _line_id(s, sus_h)
            )

            score = (
                sim * 0.45
                + (1 - dx) * 20
                + (1 - dy) * 20
                + w_ratio * 8
                + h_ratio * 7
            )

            # Huge penalties

            if line_diff > 1:
                score -= 50

            if dx > 0.20:
                score -= 50

            if dy > 0.08:
                score -= 50

            row.append(-score)

        cost.append(row)

    rows, cols = linear_sum_assignment(cost)

    matches = []

    for r, c in zip(rows, cols):

        score = -cost[r][c]

        if score < 40:
            continue

        matches.append(
            {
                "reference": ref_boxes[r],
                "suspect": sus_boxes[c],
                "score": score,
            }
        )

    return matches
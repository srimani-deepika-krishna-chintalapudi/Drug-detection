'''from typing import List, Dict
import re


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


def box_height(box):
    x1, y1, x2, y2 = box
    return max(1, y2 - y1)


def box_width(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1)


def signature(text):
    text = str(text or "")

    return {
        "chars": len(text),
        "uppercase": sum(c.isupper() for c in text),
        "lowercase": sum(c.islower() for c in text),
        "digits": sum(c.isdigit() for c in text),
        "dots": text.count("."),
        "commas": text.count(","),
        "spaces": text.count(" "),
        "specials": sum(not c.isalnum() and not c.isspace() for c in text),
    }


def signature_difference(ref_sig, sus_sig):
    diffs = []

    for key in ref_sig:
        if ref_sig[key] != sus_sig[key]:
            diffs.append(
                f"{key}: reference={ref_sig[key]}, uploaded={sus_sig[key]}"
            )

    return diffs


def group_boxes_into_lines(boxes):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    heights = [box_height(b["bbox"]) for b in boxes]
    median_h = sorted(heights)[len(heights) // 2] if heights else 20

    lines = []

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        cy = (y1 + y2) / 2

        placed = False

        for line in lines:
            line_cy = sum(
                (b["bbox"][1] + b["bbox"][3]) / 2 for b in line
            ) / max(1, len(line))

            if abs(cy - line_cy) <= median_h * 0.75:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    final_lines = []

    for line in lines:
        line = sorted(line, key=lambda b: b["bbox"][0])

        x1 = min(b["bbox"][0] for b in line)
        y1 = min(b["bbox"][1] for b in line)
        x2 = max(b["bbox"][2] for b in line)
        y2 = max(b["bbox"][3] for b in line)

        text = normalize_text(" ".join(b["text"] for b in line))

        is_vertical = any(b.get("is_vertical") for b in line)

        final_lines.append({
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "boxes": line,
            "is_vertical": is_vertical,
            "signature": signature(text),
        })

    return sorted(final_lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))


def split_columns(lines, image_shape):
    if not lines:
        return []

    h, w = image_shape[:2]

    centers = []
    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        centers.append((x1 + x2) / 2)

    centers = sorted(centers)

    if len(centers) < 2:
        for line in lines:
            line["column_id"] = 1
        return lines

    gaps = []
    for i in range(len(centers) - 1):
        gaps.append((centers[i + 1] - centers[i], centers[i], centers[i + 1]))

    biggest_gap, left_x, right_x = max(gaps, key=lambda x: x[0])

    # Only split if there is a meaningful gap.
    if biggest_gap < w * 0.12:
        for line in lines:
            line["column_id"] = 1
        return lines

    split_x = (left_x + right_x) / 2

    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        cx = (x1 + x2) / 2
        line["column_id"] = 1 if cx < split_x else 2

    return lines


def add_line_numbers(lines):
    grouped = {}

    for line in lines:
        grouped.setdefault(line["column_id"], []).append(line)

    final = []

    for col_id, col_lines in grouped.items():
        col_lines = sorted(col_lines, key=lambda l: l["bbox"][1])

        for idx, line in enumerate(col_lines, 1):
            line["line_no"] = idx
            final.append(line)

    return final


def normalized_bbox(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox

    return [
        x1 / max(1, w),
        y1 / max(1, h),
        x2 / max(1, w),
        y2 / max(1, h),
    ]


def normalized_center(bbox, image_shape):
    nb = normalized_bbox(bbox, image_shape)
    x1, y1, x2, y2 = nb

    return (x1 + x2) / 2, (y1 + y2) / 2


def position_distance(ref_line, sus_line, ref_shape, sus_shape):
    rcx, rcy = normalized_center(ref_line["bbox"], ref_shape)
    scx, scy = normalized_center(sus_line["bbox"], sus_shape)

    dx = abs(rcx - scx)
    dy = abs(rcy - scy)

    return (dx * dx + dy * dy) ** 0.5


def prepare_lines(boxes, image_shape):
    lines = group_boxes_into_lines(boxes)
    lines = split_columns(lines, image_shape)
    lines = add_line_numbers(lines)

    return lines


def compare_ocr_lines(
    ref_boxes: List[Dict],
    sus_boxes: List[Dict],
    ref_shape,
    sus_shape,
):
    issues = []

    ref_lines = prepare_lines(ref_boxes, ref_shape)
    sus_lines = prepare_lines(sus_boxes, sus_shape)

    used_sus = set()

    for ref_line in ref_lines:
        best_idx = None
        best_score = -1

        for i, sus_line in enumerate(sus_lines):
            if i in used_sus:
                continue

            if ref_line.get("is_vertical") != sus_line.get("is_vertical"):
                continue

            # Prefer same column and same line number.
            same_column = ref_line["column_id"] == sus_line["column_id"]
            same_line = ref_line["line_no"] == sus_line["line_no"]

            dist = position_distance(ref_line, sus_line, ref_shape, sus_shape)
            pos_score = max(0, 100 * (1 - dist * 4))

            structure_score = 0
            if same_column:
                structure_score += 30
            if same_line:
                structure_score += 30

            final_score = pos_score + structure_score

            if final_score > best_score:
                best_score = final_score
                best_idx = i

        if best_idx is None or best_score < 45:
            issues.append({
                "issue_type": "line_missing",
                "reference": ref_line["text"],
                "uploaded": "",
                "difference": (
                    f"Line missing or moved: column {ref_line['column_id']}, "
                    f"line {ref_line['line_no']}"
                ),
                "severity": "Medium",
                "confidence": 80,
                "ref_bbox": ref_line["bbox"],
                "suspect_bbox": None,
            })
            continue

        used_sus.add(best_idx)
        sus_line = sus_lines[best_idx]

        ref_text = ref_line["text"]
        sus_text = sus_line["text"]

        if ref_text == sus_text:
            continue

        ref_sig = ref_line["signature"]
        sus_sig = sus_line["signature"]
        sig_diffs = signature_difference(ref_sig, sus_sig)

        if sig_diffs:
            issues.append({
                "issue_type": "line_signature",
                "reference": ref_text,
                "uploaded": sus_text,
                "difference": (
                    f"Column {ref_line['column_id']} line {ref_line['line_no']} differs. "
                    + "; ".join(sig_diffs)
                ),
                "severity": "Medium",
                "confidence": 82,
                "ref_bbox": ref_line["bbox"],
                "suspect_bbox": sus_line["bbox"],
            })

    return issues'''
import math
import re
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment


CRITICAL_KEYWORDS = [
    "mrp", "batch", "bno", "b no", "expiry", "exp", "mfg", "mfd",
    "manufactured", "license", "lic", "composition", "barcode",
    "dolo", "pan", "paracetamol", "pantoprazole", "domperidone",
    "tablet", "tablets", "capsule", "capsules", "mg", "ip"
]

IGNORE_TOKENS = {"", ".", ",", "-", ":", "/", "|", "i", "l", "1", "o", "0"}


def normalize_text(text):
    text = str(text or "").lower()
    text = text.replace("₹", "rs")
    text = text.replace("-", " ")
    text = re.sub(r"[,:;(){}\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_text(text):
    text = normalize_text(text)

    replacements = {
        "m.r.p": "mrp",
        "m r p": "mrp",
        "b.no": "batch",
        "b no": "batch",
        "bno": "batch",
        "exp": "expiry",
        "mfd": "mfg",
        "tab ": "tablet ",
        "tabs": "tablets",
        "ip.": "ip",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def text_similarity(a, b):
    a = canonical_text(a)
    b = canonical_text(b)

    if not a or not b:
        return 0.0

    direct = SequenceMatcher(None, a, b).ratio()

    aw = set(a.split())
    bw = set(b.split())
    overlap = len(aw & bw) / max(1, min(len(aw), len(bw)))

    return max(direct, overlap)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1), max(1, y2 - y1)


def detect_orientation(bbox):
    w, h = bbox_size(bbox)
    return "vertical" if h / w > 2.2 else "horizontal"


def is_critical_text(text):
    t = canonical_text(text)
    return any(k in t for k in CRITICAL_KEYWORDS)


def is_useful_box(box):
    text = canonical_text(box.get("text", ""))
    conf = float(box.get("confidence", box.get("conf", 0.0)) or 0.0)

    if text in IGNORE_TOKENS:
        return False

    if len(text) < 2:
        return False

    if conf < 0.40 and not is_critical_text(text):
        return False

    return True


def merge_boxes_to_lines(boxes, image_shape):
    useful = [b for b in boxes if is_useful_box(b)]

    if not useful:
        return []

    useful = sorted(useful, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    horizontal = []
    vertical = []

    for b in useful:
        if detect_orientation(b["bbox"]) == "vertical":
            vertical.append(b)
        else:
            horizontal.append(b)

    lines = []

    def make_line(line_boxes, orientation):
        x1 = min(b["bbox"][0] for b in line_boxes)
        y1 = min(b["bbox"][1] for b in line_boxes)
        x2 = max(b["bbox"][2] for b in line_boxes)
        y2 = max(b["bbox"][3] for b in line_boxes)

        text = " ".join(str(b.get("text", "")).strip() for b in line_boxes)
        conf = sum(
            float(b.get("confidence", b.get("conf", 0.0)) or 0.0)
            for b in line_boxes
        ) / max(1, len(line_boxes))

        bbox = [x1, y1, x2, y2]

        lines.append({
            "text": text,
            "normalized_text": canonical_text(text),
            "confidence": conf,
            "bbox": bbox,
            "orientation": orientation,
            "is_critical": is_critical_text(text),
            "source_boxes": line_boxes,
        })

    if horizontal:
        avg_h = sum(bbox_size(b["bbox"])[1] for b in horizontal) / max(1, len(horizontal))
        threshold = max(12, avg_h * 0.85)

        temp = []

        for b in horizontal:
            x1, y1, x2, y2 = b["bbox"]
            cy = (y1 + y2) / 2
            placed = False

            for line in temp:
                line_cy = sum((x["bbox"][1] + x["bbox"][3]) / 2 for x in line) / len(line)

                if abs(cy - line_cy) <= threshold:
                    line.append(b)
                    placed = True
                    break

            if not placed:
                temp.append([b])

        for line in temp:
            line = sorted(line, key=lambda b: b["bbox"][0])
            make_line(line, "horizontal")

    for b in vertical:
        make_line([b], "vertical")

    return sorted(lines, key=lambda x: (x["bbox"][1], x["bbox"][0]))


def enrich(lines, image_shape):
    h, w = image_shape[:2]
    enriched = []

    for item in lines:
        bbox = item["bbox"]
        bw, bh = bbox_size(bbox)
        cx, cy = bbox_center(bbox)

        new = dict(item)
        new["norm_center"] = [cx / max(1, w), cy / max(1, h)]
        new["norm_size"] = [bw / max(1, w), bh / max(1, h)]

        enriched.append(new)

    return enriched


def position_distance(a, b):
    ax, ay = a["norm_center"]
    bx, by = b["norm_center"]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def size_difference(a, b):
    aw, ah = a["norm_size"]
    bw, bh = b["norm_size"]

    dw = abs(aw - bw) / max(aw, bw, 1e-6)
    dh = abs(ah - bh) / max(ah, bh, 1e-6)

    return (dw + dh) / 2


def can_match(a, b):
    if a["orientation"] != b["orientation"]:
        return False

    sim = text_similarity(a.get("text", ""), b.get("text", ""))
    dist = position_distance(a, b)
    size_diff = size_difference(a, b)

    if sim >= 0.78:
        return True

    if sim >= 0.62 and dist <= 0.18:
        return True

    if dist > 0.14:
        return False

    if size_diff > 1.20:
        return False

    return sim >= 0.35


def match_cost(a, b):
    if not can_match(a, b):
        return 9999

    pos = position_distance(a, b)
    size = size_difference(a, b)
    sim = text_similarity(a.get("text", ""), b.get("text", ""))

    return ((1 - sim) * 0.70) + (pos * 0.25) + (size * 0.05)

def should_report_text_mismatch(a, b, sim):
    a_text = a.get("normalized_text", "")
    b_text = b.get("normalized_text", "")

    if not a_text or not b_text:
        return False

    if a_text == b_text:
        return False

    # Always report number/code changes
    if any(ch.isdigit() for ch in a_text + b_text):
        return True

    important_words = [
        "village", "khurd", "bhatauli", "bhatouli",
        "manufactured", "manufacturer", "india", "baddi",
        "solan", "mumbai", "alkem", "abbott",
        "contains", "dosage", "physician"
    ]

    combined = f"{a_text} {b_text}".lower()

    if any(w in combined for w in important_words):
        return True

    if a_text in b_text or b_text in a_text:
        return False

    return sim < 0.72

def strict_match_ocr(ref_boxes, sus_boxes, ref_shape, sus_shape):
    ref_lines = merge_boxes_to_lines(ref_boxes, ref_shape)
    sus_lines = merge_boxes_to_lines(sus_boxes, sus_shape)

    ref = enrich(ref_lines, ref_shape)
    sus = enrich(sus_lines, sus_shape)

    issues = []

    if not ref:
        return issues

    if not sus:
        return [{
            "issue_type": "missing_text",
            "reference": r.get("text", ""),
            "uploaded": "",
            "difference": "Text missing in suspect image",
            "severity": "High" if r["is_critical"] else "Medium",
            "confidence": 90,
            "ref_bbox": r.get("bbox"),
            "suspect_bbox": None,
        } for r in ref if r["is_critical"]]

    cost_matrix = [[match_cost(r, s) for s in sus] for r in ref]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_ref = set()
    matched_sus = set()

    for r_idx, s_idx in zip(row_ind, col_ind):
        cost = cost_matrix[r_idx][s_idx]

        if cost >= 9999:
            continue

        r = ref[r_idx]
        s = sus[s_idx]

        matched_ref.add(r_idx)
        matched_sus.add(s_idx)

        sim = text_similarity(r.get("text", ""), s.get("text", ""))
        dist = position_distance(r, s)
        size_diff = size_difference(r, s)

        if should_report_text_mismatch(r, s, sim):
            issues.append({
                "issue_type": "text_mismatch",
                "reference": r.get("text", ""),
                "uploaded": s.get("text", ""),
                "difference": f"Text mismatch. Similarity={sim:.2f}",
                "severity": "High" if r["is_critical"] else "Medium",
                "confidence": int((1 - sim) * 100),
                "ref_bbox": r.get("bbox"),
                "suspect_bbox": s.get("bbox"),
            })

        if (
            r["orientation"] == "vertical"
            and s["orientation"] == "vertical"
            and sim >= 0.65
        ):
            rx, ry = r["norm_center"]
            sx, sy = s["norm_center"]

            x_shift = abs(rx - sx)
            y_shift = abs(ry - sy)

            if x_shift > 0.025 or y_shift > 0.035:
                issues.append({
                    "issue_type": "vertical_text_alignment",
                    "reference": r.get("text", ""),
                    "uploaded": s.get("text", ""),
                    "difference": (
                        f"Vertical text alignment changed. "
                        f"x shift={x_shift:.3f}, y shift={y_shift:.3f}"
                    ),
                    "severity": "Medium",
                    "confidence": 78,
                    "ref_bbox": r.get("bbox"),
                    "suspect_bbox": s.get("bbox"),
                })

        if (r["is_critical"] or s["is_critical"]) and sim >= 0.65 and dist > 0.085:
            issues.append({
                "issue_type": "position_mismatch",
                "reference": r.get("text", ""),
                "uploaded": s.get("text", ""),
                "difference": f"Important text position changed. Distance={dist:.3f}",
                "severity": "Medium",
                "confidence": 75,
                "ref_bbox": r.get("bbox"),
                "suspect_bbox": s.get("bbox"),
            })

        if (r["is_critical"] or s["is_critical"]) and sim >= 0.65 and size_diff > 0.95:
            issues.append({
                "issue_type": "size_mismatch",
                "reference": r.get("text", ""),
                "uploaded": s.get("text", ""),
                "difference": f"Important text box size changed. Difference={size_diff:.2f}",
                "severity": "Medium",
                "confidence": 70,
                "ref_bbox": r.get("bbox"),
                "suspect_bbox": s.get("bbox"),
            })

    for i, r in enumerate(ref):
        if i in matched_ref:
            continue

        if not r["is_critical"]:
            continue

        best_sim = 0.0
        best_s = None

        for s in sus:
            if s["orientation"] != r["orientation"]:
                continue

            sim = text_similarity(r.get("text", ""), s.get("text", ""))

            if sim > best_sim:
                best_sim = sim
                best_s = s

        if best_sim >= 0.55 and best_s is not None:
            issues.append({
                "issue_type": "position_mismatch",
                "reference": r.get("text", ""),
                "uploaded": best_s.get("text", ""),
                "difference": f"Critical text found but position/line grouping changed. Similarity={best_sim:.2f}",
                "severity": "Medium",
                "confidence": 72,
                "ref_bbox": r.get("bbox"),
                "suspect_bbox": best_s.get("bbox"),
            })
            continue

        issues.append({
            "issue_type": "missing_text",
            "reference": r.get("text", ""),
            "uploaded": "",
            "difference": "Critical reference text not found in uploaded image",
            "severity": "High",
            "confidence": 85,
            "ref_bbox": r.get("bbox"),
            "suspect_bbox": None,
        })

    for j, s in enumerate(sus):
        if j in matched_sus:
            continue

        if not s["is_critical"]:
            continue

        best_sim = 0.0

        for r in ref:
            if r["orientation"] != s["orientation"]:
                continue

            sim = text_similarity(r.get("text", ""), s.get("text", ""))

            if sim > best_sim:
                best_sim = sim

        if best_sim >= 0.55:
            continue

        issues.append({
            "issue_type": "extra_text",
            "reference": "",
            "uploaded": s.get("text", ""),
            "difference": "Extra critical text found in uploaded image",
            "severity": "High",
            "confidence": 80,
            "ref_bbox": None,
            "suspect_bbox": s.get("bbox"),
        })

    return issues
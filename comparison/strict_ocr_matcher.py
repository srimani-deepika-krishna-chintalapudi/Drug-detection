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

    dx = ax - bx
    dy = ay - by

    return (dx**2 + dy**2) ** 0.5

def size_difference(a, b):
    aw, ah = a["norm_size"]
    bw, bh = b["norm_size"]

    dw = abs(aw - bw) / max(aw, bw, 1e-6)
    dh = abs(ah - bh) / max(ah, bh, 1e-6)

    return (dw + dh) / 2

def extract_key_tokens(text):
    text = canonical_text(text)
    tokens = text.split()

    return set([
        t for t in tokens
        if len(t) > 2 and t not in IGNORE_TOKENS
    ])

def iou(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)

    return inter_area / float(a_area + b_area - inter_area)

def can_match(a, b):
    if a["orientation"] != b["orientation"]:
        return False
    if extract_key_tokens(a["text"]) != extract_key_tokens(b["text"]):
        if text_similarity(a["text"], b["text"]) < 0.7:
            return False
    
    dist = position_distance(a, b)
    overlap = iou(a, b)

    # HARD spatial reject
    if dist > 0.18:
        return False

    if overlap < 0.02:
        return False

    return True

def match_cost(a, b):
    if not can_match(a, b):
        return 9999

    tokens_a = extract_key_tokens(a.get("text", ""))
    tokens_b = extract_key_tokens(b.get("text", ""))

    if not tokens_a or not tokens_b:
        sim = 0.0
    else:
        sim = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    dist = position_distance(a, b)
    overlap = iou(a, b)
    size = size_difference(a, b)

    return (
        (1 - sim) * 0.5 +
        dist * 0.3 +
        (1 - overlap) * 0.15 +
        size * 0.05
    )

def normalize_strict(text):
    text = str(text or "").upper()
    text = re.sub(r'[^A-Z0-9]', '', text)  # remove spaces, dots, punctuation
    return text


def should_report_text_mismatch(a, b, sim):
    a_text = a.get("normalized_text", "")
    b_text = b.get("normalized_text", "")

    if not a_text or not b_text:
        return False

    # 🔴 NEW: ignore very short text (major FP source)
    if len(a_text) < 6 or len(b_text) < 6:
        return False

    # 🔴 normalize aggressively
    a_clean = normalize_strict(a_text)
    b_clean = normalize_strict(b_text)

    # exact match after normalization → ignore
    if a_clean == b_clean:
        return False

    # substring case → ignore (your MFG.FEB case)
    if a_clean in b_clean or b_clean in a_clean:
        return False

    # 🔴 relaxed similarity threshold
    if sim >= 0.90:
        return False

    # numbers still matter (batch, date etc.)
    if any(ch.isdigit() for ch in a_clean + b_clean):
        return sim < 0.95

    return sim < 0.80

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

    MAX_COST = 0.40  # tune between 0.35–0.45

    for r_idx, s_idx in zip(row_ind, col_ind):
        cost = cost_matrix[r_idx][s_idx]

        if cost >= MAX_COST:
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
            and sim >= 0.75
            and not (normalize_strict(r["text"]) == normalize_strict(s["text"]))
        ):
            rx, ry = r["norm_center"]
            sx, sy = s["norm_center"]

            x_shift = abs(rx - sx)
            y_shift = abs(ry - sy)

            if x_shift > 0.05 or y_shift > 0.30:
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

        if (r["is_critical"] or s["is_critical"]) and sim >= 0.80 and dist > 0.15:
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

        if (r["is_critical"] or s["is_critical"]) and sim >= 0.65 and size_diff > 1.3:
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
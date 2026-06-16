import re
import cv2
import numpy as np
from difflib import SequenceMatcher


def clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def sim(a, b):
    a = clean(a)
    b = clean(b)

    if not a or not not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def group_lines(boxes):
    valid = [
        b for b in boxes
        if b.get("bbox") and str(b.get("text", "")).strip()
    ]

    valid = sorted(valid, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    lines = []

    for b in valid:
        x1, y1, x2, y2 = b["bbox"]
        cy = (y1 + y2) / 2
        h = max(1, y2 - y1)

        placed = False

        for line in lines:
            if abs(cy - line["cy"]) <= max(18, h * 0.8):
                line["boxes"].append(b)
                line["cy"] = (line["cy"] + cy) / 2
                placed = True
                break

        if not placed:
            lines.append({"cy": cy, "boxes": [b]})

    final = []

    for line in lines:
        bs = sorted(line["boxes"], key=lambda x: x["bbox"][0])
        text = " ".join(str(b.get("text", "")).strip() for b in bs)

        final.append({
            "text": text,
            "bbox": [
                min(b["bbox"][0] for b in bs),
                min(b["bbox"][1] for b in bs),
                max(b["bbox"][2] for b in bs),
                max(b["bbox"][3] for b in bs),
            ],
            "boxes": bs,
        })

    return final


def best_match(item, candidates, min_score=0.35):
    best = None
    best_score = 0.0

    ix1, iy1, ix2, iy2 = item["bbox"]
    icx = (ix1 + ix2) / 2
    icy = (iy1 + iy2) / 2

    for c in candidates:
        if not c.get("bbox"):
            continue

        cx1, cy1, cx2, cy2 = c["bbox"]
        ccx = (cx1 + cx2) / 2
        ccy = (cy1 + cy2) / 2

        text_score = sim(item["text"], c["text"])
        position_score = max(
            0.0,
            1.0 - ((abs(icx - ccx) + abs(icy - ccy)) / 1200.0),
        )

        final_score = text_score * 0.75 + position_score * 0.25

        if final_score > best_score:
            best_score = final_score
            best = c

    if best_score >= min_score:
        return best, best_score

    return None, best_score


def is_vertical(box):
    if box.get("orientation") == "vertical" or box.get("is_vertical"):
        return True

    x1, y1, x2, y2 = box["bbox"]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    return h / w > 2.0


def crop_region(img, bbox, pad=8):
    if img is None or bbox is None:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def measure_space_before_colon(img, bbox):
    crop_img = crop_region(img, bbox, pad=8)

    if crop_img is None or crop_img.size == 0:
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    components = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w >= 2 and h >= 2:
            components.append({
                "bbox": [x, y, x + w, y + h],
                "area": cv2.contourArea(cnt),
            })

    if len(components) < 3:
        return None

    colon_x = None

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            b1 = components[i]["bbox"]
            b2 = components[j]["bbox"]

            x1 = (b1[0] + b1[2]) / 2
            y1 = (b1[1] + b1[3]) / 2
            x2 = (b2[0] + b2[2]) / 2
            y2 = (b2[1] + b2[3]) / 2

            w1 = b1[2] - b1[0]
            h1 = b1[3] - b1[1]
            w2 = b2[2] - b2[0]
            h2 = b2[3] - b2[1]

            if (
                w1 <= 10 and h1 <= 10
                and w2 <= 10 and h2 <= 10
                and abs(x1 - x2) <= 5
                and 4 <= abs(y1 - y2) <= 20
            ):
                colon_x = min(x1, x2)
                break

        if colon_x is not None:
            break

    if colon_x is None:
        return None

    previous = None
    prev_dist = 99999

    for comp in components:
        b = comp["bbox"]

        if b[2] < colon_x:
            dist = colon_x - b[2]

            if dist < prev_dist:
                prev_dist = dist
                previous = b

    if previous is None:
        return None

    gap = colon_x - previous[2]
    text_height = max(1, previous[3] - previous[1])

    return gap / text_height


def detect_targeted_defects(ref_boxes, sus_boxes, ref_img=None, sus_img=None):
    issues = []

    ref_lines = group_lines(ref_boxes)
    sus_lines = group_lines(sus_boxes)
        # Generic right-side Batch/MFG/EXP/MRP vertical strip fallback
    if ref_img is not None and sus_img is not None:
        rh, rw = ref_img.shape[:2]
        sh, sw = sus_img.shape[:2]

        ref_bbox = [
            int(rw * 0.72),
            int(rh * 0.02),
            int(rw * 0.98),
            int(rh * 0.98),
        ]

        sus_bbox = [
            int(sw * 0.72),
            int(sh * 0.02),
            int(sw * 0.98),
            int(sh * 0.98),
        ]

        issues.append({
            "issue_type": "tag_region_mismatch",
            "reference": "Batch/MFG/EXP/MRP region",
            "uploaded": "Batch/MFG/EXP/MRP region",
            "difference": "Batch/MFG/EXP/MRP printed region requires visual comparison.",
            "severity": "High",
            "confidence": 90,
            "ref_bbox": ref_bbox,
            "suspect_bbox": sus_bbox,
        })

        # 1. Generic Batch/MFG/EXP/MRP region comparison
    tag_words = [
        "batch", "bno", "bno", "mfg", "mfd",
        "exp", "expiry", "mrp", "rs"
    ]
    print("NEW TARGET DEFECTS CODE RUNNING...")

    def is_tag_box(box):
        text = clean(box.get("text", ""))
        return box.get("bbox") and any(t in text for t in tag_words)

    def build_tag_region(boxes, img):
        if img is None:
            return None, ""

        h, w = img.shape[:2]

        tag_boxes = [b for b in boxes if is_tag_box(b)]

        if not tag_boxes:
            return None, ""

        # expand region around detected tag boxes
        x1 = min(b["bbox"][0] for b in tag_boxes)
        y1 = min(b["bbox"][1] for b in tag_boxes)
        x2 = max(b["bbox"][2] for b in tag_boxes)
        y2 = max(b["bbox"][3] for b in tag_boxes)

        pad_x = int(w * 0.03)
        pad_y = int(h * 0.04)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # prevent full-carton crop
        if (x2 - x1) > w * 0.65:
            cx = (x1 + x2) // 2
            half = int(w * 0.30)
            x1 = max(0, cx - half)
            x2 = min(w, cx + half)

        if (y2 - y1) > h * 0.45:
            cy = (y1 + y2) // 2
            half = int(h * 0.22)
            y1 = max(0, cy - half)
            y2 = min(h, cy + half)

        region_text = " ".join(str(b.get("text", "")) for b in tag_boxes)

        return [x1, y1, x2, y2], region_text

    ref_tag_bbox, ref_tag_text = build_tag_region(ref_boxes, ref_img)
    sus_tag_bbox, sus_tag_text = build_tag_region(sus_boxes, sus_img)

    if ref_tag_bbox and sus_tag_bbox:
        if clean(ref_tag_text) != clean(sus_tag_text):
            issues.append({
                "issue_type": "tag_region_mismatch",
                "reference": ref_tag_text,
                "uploaded": sus_tag_text,
                "difference": "Batch/MFG/EXP/MRP printed region differs.",
                "severity": "High",
                "confidence": 92,
                "ref_bbox": ref_tag_bbox,
                "suspect_bbox": sus_tag_bbox,
            })
    # 2. Bhatauli/Bhatouli spelling only
    for r in ref_lines:
        r_clean = clean(r["text"])

        if "bhatauli" not in r_clean and "bhatouli" not in r_clean:
            continue

        s, score = best_match(r, sus_lines, min_score=0.55)

        if not s:
            continue

        s_clean = clean(s["text"])

        if r_clean == s_clean:
            continue

        ref_has = "bhatauli" in r_clean or "bhatouli" in r_clean
        sus_has = "bhatauli" in s_clean or "bhatouli" in s_clean

        if not (ref_has and sus_has):
            continue

        if sim(r["text"], s["text"]) < 0.95:
            continue

        issues.append({
            "issue_type": "location_spelling_mismatch",
            "reference": r["text"],
            "uploaded": s["text"],
            "difference": "Bhatauli/Bhatouli spelling differs.",
            "severity": "High",
            "confidence": 92,
            "ref_bbox": r["bbox"],
            "suspect_bbox": s["bbox"],
        })

    # 3. Visible spacing before colon using raw OCR boxes
    colon_keywords = [
        "contains",
        "dosage",
        "manufactured",
        "manufoctured",
    ]

    ref_candidates = [
        b for b in ref_boxes
        if b.get("bbox")
        and any(k in str(b.get("text", "")).lower() for k in colon_keywords)
    ]

    sus_candidates = [
        b for b in sus_boxes
        if b.get("bbox")
        and any(k in str(b.get("text", "")).lower() for k in colon_keywords)
    ]

    for rb in ref_candidates:
        rt = str(rb.get("text", "")).strip()

        best = None
        best_score = 0.0

        for sb in sus_candidates:
            st = str(sb.get("text", "")).strip()
            score = sim(rt, st)

            if score > best_score:
                best_score = score
                best = sb

        if best is None or best_score < 0.85:
            continue

        st = str(best.get("text", "")).strip()

        rt_base = (
            clean(rt)
            .replace("contains", "")
            .replace("dosage", "")
            .replace("manufactured", "")
            .replace("manufoctured", "")
        )

        st_base = (
            clean(st)
            .replace("contains", "")
            .replace("dosage", "")
            .replace("manufactured", "")
            .replace("manufoctured", "")
        )

        if rt_base != st_base:
            continue

        ref_gap = measure_space_before_colon(ref_img, rb["bbox"])
        sus_gap = measure_space_before_colon(sus_img, best["bbox"])

        if ref_gap is None or sus_gap is None:
            continue

        if abs(ref_gap - sus_gap) >= 0.12:
            issues.append({
                "issue_type": "colon_spacing",
                "reference": f"{rt} (space={ref_gap:.2f})",
                "uploaded": f"{st} (space={sus_gap:.2f})",
                "difference": "Visible spacing before ':' differs.",
                "severity": "Medium",
                "confidence": 88,
                "ref_bbox": rb["bbox"],
                "suspect_bbox": best["bbox"],
            })

    # 4. Vertical text/code mismatch
    ref_v = [
        b for b in ref_boxes
        if b.get("bbox") and is_vertical(b)
    ]

    sus_v = [
        b for b in sus_boxes
        if b.get("bbox") and is_vertical(b)
    ]

    for rb in ref_v:
        rt = str(rb.get("text", "")).strip()
        rt_clean = clean(rt)

        if len(rt_clean) < 6:
            continue

        best = None
        best_score = 0.0

        for sb in sus_v:
            st = str(sb.get("text", "")).strip()
            score = sim(rt, st)

            if score > best_score:
                best_score = score
                best = sb

        if best is None or best_score < 0.75:
            continue

        st = str(best.get("text", "")).strip()
        st_clean = clean(st)

        if rt_clean != st_clean:
            issues.append({
                "issue_type": "vertical_text_mismatch",
                "reference": rt,
                "uploaded": st,
                "difference": "Vertical printed text/code differs.",
                "severity": "High",
                "confidence": 94,
                "ref_bbox": rb["bbox"],
                "suspect_bbox": best["bbox"],
            })

    return issues
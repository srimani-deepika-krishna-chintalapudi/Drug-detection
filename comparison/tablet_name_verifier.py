import re
import numpy as np
from rapidfuzz import fuzz

print("TABLET_NAME_VERIFIER FILE LOADED")

def _clean(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

def _get_bbox(box):
    bbox = box.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return [int(v) for v in bbox]

def derive_tablet_name(ref_ocr_boxes):
    candidates = []
    blacklist = {
        "warning", "caution", "composition", "contains", "schedule", "mfg", "exp",
        "batch", "dosage", "physician", "mrp", "lic", "licence", "license", "marketed",
        "manufactured", "store", "b.no", "bno", "lot", "expiry", "retail", "price",
        "keep out of reach", "directed", "swallowed", "chewed"
    }
    for box in ref_ocr_boxes:
        text = str(box.get("text", "")).strip()
        bbox = box.get("bbox")
        if not text or not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if h < 10 or w < 10:
            continue
        if len(text) < 4:
            continue
            
        text_lower = text.lower()
        if any(word in text_lower for word in blacklist):
            continue
            
        # Prioritize by bounding box area (largest text block usually is the brand/medicine name)
        area = w * h
        candidates.append((area, text))
    
    candidates.sort(reverse=True, key=lambda x: x[0])
    print("Tablet fallback candidates:", candidates[:8])
    if candidates:
        final_target = candidates[0][1]
        print("Tablet target name used:", final_target)
        return final_target
    return None

def _find_best_name_box(ocr_boxes, medicine_name, label=""):
    target = _clean(medicine_name)
    target_digits = re.sub(r"\D+", "", target)

    best = None
    best_score = -1

    print(f"---- {label} Tablet Candidate Scores ----")
    print("target:", medicine_name, "| cleaned:", target)

    for box in ocr_boxes:
        text = str(box.get("text", "")).strip()
        clean_text = _clean(text)

        if not clean_text:
            continue

        # If target has strength like 16, 500, 650, prefer boxes containing that number.
        cand_digits = re.sub(r"\D+", "", clean_text)
        if target_digits and target_digits not in cand_digits:
            digit_penalty = 35
        else:
            digit_penalty = 0

        # Do NOT use partial_ratio here. It made "Vertin" match "Vertin 16" as 100.
        score = fuzz.ratio(target, clean_text)

        # Prefer full medicine-name boxes, not vertical side brand-only boxes.
        if box.get("is_vertical") or box.get("orientation") == "vertical":
            score -= 20

        score -= digit_penalty

        # Prefer candidates that contain the full cleaned target.
        if target in clean_text or clean_text in target:
            score += 10

        print(score, "|", text, "|", box.get("bbox"))

        if score > best_score:
            best_score = score
            best = box

    print("BEST:", best_score, "|", best)
    print("----------------------------------------")

    if best is None or best_score < 55:
        return None, best_score

    return best, best_score

def get_anchor_distances(ref_boxes, sus_boxes):
    anchor_kws = ["batch", "mrp", "exp", "mfg", "lic"]
    
    ref_anchors = {}
    sus_anchors = {}
    
    for kw in anchor_kws:
        for b in ref_boxes:
            if kw in _clean(b.get("text", "")):
                ref_anchors[kw] = _get_bbox(b)
                break
        for b in sus_boxes:
            if kw in _clean(b.get("text", "")):
                sus_anchors[kw] = _get_bbox(b)
                break
                
    common_kws = [k for k in anchor_kws if k in ref_anchors and k in sus_anchors]
    
    if len(common_kws) >= 2:
        k1, k2 = common_kws[0], common_kws[1]
        rb1, rb2 = ref_anchors[k1], ref_anchors[k2]
        sb1, sb2 = sus_anchors[k1], sus_anchors[k2]
        
        def center(b):
            return ((b[0]+b[2])/2, (b[1]+b[3])/2)
            
        rc1, rc2 = center(rb1), center(rb2)
        sc1, sc2 = center(sb1), center(sb2)
        
        ref_dist = ((rc1[0]-rc2[0])**2 + (rc1[1]-rc2[1])**2)**0.5
        sus_dist = ((sc1[0]-sc2[0])**2 + (sc1[1]-sc2[1])**2)**0.5
        
        if ref_dist > 10 and sus_dist > 10:
            return ref_dist, sus_dist
            
    return None, None

def verify_tablet_name(
    ref_ocr_boxes,
    sus_ocr_boxes,
    reference_image,
    suspect_image,
    medicine_name,
    threshold_percent=5.0,
):
    if reference_image is None or suspect_image is None:
        print("Tablet verifier skipped: image is None")
        return []

    # If medicine_name is empty, derive it from reference OCR
    if not medicine_name:
        medicine_name = derive_tablet_name(ref_ocr_boxes)
        print("Tablet verifier derived medicine_name:", medicine_name)

    if not medicine_name:
        print("Tablet verifier skipped: medicine_name is empty and could not be derived")
        return []

    ref_h, ref_w = reference_image.shape[:2]
    sus_h, sus_w = suspect_image.shape[:2]

    ref_box, ref_score = _find_best_name_box(
        ref_ocr_boxes,
        medicine_name,
        label="REFERENCE",
    )
    sus_box, sus_score = _find_best_name_box(
        sus_ocr_boxes,
        medicine_name,
        label="SUSPECT",
    )

    if ref_box is None or sus_box is None:
        print("Tablet verifier skipped: best box not found")
        print("ref_score:", ref_score, "sus_score:", sus_score)
        return []

    ref_text = str(ref_box.get("text", ""))
    sus_text = str(sus_box.get("text", ""))

    ref_bbox = _get_bbox(ref_box)
    sus_bbox = _get_bbox(sus_box)

    if ref_bbox is None or sus_bbox is None:
        print("Tablet verifier skipped: bbox missing")
        return []

    ref_tablet_width = max(1, ref_bbox[2] - ref_bbox[0])
    sus_tablet_width = max(1, sus_bbox[2] - sus_bbox[0])

    # Try to calculate normalization using anchor distances
    ref_dist, sus_dist = get_anchor_distances(ref_ocr_boxes, sus_ocr_boxes)
    
    if ref_dist and sus_dist:
        print(f"Tablet verifier: Normalizing using anchor distances: ref_dist={ref_dist:.2f}, sus_dist={sus_dist:.2f}")
        ref_span = ref_tablet_width / ref_dist
        sus_span = sus_tablet_width / sus_dist
    else:
        # Fallback to image width normalization
        print("Tablet verifier: Fallback to image width normalization")
        ref_span = ref_tablet_width / ref_w
        sus_span = sus_tablet_width / sus_w

    variance_percent = abs(ref_span - sus_span) / max(ref_span, 1e-6) * 100

    ref_clean = _clean(ref_text)
    sus_clean = _clean(sus_text)

    print("========== TABLET NAME DEBUG ==========")
    print("medicine_name:", medicine_name)
    print("ref_text:", ref_text)
    print("sus_text:", sus_text)
    print("ref_clean:", ref_clean)
    print("sus_clean:", sus_clean)
    print("ref_bbox:", ref_bbox)
    print("sus_bbox:", sus_bbox)
    print("ref_span:", ref_span)
    print("sus_span:", sus_span)
    print("variance_percent:", variance_percent)
    print("threshold_percent:", threshold_percent)
    print("======================================")

    # Classification
    is_text_identical = (ref_clean == sus_clean)
    
    if is_text_identical and variance_percent <= threshold_percent:
        # MATCH or MINOR VARIANCE: OK
        return []

    x1, y1, x2, y2 = sus_bbox
    reason = "Tablet name text is different or spacing/span exceeds tolerance."

    issue = {
        "issue_type": "tablet_name_mismatch",
        "category": "tablet_name_verification",
        "severity": "high",
        "reference_text": ref_text,
        "suspect_text": sus_text,
        "reference_span": round(ref_span, 4),
        "suspect_span": round(sus_span, 4),
        "variance_percent": round(variance_percent, 2),
        "status": "DIFFERENT",
        "reason": reason,
        "ref_bbox": ref_bbox,
        "suspect_bbox": sus_bbox,
        "bbox": sus_bbox,
        "fault_bbox_yxyx": [y1, x1, y2, x2],
        "confidence": round(min(ref_score, sus_score) / 100, 2),
        # Ensuring compatibility with _strict_meaningful_issue_filter without breaking it
        "difference": f"{reason} Reference name: '{ref_text}' | Uploaded name: '{sus_text}' | Variance: {variance_percent:.2f}%",
        "reference": ref_text,
        "uploaded": sus_text
    }

    return [issue]
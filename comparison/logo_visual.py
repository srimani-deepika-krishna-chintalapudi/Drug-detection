import cv2
import numpy as np


def _blue_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # blue / purple printed brand ink
    lower = np.array([90, 40, 40])
    upper = np.array([150, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _blue_components(img):
    mask = _blue_mask(img)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    comps = []

    h, w = img.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if area < 80:
            continue

        if bw < 20 or bh < 10:
            continue

        comps.append({
            "bbox": [x, y, x + bw, y + bh],
            "area": area,
            "center": [(x + bw / 2) / w, (y + bh / 2) / h],
            "size": [bw / w, bh / h],
        })

    return comps


def _nearest_component(comp, others):
    best = None
    best_dist = 999

    cx, cy = comp["center"]

    for o in others:
        ox, oy = o["center"]
        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5

        if dist < best_dist:
            best_dist = dist
            best = o

    return best, best_dist


def detect_logo_visual_differences(ref_img, sus_img):
    issues = []

    ref_comps = _blue_components(ref_img)
    sus_comps = _blue_components(sus_img)

    if not ref_comps or not sus_comps:
        return issues

    for r in ref_comps:
        s, dist = _nearest_component(r, sus_comps)

        if s is None:
            continue

        rw, rh = r["size"]
        sw, sh = s["size"]

        width_diff = abs(rw - sw) / max(rw, sw, 1e-6)
        height_diff = abs(rh - sh) / max(rh, sh, 1e-6)

        # logo/blue brand mark shifted or scaled
        if dist > 0.045 or width_diff > 0.35 or height_diff > 0.35:
            issues.append({
                "issue_type": "logo_layout",
                "reference": "blue brand/logo print",
                "uploaded": "blue brand/logo print",
                "difference": (
                    f"Visible logo/brand print layout difference. "
                    f"position shift={dist:.3f}, width diff={width_diff:.2f}, "
                    f"height diff={height_diff:.2f}"
                ),
                "severity": "Medium",
                "confidence": 82,
                "ref_bbox": r["bbox"],
                "suspect_bbox": s["bbox"],
            })

    return issues
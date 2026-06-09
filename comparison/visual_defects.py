import cv2
import numpy as np


def _resize_same(ref_img, sus_img):
    h, w = ref_img.shape[:2]
    return ref_img, cv2.resize(sus_img, (w, h), interpolation=cv2.INTER_AREA)


def _ignore_big_layout_regions(contour, img_shape):
    h, w = img_shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    area = bw * bh

    # Ignore huge layout shifts; keep small local print defects.
    if area > (w * h) * 0.015:
        return True

    if bw > w * 0.20 or bh > h * 0.20:
        return True

    return False


def detect_visual_defects(ref_img, sus_img):
    issues = []

    ref_img, sus_img = _resize_same(ref_img, sus_img)

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    sus_gray = cv2.cvtColor(sus_img, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
    sus_gray = cv2.GaussianBlur(sus_gray, (3, 3), 0)

    diff = cv2.absdiff(ref_gray, sus_gray)

    _, th = cv2.threshold(diff, 38, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if _ignore_big_layout_regions(c, ref_img.shape):
            continue

        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < 35:
            continue

        # Keep small text defects/dots/broken print.
        if area > 5000:
            continue

        issues.append({
            "issue_type": "local_visual_print_defect",
            "reference": "local print region",
            "uploaded": "local print region",
            "difference": (
                f"Small local print/character defect detected. "
                f"Changed region size={w}x{h}px."
            ),
            "severity": "Medium",
            "confidence": 82,
            "ref_bbox": [x, y, x + w, y + h],
            "suspect_bbox": [x, y, x + w, y + h],
        })

    return issues
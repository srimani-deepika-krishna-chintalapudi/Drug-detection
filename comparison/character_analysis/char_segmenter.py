import cv2
import numpy as np


def binarize_text_crop(crop):
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th


def extract_character_boxes(crop):
    th = binarize_text_crop(crop)

    if th is None:
        return [], None

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        img_h, img_w = th.shape
        if area < 0.001 * img_h * img_w:
            continue
        if h < 0.35 * img_h:
            continue
        
        if w < 0.02 * img_w:
            continue
        
        if h / max(w, 1) > 25:
            continue

        aspect = h / max(w, 1)
        if aspect < 0.5:
            continue

        boxes.append([x, y, x + w, y + h])

    boxes = sorted(boxes, key=lambda b: b[0])

    return boxes, th


def draw_character_boxes(crop, boxes):
    if crop is None:
        return None

    vis = crop.copy()
    vis = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            str(i),
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return vis
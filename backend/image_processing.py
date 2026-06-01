import cv2
import numpy as np


def enhance_color_image(bgr):
    """Mild enhancement only. Keeps visual identity intact."""
    img = bgr.copy()

    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, 3, 3, 7, 21
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.4,
        tileGridSize=(8, 8)
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    contrast = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrast, 1.25, blur, -0.25, 0)

    return sharp


def resize_for_ocr(bgr, max_width=1800):
    h, w = bgr.shape[:2]

    if w <= max_width:
        return bgr.copy()

    scale = max_width / float(w)
    new_h = int(h * scale)

    return cv2.resize(
        bgr,
        (max_width, new_h),
        interpolation=cv2.INTER_CUBIC
    )


def reduce_shadow(bgr):
    """
    Reduces uneven lighting/shadows while preserving color.
    Useful for mobile camera carton photos.
    """
    planes = cv2.split(bgr)
    result_planes = []

    for plane in planes:
        dilated = cv2.dilate(
            plane,
            np.ones((7, 7), np.uint8)
        )
        background = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, background)
        normalized = cv2.normalize(
            diff,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        )
        result_planes.append(normalized)

    return cv2.merge(result_planes)


def sharpen_for_ocr(bgr):
    """
    Stronger sharpening for blurry OCR images.
    Should be used only in scan mode.
    """
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.3)
    sharp = cv2.addWeighted(bgr, 1.7, blur, -0.7, 0)
    return sharp


def enhance_for_blurry_text(bgr):
    """
    CamScanner-style OCR enhancement.
    More aggressive than enhance_color_image.
    Use only when scan mode is enabled.
    """
    img = resize_for_ocr(bgr)
    img = reduce_shadow(img)
    img = sharpen_for_ocr(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )
    l2 = clahe.apply(l)

    merged = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return enhanced


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 20 or max_height < 20:
        return image.copy()

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped


def detect_document_contour(bgr):
    """
    Detects largest rectangular document/carton-like region.
    Returns 4-point contour in original image coordinates or None.
    """
    original = bgr.copy()
    h, w = original.shape[:2]

    target_h = 700
    ratio = h / float(target_h)

    resized = cv2.resize(
        original,
        (int(w / ratio), target_h),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = resized.shape[0] * resized.shape[1]

    for c in contours[:10]:
        area = cv2.contourArea(c)

        if area < image_area * 0.12:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            pts *= ratio
            return pts

    return None


def document_scan_mode(bgr):
    """
    Optional CamScanner-style mode.
    Use only for blurry, tilted, shadowed, or low-light mobile photos.

    It:
    - detects document/carton boundary if possible
    - straightens perspective
    - reduces shadows
    - sharpens text
    - improves OCR contrast
    """
    original = bgr.copy()

    contour = detect_document_contour(original)

    if contour is not None:
        scanned = four_point_transform(original, contour)
    else:
        scanned = original

    scanned = enhance_for_blurry_text(scanned)

    return scanned


def draw_ocr_overlay(bgr, boxes, color=(0, 180, 0)):
    out = bgr.copy()

    for item in boxes:
        bbox = item.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        text = str(item.get("text", ""))[:28]

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            out,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return out


def crop_bbox(bgr, bbox, pad=8):
    h, w = bgr.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return bgr.copy()

    return bgr[y1:y2, x1:x2].copy()
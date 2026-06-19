import re


def extract_digit_sequences(text):
    """
    Extract meaningful digit groups from OCR text.
    Example:
    'MRP Rs. 120.50' -> ['120', '50']
    'Batch AB2345' -> ['2345']
    """
    return re.findall(r"\d+", str(text or ""))


def digit_sequences_differ(ref_text, sus_text):
    """
    Returns True only when both texts contain digits and the digit sequences differ.
    This prevents strength, MRP, batch, MFG, EXP, license values from being treated
    as simple spelling/OCR noise.
    """
    ref_digits = extract_digit_sequences(ref_text)
    sus_digits = extract_digit_sequences(sus_text)

    if not ref_digits and not sus_digits:
        return False

    if not ref_digits or not sus_digits:
        return True

    return ref_digits != sus_digits


def make_content_mismatch_issue(ref_text, sus_text, ref_bbox, sus_bbox):
    """
    Standard issue object for numeric/content mismatch.
    Compatible with your existing pipeline issue format.
    """
    x1, y1, x2, y2 = sus_bbox

    reason = (
        "Numeric/content mismatch detected. "
        f"Expected=[{ref_text}], Actual=[{sus_text}]. "
        "Digit sequence differs, so this is treated as a real content defect."
    )

    return {
        "issue_type": "content_mismatch",
        "category": "text_verification",
        "severity": "high",
        "title": "Numeric content mismatch detected",
        "description": reason,
        "difference": reason,
        "reference": str(ref_text or ""),
        "uploaded": str(sus_text or ""),
        "reference_text": str(ref_text or ""),
        "suspect_text": str(sus_text or ""),
        "ref_bbox": ref_bbox,
        "suspect_bbox": sus_bbox,
        "bbox": sus_bbox,
        "fault_bbox_yxyx": [y1, x1, y2, x2],
        "confidence": 0.95,
    }
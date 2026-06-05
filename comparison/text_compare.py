from dataclasses import dataclass, asdict
from typing import List, Dict
from difflib import SequenceMatcher
from rapidfuzz import fuzz
import re
import string

try:
    from comparison.dictionary_correction import correct_word
except Exception:
    def correct_word(x):
        return x


@dataclass
class TextIssue:
    reference: str
    uploaded: str
    difference: str
    severity: str
    confidence: float
    ref_bbox: list | None = None
    suspect_bbox: list | None = None
    issue_type: str = "text"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def tokenize(text: str) -> List[str]:
    return re.findall(
        r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*|[^\w\s]",
        text or "",
    )


def box_size(bbox):
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1), max(1, y2 - y1)


def is_vertical_bbox(bbox) -> bool:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return h / w > 2.2


def box_orientation(box: Dict) -> str:
    if "is_vertical" in box:
        return "vertical" if box.get("is_vertical") else "horizontal"

    bbox = box.get("bbox")
    if not bbox:
        return "unknown"

    return "vertical" if is_vertical_bbox(bbox) else "horizontal"


def size_score(ref_bbox, sus_bbox):
    rw, rh = box_size(ref_bbox)
    sw, sh = box_size(sus_bbox)

    width_ratio = min(rw, sw) / max(rw, sw)
    height_ratio = min(rh, sh) / max(rh, sh)

    return 100 * ((width_ratio + height_ratio) / 2)


def line_similarity(a: str, b: str) -> int:
    return max(
        fuzz.ratio(a.lower(), b.lower()),
        fuzz.token_sort_ratio(a.lower(), b.lower()),
        fuzz.partial_ratio(a.lower(), b.lower()),
    )


def position_similarity(ref_bbox, sus_bbox):
    rx1, ry1, rx2, ry2 = ref_bbox
    sx1, sy1, sx2, sy2 = sus_bbox

    rcx = (rx1 + rx2) / 2
    rcy = (ry1 + ry2) / 2

    scx = (sx1 + sx2) / 2
    scy = (sy1 + sy2) / 2

    dx = abs(rcx - scx)
    dy = abs(rcy - scy)

    dist = (dx * dx + dy * dy) ** 0.5

    return max(0, 100 - dist / 5)


def combined_match_score(ref_text, sus_text, ref_bbox, sus_bbox):
    text_score = line_similarity(ref_text, sus_text)
    pos_score = position_similarity(ref_bbox, sus_bbox)
    size = size_score(ref_bbox, sus_bbox)

    return (
        text_score * 0.60
        + pos_score * 0.25
        + size * 0.15
    )


def classify_edit(ref: str, sus: str) -> str:
    ref = correct_word(ref)
    sus = correct_word(sus)

    ref = normalize_text(ref)
    sus = normalize_text(sus)

    if ref == sus:
        return ""

    if ref.lower() == sus.lower() and ref != sus:
        return f"Capitalization change: '{ref}' vs '{sus}'"

    if ref.replace(" ", "") == sus.replace(" ", ""):
        return "Spacing change"

    ref_no_punct = "".join(c for c in ref if c not in string.punctuation)
    sus_no_punct = "".join(c for c in sus if c not in string.punctuation)

    if ref_no_punct == sus_no_punct and ref != sus:
        return "Punctuation change"

    sm = SequenceMatcher(None, ref, sus)
    parts = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        a = ref[i1:i2]
        b = sus[j1:j2]

        if tag == "delete":
            parts.append(f"Missing character(s): '{a}'")
        elif tag == "insert":
            parts.append(f"Extra character(s): '{b}'")
        elif tag == "replace":
            parts.append(f"Substituted '{a}' with '{b}'")

    return "; ".join(parts) if parts else "Text differs"


def severity_for(ref: str, sus: str, field_name: str = "text") -> str:
    sensitive_words = [
        "tablet", "tablets", "capsule", "capsules",
        "mg", "ml", "mcg", "ip", "bp", "usp",
        "paracetamol", "cetirizine", "amoxicillin",
        "pantoprazole", "domperidone",
        "batch", "mfg", "mfd", "exp", "expiry",
        "manufacturer", "dosage", "composition",
        "barcode", "qr", "mrp",
    ]

    combined = f"{ref} {sus} {field_name}".lower()
    return "High" if any(w in combined for w in sensitive_words) else "Medium"


def compare_strings(ref: str, sus: str, field_name: str = "text") -> TextIssue | None:
    ref = normalize_text(ref)
    sus = normalize_text(sus)

    if ref == sus:
        return None

    if not ref and sus:
        diff = f"Extra text '{sus}'"
    elif ref and not sus:
        diff = f"Missing text '{ref}'"
    else:
        diff = classify_edit(ref, sus)

    return TextIssue(
        reference=ref,
        uploaded=sus,
        difference=diff,
        severity=severity_for(ref, sus, field_name),
        confidence=95.0,
        issue_type=field_name,
    )


def compare_token_lists(ref_text: str, sus_text: str, field_name: str = "text") -> List[TextIssue]:
    ref_tokens = tokenize(ref_text)
    sus_tokens = tokenize(sus_text)

    issues: List[TextIssue] = []
    sm = SequenceMatcher(None, ref_tokens, sus_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        ref_chunk = ref_tokens[i1:i2]
        sus_chunk = sus_tokens[j1:j2]

        if tag == "replace" and len(ref_chunk) == len(sus_chunk):
            for rw, sw in zip(ref_chunk, sus_chunk):
                issue = compare_strings(rw, sw, field_name)
                if issue:
                    issues.append(issue)

        elif tag == "delete":
            ref_part = " ".join(ref_chunk)
            issues.append(TextIssue(
                reference=ref_part,
                uploaded="",
                difference=f"Missing text '{ref_part}'",
                severity="High",
                confidence=95.0,
                issue_type="missing_text",
            ))

        elif tag == "insert":
            sus_part = " ".join(sus_chunk)
            issues.append(TextIssue(
                reference="",
                uploaded=sus_part,
                difference=f"Extra text '{sus_part}'",
                severity="Medium",
                confidence=80.0,
                issue_type="extra_text",
            ))

        else:
            ref_part = " ".join(ref_chunk)
            sus_part = " ".join(sus_chunk)
            issue = compare_strings(ref_part, sus_part, field_name)
            if issue:
                issues.append(issue)

    return issues

def match_ocr_boxes(
    ref_boxes: List[Dict],
    sus_boxes: List[Dict],
    threshold: int = 65,
) -> List[TextIssue]:

    issues: List[TextIssue] = []
    used_sus = set()

    for rb in ref_boxes:

        ref_text = normalize_text(rb.get("text", ""))
        ref_bbox = rb.get("bbox")

        if not ref_text or len(ref_text) < 2 or not ref_bbox:
            continue

        best_idx = None
        best_score = -1

        for i, sb in enumerate(sus_boxes):

            if i in used_sus:
                continue

            sus_text = normalize_text(sb.get("text", ""))
            sus_bbox = sb.get("bbox")

            if not sus_text or len(sus_text) < 2 or not sus_bbox:
                continue

            # Vertical text only matches vertical text
            if box_orientation(rb) != box_orientation(sb):
                continue

            rx1, ry1, rx2, ry2 = ref_bbox
            sx1, sy1, sx2, sy2 = sus_bbox

            ref_cx = (rx1 + rx2) / 2
            sus_cx = (sx1 + sx2) / 2

            if abs(ref_cx - sus_cx) > 1200:
                continue

            score = combined_match_score(
                ref_text,
                sus_text,
                ref_bbox,
                sus_bbox,
            )

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None or best_score < threshold:
            issues.append(TextIssue(
                reference=ref_text,
                uploaded="",
                difference=f"Missing text '{ref_text}'",
                severity="High",
                confidence=95.0,
                ref_bbox=ref_bbox,
                suspect_bbox=None,
                issue_type="missing_text",
            ))
            continue

        used_sus.add(best_idx)
        sb = sus_boxes[best_idx]
        sus_text = normalize_text(sb.get("text", ""))

        if ref_text != sus_text:
            token_issues = compare_token_lists(ref_text, sus_text, field_name="text")

            if not token_issues:
                issue = compare_strings(ref_text, sus_text)
                token_issues = [issue] if issue else []

            for issue in token_issues:
                issue.ref_bbox = ref_bbox
                issue.suspect_bbox = sb.get("bbox")
                issue.confidence = round(min(95.0, max(70.0, best_score)), 2)
                issues.append(issue)

    for i, sb in enumerate(sus_boxes):
        if i in used_sus:
            continue

        sus_text = normalize_text(sb.get("text", ""))
        sus_bbox = sb.get("bbox")
        
        if not sus_text or len(sus_text) < 2 or not sus_bbox:
            continue

        issues.append(TextIssue(
            reference="",
            uploaded=sus_text,
            difference=f"Extra text '{sus_text}'",
            severity="Medium",
            confidence=80.0,
            ref_bbox=None,
            suspect_bbox=sus_bbox,
            issue_type="extra_text",
        ))

    return issues


def compare_full_text(ref_text: str, sus_text: str) -> List[TextIssue]:
    return []


def issues_to_dicts(issues):
    return [asdict(x) for x in issues]
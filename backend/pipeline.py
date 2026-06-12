from pathlib import Path
import cv2
from comparison.vertical_text_compare import detect_vertical_text_differences
import numpy as npS
from comparison.line_spacing import detect_line_spacing_differences
from comparison.auto_label_regions import detect_auto_label_region_differences
import re
from comparison.spacing_compare import detect_spacing_differences
from comparison.space_visual import detect_space_differences
from comparison.target_defects import detect_targeted_defects
from comparison.local_print_defects import detect_local_print_defects
from comparison.local_alignment import detect_local_alignment
from comparison.punctuation_visual import detect_visual_punctuation
from comparison.logo_visual import detect_logo_visual_differences

from utils.config import OUTPUT_DIR
from utils.image_io import read_bgr, write_image
from backend.image_processing import (
    enhance_color_image,
    document_scan_mode,
    draw_ocr_overlay,
    crop_bbox,
)

from ocr.ocr_engine import PaddleOCREngine
from ocr.field_extractor import extract_fields
from comparison.strict_ocr_matcher import strict_match_ocr
from comparison.typography import compare_typography
from comparison.strip_quality import detect_strip_quality_issues
from comparison.crop_match import detect_crop_based_differences

from backend.scoring import score_authenticity, verdict_from_score
from reports.pdf_report import generate_pdf_report
from database.db import create_comparison, add_difference


def _ocr_text(ocr_obj):
    return getattr(ocr_obj, "full_text", "") or ""


def _get_medicine_name(fields):
    med = fields.get("medicine", {})
    return med.get("name") if isinstance(med, dict) else None


def _get_medicine_confidence(fields):
    med = fields.get("medicine", {})
    return float(med.get("confidence") or 0) if isinstance(med, dict) else 0.0


def _is_same_image(img1, img2):
    if img1.shape != img2.shape:
        return False
    return float(np.mean(cv2.absdiff(img1, img2))) < 1.0


def _norm_for_visual(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text.strip()


def _same_text_only(ref_text, sus_text):
    a = _norm_for_visual(ref_text)
    b = _norm_for_visual(sus_text)

    if not a or not b:
        return False

    if len(a) < 4 or len(b) < 4:
        return False

    return a == b


def _same_base_text_ignore_punctuation(ref_text, sus_text):
    return _norm_for_visual(ref_text) == _norm_for_visual(sus_text)


def _filter_low_quality_issues(issues):
    clean = []

    visual_issue_types = {
        "word_width_spacing",
        "text_width_spacing","space_difference",
        "letter_spacing","auto_label_region_difference",
        "code_mismatch","spelling_mismatch",
        "location_spelling_mismatch", "tag_value_mismatch", "location_spelling_mismatch","vertical_text_mismatch","vertical_text_alignment",
        "colon_spacing",
        "crop_typography",
        "typography",
        "font_size",
        "font_style",
        "alignment",
        "local_print_defect",
        "position_mismatch",
        "size_mismatch",
        "boldness",
        "logo_layout",
        "vertical_text_mismatch",
        "local_alignment",
        "vertical_text_alignment",
    }

    for issue in issues:
        issue_type = issue.get("issue_type", "")
        confidence = float(issue.get("confidence", 0) or 0)

        ref_text = str(issue.get("reference", "") or "").strip()
        sus_text = str(issue.get("uploaded", "") or "").strip()

        has_ref = bool(issue.get("ref_bbox"))
        has_sus = bool(issue.get("suspect_bbox"))

        if issue_type == "medicine":
            clean.append(issue)
            continue

        if confidence < 65:
            continue

        if issue_type == "punctuation":
            if not _same_base_text_ignore_punctuation(ref_text, sus_text):
                continue

            issue["severity"] = "Medium"
            issue["confidence"] = max(confidence, 85)
            clean.append(issue)
            continue
        
        if issue_type == "text_mismatch":
            if not has_ref or not has_sus:
                continue

            ref_norm = _norm_for_visual(ref_text)
            sus_norm = _norm_for_visual(sus_text)

            if ref_norm == sus_norm:
                continue

            clean.append(issue)
            continue
        
        if issue_type in visual_issue_types:
            if not has_ref or not has_sus:
                continue

            allow_different_text_types = {
                "logo_layout",
                "colon_spacing","auto_label_region_difference",
                "vertical_text_mismatch",
                "location_spelling_mismatch","spelling_mismatch","word_width_spacing",
                "code_mismatch","space_difference",
                "tag_value_mismatch",
                "vertical_text_alignment",
            }

            if issue_type not in allow_different_text_types:
                if not _same_text_only(ref_text, sus_text):
                    continue

        if issue_type in ["missing_text", "extra_text"]:
            text = ref_text or sus_text
            norm = _norm_for_visual(text)

            if len(norm) < 5:
                continue

            important = [
                "batch", "bno", "mfg", "mfd", "exp", "expiry", "mrp", "pantoprazole", "domperidone", "paracetamol", "tablet", "capsule", "mg",
                "village", "bhatauli", "bhatouli", "khurd", "manufactured", "manufacturer", "india", "baddi",
                "solan", "mumbai", "alkem", "abbott", "lic", "license", "regd", "reg", "no", 
                "number", "village", "bhatauli", "bhatouli", "khurd", "manufactured", "manufacturer", "india", "baddi",
                "solan", "mumbai", "alkem", "abbott", "contains", "dosage", "physician", "lic", "license"
            ]

            if not any(x in norm for x in important):
                continue

        clean.append(issue)

    return clean


def _rotate_crop_if_vertical(crop, bbox):
    if crop is None or bbox is None:
        return crop

    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    if h / w > 2.0:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop


def _dedupe_issues(issues):
    seen = set()
    unique = []

    for issue in issues:
        key = (
            issue.get("issue_type", ""),
            issue.get("reference", ""),
            issue.get("uploaded", ""),
            issue.get("difference", ""),
            str(issue.get("ref_bbox")),
            str(issue.get("suspect_bbox")),
        )

        if key not in seen:
            seen.add(key)
            unique.append(issue)

    return unique


def _bbox_iou(a, b):
    if not a or not b:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / max(1, area_a + area_b - inter)


def _merge_same_crop_issues(issues):
    merged = []

    for issue in issues:
        added = False
        ref_box = issue.get("ref_bbox")
        sus_box = issue.get("suspect_bbox")

        for existing in merged:
            ref_iou = _bbox_iou(ref_box, existing.get("ref_bbox"))
            sus_iou = _bbox_iou(sus_box, existing.get("suspect_bbox"))

            if ref_iou > 0.45 or sus_iou > 0.45:
                existing_types = existing.get("issue_type", "")
                new_type = issue.get("issue_type", "")

                if new_type and new_type not in existing_types:
                    existing["issue_type"] = existing_types + " + " + new_type

                old_diff = existing.get("difference", "")
                new_diff = issue.get("difference", "")

                if new_diff and new_diff not in old_diff:
                    existing["difference"] = old_diff + " | " + new_diff

                existing["confidence"] = max(
                    float(existing.get("confidence", 0) or 0),
                    float(issue.get("confidence", 0) or 0),
                )

                if issue.get("severity") == "High":
                    existing["severity"] = "High"

                added = True
                break

        if not added:
            merged.append(issue)

    return merged


def compare_cartons(authentic_path: Path, suspect_path: Path, scan_mode=False):
    authentic_bgr_original = read_bgr(authentic_path)
    suspect_bgr_original = read_bgr(suspect_path)

    same_image = _is_same_image(authentic_bgr_original, suspect_bgr_original)

    if scan_mode:
        authentic_bgr = document_scan_mode(authentic_bgr_original)
        suspect_bgr = document_scan_mode(suspect_bgr_original)
    else:
        authentic_bgr = authentic_bgr_original
        suspect_bgr = suspect_bgr_original

    authentic_enh = enhance_color_image(authentic_bgr)
    suspect_enh = enhance_color_image(suspect_bgr)

    engine = PaddleOCREngine()

    ref_ocr = engine.run(authentic_bgr, "authentic")
    sus_ocr = engine.run(suspect_bgr, "suspect")

    ref_text = _ocr_text(ref_ocr)
    sus_text = _ocr_text(sus_ocr)

    ref_fields = extract_fields(ref_text, ref_ocr.boxes)
    sus_fields = extract_fields(sus_text, sus_ocr.boxes)

    ref_med_name = _get_medicine_name(ref_fields)
    sus_med_name = _get_medicine_name(sus_fields)
    ref_med_conf = _get_medicine_confidence(ref_fields)

    medicine_block = False
    medicine_message = ""

    if ref_med_name and sus_med_name and ref_med_name != sus_med_name:
        medicine_block = True
        medicine_message = (
            f"Medicine mismatch: authentic appears to be {ref_med_name}, "
            f"suspect appears to be {sus_med_name}. Comparison blocked."
        )

    run_dir = OUTPUT_DIR / f"run_{authentic_path.stem[:8]}_{suspect_path.stem[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "authentic_original": authentic_path,
        "suspect_original": suspect_path,
        "authentic_processed": write_image(run_dir / "authentic_processed.png", authentic_bgr),
        "suspect_processed": write_image(run_dir / "suspect_processed.png", suspect_bgr),
        "authentic_enhanced": write_image(run_dir / "authentic_enhanced.png", authentic_enh),
        "suspect_enhanced": write_image(run_dir / "suspect_enhanced.png", suspect_enh),
        "authentic_ocr_overlay": write_image(
            run_dir / "authentic_ocr_overlay.png",
            draw_ocr_overlay(authentic_bgr, ref_ocr.boxes),
        ),
        "suspect_ocr_overlay": write_image(
            run_dir / "suspect_ocr_overlay.png",
            draw_ocr_overlay(suspect_bgr, sus_ocr.boxes),
        ),
    }

    issues = []

    if same_image:
        score = 100.0
        verdict = "Original"

    elif medicine_block:
        issues.append({
            "issue_type": "medicine",
            "reference": ref_med_name,
            "uploaded": sus_med_name,
            "difference": medicine_message,
            "severity": "High",
            "confidence": 95,
            "ref_bbox": None,
            "suspect_bbox": None,
        })

        score = score_authenticity(issues)
        verdict = verdict_from_score(score)

    else:
        detectors = [
            (
                "Targeted Defects",
                lambda: detect_targeted_defects(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr,
                    suspect_bgr,
                ),
            ),
            (
                "Space Difference",
                lambda: detect_space_differences(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                ),
            ),
            (
                "Generic Spacing Comparison",
                lambda: detect_spacing_differences(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                ),
            ),
            (
                "Auto Label Region Comparison",
                lambda: detect_auto_label_region_differences(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr,
                    suspect_bgr,
                ),
            ),
            (
                "Strict OCR comparison",
                lambda: strict_match_ocr(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr.shape,
                    suspect_bgr.shape,
                ),
            ),
            (
                "Generic Vertical Text Comparison",
                lambda: detect_vertical_text_differences(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr.shape,
                    suspect_bgr.shape,
                ),
            ),
            (
                "Letter Spacing",
                lambda: compare_letter_spacing(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr,
                    suspect_bgr,
                ),
            ),
        ]

        for name, detector in detectors:
            try:
                found = detector()
                if found is None:
                    found = []

                print(f"{name}: {len(found)} issues")
                issues.extend(found)

            except Exception as e:
                print(f"{name} failed:", e)

        issues = _filter_low_quality_issues(issues)
        issues = _dedupe_issues(issues)
        issues = _merge_same_crop_issues(issues)

        score = score_authenticity(issues)
        verdict = verdict_from_score(score)
                

    evidence_pairs = {}

    for idx, issue in enumerate(issues):
        ref_crop_path = None
        sus_crop_path = None

        if issue.get("ref_bbox"):
            try:
                ref_crop = crop_bbox(authentic_bgr, issue["ref_bbox"])
                ref_crop = _rotate_crop_if_vertical(ref_crop, issue["ref_bbox"])
                ref_crop_path = write_image(run_dir / f"evidence_{idx + 1}_ref.png", ref_crop)
            except Exception as e:
                print(f"Reference crop failed for issue {idx + 1}: {e}")

        if issue.get("suspect_bbox"):
            try:
                sus_crop = crop_bbox(suspect_bgr, issue["suspect_bbox"])
                sus_crop = _rotate_crop_if_vertical(sus_crop, issue["suspect_bbox"])
                sus_crop_path = write_image(run_dir / f"evidence_{idx + 1}_suspect.png", sus_crop)
            except Exception as e:
                print(f"Suspect crop failed for issue {idx + 1}: {e}")

        if ref_crop_path or sus_crop_path:
            evidence_pairs[idx] = {
                "ref": ref_crop_path,
                "sus": sus_crop_path,
            }

    report_path = run_dir / "authentication_report.pdf"

    generate_pdf_report(
        report_path,
        authentic_path,
        suspect_path,
        ref_ocr,
        sus_ocr,
        ref_fields,
        sus_fields,
        issues,
        score,
        verdict,
        evidence_pairs,
    )

    comparison_id = create_comparison(
        authentic_path,
        suspect_path,
        ref_med_name,
        ref_med_conf,
        score,
        verdict,
        report_path,
    )

    for idx, issue in enumerate(issues):
        pair = evidence_pairs.get(idx, {})
        add_difference(
            comparison_id,
            issue,
            pair.get("ref"),
            pair.get("sus"),
        )

    return {
        "comparison_id": comparison_id,
        "paths": {k: str(v) for k, v in paths.items()},
        "ref_ocr": ref_ocr,
        "sus_ocr": sus_ocr,
        "ref_fields": ref_fields,
        "sus_fields": sus_fields,
        "issues": issues,
        "evidence_pairs": {
            k: {kk: str(vv) if vv else None for kk, vv in val.items()}
            for k, val in evidence_pairs.items()
        },
        "score": score,
        "verdict": verdict,
        "report_path": str(report_path),
        "medicine_blocked": medicine_block,
        "medicine_message": medicine_message,
        "scan_mode": scan_mode,
    }
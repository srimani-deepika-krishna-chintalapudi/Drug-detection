from pathlib import Path
import cv2
from comparison.spelling_mismatch_detector import detect_spelling_mismatches
from comparison.text_pair_matcher import get_matched_text_pairs
from comparison.text_spacing_detector import detect_text_spacing_differences
from comparison.char_symbol_gap_detector import detect_char_symbol_gap_differences
from comparison.spacing_mismatch_detector import detect_spacing_mismatches
from comparison.tablet_name_verifier import verify_tablet_name
from comparison.strict_layout_string_detector import detect_strict_layout_string_differences
from comparison.generic_text_spacing import detect_generic_text_spacing
from comparison.carton_spacing import detect_carton_spacing_issues
from comparison.character_analysis.char_spacing import compare_character_spacing
from comparison.vertical_text_compare import detect_vertical_text_differences
import numpy as np
from comparison.spacing_generic import detect_generic_spacing_issues
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

import os
print(f"PIPELINE FILE LOADED: {__file__}")
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
        "word_width_spacing","char_symbol_spacing_mismatch",
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
        issue_type = str(issue.get("issue_type", "")).lower()
        if issue_type == "spelling_mismatch":
            clean.append(issue)
            continue
        if issue_type == "text_spacing_mismatch":
            clean.append(issue)
            continue

        if issue_type in {
            "strict_spacing_mismatch",
            "strict_string_mismatch","char_symbol_spacing_mismatch",
            "vertical_orientation_mismatch",
            "vertical_layout_shift",
            "spacing_mismatch",
        }:
            clean.append(issue)
            continue

        text_blob = " ".join([
            str(issue.get("reference_text", "")),
            str(issue.get("suspect_text", "")),
            str(issue.get("reference", "")),
            str(issue.get("uploaded", "")),
            str(issue.get("description", "")),
            str(issue.get("difference", "")),
        ]).lower()

        address_words = [
            "baddi", "solan", "h.p", "hp", "distt", "district",
            "village", "road", "plot", "industrial", "area",
            "pradesh", "india", "khurd", "bhatauli", "bhatouli"
        ]

        is_address_issue = any(w in text_blob for w in address_words)

        if is_address_issue and issue_type in [
            "word_width_spacing",
            "letter_spacing_change",
            "spacing_difference",
            "generic_spacing_difference",
            "auto_label_region_difference",
        ]:
            print("REMOVED ADDRESS/LOCATION FALSE POSITIVE:", text_blob[:120])
            continue
        if is_address_issue and issue_type in ["missing_text", "extra_text", "text_mismatch"]:
            clean.append(issue)
            continue
        confidence = float(issue.get("confidence", 0) or 0)
        if issue_type in ["character_spacing","letter_spacing_change","word_width_spacing"]:
            issue.setdefault("ref_bbox", issue.get("bbox") or issue.get("reference_bbox"))
            issue.setdefault("suspect_bbox", issue.get("bbox") or issue.get("uploaded_bbox"))

            if issue.get("ref_bbox") and issue.get("suspect_bbox"):
                clean.append(issue)

            continue

        ref_text = str(issue.get("reference", "") or "").strip()
        sus_text = str(issue.get("uploaded", "") or "").strip()

        has_ref = bool(issue.get("ref_bbox"))
        has_sus = bool(issue.get("suspect_bbox"))

        if issue_type in ["medicine", "tablet_name_mismatch"]:
            clean.append(issue)
            continue

        if confidence < 65:
            continue
        if issue_type in ["space_difference","character_spacing","word_or_symbol_spacing",]:
            clean.append(issue)
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
                "logo_layout","char_symbol_spacing_mismatch",
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

    if h > w * 1.2:
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
        issue_type = str(issue.get("issue_type", "")).lower()

        if issue_type == "spelling_mismatch":
            merged.append(issue)
            continue

        if issue_type == "text_spacing_mismatch":
            merged.append(issue)
            continue

        if issue_type in {
            "tablet_name_mismatch",
            "strict_spacing_mismatch",
            "char_symbol_spacing_mismatch",
            "strict_string_mismatch",
            "vertical_orientation_mismatch",
            "vertical_layout_shift",
            "spacing_mismatch",
        }:
            merged.append(issue)
            continue

        for existing in merged:
            if existing.get("issue_type") == "tablet_name_mismatch":
                continue

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

                if issue.get("severity") == "High" or issue.get("severity") == "high":
                    existing["severity"] = "High"

                added = True
                break

        if not added:
            merged.append(issue)

    return merged

def _strict_meaningful_issue_filter(issues):
    clean = []

    weak_spacing_types = {
        "space_difference",
        "character_spacing",
        "word_or_symbol_spacing",
        "carton_spacing_difference",
    }

    for issue in issues:

        print(
            "STRICT CHECK:",
            issue.get("issue_type"),
            issue.get("confidence"),
            issue.get("difference"),
        )

        issue_type = str(issue.get("issue_type", "")).lower()
        diff = str(issue.get("difference", "")).lower()
        confidence = float(issue.get("confidence", 0) or 0)

        if issue_type == "spelling_mismatch":
            print("KEPT:", issue_type)
            clean.append(issue)
            continue

        if issue_type == "text_spacing_mismatch":
            print("KEPT:", issue_type)
            clean.append(issue)
            continue

        if issue_type in {
            "strict_spacing_mismatch",
            "strict_string_mismatch","char_symbol_spacing_mismatch",
            "vertical_orientation_mismatch",
            "vertical_layout_shift",
            "spacing_mismatch",
        }:
            print("KEPT:", issue_type)
            clean.append(issue)
            continue

        if issue_type in {
            "strict_spacing_mismatch",
            "strict_string_mismatch",
            "vertical_orientation_mismatch",
            "vertical_layout_shift",
        }:
            print("KEPT:", issue_type)
            clean.append(issue)
            continue

        if issue_type == "spacing_mismatch":
            if (
                issue.get("severity") == "high"
                or confidence >= 0.75
            ):
                print("KEPT: spacing_mismatch")
                clean.append(issue)
            continue

        # Strict spacing rule
        if issue_type in weak_spacing_types:
            if confidence < 90:
                print("REMOVED: low confidence spacing issue")
                continue

            minor_words = [
                "minor",
                "mean gap",
                "normalized gap",
                "single extra space",
                "padding",
                "slight",
            ]

            if any(w in diff for w in minor_words):
                print("REMOVED: weak spacing difference")
                continue

        # Strict spelling/text rule
        if issue_type in {
            "text_mismatch",
            "ocr_text_mismatch",
            "spelling_difference",
            "word_difference",
        }:
            ref = str(issue.get("reference", "")).strip()
            sus = str(issue.get("uploaded", "")).strip()

            if not ref or not sus:
                clean.append(issue)
                continue

            if len(ref) >= 5 and len(sus) >= 5:
                from rapidfuzz import fuzz

                similarity = fuzz.ratio(ref.lower(), sus.lower())

                print("TEXT SIMILARITY:", similarity)

                if similarity >= 88:
                    print("REMOVED: likely valid variant")
                    continue

            if confidence < 85:
                print("REMOVED: low confidence text issue")
                continue

        print("KEPT:", issue_type)
        clean.append(issue)

    return clean

def compare_cartons(authentic_path: Path, suspect_path: Path, scan_mode=False,medicine_name=None):
    authentic_bgr_original = read_bgr(authentic_path)
    suspect_bgr_original = read_bgr(suspect_path)
    print("COMPARE_CARTONS CALLED")
    print(f"medicine_name received in compare_cartons: {medicine_name}")
    print("### REAL COMPARE_CARTONS RUNNING ###")
    print("### PIPELINE FILE:", __file__)
    print("### MEDICINE NAME RECEIVED:", medicine_name)

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
                "Text Spacing Detector",
                lambda: detect_text_spacing_differences(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr,
                    suspect_bgr,
                    medicine_name=medicine_name,
                    gap_threshold_percent=12.0,
                ),
            ),
            #(
             #   "Space Difference",
              #  lambda: detect_space_differences(
               #     ref_ocr.boxes,
                #    sus_ocr.boxes,
                 #   authentic_bgr,
                  #  suspect_bgr,
                #),
            #),
            #(
             #   "Generic Text Character Spacing",
              #  lambda: detect_generic_text_spacing(
               #     ref_ocr.boxes,
                #    sus_ocr.boxes,
                 #   authentic_bgr,
                  #  suspect_bgr,
                #),
            #),
            #(
             #   "Generic Spacing Comparison",
              #  lambda: detect_spacing_differences(
               #     ref_ocr.boxes,
                #    sus_ocr.boxes,
                #),
            #),
            #(
             #   "Auto Label Region Comparison",
              #  lambda: detect_auto_label_region_differences(
               #     ref_ocr.boxes,
                #    sus_ocr.boxes,
                 #   authentic_bgr,
                  #  suspect_bgr,
            #),
            #),
            #(
             #   "Dedicated Spacing Mismatch",
              ##     ref_ocr.boxes,
                #    sus_ocr.boxes,
                 ##  suspect_bgr,
                   # medicine_name=medicine_name,
                    #threshold_percent=5.0,
                #),
            #),
            #(
             #   "Strict Layout String Detector",
              ###    sus_ocr.boxes,
                 #   authentic_bgr,
                  ## medicine_name=medicine_name,
              #  ),
            #),
            #(
             #   "Char Symbol Gap Detector",
              #  lambda: detect_char_symbol_gap_differences(
               #     ref_ocr.boxes,
                #    sus_ocr.boxes,
                 #   authentic_bgr,
                  #  suspect_bgr,
                   # medicine_name=medicine_name,
                    #threshold_percent=35.0,
               # ),
            #),
            (
            "Spelling Mismatch Detector",
            lambda: detect_spelling_mismatches(
                ref_ocr.boxes,
                sus_ocr.boxes,
                authentic_bgr,
                suspect_bgr,
                medicine_name=medicine_name,
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
                "Text Pair Matching",
                lambda: get_matched_text_pairs(
                    ref_ocr.boxes,
                    sus_ocr.boxes,
                    authentic_bgr.shape,
                    suspect_bgr.shape,
                )
            )
     ]

        for name, detector in detectors:
            try:
                found = detector()
                if found is None:
                    found = []

                print(f"{name}: {len(found)} issues")

                if name == "Space Difference":
                    print("SPACE DIFFERENCE DEBUG:", found)

                issues.extend(found)

            except Exception as e:
                print(f"{name} failed:", e)

        # PUT TABLET NAME VERIFIER HERE — OUTSIDE THE LOOP
        try:
            print("TABLET NAME VERIFIER RUNNING")
            print("Tablet target name from UI:", medicine_name)

            tablet_name_issues = verify_tablet_name(
                ref_ocr.boxes,
                sus_ocr.boxes,
                authentic_bgr,
                suspect_bgr,
                medicine_name,
                threshold_percent=5.0,
            )
            print(f"Tablet Name Verification: {len(tablet_name_issues)} issues")
            issues.extend(tablet_name_issues)

        except Exception as e:
            print("Tablet Name Verification failed:", e)
        
        issues = _filter_low_quality_issues(issues)
        issues = _dedupe_issues(issues)
        issues = _merge_same_crop_issues(issues)
        print(
            "BEFORE STRICT FILTER:",
            len(issues),
            [i.get("issue_type") for i in issues],
        )

        issues = _strict_meaningful_issue_filter(issues)

        print(
            "AFTER STRICT FILTER:",
            len(issues),
            [i.get("issue_type") for i in issues],
        )

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
                ref_crop_path = write_image(
                    run_dir / f"evidence_{idx + 1}_ref.png",
                    ref_crop,
                )
            except Exception as e:
                print(f"Reference crop failed for issue {idx + 1}: {e}")


        if issue.get("suspect_bbox"):
            try:
                sus_crop = crop_bbox(suspect_bgr, issue["suspect_bbox"])
                sus_crop = _rotate_crop_if_vertical(sus_crop, issue["suspect_bbox"])
                sus_crop_path = write_image(
                    run_dir / f"evidence_{idx + 1}_suspect.png",
                    sus_crop,
                )
            except Exception as e:
                print(f"Suspect crop failed for issue {idx + 1}: {e}")

        if ref_crop_path or sus_crop_path:
            evidence_id = issue.get("evidence_id") or f"evidence_{idx}"
            issue["evidence_id"] = evidence_id

            evidence_pairs[evidence_id] = {
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
        evidence_id = issue.get("evidence_id")
        pair = evidence_pairs.get(evidence_id, {})

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
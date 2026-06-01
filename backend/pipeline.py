from pathlib import Path
import cv2
import numpy as np

from utils.config import OUTPUT_DIR
from comparison.strip_quality import detect_strip_quality_issues
from utils.image_io import read_bgr, write_image
from backend.image_processing import (
    enhance_color_image,
    document_scan_mode,
    draw_ocr_overlay,
    crop_bbox,
)
from ocr.ocr_engine import PaddleOCREngine, choose_best_ocr
from ocr.field_extractor import extract_fields
from comparison.typography import compare_typography
from comparison.text_compare import match_ocr_boxes, issues_to_dicts, compare_full_text
from visual_analysis.visual_compare import compare_overall_visual
from backend.scoring import score_authenticity, verdict_from_score
from reports.pdf_report import generate_pdf_report
from database.db import create_comparison, add_difference


def _ocr_text(ocr_obj):
    return getattr(ocr_obj, "full_text", None) or getattr(ocr_obj, "text", "") or ""


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


def _dedupe_issues(issues):
    seen, unique = set(), []

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

    ref_o1 = engine.run(authentic_bgr, "authentic_original_or_scanned")
    ref_o2 = engine.run(authentic_enh, "authentic_enhanced")
    sus_o1 = engine.run(suspect_bgr, "suspect_original_or_scanned")
    sus_o2 = engine.run(suspect_enh, "suspect_enhanced")

    ref_ocr = choose_best_ocr(ref_o1, ref_o2)
    sus_ocr = choose_best_ocr(sus_o1, sus_o2)

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
        "authentic_processed": write_image(
            run_dir / "authentic_processed.png",
            authentic_bgr,
        ),
        "suspect_processed": write_image(
            run_dir / "suspect_processed.png",
            suspect_bgr,
        ),
        "authentic_enhanced": write_image(
            run_dir / "authentic_enhanced.png",
            authentic_enh,
        ),
        "suspect_enhanced": write_image(
            run_dir / "suspect_enhanced.png",
            suspect_enh,
        ),
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
        issues = []
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
        box_issues = match_ocr_boxes(ref_ocr.boxes, sus_ocr.boxes)
        issues.extend(issues_to_dicts(box_issues))

        try:
            full_text_issues = compare_full_text(ref_text, sus_text)
            issues.extend(issues_to_dicts(full_text_issues))
        except Exception as e:
            print("Full text comparison failed:", e)
        
        try:
            strip_issues = detect_strip_quality_issues(
                ref_ocr.boxes,
                sus_ocr.boxes,
                authentic_bgr,
                suspect_bgr,
            )
            issues.extend(strip_issues)
        except Exception as e:
            print("Strip quality comparison failed:", e)

        try:
            typo_issues = compare_typography(
                ref_ocr.boxes,
                sus_ocr.boxes,
                authentic_bgr,
                suspect_bgr,
            )
            issues.extend(typo_issues)
        except Exception as e:
            print("Typography comparison failed:", e)

        # Visual comparison disabled for now.
# It can create false positives unless both images are perfectly aligned.
# Enable later only after proper region-level alignment.

        issues = _dedupe_issues(issues)
        score = score_authenticity(issues)
        verdict = verdict_from_score(score)

    evidence_pairs = {}

    for idx, issue in enumerate(issues):
        ref_crop_path = None
        sus_crop_path = None

        if issue.get("ref_bbox"):
            try:
                ref_crop = crop_bbox(authentic_bgr, issue["ref_bbox"])
                ref_crop_path = write_image(
                    run_dir / f"evidence_{idx + 1}_ref.png",
                    ref_crop,
                )
            except Exception as e:
                print(f"Reference crop failed for issue {idx + 1}: {e}")

        if issue.get("suspect_bbox"):
            try:
                sus_crop = crop_bbox(suspect_bgr, issue["suspect_bbox"])
                sus_crop_path = write_image(
                    run_dir / f"evidence_{idx + 1}_suspect.png",
                    sus_crop,
                )
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
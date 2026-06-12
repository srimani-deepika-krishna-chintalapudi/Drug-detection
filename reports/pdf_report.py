from pathlib import Path
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from PIL import Image as PILImage


def _img(path, max_width=2.4 * inch, max_height=2.4 * inch):
    if path is None:
        return Paragraph("Image unavailable", getSampleStyleSheet()["Normal"])

    p = Path(path)

    if not p.exists():
        return Paragraph("Image unavailable", getSampleStyleSheet()["Normal"])

    try:
        im = PILImage.open(p)
        w, h = im.size

        if w <= 0 or h <= 0:
            return Paragraph("Invalid image", getSampleStyleSheet()["Normal"])

        scale = min(max_width / w, max_height / h)

        return Image(
            str(p),
            width=w * scale,
            height=h * scale,
        )

    except Exception:
        return Paragraph("Image unavailable", getSampleStyleSheet()["Normal"])


def _safe_text(value):
    if value is None:
        return ""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_evidence_pair(evidence_pairs, idx_zero_based):
    if isinstance(evidence_pairs, dict):
        return evidence_pairs.get(idx_zero_based)

    if isinstance(evidence_pairs, list) and idx_zero_based < len(evidence_pairs):
        return evidence_pairs[idx_zero_based]

    return None


def _append_final_difference_summary_table(story, issues, evidence_pairs, styles):
    story.append(PageBreak())
    story.append(Paragraph("All Differences Summary Table", styles["Heading2"]))
    story.append(Spacer(1, 8))

    rows = [[
        Paragraph("<b>Difference</b>", styles["Normal"]),
        Paragraph("<b>Cropped Image of Authentic Carton</b>", styles["Normal"]),
        Paragraph("<b>Cropped Image of Uploaded Carton</b>", styles["Normal"]),
    ]]

    if not issues:
        rows.append([
            Paragraph("No significant differences detected.", styles["Normal"]),
            Paragraph("N/A", styles["Normal"]),
            Paragraph("N/A", styles["Normal"]),
        ])
    else:
        for idx, issue in enumerate(issues):
            pair = _get_evidence_pair(evidence_pairs, idx)

            ref_path = None
            sus_path = None

            if isinstance(pair, dict):
                ref_path = pair.get("ref")
                sus_path = pair.get("sus")

            difference = (
                issue.get("difference")
                or issue.get("message")
                or issue.get("issue_type")
                or "Difference detected"
            )

            rows.append([
                Paragraph(_safe_text(difference), styles["Normal"]),
                _img(ref_path, 1.7 * inch, 1.2 * inch),
                _img(sus_path, 1.7 * inch, 1.2 * inch),
            ])

    table = Table(
        rows,
        colWidths=[3.1 * inch, 1.7 * inch, 1.7 * inch],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(table)


def generate_pdf_report(
    output_path,
    authentic_path,
    suspect_path,
    ocr_ref,
    ocr_sus,
    fields_ref,
    fields_sus,
    issues,
    score,
    verdict,
    evidence_pairs,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Pharmaceutical Carton Authentication Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Authenticity Score:</b> {score:.1f}/100", styles["Normal"]))
    story.append(Paragraph(f"<b>Final Verdict:</b> {_safe_text(verdict)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(
        Table(
            [
                [
                    _img(authentic_path, 2.8 * inch, 3.4 * inch),
                    _img(suspect_path, 2.8 * inch, 3.4 * inch),
                ]
            ],
            colWidths=[3.1 * inch, 3.1 * inch],
        )
    )

    story.append(Spacer(1, 14))

    story.append(Paragraph("Extracted OCR Fields", styles["Heading2"]))

    rows = [["Field", "Authentic", "Suspect"]]
    for k in sorted(set(fields_ref.keys()) | set(fields_sus.keys())):
        rows.append(
            [
                Paragraph(_safe_text(k), styles["Normal"]),
                Paragraph(_safe_text(fields_ref.get(k)), styles["Normal"]),
                Paragraph(_safe_text(fields_sus.get(k)), styles["Normal"]),
            ]
        )

    tbl = Table(rows, colWidths=[1.5 * inch, 2.35 * inch, 2.35 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(tbl)
    story.append(PageBreak())

    story.append(Paragraph("Detected Differences", styles["Heading2"]))
    story.append(Spacer(1, 8))

    if not issues:
        story.append(Paragraph("No significant differences detected.", styles["Normal"]))

    for idx, issue in enumerate(issues, 1):
        block = []

        block.append(
            Paragraph(
                f"<b>Difference {idx}: {_safe_text(issue.get('severity', ''))}</b>",
                styles["Heading3"],
            )
        )

        data = [
            ["Reference", Paragraph(_safe_text(issue.get("reference", "")), styles["Normal"])],
            ["Uploaded", Paragraph(_safe_text(issue.get("uploaded", "")), styles["Normal"])],
            ["Difference", Paragraph(_safe_text(issue.get("difference", "")), styles["Normal"])],
            ["Confidence", Paragraph(_safe_text(issue.get("confidence", "")), styles["Normal"])],
        ]

        t = Table(data, colWidths=[1.2 * inch, 5.2 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        block.append(t)

        pair = None
        if isinstance(evidence_pairs, dict):
            pair = evidence_pairs.get(idx - 1)
        elif isinstance(evidence_pairs, list) and idx - 1 < len(evidence_pairs):
            pair = evidence_pairs[idx - 1]

        if pair:
            block.append(Spacer(1, 6))

            ref_path = pair.get("ref") if isinstance(pair, dict) else None
            sus_path = pair.get("sus") if isinstance(pair, dict) else None

            evidence_table = Table(
                [
                    [
                        _img(ref_path, 2.3 * inch, 2.3 * inch),
                        _img(sus_path, 2.3 * inch, 2.3 * inch),
                    ]
                ],
                colWidths=[2.7 * inch, 2.7 * inch],
            )
            evidence_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ]
                )
            )

            block.append(evidence_table)

        block.append(Spacer(1, 12))
        story.append(KeepTogether(block))

    _append_final_difference_summary_table(story, issues, evidence_pairs, styles)

    doc.build(story)
    return output_path
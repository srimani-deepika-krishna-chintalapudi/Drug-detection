# Pharmaceutical Carton Authentication System

## Overview

This project compares an uploaded pharmaceutical carton image against an authentic reference carton image and identifies potential differences such as:

- Text mismatches
- Typography variations
- Alignment issues
- Layout inconsistencies
- Crop-based differences
- Visual anomalies

The system is designed to help identify suspicious or counterfeit pharmaceutical packaging by comparing a suspect carton with a trusted reference image.

---

## Features

- OCR-based text extraction using PaddleOCR
- Text comparison and mismatch detection
- Typography analysis
- Alignment and spacing verification
- Crop-based visual comparison
- OCR correction and normalization
- Difference highlighting
- Report generation
- Reference image management

---

## What This Project Does

The system performs the following workflow:

1. Accepts an uploaded carton image.
2. Loads the authentic reference carton image.
3. Extracts text from both images using OCR.
4. Compares text content.
5. Compares typography and spacing.
6. Detects alignment issues.
7. Detects crop-level visual differences.
8. Highlights suspicious regions.
9. Generates a visual verification report.

---

## Project Structure

```text
pharma_carton_auth_system/
│
├── backend/
│   ├── pipeline.py
│   └── remove.py
│
├── comparison/
│   ├── anchor_match.py
│   ├── crop_match.py
│   ├── dictionary_correction.py
│   ├── line_compare.py
│   ├── strict_ocr_matcher.py
│   ├── text_compare.py
│   └── typography.py
│
├── ocr/
│   ├── ocr_engine.py
│   ├── field_extractor.py
│   └── crop_refiner.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── reference_images/
│   ├── uploads/
│   ├── outputs/
│   └── reference_db.json
│
├── database/
│   └── carton_auth.db
│
└── README.md
```

---

# Main Modules

## Frontend

### `frontend/app.py`

This is the user-facing application.

Responsibilities:

- Upload carton images
- Select reference images
- Trigger authentication
- Display verification results
- Display generated reports

This is the entry point for users interacting with the system.

---

## Backend

### `backend/pipeline.py`

This is the main processing pipeline and core controller of the application.

Responsibilities:

- Load reference image
- Load uploaded image
- Run OCR
- Execute comparison modules
- Collect detected issues
- Filter false positives
- Generate final verification results

This file acts as the central orchestrator of the system.

---

### `backend/remove.py`

Handles image preprocessing and cleanup.

Possible responsibilities:

- Noise reduction
- Image enhancement
- Background cleanup
- Preprocessing before OCR

---

# OCR Module

## `ocr/ocr_engine.py`

Performs Optical Character Recognition (OCR).

Extracts:

- Text
- Confidence scores
- Bounding boxes
- Text locations
- Text orientation

Uses PaddleOCR to read printed information from pharmaceutical cartons.

---

## `ocr/field_extractor.py`

Extracts important structured information from OCR results.

Examples:

- Medicine name
- Batch number
- Manufacturing date
- Expiry date
- MRP
- Manufacturer information
- License details

Converts raw OCR text into meaningful business data.

---

## `ocr/crop_refiner.py`

Improves OCR performance on difficult text regions.

Functions include:

- Region cropping
- Region enhancement
- Rotation correction
- OCR refinement

Useful for blurry or vertically printed text.

---

# Comparison Module

## `comparison/text_compare.py`

Compares text extracted from reference and suspect cartons.

Detects:

- Missing text
- Additional text
- Spelling differences
- Character substitutions
- OCR mismatches

### Example

```text
Reference : TABLET
Uploaded  : TABLETS
```

---

## `comparison/typography.py`

Detects visual differences in printed text appearance.

Compares:

- Font size
- Text thickness
- Boldness
- Character spacing
- Width and height
- Alignment
- Contrast

Useful when text content is identical but print quality differs.

### Example

```text
Reference : PARACETAMOL
Uploaded  : PARACETAMOL

Text is identical,
but spacing or thickness differs.
```

---

## `comparison/line_compare.py`

Performs line-level layout comparison.

Detects:

- Horizontal shifts
- Vertical shifts
- Line spacing issues
- Block misalignment

Helps identify layout inconsistencies often found in counterfeit packaging.

---

## `comparison/anchor_match.py`

Matches stable reference points between images.

Examples:

- Medicine name
- Logo location
- Header regions
- Key labels

These anchor points help align images before comparison.

---

## `comparison/crop_match.py`

Performs crop-based image comparison.

Instead of comparing full images, important regions are compared individually.

Detects:

- Missing symbols
- Layout changes
- Print defects
- Region-level differences

Improves detection accuracy.

---

## `comparison/strict_ocr_matcher.py`

Performs strict text matching.

Purpose:

- Reduce OCR matching errors
- Prevent incorrect fuzzy matches
- Improve comparison reliability

---

## `comparison/dictionary_correction.py`

Corrects common OCR recognition mistakes.

Examples:

```text
0 → O
1 → I
5 → S
8 → B
```

Improves OCR accuracy before comparison.

---

# Data Storage

## `data/reference_images/`

Stores authentic carton images used as trusted references.

---

## `data/uploads/`

Stores images uploaded during verification.

These files are generated dynamically and should generally not be committed to GitHub.

---

## `data/outputs/`

Stores generated outputs such as:

- Difference images
- Cropped issue images
- Reports

These files are generated dynamically and should generally not be committed to GitHub.

---

## `data/reference_db.json`

Stores metadata about reference cartons.

Examples:

- Product names
- Reference image mapping
- Product configuration information

---

# Database

## `database/carton_auth.db`

Local SQLite database.

May store:

- Verification history
- Product information
- Uploaded image records
- Authentication results

---

# Files Ignored by Git

The following files should not normally be committed:

```text
__pycache__/
*.pyc

.venv/

data/uploads/
data/outputs/

*.db
```

These are managed through `.gitignore`.

---

# Installation

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Frontend

```bash
python frontend/app.py
```

## Run FastAPI Backend (if applicable)

```bash
uvicorn backend.main:app --reload
```

Use the appropriate startup command based on your application structure.

---

# Authentication Workflow

1. Start the application.
2. Upload suspect carton image.
3. Load authentic reference image.
4. Execute comparison process.
5. Review detected differences.
6. Examine highlighted regions.
7. Generate verification report.
8. Determine whether packaging is suspicious.

---

# Limitations

This system performs packaging-level verification only.

It does **not** guarantee counterfeit medicine detection.

Additional validation methods may include:

- QR code verification
- Barcode verification
- Manufacturer database verification
- Batch verification
- Regulatory validation
- Laboratory testing

---

# Future Enhancements

- Deep learning-based counterfeit detection
- Logo verification module
- Barcode validation
- QR code validation
- Multi-product reference database
- Automated confidence scoring
- Explainable AI reporting
- Cloud deployment
- Mobile application support

---

# Author

Developed as a Pharmaceutical Carton Authentication and Verification System for detecting visual differences between authentic and suspect medicine packaging.
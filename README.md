Pharmaceutical Carton Authentication System

This project compares an uploaded medicine carton image with an authentic reference carton image and detects possible differences such as text mismatch, font/typography changes, alignment issues, crop differences, and visual inconsistencies.

It is mainly built for verifying pharmaceutical cartons like PAN-D by comparing fake/suspect carton images against a stored original reference image.

---

## What this project does

The system:

1. Takes an uploaded carton image.
2. Loads the original/authentic reference image.
3. Runs OCR to extract printed text from both images.
4. Compares text, line positions, spacing, alignment, and typography.
5. Detects visual/crop-based differences.
6. Highlights suspicious regions.
7. Generates output images/reports showing detected issues.

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

###Main Files Explained


##frontend/app.py

This is the user interface part of the project.

It allows the user to:

Upload carton images.
Select or compare against a reference image.
Run the verification process.
View detected differences.
See output images and report results.

In simple words, this is the screen/app that the user interacts with.

##backend/pipeline.py

This is the main processing pipeline.

It controls the full authentication flow:

Loads the reference image.
Loads the uploaded/suspect image.
Runs OCR.
Calls comparison modules.
Collects all detected issues.
Filters unwanted/false detections.
Sends final result back to the frontend.

This is the brain of the backend.

##backend/remove.py

This file is used for image cleanup or preprocessing.

Depending on the implementation, it may help remove unwanted background/noise or prepare the image before OCR/comparison.

###OCR Files

##ocr/ocr_engine.py

This file handles OCR.

OCR means Optical Character Recognition.

It reads text from carton images using OCR tools like PaddleOCR.

It extracts:

Text
Confidence score
Bounding boxes
Text position
Text orientation

This is important because most differences in medicine cartons are related to printed text.

##ocr/field_extractor.py

This file extracts important text fields from OCR results.

For example:

Medicine name
Batch number
MRP
Manufacturing date
Expiry date
Manufacturer details
License number

It helps convert raw OCR text into useful fields.

##ocr/crop_refiner.py

This file improves cropped OCR regions.

It helps when small or blurry text is not detected properly.

It may crop specific text regions, rotate them if needed, and make them easier for OCR or comparison.

###Comparison Files

##comparison/text_compare.py

This compares extracted text from the reference image and uploaded image.

It checks things like:

Missing text
Extra text
Wrong spelling
Changed characters
Different words
OCR-based text mismatches

Example:

Reference: TABLET
Uploaded: TABLETS

This file detects that difference.

##comparison/typography.py

This checks visual text style differences.

It compares things like:

Font size
Font thickness
Boldness
Spacing
Text height
Text width
Alignment
Color/contrast difference

This is useful when the text is the same but the printing style is different.

Example:

Reference and fake both say "PARACETAMOL"
but fake text is shifted, thinner, or spaced differently.
comparison/line_compare.py

This compares line-level alignment.

It checks whether text lines are placed correctly.

It helps detect:

Text shifted up/down
Text shifted left/right
Different line spacing
Misaligned printed blocks

This is important because fake cartons may copy the same text but fail in exact placement.

##comparison/anchor_match.py

This file matches important fixed points between the reference and uploaded images.

These fixed points are called anchors.

Examples:

Medicine name location
Logo location
Main text block location
Important label position

It helps align both images before comparing them.

##comparison/crop_match.py

This compares cropped regions of the reference and uploaded image.

Instead of comparing the whole image at once, it compares smaller important parts.

This improves accuracy because small differences can be missed in full-image comparison.

It helps detect:

Printing differences
Missing symbols
Layout changes
Region-level mismatches
comparison/strict_ocr_matcher.py

This performs stricter OCR text matching.

It is useful when normal fuzzy matching is too lenient.

It helps reduce cases where wrong text is accepted just because it looks somewhat similar.

##comparison/dictionary_correction.py

OCR sometimes reads text incorrectly.

Example:

0 instead of O
1 instead of I
5 instead of S

This file corrects common OCR mistakes before comparison.

It improves text comparison accuracy.

###Data Folder

##data/reference_images/

Stores original/authentic carton images.

These images are used as the trusted reference for comparison.

##data/uploads/

Stores images uploaded by the user during testing.

These should usually not be pushed to GitHub.

##data/outputs/

Stores generated results.

This may include:

Highlighted difference images
Cropped issue images
Report outputs

These should usually not be pushed to GitHub.

##data/reference_db.json

Stores reference image details and metadata.

It may contain information about which product uses which reference image.

###Database

##database/carton_auth.db

This is the local database file.

It may store:

Uploaded image details
Verification history
Product information
Result records

Usually, .db files should not be pushed unless needed for demo/sample data.

Files that should not be pushed to GitHub

The following files are generated automatically and should be ignored:

__pycache__/
*.pyc
data/uploads/
data/outputs/
*.db
.venv/

These are ignored using .gitignore.

---How to run the project

Activate virtual environment:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---Run the frontend/app:

###python frontend/app.py

If the backend uses FastAPI, run:

uvicorn backend.main:app --reload

Use the correct command depending on your actual app structure.

Basic workflow
Start the app.
Upload suspect carton image.
Select authentic reference image.
Run comparison.
View detected issues.
Check highlighted image/report.
Decide whether carton is authentic or suspicious.
Important Note

This project does not guarantee 100% fake medicine detection.

It mainly detects visible packaging differences.

Final verification should also include:

QR/barcode validation
Batch number verification
Manufacturer database check
Chemical/lab testing
Regulatory verification

This system is useful as a first-level visual authentication tool.
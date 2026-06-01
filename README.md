# Pharmaceutical Carton Authentication System

Offline Windows-friendly Streamlit application for comparing an authentic carton image with a suspect carton image.

## What v1 does well

- Preserves uploaded images exactly.
- Creates only a mild enhanced color image and OCR overlay.
- Runs OCR on original and enhanced image, then chooses the better result.
- Extracts medicine-related fields using OCR + fuzzy matching.
- Blocks comparison when medicines appear different.
- Detects text differences: missing, extra, substituted characters, punctuation, spacing, and capitalization.
- Produces visual evidence crops for each difference.
- Calculates conservative authenticity score.
- Generates a PDF report.

## What v1 intentionally does not do

- No automatic perspective correction.
- No aggressive thresholding.
- No black-and-white conversion.
- No blind pixel-level comparison.
- No paid APIs, Docker, cloud services, or external endpoints.

## Setup on Windows

Create and activate environment:

```powershell
cd pharma_carton_auth_system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PaddlePaddle fails, install the CPU wheel directly:

```powershell
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Run:

```powershell
streamlit run frontend/app.py
```

## Important Windows barcode note

`pyzbar` may require the ZBar runtime. OCR comparison works without it. Barcode/QR decoding will show as unavailable if ZBar is not installed.

## Project layout

```text
frontend/
backend/
ocr/
comparison/
visual_analysis/
database/
reports/
utils/
data/reference_images/
data/uploads/
data/outputs/
```

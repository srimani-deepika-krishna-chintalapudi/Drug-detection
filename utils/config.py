from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
OUTPUT_DIR = DATA_DIR / 'outputs'
REFERENCE_DIR = DATA_DIR / 'reference_images'
DB_PATH = BASE_DIR / 'database' / 'carton_auth.db'

for p in [UPLOAD_DIR, OUTPUT_DIR, REFERENCE_DIR, DB_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

MEDICINE_CONFIDENCE_THRESHOLD = 80
TEXT_MATCH_THRESHOLD = 85
OCR_MIN_CONFIDENCE = 0.40
VISUAL_DIFF_THRESHOLD = 0.18

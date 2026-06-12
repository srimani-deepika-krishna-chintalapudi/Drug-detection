import sqlite3
from pathlib import Path
from datetime import datetime
from utils.config import DB_PATH


def init_db():
    schema = Path(__file__).with_name('schema.sql').read_text(encoding='utf-8')
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema)


def create_comparison(authentic_image, suspect_image, medicine_name, medicine_confidence, score, verdict, report_path=None):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''INSERT INTO comparisons(created_at, authentic_image, suspect_image, medicine_name, medicine_confidence, authenticity_score, verdict, report_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (datetime.now().isoformat(timespec='seconds'), str(authentic_image), str(suspect_image), medicine_name, medicine_confidence, score, verdict, str(report_path) if report_path else None))
        return cur.lastrowid


def add_difference(comparison_id, issue, ref_crop=None, suspect_crop=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''INSERT INTO differences(comparison_id, issue_type, reference_text, suspect_text, difference, severity, confidence, ref_crop_path, suspect_crop_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (comparison_id, issue.get('issue_type'), issue.get('reference',''), issue.get('uploaded',''), issue.get('difference',''), issue.get('severity',''), issue.get('confidence',0), str(ref_crop) if ref_crop else None, str(suspect_crop) if suspect_crop else None))

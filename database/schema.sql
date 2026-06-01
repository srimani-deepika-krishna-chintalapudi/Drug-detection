CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    authentic_image TEXT NOT NULL,
    suspect_image TEXT NOT NULL,
    medicine_name TEXT,
    medicine_confidence REAL,
    authenticity_score REAL,
    verdict TEXT,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS differences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id INTEGER NOT NULL,
    issue_type TEXT,
    reference_text TEXT,
    suspect_text TEXT,
    difference TEXT,
    severity TEXT,
    confidence REAL,
    ref_crop_path TEXT,
    suspect_crop_path TEXT,
    FOREIGN KEY(comparison_id) REFERENCES comparisons(id)
);

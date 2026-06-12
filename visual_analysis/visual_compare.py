from typing import Dict, List
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def color_distance_lab(a_bgr, b_bgr):
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    return float(np.linalg.norm(a - b))


def compare_overall_visual(ref_bgr, sus_bgr):
    # Resize suspect to reference only for coarse visual estimate, not for evidence.
    h, w = ref_bgr.shape[:2]
    sus_r = cv2.resize(sus_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gray1 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(sus_r, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    color_delta = color_distance_lab(ref_bgr, sus_r)
    issues = []
    if color_delta > 18:
        issues.append({'issue_type':'color','difference':f'Visible color shift detected, LAB distance {color_delta:.1f}', 'severity':'Low', 'confidence':min(90, color_delta*3), 'ref_bbox':None, 'suspect_bbox':None})
    if score < 0.72:
        issues.append({'issue_type':'layout','difference':f'Large visual/layout mismatch, SSIM {score:.2f}', 'severity':'Medium', 'confidence':round((1-score)*100,2), 'ref_bbox':None, 'suspect_bbox':None})
    return {'ssim': float(score), 'color_delta': color_delta, 'issues': issues}

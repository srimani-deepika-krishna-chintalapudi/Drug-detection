def score_authenticity(issues, visual_result=None):
    score = 100.0
    weights = {
        'dosage': 25, 'manufacturer': 22, 'composition': 20, 'barcode': 25, 'qr': 25,
        'missing_text': 18, 'extra_text': 10, 'text': 14, 'typography': 7,
        'logo': 18, 'layout': 10, 'color': 4
    }
    severity_factor = {'High': 1.0, 'Medium': 0.55, 'Low': 0.25}
    for issue in issues:
        it = str(issue.get('issue_type', 'text')).lower()
        base = next((v for k, v in weights.items() if k in it), weights.get(it, 10))
        factor = severity_factor.get(issue.get('severity'), 0.5)
        conf = min(1, float(issue.get('confidence', 70)) / 100)
        score -= base * factor * conf
    return max(0, min(100, score))


def verdict_from_score(score):
    if score >= 95:
        return 'Original'
    if score >= 70:
        return 'Suspect'
    return 'Potential Counterfeit'

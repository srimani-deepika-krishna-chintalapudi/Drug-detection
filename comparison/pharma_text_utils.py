import re

def is_critical_pharma_text(text, medicine_name=None):
    text_lower = str(text or "").lower()
    
    if medicine_name:
        med_clean = re.sub(r"[^a-z0-9]", "", medicine_name.lower())
        text_clean = re.sub(r"[^a-z0-9]", "", text_lower)
        if len(med_clean) >= 4 and med_clean in text_clean:
            return True

    critical_terms = [
        "mrp", "mfg", "exp", "batch", "b.no", "bno", "lic", "license",
        "schedule", "composition", "mg", "ml", "tablet", "capsule", "syrup"
    ]
    for term in critical_terms:
        if term in text_lower:
            return True
    return False

def is_address_or_location_text(text):
    text_lower = str(text or "").lower()
    address_terms = [
        "baddi", "solan", "h.p.", "hp", "distt", "village", "road", "plot", 
        "industrial", "area", "pradesh", "india", "khurd", "bhatouli", 
        "bhatauli", "mumbai", "khasra", "tehsil", "mandal", "nagar", "marg",
        "estate", "sector", "phase"
    ]
    for term in address_terms:
        if term in text_lower:
            return True
    return False

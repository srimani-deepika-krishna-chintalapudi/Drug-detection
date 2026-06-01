import re
from typing import Dict, List
from rapidfuzz import fuzz, process


# Brand names / trade names.
# These are the carton identity.
BRAND_DICTIONARY = [
    "dolo-650",
    "dolo",
    "pan-d",
    "cetirizine",
    "amoxicillin",
    "azithromycin",
    "ibuprofen",
    "metformin",
    "atorvastatin",
]

# Composition / generic drug names.
# These should NOT override brand name.
COMPOSITION_DICTIONARY = [
    "paracetamol",
    "acetaminophen",
    "pantoprazole",
    "domperidone",
    "cetirizine hydrochloride",
    "amoxicillin",
    "azithromycin",
    "ibuprofen",
    "metformin",
    "atorvastatin",
]

COMMON_CORRECTIONS = {
    "cetrizine": "cetirizine",
    "cetrezine": "cetirizine",
    "amoxycillin": "amoxicillin",
    "paracetemol": "paracetamol",
    "parocetamol": "paracetamol",
    "poracetamol": "paracetamol",
    "pan d": "pan-d",
    "dolo 650": "dolo-650",
    "dolo650": "dolo-650",
    "dolo-65o": "dolo-650",
    "dolo-o50": "dolo-650",
}

FIELD_PATTERNS = {
    "batch_number": r"(?i)(batch|b\.no|b no|lot)[:\s\-]*([A-Z0-9\-/]+)",
    "manufacturing_date": r"(?i)(mfg|mfd|manufactur(?:e|ing))[:\s\-]*([0-9]{1,2}[./\-][0-9]{2,4})",
    "expiry_date": r"(?i)(exp|expiry|use before)[:\s\-]*([0-9]{1,2}[./\-][0-9]{2,4})",
    "dosage": r"(?i)\b([0-9]+\s?(?:mg|mcg|g|ml|iu))\b",
}


def normalize_for_search(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\n", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9\-./ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_brand_variant(text: str) -> str:
    text = normalize_for_search(text)
    text = text.replace("dolo 650", "dolo-650")
    text = text.replace("dolo650", "dolo-650")
    text = text.replace("pan d", "pan-d")
    return text


def find_exact_brand(text: str):
    text = normalize_brand_variant(text)

    # Strong brand patterns first.
    if re.search(r"\bdolo\s*[-]?\s*650\b", text):
        return {
            "name": "dolo-650",
            "confidence": 100.0,
            "source": "brand_exact:dolo-650",
        }

    if re.search(r"\bpan\s*[-]?\s*d\b", text):
        return {
            "name": "pan-d",
            "confidence": 100.0,
            "source": "brand_exact:pan-d",
        }

    for brand in BRAND_DICTIONARY:
        if re.search(rf"\b{re.escape(brand)}\b", text):
            return {
                "name": brand,
                "confidence": 100.0,
                "source": f"brand_exact:{brand}",
            }

    return None


def find_fuzzy_brand(text: str):
    text = normalize_brand_variant(text)

    candidates = []
    words = text.split()

    # Whole text candidate.
    candidates.append(text)

    # Single and pair tokens.
    for i in range(len(words)):
        candidates.append(words[i])
        if i + 1 < len(words):
            candidates.append(words[i] + " " + words[i + 1])
            candidates.append(words[i] + "-" + words[i + 1])

    best = None

    for cand in candidates:
        match = process.extractOne(
            cand,
            BRAND_DICTIONARY,
            scorer=fuzz.ratio,
        )

        if match:
            name, score, _ = match
            if best is None or score > best[1]:
                best = (name, score, cand)

    if best and best[1] >= 82:
        return {
            "name": best[0],
            "confidence": float(best[1]),
            "source": f"brand_fuzzy:{best[2]}",
        }

    return None


def find_composition(text: str):
    text = normalize_for_search(text)

    found = []

    for wrong, right in COMMON_CORRECTIONS.items():
        if wrong in text and right in COMPOSITION_DICTIONARY:
            found.append(right)

    for comp in COMPOSITION_DICTIONARY:
        if comp in text:
            found.append(comp)

    # Fuzzy typo recovery for composition words.
    words = text.split()
    for word in words:
        match = process.extractOne(
            word,
            COMPOSITION_DICTIONARY,
            scorer=fuzz.ratio,
        )
        if match:
            name, score, _ = match
            if score >= 86:
                found.append(name)

    unique = []
    for x in found:
        if x not in unique:
            unique.append(x)

    return ", ".join(unique) if unique else None


def identify_medicine(full_text: str):
    """
    Important rule:
    Brand/trade name wins over composition.

    Example:
    Dolo-650 carton contains Paracetamol.
    Medicine name should be Dolo-650, NOT Paracetamol.
    """

    text = normalize_brand_variant(full_text)

    # Apply corrections, but don't let composition correction override brand.
    corrected_text = text
    for wrong, right in COMMON_CORRECTIONS.items():
        corrected_text = corrected_text.replace(wrong, right)

    exact_brand = find_exact_brand(corrected_text)
    if exact_brand:
        return exact_brand

    fuzzy_brand = find_fuzzy_brand(corrected_text)
    if fuzzy_brand:
        return fuzzy_brand

    composition = find_composition(corrected_text)
    if composition:
        return {
            "name": None,
            "composition": composition,
            "confidence": 60.0,
            "source": "composition_only_not_brand",
        }

    return {
        "name": None,
        "confidence": 0.0,
        "source": "none",
    }


def extract_fields(full_text: str, boxes: List[Dict]) -> Dict:
    text = str(full_text or "").replace("\n", " ")

    medicine_info = identify_medicine(full_text)

    fields = {
        "medicine": medicine_info,
        "dosage": None,
        "manufacturer": None,
        "composition": None,
        "warnings": None,
        "batch_number": None,
        "manufacturing_date": None,
        "expiry_date": None,
        "barcode_text": None,
        "qr_text": None,
    }

    for key, pat in FIELD_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            fields[key] = m.group(m.lastindex)

    comp = re.search(
        r"(?i)(composition|contains|each.*contains)[:\s]*(.{5,160})",
        text,
    )
    if comp:
        fields["composition"] = comp.group(2).strip()
    else:
        detected_comp = medicine_info.get("composition") or find_composition(full_text)
        if detected_comp:
            fields["composition"] = detected_comp

    warn = re.search(
        r"(?i)(warning|caution|schedule|keep out of reach)(.{0,160})",
        text,
    )
    if warn:
        fields["warnings"] = (warn.group(1) + warn.group(2)).strip()

    manu = re.search(
        r"(?i)(manufactured by|mfd by|marketed by|made in india by)[:\s]*(.{3,100})",
        text,
    )
    if manu:
        fields["manufacturer"] = manu.group(2).strip()

    return fields
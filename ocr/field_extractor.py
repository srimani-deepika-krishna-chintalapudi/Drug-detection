import re
from typing import Dict, List
from rapidfuzz import fuzz, process


BRAND_DICTIONARY = [
    "dolo-650", "dolo", "pan-d", "cetirizine", "amoxicillin",
    "azithromycin", "ibuprofen", "metformin", "atorvastatin",
]

COMPOSITION_DICTIONARY = [
    "paracetamol", "acetaminophen", "pantoprazole", "domperidone",
    "cetirizine hydrochloride", "amoxicillin", "azithromycin",
    "ibuprofen", "metformin", "atorvastatin",
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
    "8.no": "b.no",
    "8 no": "b.no",
    "bn0": "bno",
    "b.n0": "b.no",
    "batch n0": "batch no",
    "botch": "batch",
    "8atch": "batch",
}

FIELD_PATTERNS = {
    "batch_number": r"(?i)(batch|b\.?\s*no\.?|bno|lot)[^\w]{0,10}([A-Z0-9\-/]{3,24})",
    "manufacturing_date": r"(?i)(mfg|mfd|manufactur(?:e|ing))[^\d]{0,12}([0-9]{1,2}[./\-][0-9]{2,4})",
    "expiry_date": r"(?i)(exp|expiry|use before)[^\d]{0,12}([0-9]{1,2}[./\-][0-9]{2,4})",
    "dosage": r"(?i)\b([0-9]+\s?(?:mg|mcg|g|ml|iu))\b",
    "mrp": r"(?i)(mrp|m\.r\.p\.?)[^\d]{0,12}([0-9]+(?:\.[0-9]{1,2})?)",
}


def normalize_for_search(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\n", " ")
    text = text.replace("–", "-").replace("—", "-")

    for wrong, right in COMMON_CORRECTIONS.items():
        text = text.replace(wrong, right)

    text = re.sub(r"[^a-z0-9\-./ ₹rs]+", " ", text)
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

    if re.search(r"\bdolo\s*[-]?\s*650\b", text):
        return {"name": "dolo-650", "confidence": 100.0, "source": "brand_exact:dolo-650"}

    if re.search(r"\bpan\s*[-]?\s*d\b", text):
        return {"name": "pan-d", "confidence": 100.0, "source": "brand_exact:pan-d"}

    for brand in BRAND_DICTIONARY:
        if re.search(rf"\b{re.escape(brand)}\b", text):
            return {"name": brand, "confidence": 100.0, "source": f"brand_exact:{brand}"}

    return None


def find_fuzzy_brand(text: str):
    text = normalize_brand_variant(text)
    words = text.split()

    candidates = [text]

    for i in range(len(words)):
        candidates.append(words[i])
        if i + 1 < len(words):
            candidates.append(words[i] + " " + words[i + 1])
            candidates.append(words[i] + "-" + words[i + 1])

    best = None

    for cand in candidates:
        match = process.extractOne(cand, BRAND_DICTIONARY, scorer=fuzz.ratio)
        if match:
            name, score, _ = match
            if best is None or score > best[1]:
                best = (name, score, cand)

    if best and best[1] >= 82:
        return {"name": best[0], "confidence": float(best[1]), "source": f"brand_fuzzy:{best[2]}"}

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

    for word in text.split():
        match = process.extractOne(word, COMPOSITION_DICTIONARY, scorer=fuzz.ratio)
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
    text = normalize_brand_variant(full_text)

    exact_brand = find_exact_brand(text)
    if exact_brand:
        return exact_brand

    fuzzy_brand = find_fuzzy_brand(text)
    if fuzzy_brand:
        return fuzzy_brand

    composition = find_composition(text)
    if composition:
        return {
            "name": None,
            "composition": composition,
            "confidence": 60.0,
            "source": "composition_only_not_brand",
        }

    return {"name": None, "confidence": 0.0, "source": "none"}


def clean_candidate_value(value: str) -> str:
    value = str(value or "").strip()
    value = value.replace("_", "")
    value = value.replace("—", "-").replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :.-")


def is_date_like(value: str) -> bool:
    value = clean_candidate_value(value)
    return bool(re.fullmatch(r"[0-9]{1,2}[./\-][0-9]{2,4}", value))


def is_batch_like(value: str) -> bool:
    value = clean_candidate_value(value).upper()

    if re.search(r"(B\.?NO|BATCH|LOT|MFG|MFD|EXP|MRP|RS|₹)", value, re.I):
        return False

    if is_date_like(value):
        return False

    if is_mrp_like(value):
        return False

    return bool(re.fullmatch(r"[A-Z0-9\-\/]{3,24}", value, re.I))


def is_mrp_like(value: str) -> bool:
    value = clean_candidate_value(value)
    return bool(re.fullmatch(r"(₹|RS\.?)?\s*[0-9]+(?:\.[0-9]{1,2})?", value, re.I))


def nearest_value_from_boxes(boxes: List[Dict], keywords: List[str], value_type="text"):
    if not boxes:
        return None

    for i, box in enumerate(boxes):
        label = str(box.get("text", "")).upper()
        label_norm = label.replace(".", "").replace(" ", "")

        keyword_hit = False
        for k in keywords:
            k_norm = k.upper().replace(".", "").replace(" ", "")
            if k_norm in label_norm:
                keyword_hit = True
                break

        if not keyword_hit:
            continue

        candidates = []

        for j in range(max(0, i - 10), min(len(boxes), i + 15)):
            if j == i:
                continue

            candidate = clean_candidate_value(boxes[j].get("text", ""))

            if not candidate:
                continue

            if value_type == "date" and is_date_like(candidate):
                candidates.append(candidate)

            elif value_type == "batch" and is_batch_like(candidate):
                candidates.append(candidate)

            elif value_type == "mrp" and is_mrp_like(candidate):
                candidates.append(candidate)

            elif value_type == "text":
                candidates.append(candidate)

        if candidates:
            return candidates[0]

    return None


def extract_batch_number(text):
    text = normalize_for_search(text)

    text = text.replace("b no", "bno")
    text = text.replace("b. no", "bno")
    text = text.replace("b.no", "bno")
    text = text.replace("batch no", "batch")
    text = text.replace("lot no", "lot")

    patterns = [
        r"\b(?:batch|bno|lot)\s*[:\-]?\s*([a-z0-9\/\-]{3,24})",
        r"\b([a-z]{1,4}\d{3,12})\b",
        r"\b([0-9]{2,5}[a-z]{1,4}[0-9]{1,8})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = clean_candidate_value(match.group(1)).upper()
            if is_batch_like(candidate):
                return candidate

    return None


def extract_fields(full_text: str, boxes: List[Dict]) -> Dict:
    text = str(full_text or "").replace("\n", " ")
    normalized_text = normalize_for_search(text)

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
        "mrp": None,
        "barcode_text": None,
        "qr_text": None,
    }

    for key, pat in FIELD_PATTERNS.items():
        m = re.search(pat, normalized_text, re.I)
        if m:
            fields[key] = clean_candidate_value(m.group(m.lastindex)).upper()

    if not fields["batch_number"]:
        fields["batch_number"] = extract_batch_number(normalized_text)

    if not fields["batch_number"]:
        fields["batch_number"] = nearest_value_from_boxes(
            boxes,
            ["B.NO", "B NO", "BNO", "BATCH", "LOT", "LOT NO"],
            value_type="batch",
        )

    if not fields["manufacturing_date"]:
        fields["manufacturing_date"] = nearest_value_from_boxes(
            boxes,
            ["MFG", "MFD", "MFG.", "MFD.", "MANUFACTURED"],
            value_type="date",
        )

    if not fields["expiry_date"]:
        fields["expiry_date"] = nearest_value_from_boxes(
            boxes,
            ["EXP", "EXPIRY", "EXP.", "EXPIRY.", "USE BEFORE"],
            value_type="date",
        )

    if not fields["mrp"]:
        fields["mrp"] = nearest_value_from_boxes(
            boxes,
            ["MRP", "M.R.P", "M R P"],
            value_type="mrp",
        )

    comp = re.search(r"(?i)(composition|contains|each.*contains)[:\s]*(.{5,160})", text)
    if comp:
        fields["composition"] = comp.group(2).strip()
    else:
        detected_comp = medicine_info.get("composition") or find_composition(full_text)
        if detected_comp:
            fields["composition"] = detected_comp

    warn = re.search(r"(?i)(warning|caution|schedule|keep out of reach)(.{0,160})", text)
    if warn:
        fields["warnings"] = (warn.group(1) + warn.group(2)).strip()

    manu = re.search(
        r"(?i)(manufactured by|mfd by|marketed by|made in india by)[:\s]*(.{3,100})",
        text,
    )
    if manu:
        fields["manufacturer"] = manu.group(2).strip()

    return fields
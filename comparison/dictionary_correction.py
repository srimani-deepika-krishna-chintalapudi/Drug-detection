from rapidfuzz import process

MEDICINE_WORDS = [
    "PARACETAMOL",
    "TABLET",
    "TABLETS",
    "CAPSULE",
    "CAPSULES",
    "DOMPERIDONE",
    "PANTOPRAZOLE",
    "DOLO",
    "PAN-D"
]

def correct_word(word):
    if len(word) < 4:
        return word

    match = process.extractOne(
        word.upper(),
        MEDICINE_WORDS
    )

    if match and match[1] > 90:
        return match[0]

    return word
from pathlib import Path
import json
import streamlit as st

from backend.pipeline import compare_cartons
from utils.image_io import save_uploaded_file


MEDICINE_NAMES = [
    "Dolo-650",
    "Dolo",
    "Dolo 500",
    "Dolo 250",
    "Dolo Cold",
    "Dolo Drops",
    "Pan-D",
    "Pan D",
    "Pan D 650",
    "Paracetamol",
    "Cetirizine",
    "Amoxicillin",
    "Azithromycin",
    "Ibuprofen",
    "Metformin",
    "Atorvastatin",
]

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REFERENCE_DIR = DATA_DIR / "reference_images"
REFERENCE_DB = DATA_DIR / "reference_db.json"

ADMIN_EMAIL = "admin@ex.com"
ADMIN_PASSWORD = "admin123"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_reference_db():
    if not REFERENCE_DB.exists():
        return {}
    try:
        return json.loads(REFERENCE_DB.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_reference_db(db):
    REFERENCE_DB.write_text(json.dumps(db, indent=4), encoding="utf-8")


def normalize_name(name):
    return str(name or "").strip().lower().replace(" ", "").replace("-", "")


def get_all_medicine_names():
    db = load_reference_db()
    names = [item["medicine_name"] for item in db.values()]

    for name in MEDICINE_NAMES:
        if name not in names:
            names.append(name)

    return sorted(names)


def find_reference(medicine_name):
    db = load_reference_db()
    key = normalize_name(medicine_name)

    if key in db:
        ref_path = Path(db[key]["image_path"])
        if ref_path.exists():
            return db[key], ref_path

    return None, None


def save_reference_image(medicine_name, uploaded_file):
    db = load_reference_db()

    key = normalize_name(medicine_name)
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    ref_path = REFERENCE_DIR / f"{key}{suffix}"

    with open(ref_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    db[key] = {
        "medicine_name": medicine_name.strip(),
        "image_path": str(ref_path),
    }

    save_reference_db(db)
    return ref_path


def medicine_autocomplete(label="Medicine name", key_prefix="medicine"):
    all_medicines = get_all_medicine_names()

    selected = st.selectbox(
        label,
        options=[""] + all_medicines,
        index=0,
        key=f"{key_prefix}_selectbox",
        placeholder="Type medicine name..."
    )

    return selected


st.set_page_config(
    page_title="Pharma Carton Authentication",
    layout="wide",
)

st.title("Pharmaceutical Carton Authentication System")

portal = st.sidebar.radio(
    "Choose Portal",
    ["User Portal", "Admin Portal"]
)


if portal == "User Portal":
    st.header("User Portal")

    selected_medicine = medicine_autocomplete(
        "Enter medicine name",
        key_prefix="user_medicine"
    )

    scan_mode = st.checkbox(
        "Use Document Scan Mode (CamScanner-style)",
        value=False
    )

    with st.expander("📘 When should I use Document Scan Mode?"):
        st.markdown("""
### ✅ Recommended
- Blurry photos
- Tilted / angled cartons
- Shadowed images
- Low-light photos
- Mobile camera captures

### ❌ Not Recommended
- Already clear images
- Downloaded images
- Reference images
- High-quality carton photos

### What happens?
1. Detect carton/document area
2. Straighten perspective
3. Reduce shadows
4. Sharpen blurry text
5. Improve OCR extraction

⚠️ For clear images, keep this OFF to avoid unnecessary image modifications.
""")

    suspect_file = st.file_uploader(
        "Upload carton image for verification",
        type=["jpg", "jpeg", "png", "webp"],
        key="suspect_upload",
    )

    if st.button("Verify Carton", type="primary"):
        if not selected_medicine.strip():
            st.error("Please enter medicine name.")
            st.stop()

        ref_meta, ref_path = find_reference(selected_medicine)

        if ref_path is None:
            st.error(
                "No reference image found for this medicine. "
                "Ask admin to upload it first."
            )
            st.stop()

        if suspect_file is None:
            st.error("Please upload carton image.")
            st.stop()

        suspect_path = save_uploaded_file(suspect_file, UPLOAD_DIR)

        st.success(f"Reference found: {ref_meta['medicine_name']}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Authentic Reference")
            st.image(str(ref_path), width="stretch")

        with col2:
            st.subheader("Uploaded Carton")
            st.image(str(suspect_path), width="stretch")

        with st.spinner("Comparing carton with reference..."):
            result = compare_cartons(
                ref_path,
                suspect_path,
                scan_mode=scan_mode
            )

        st.success(
            f"Verdict: {result['verdict']} | "
            f"Score: {result['score']:.1f}/100"
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Images", "OCR Fields", "Differences", "Report"]
        )

        with tab1:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Authentic Reference")
                st.image(str(ref_path), width="stretch")

                st.markdown("### Authentic OCR Overlay")
                st.image(
                    result["paths"]["authentic_ocr_overlay"],
                    width="stretch"
                )

            with c2:
                st.markdown("### Uploaded Carton")
                st.image(str(suspect_path), width="stretch")

                st.markdown("### Uploaded OCR Overlay")
                st.image(
                    result["paths"]["suspect_ocr_overlay"],
                    width="stretch"
                )

        with tab2:
            st.subheader("Extracted OCR Fields")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Authentic OCR Text")
                st.text_area(
                    "Authentic Full OCR",
                    value=getattr(result["ref_ocr"], "full_text", ""),
                    height=250,
                    disabled=True,
                )

                st.markdown("### Authentic Extracted Fields")
                st.json(result["ref_fields"])

            with c2:
                st.markdown("### Uploaded OCR Text")
                st.text_area(
                    "Uploaded Full OCR",
                    value=getattr(result["sus_ocr"], "full_text", ""),
                    height=250,
                    disabled=True,
                )

                st.markdown("### Uploaded Extracted Fields")
                st.json(result["sus_fields"])

        with tab3:
            st.subheader("Detected Differences with Evidence")

            if not result["issues"]:
                st.info("No significant differences detected.")
            else:
                for i, issue in enumerate(result["issues"], 1):
                    st.markdown(
                        f"## Difference {i}: "
                        f"{issue.get('severity', '')} - "
                        f"{issue.get('issue_type', '')}"
                    )

                    st.write("**Severity:**", issue.get("severity"))
                    st.write("**Reference:**", issue.get("reference"))
                    st.write("**Uploaded:**", issue.get("uploaded"))
                    st.write("**Difference:**", issue.get("difference"))
                    st.write("**Confidence:**", issue.get("confidence"))

                    pair = (
                        result["evidence_pairs"].get(str(i - 1))
                        or result["evidence_pairs"].get(i - 1)
                    )

                    if pair:
                        c1, c2 = st.columns(2)

                        with c1:
                            st.markdown("### Reference Evidence")
                            if pair.get("ref"):
                                st.image(pair["ref"], width="stretch")
                            else:
                                st.info("No reference crop available")

                        with c2:
                            st.markdown("### Uploaded Evidence")
                            if pair.get("sus"):
                                st.image(pair["sus"], width="stretch")
                            else:
                                st.info("No uploaded crop available")
                    else:
                        st.info("No evidence crop available for this difference.")

                    st.divider()

        with tab4:
            st.subheader("PDF Report")

            st.write(f"Verdict: **{result['verdict']}**")
            st.write(f"Score: **{result['score']:.1f}/100**")

            if result.get("report_path"):
                with open(result["report_path"], "rb") as f:
                    st.download_button(
                        "Download PDF Report",
                        f,
                        file_name="authentication_report.pdf",
                        mime="application/pdf",
                    )


elif portal == "Admin Portal":
    st.header("Admin Portal")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        email = st.text_input("Admin Email")
        password = st.text_input("Admin Password", type="password")

        if st.button("Login as Admin"):
            if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Admin login successful.")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")

    else:
        st.success("Logged in as Admin")

        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.subheader("Upload Authentic Reference Image")

        medicine_name = medicine_autocomplete(
            "Medicine name",
            key_prefix="admin_medicine"
        )

        reference_file = st.file_uploader(
            "Upload authentic reference carton",
            type=["jpg", "jpeg", "png", "webp"],
            key="reference_upload",
        )

        if st.button("Save Reference Image"):
            if not medicine_name.strip():
                st.error("Please enter medicine name.")
                st.stop()

            if reference_file is None:
                st.error("Please upload reference image.")
                st.stop()

            ref_path = save_reference_image(medicine_name, reference_file)
            st.success(f"Reference image saved for {medicine_name}")
            st.image(str(ref_path), caption="Saved Reference", width="stretch")

        st.subheader("Current Reference Database")

        db = load_reference_db()

        if not db:
            st.info("No reference images uploaded yet.")
        else:
            for key, item in db.items():
                st.markdown(f"### {item['medicine_name']}")
                st.write(item["image_path"])

                if Path(item["image_path"]).exists():
                    st.image(item["image_path"], width=300)
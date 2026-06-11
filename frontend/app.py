from pathlib import Path
import json
import sys
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline import compare_cartons
from utils.image_io import save_uploaded_file


MEDICINE_NAMES = [
    "Dolo-650", "Dolo", "Dolo 500", "Dolo 250", "Dolo Cold", "Dolo Drops",
    "Pan-D", "Pan D", "Pan D 650", "Paracetamol", "Cetirizine",
    "Amoxicillin", "Azithromycin", "Ibuprofen", "Metformin", "Atorvastatin",
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
        placeholder="Type medicine name...",
    )

    return selected


def inject_css():
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f766e, #2563eb);
        padding: 28px;
        border-radius: 22px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
    }

    .hero p {
        margin-top: 8px;
        font-size: 16px;
        opacity: 0.95;
    }

    .section-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    .diff-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 18px;
        border-left: 7px solid #2563eb;
    }

    .diff-card.high {
        border-left-color: #dc2626;
    }

    .diff-card.medium {
        border-left-color: #f59e0b;
    }

    .diff-card.low {
        border-left-color: #2563eb;
    }

    .diff-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    }

    .admin-box {
        background: #f8fafc;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(
    page_title="Pharma Carton Authentication",
    page_icon="💊",
    layout="wide",
)

inject_css()

st.markdown("""
<div class="hero">
    <h1>💊 Pharmaceutical Carton Authentication System</h1>
    <p>OCR-powered carton verification with typography, spacing, alignment and evidence-based counterfeit analysis.</p>
</div>
""", unsafe_allow_html=True)

portal = st.sidebar.radio(
    "Choose Portal",
    ["User Portal", "Admin Portal"],
)


if portal == "User Portal":
    st.markdown("## 🔍 User Verification Portal")

    with st.container():
        selected_medicine = medicine_autocomplete(
            "Select medicine name",
            key_prefix="user_medicine",
        )

        scan_mode = st.checkbox(
            "Use Document Scan Mode",
            value=False,
            help="Use only for blurry, tilted, low-light, or mobile-captured carton images.",
        )

        with st.expander("📘 When should I use Document Scan Mode?"):
            st.markdown("""
**Use it for:** blurry photos, tilted cartons, shadows, low light, mobile camera captures.  
**Avoid it for:** clear images, downloaded images, reference-quality photos.
""")

        suspect_file = st.file_uploader(
            "Upload carton image for verification (JPG, PNG, WEBP)",
            key="suspect_upload",
        )

    if st.button("🚀 Verify Carton", type="primary", width="stretch"):
        if not selected_medicine.strip():
            st.error("Please enter medicine name.")
            st.stop()

        ref_meta, ref_path = find_reference(selected_medicine)

        if ref_path is None:
            st.error("No reference image found for this medicine. Ask admin to upload it first.")
            st.stop()

        if suspect_file is None:
            st.error("Please upload carton image.")
            st.stop()

        allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
        if Path(suspect_file.name).suffix.lower() not in allowed_ext:
            st.error("Unsupported file type. Please upload a JPG, PNG, or WEBP image.")
            st.stop()

        suspect_path = save_uploaded_file(suspect_file, UPLOAD_DIR)

        st.success(f"Reference found: {ref_meta['medicine_name']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Authentic Reference")
            st.image(str(ref_path), use_column_width=True)

        with col2:
            st.markdown("### Uploaded Carton")
            st.image(str(suspect_path), use_column_width=True)

        with st.spinner("Analyzing carton. Running OCR, visual checks and report generation..."):
            result = compare_cartons(
                ref_path,
                suspect_path,
                scan_mode=scan_mode,
            )

        st.markdown("## ✅ Verification Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Verdict", result["verdict"])

        with c2:
            st.metric("Authenticity Score", f"{result['score']:.1f}%")

        with c3:
            st.metric("Differences Found", len(result["issues"]))

        tab1, tab2, tab3, tab4 = st.tabs([
            "🖼 Images",
            "📄 OCR Analysis",
            "⚠ Differences",
            "📑 Report",
        ])

        with tab1:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### Authentic Reference")
                st.image(str(ref_path), use_column_width=True)

                st.markdown("### Authentic OCR Overlay")
                st.image(result["paths"]["authentic_ocr_overlay"], use_column_width=True)

            with c2:
                st.markdown("### Uploaded Carton")
                st.image(str(suspect_path), use_column_width=True)

                st.markdown("### Uploaded OCR Overlay")
                st.image(result["paths"]["suspect_ocr_overlay"], use_column_width=True)

        with tab2:
            st.markdown("### Extracted OCR Fields")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Authentic OCR Text")
                st.text_area(
                    "Authentic Full OCR",
                    value=getattr(result["ref_ocr"], "full_text", ""),
                    height=280,
                    disabled=True,
                )

                st.markdown("#### Authentic Extracted Fields")
                st.json(result["ref_fields"])

            with c2:
                st.markdown("#### Uploaded OCR Text")
                st.text_area(
                    "Uploaded Full OCR",
                    value=getattr(result["sus_ocr"], "full_text", ""),
                    height=280,
                    disabled=True,
                )

                st.markdown("#### Uploaded Extracted Fields")
                st.json(result["sus_fields"])

        with tab3:
            st.markdown("### Detected Differences with Evidence")

            if not result["issues"]:
                st.info("No significant differences detected.")
            else:
                for i, issue in enumerate(result["issues"], 1):
                    severity = str(issue.get("severity", "Medium")).lower()

                    st.markdown(f"""
                    <div class="diff-card {severity}">
                        <h4>Difference {i}: {issue.get("issue_type", "Issue")}</h4>
                        <b>Severity:</b> {issue.get("severity", "")}<br>
                        <b>Reference:</b> {issue.get("reference", "")}<br>
                        <b>Uploaded:</b> {issue.get("uploaded", "")}<br>
                        <b>Finding:</b> {issue.get("difference", "")}<br>
                        <b>Confidence:</b> {issue.get("confidence", "")}%
                    </div>
                    """, unsafe_allow_html=True)

                    pair = (
                        result["evidence_pairs"].get(str(i - 1))
                        or result["evidence_pairs"].get(i - 1)
                    )

                    if pair:
                        ec1, ec2 = st.columns(2)

                        with ec1:
                            if pair.get("ref"):
                                st.image(
                                    pair["ref"],
                                    caption="Reference Evidence",
                                    use_column_width=True,
                                )
                            else:
                                st.info("No reference crop available.")

                        with ec2:
                            if pair.get("sus"):
                                st.image(
                                    pair["sus"],
                                    caption="Uploaded Evidence",
                                    use_column_width=True,
                                )
                            else:
                                st.info("No uploaded crop available.")
                    else:
                        st.info("No evidence crop available for this difference.")

        with tab4:
            st.markdown("### Authentication Report")

            ref_conf = getattr(result["ref_ocr"], "avg_confidence", 0) * 100
            sus_conf = getattr(result["sus_ocr"], "avg_confidence", 0) * 100

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("Authenticity Score", f"{result['score']:.2f}%")

            with r2:
                st.metric("Reference OCR Confidence", f"{ref_conf:.2f}%")

            with r3:
                st.metric("Uploaded OCR Confidence", f"{sus_conf:.2f}%")

            st.markdown(f"**Verdict:** {result['verdict']}")

            if result.get("report_path"):
                with open(result["report_path"], "rb") as f:
                    st.download_button(
                        "📥 Download PDF Report",
                        f,
                        file_name="authentication_report.pdf",
                        mime="application/pdf",
                        width="stretch",
                    )


elif portal == "Admin Portal":
    st.markdown("## 🛡 Admin Portal")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:
        st.markdown('<div class="admin-box">', unsafe_allow_html=True)

        email = st.text_input("Admin Email")
        password = st.text_input("Admin Password", type="password")

        if st.button("Login as Admin", type="primary", width="stretch"):
            if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Admin login successful.")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.success("Logged in as Admin")

        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.markdown("### Upload Authentic Reference Image")

        medicine_name = st.text_input(
            "Medicine name",
            placeholder="Enter new or existing medicine name",
            key="admin_medicine_name",
        )

        reference_file = st.file_uploader(
            "Upload authentic reference carton (JPG, PNG, WEBP)",
            key="reference_upload",
        )

        if st.button("Save Reference Image", type="primary", width="stretch"):
            medicine_name_clean = medicine_name.strip()

            if not medicine_name_clean:
                st.error("Please enter medicine name.")
                st.stop()

            if reference_file is None:
                st.error("Please upload reference image.")
                st.stop()

            allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
            if Path(reference_file.name).suffix.lower() not in allowed_ext:
                st.error("Unsupported file type. Please upload a JPG, PNG, or WEBP image.")
                st.stop()

            ref_path = save_reference_image(medicine_name_clean, reference_file)

            st.success(f"Reference image saved for {medicine_name_clean}")
            st.image(str(ref_path), caption="Saved Reference", use_column_width=True)

            st.rerun()

        st.markdown("### Current Reference Database")

        db = load_reference_db()

        if not db:
            st.info("No reference images uploaded yet.")
        else:
            for key, item in db.items():
                with st.expander(item["medicine_name"]):
                    st.write(item["image_path"])

                    if Path(item["image_path"]).exists():
                        st.image(item["image_path"], width=350)
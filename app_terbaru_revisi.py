from pathlib import Path
import json
import math
import joblib
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor


# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "xgboost_model.json"
METADATA_PATH = MODEL_DIR / "metadata.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DATA_PATH = DATA_DIR / "data_penumpang.csv"

REQUIRED_COLUMNS = [
    "Tanggal",
    "Waktu",
    "Nomor Bus",
    "Rute/Koridor",
    "Jumlah Penumpang",
]

def load_background():
    image_path = Path(__file__).parent / "background_transjakarta.jpg"

    if not image_path.exists():
        return ""

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    return encoded


background_image = load_background()

st.set_page_config(
    page_title="Prediksi Penumpang Transjakarta",
    page_icon="🚌",
    layout="wide",
)


# ============================================================
# 2. CSS - PERBAIKAN KONTRAS SELURUH KOMPONEN
# ============================================================


st.markdown(
    """
    <style>
    /* ===== TEMA BIRU TRANSJAKARTA ===== */

    .stApp {
        background:
            linear-gradient(
                rgba(5, 20, 35, 0.78),
                rgba(5, 20, 35, 0.86)
            ),
            url("data:image/jpeg;base64,{}")
            center center / cover fixed no-repeat;
    }

    section.main,
    section.main .block-container {
        background: transparent !important;
        color: #FFFFFF !important;
    }

    /* Semua teks utama */
    section.main h1,
    section.main h2,
    section.main h3,
    section.main h4,
    section.main p,
    section.main li,
    section.main label,
    section.main small,
    section.main span,
    section.main [data-testid="stMarkdownContainer"],
    section.main [data-testid="stMarkdownContainer"] *,
    section.main [data-testid="stCaptionContainer"],
    section.main [data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
    }

    /* Sidebar bergaya dashboard gelap */
    section[data-testid="stSidebar"] {
        background: #262626 !important;
        border-right: 1px solid #383838 !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #262626 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #E8E8E8 !important;
    }

    /* Header navigasi */
    .sidebar-heading {
        padding: 7px 4px 13px 4px;
        margin-bottom: 12px;
        border-bottom: 1px solid #414141;
    }

    .sidebar-heading-title {
        color: #F4F4F4;
        font-size: 15px;
        font-weight: 750;
        letter-spacing: 1.1px;
    }

    .sidebar-heading-caption {
        color: #BDBDBD;
        font-size: 11px;
        line-height: 1.45;
        margin-top: 5px;
    }

    /* Kotak pencarian */
    section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #333333 !important;
        color: #F2F2F2 !important;
        border: 1px solid #3D3D3D !important;
        border-radius: 8px !important;
        min-height: 37px !important;
        font-size: 12px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
        color: #AFAFAF !important;
    }

    section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
        border-color: #777777 !important;
        box-shadow: none !important;
    }

    /* Navigasi sidebar: satu ikon khusus untuk setiap menu */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px !important;
    }

    /* Hilangkan lingkaran radio bawaan Streamlit, pertahankan teks */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label input[type="radio"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:not([data-testid="stMarkdownContainer"]) {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Baris menu */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        min-height: 41px !important;
        padding: 8px 11px !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: background 0.18s ease, border-color 0.18s ease !important;
    }

    /* Ikon tunggal: ikon dan indikator radio digabung menjadi satu */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label::before {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 22px !important;
        min-width: 22px !important;
        height: 22px !important;
        margin-right: 11px !important;
        color: #BDBDBD !important;
        font-family: "Segoe UI Symbol", "Arial Unicode MS", sans-serif !important;
        font-size: 18px !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        background: transparent !important;
        border: 0 !important;
        border-radius: 0 !important;
    }

    /* Beranda, Unggah Dataset, Prapemrosesan,
       Prediksi, Rekomendasi Layanan, Evaluasi Model */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(1)::before { content: "⌂" !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(2)::before { content: "↑" !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(3)::before { content: "⚙" !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(4)::before { content: "▥" !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(5)::before { content: "⇄" !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(6)::before { content: "✓" !important; }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #D8D8D8 !important;
        font-size: 13px !important;
        font-weight: 450 !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #343434 !important;
        border-color: #3C3C3C !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #383838 !important;
        border-color: #414141 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::before {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Kartu metrik */
    section.main [data-testid="stMetric"] {
        background: rgba(0, 30, 65, 0.65) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 5px 18px rgba(0,0,0,0.16);
    }

    section.main [data-testid="stMetric"] *,
    section.main [data-testid="stMetricLabel"],
    section.main [data-testid="stMetricValue"],
    section.main [data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }

    /* Tombol */
    section.main .stButton > button,
    section.main .stDownloadButton > button,
    section.main [data-testid="stFormSubmitButton"] button {
        background: #00AEEF !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.2s ease;
    }

    section.main .stButton > button *,
    section.main .stDownloadButton > button *,
    section.main [data-testid="stFormSubmitButton"] button * {
        color: #FFFFFF !important;
    }

    section.main .stButton > button:hover,
    section.main .stDownloadButton > button:hover,
    section.main [data-testid="stFormSubmitButton"] button:hover {
        background: #008FC7 !important;
        border-color: #FFFFFF !important;
    }

    /* Input: latar putih agar nilai tetap mudah dibaca */
    section.main input,
    section.main textarea {
        background: #FFFFFF !important;
        color: #003B70 !important;
        border: 1px solid #B7D7EC !important;
        border-radius: 8px !important;
    }

    section.main input::placeholder,
    section.main textarea::placeholder {
        color: #617C91 !important;
    }

    /* Dropdown */
    section.main [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #B7D7EC !important;
    }

    section.main [data-baseweb="select"] *,
    section.main [data-baseweb="input"] *,
    section.main [data-baseweb="textarea"] * {
        color: #003B70 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #FFFFFF !important;
    }

    [data-baseweb="menu"] *,
    [role="listbox"] * {
        color: #003B70 !important;
    }

    /* Area unggah dataset */
    section.main [data-testid="stFileUploaderDropzone"] {
        background: rgba(0, 35, 75, 0.7) !important;
        border: 1px dashed rgba(255,255,255,0.65) !important;
        border-radius: 12px !important;
    }

    section.main [data-testid="stFileUploaderDropzone"] *,
    section.main [data-testid="stFileUploader"] button,
    section.main [data-testid="stFileUploader"] button * {
        color: #FFFFFF !important;
    }

    section.main [data-testid="stFileUploader"] button {
        background: #003B70 !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 8px !important;
    }

    /* Notifikasi sukses */
    section.main [data-testid="stAlert"][kind="success"] {
        background: #D9F5E5 !important;
        border: 1px solid #7BCDA0 !important;
    }

    section.main [data-testid="stAlert"][kind="success"] *,
    section.main [data-testid="stAlert"][kind="success"] p {
        color: #145C38 !important;
    }

    /* Notifikasi peringatan */
    section.main [data-testid="stAlert"][kind="warning"] {
        background: #FFF3CD !important;
        border: 1px solid #D8B65B !important;
    }

    section.main [data-testid="stAlert"][kind="warning"] *,
    section.main [data-testid="stAlert"][kind="warning"] p {
        color: #59420B !important;
    }

    /* Notifikasi error */
    section.main [data-testid="stAlert"][kind="error"] {
        background: #FDE2E2 !important;
        border: 1px solid #E5A0A0 !important;
    }

    section.main [data-testid="stAlert"][kind="error"] *,
    section.main [data-testid="stAlert"][kind="error"] p {
        color: #842029 !important;
    }

    /* Notifikasi informasi */
    section.main [data-testid="stAlert"][kind="info"] {
        background: #DCEFFA !important;
        border: 1px solid #A0CCE7 !important;
    }

    section.main [data-testid="stAlert"][kind="info"] *,
    section.main [data-testid="stAlert"][kind="info"] p {
        color: #164B68 !important;
    }

    /* Tabel */
    section.main [data-testid="stDataFrame"],
    section.main [data-testid="stTable"] {
        background: #FFFFFF !important;
        border: 1px solid #B7D7EC !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Radio dan checkbox */
    section.main [data-testid="stRadio"] label,
    section.main [data-testid="stCheckbox"] label {
        color: #FFFFFF !important;
    }

    section.main input[type="radio"],
    section.main input[type="checkbox"] {
        accent-color: #00AEEF !important;
    }

    /* Garis pemisah */
    section.main hr {
        border-color: rgba(255,255,255,0.25) !important;
    }


    /* Siluet bus sebagai watermark latar, tidak menutupi konten */
    .stApp {
        background-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='900' height='260' viewBox='0 0 900 260'><g fill='none' stroke='%23ffffff' stroke-width='7' opacity='.12' stroke-linecap='round' stroke-linejoin='round'><path d='M90 65 Q90 35 130 35 H690 Q735 35 750 75 V185 H90 Z'/><path d='M125 70 H700 V145 H125 Z M90 165 H750'/><path d='M160 80 V135 M275 80 V135 M390 80 V135 M505 80 V135 M620 80 V135'/><path d='M120 185 H720'/><circle cx='205' cy='190' r='28'/><circle cx='640' cy='190' r='28'/></g></svg>"),
            linear-gradient(135deg, #00529B 0%, #003B70 55%, #002B50 100%) !important;
        background-repeat: no-repeat, no-repeat !important;
        background-position: right 95% bottom 2%, center !important;
        background-size: min(78vw, 900px) auto, cover !important;
        background-attachment: fixed !important;
    }
    section.main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .tj-section {
        background: rgba(0, 35, 75, 0.58);
        border: 1px solid rgba(255,255,255,.20);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: .6rem 0 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. FUNGSI BANTUAN
# ============================================================

def baca_dataset(file):
    nama = file.name.lower()

    if nama.endswith(".csv"):
        return pd.read_csv(file)

    if nama.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Format file tidak didukung.")


def preprocess_data(df):
    df = df.copy()

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Kolom dataset belum lengkap: "
            + ", ".join(missing)
        )

    df["Tanggal"] = pd.to_datetime(
        df["Tanggal"],
        dayfirst=True,
        errors="coerce",
    )

    waktu_asli = df["Waktu"].astype(str).str.strip()

    waktu_parsed = pd.to_datetime(
        waktu_asli,
        format="mixed",
        errors="coerce",
    )

    if waktu_parsed.isna().all():
        waktu_parsed = pd.to_datetime(
            waktu_asli,
            errors="coerce",
        )

    df["Waktu"] = (
        waktu_parsed.dt.hour * 60
        + waktu_parsed.dt.minute
    )

    df["Jumlah Penumpang"] = pd.to_numeric(
        df["Jumlah Penumpang"],
        errors="coerce",
    )

    jumlah_awal = len(df)

    df = df.drop_duplicates()

    jumlah_duplikat = jumlah_awal - len(df)

    invalid = (
        df["Tanggal"].isna()
        | df["Waktu"].isna()
        | df["Jumlah Penumpang"].isna()
        | (df["Jumlah Penumpang"] < 0)
    )

    jumlah_invalid = int(invalid.sum())

    df = df.loc[~invalid].copy()

    df["Waktu"] = df["Waktu"].astype(int)
    df["Jumlah Penumpang"] = (
        df["Jumlah Penumpang"].astype(float)
    )

    return df, jumlah_duplikat, jumlah_invalid


def klasifikasi_periode(menit):
    if 420 <= menit <= 570:
        return "Pagi"

    if 960 <= menit <= 1140:
        return "Sore"

    return "Di luar jam sibuk"


def baca_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. "
            "Jalankan skrip training terlebih dahulu."
        )

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    return model


def baca_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Metadata model tidak ditemukan. Jalankan train_model_terbaru_revisi.py terlebih dahulu."
        )
    return joblib.load(METADATA_PATH)


def bentuk_fitur_prediksi(result, metadata):
    """Gunakan rekayasa fitur dan urutan kolom yang sama dengan training."""
    work = result.copy()
    dates = pd.to_datetime(work["Tanggal"], errors="coerce")
    work["Waktu Menit"] = pd.to_numeric(work["Waktu (menit)"], errors="coerce")
    work["Hari dalam minggu"] = dates.dt.dayofweek
    work["Hari dalam bulan"] = dates.dt.day
    work["Bulan"] = dates.dt.month

    categories = metadata.get("route_categories", ["Koridor 2", "Rute 2A"])
    route_values = work["Rute/Koridor"].astype(str).str.strip().str.casefold()
    for route in categories:
        work[f"Rute__{route}"] = (route_values == route.casefold()).astype(int)

    columns = metadata.get("feature_columns")
    if not columns:
        raise ValueError("Daftar fitur tidak ada di metadata. Jalankan ulang training.")
    return work.reindex(columns=columns, fill_value=0).astype(float)


def buat_titik_prediksi(tanggal_mulai, waktu_pilihan):
    rows = []

    for hari in range(7):
        tanggal = (
            pd.Timestamp(tanggal_mulai)
            + pd.Timedelta(days=hari)
        )

        for waktu in waktu_pilihan:
            jam, menit = map(int, waktu.split(":"))
            menit_harian = jam * 60 + menit

            for rute in ["Koridor 2", "Rute 2A"]:
                rows.append({
                    "Tanggal": tanggal,
                    "Waktu": waktu,
                    "Waktu (menit)": menit_harian,
                    "Periode": klasifikasi_periode(menit_harian),
                    "Rute/Koridor": rute,
                })

    return pd.DataFrame(rows)


def tampilkan_status_gagal(judul, exc, tindakan=None):
    """Menampilkan kegagalan dengan penyebab dan langkah pemeriksaan."""
    pesan = str(exc).strip() or "Tidak ada detail kesalahan dari sistem."
    st.error(f"{judul}: {pesan}")
    st.markdown("**Langkah pemeriksaan:**")
    if tindakan:
        for langkah in tindakan:
            st.markdown(f"- {langkah}")
    else:
        st.markdown(
            "- Periksa kembali format dan isi data.\n"
            "- Pastikan file model dan metadata tersedia.\n"
            "- Jalankan ulang proses setelah penyebab diperbaiki."
        )


def tampilkan_progres(tahapan):
    """Progress bar sederhana untuk tahapan UI yang benar-benar dijalankan."""
    progress = st.progress(0, text="Memulai proses...")
    status = st.status("Proses sedang berjalan", expanded=True)
    try:
        for nilai, pesan in tahapan:
            status.write(pesan)
            progress.progress(nilai, text=pesan)
        status.update(label="Proses selesai", state="complete", expanded=False)
        progress.progress(100, text="Selesai")
    except Exception:
        status.update(label="Proses gagal", state="error", expanded=True)
        raise


# ============================================================
# 4. SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "processed" not in st.session_state:
    st.session_state.processed = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "rekomendasi_operasional" not in st.session_state:
    st.session_state.rekomendasi_operasional = None


# ============================================================
# 5. HEADER DAN NAVIGASI
# ============================================================

st.title("Prediksi Jumlah Penumpang Transjakarta")

st.caption(
    "XGBoost Regression | Halte Pulo Gadung | "
    "Koridor 2 dan Rute 2A"
)

st.sidebar.markdown(
    """
    <div class="sidebar-heading">
        <div class="sidebar-heading-title">MENU UTAMA</div>
        <div class="sidebar-heading-caption">Sistem Prediksi Penumpang Transjakarta</div>
    </div>
    """,
    unsafe_allow_html=True,
)

menu_halaman = [
    "Beranda",
    "Unggah Dataset",
    "Prapemrosesan",
    "Prediksi",
    "Rekomendasi Layanan",
    "Evaluasi Model",
]

kata_kunci_menu = st.sidebar.text_input(
    "Cari menu",
    placeholder="⌕  Search",
    label_visibility="collapsed",
    key="pencarian_menu_sidebar",
)

menu_tampil = [
    menu for menu in menu_halaman
    if kata_kunci_menu.strip().lower() in menu.lower()
]

if not menu_tampil:
    st.sidebar.caption("Menu tidak ditemukan.")
    menu_tampil = menu_halaman

if "halaman_aktif" not in st.session_state:
    st.session_state.halaman_aktif = "Beranda"

if st.session_state.halaman_aktif not in menu_tampil:
    st.session_state.halaman_aktif = menu_tampil[0]

# ==========================================
# NAVIGASI SIDEBAR
# ==========================================

# Menentukan halaman awal
if "halaman_aktif" not in st.session_state:
    st.session_state.halaman_aktif = "Beranda"

# Judul sidebar
st.sidebar.markdown(
    """
    <div class="sidebar-heading">
        <div class="sidebar-heading-title">MENU UTAMA</div>
        <div class="sidebar-heading-caption">
            Sistem Prediksi Penumpang Transjakarta
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Daftar menu dan ikon
menu_navigasi = [
    ("Beranda", "🏠"),
    ("Unggah Dataset", "📤"),
    ("Prapemrosesan", "⚙️"),
    ("Prediksi", "📊"),
    ("Rekomendasi Layanan", "🚌"),
    ("Evaluasi Model", "📈"),
]

# Membuat tombol navigasi
for nama_menu, ikon_menu in menu_navigasi:

    if st.sidebar.button(
        f"{ikon_menu}  {nama_menu}",
        key=f"menu_{nama_menu}",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.halaman_aktif == nama_menu
            else "secondary"
        )
    ):
        st.session_state.halaman_aktif = nama_menu
        st.rerun()

# Menentukan halaman yang sedang dibuka
page = st.session_state.halaman_aktif

# ============================================================
# 6. BERANDA
# ============================================================

if page == "Beranda":

    st.subheader("Beranda")

    st.write(
        "Aplikasi ini digunakan untuk mengolah dataset "
        "pengamatan, memprediksi jumlah penumpang menggunakan "
        "XGBoost Regression, mengevaluasi model, dan menyajikan "
        "simulasi rekomendasi layanan bus."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Metode", "XGBoost Regression")
    col2.metric("Periode Prediksi", "7 hari")
    col3.metric("Lokasi Pengamatan", "Pulo Gadung")

    st.markdown("### Status komponen aplikasi")
    status_cols = st.columns(3)
    status_cols[0].metric("File model", "Tersedia" if MODEL_PATH.exists() else "Belum ada")
    status_cols[1].metric("Metadata", "Tersedia" if METADATA_PATH.exists() else "Belum ada")
    status_cols[2].metric("Metrik evaluasi", "Tersedia" if METRICS_PATH.exists() else "Belum ada")

    st.markdown("### Alur penggunaan")

    st.markdown(
        """
        1. Unggah dataset pengamatan.
        2. Periksa dan proses data.
        3. Jalankan prediksi menggunakan model terlatih.
        4. Tinjau grafik dan hasil prediksi.
        5. Masukkan parameter operasional untuk simulasi kebutuhan armada.
        6. Periksa metrik evaluasi model.
        """
    )


# ============================================================
# 7. UNGGAH DATASET
# ============================================================

elif page == "Unggah Dataset":

    st.subheader("Unggah Dataset Pengamatan")

    uploaded = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
        key="file_dataset_pengamatan",
    )

    if uploaded is not None:
        try:
            progress = st.progress(0, text="Menyiapkan proses unggah dataset...")

            with st.status("Unggah dataset sedang diproses", expanded=True) as proses:
                proses.write("Tahap 1/3: membaca file CSV atau Excel.")
                raw = baca_dataset(uploaded)
                progress.progress(40, text="File berhasil dibaca")

                proses.write("Tahap 2/3: memeriksa struktur dan kolom dataset.")
                missing = [
                    column
                    for column in REQUIRED_COLUMNS
                    if column not in raw.columns
                ]
                progress.progress(75, text="Pemeriksaan struktur dataset selesai")

                proses.write("Tahap 3/3: menyimpan dataset ke dalam sesi aplikasi.")
                st.session_state.dataset = raw
                st.session_state.processed = None
                st.session_state.predictions = None
                st.session_state.rekomendasi_operasional = None
                progress.progress(100, text="Proses unggah selesai")

                if missing:
                    proses.update(
                        label="Dataset terbaca, tetapi kolom wajib belum lengkap",
                        state="complete",
                        expanded=False,
                    )
                else:
                    proses.update(
                        label="Dataset berhasil dibaca dan divalidasi",
                        state="complete",
                        expanded=False,
                    )

            st.success("Dataset berhasil dibaca.")
            st.write("Jumlah baris:", len(raw))
            st.write("Jumlah kolom:", len(raw.columns))

            if missing:
                st.error(
                    "Kolom wajib belum ditemukan: "
                    + ", ".join(missing)
                )
            else:
                st.success("Semua kolom wajib ditemukan.")

            st.dataframe(
                raw,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"Dataset gagal dibaca: {exc}")

    if st.session_state.dataset is not None:
        st.markdown("### Pratinjau dataset")

        st.dataframe(
            st.session_state.dataset.head(10),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 8. PRAPEMROSESAN
# ============================================================

elif page == "Prapemrosesan":

    st.subheader("Prapemrosesan Data")

    if st.session_state.dataset is None:
        st.warning("Unggah dataset terlebih dahulu.")

    else:
        if st.button("Proses Dataset"):

            try:
                progress = st.progress(0, text="Memeriksa struktur dataset...")
                with st.status("Prapemrosesan sedang berjalan", expanded=True) as proses:
                    proses.write("Tahap 1/3: memeriksa kolom dan format data.")
                    progress.progress(25, text="Struktur dataset diperiksa")
                    clean, duplicates, invalid = preprocess_data(
                        st.session_state.dataset
                    )

                    proses.write("Tahap 2/3: membersihkan data duplikat dan tidak valid.")
                    progress.progress(60, text="Pembersihan data selesai")

                    proses.write("Tahap 3/3: menyaring data pada jam sibuk pagi dan sore.")
                    clean = clean[
                        clean["Waktu"].between(420, 570)
                        | clean["Waktu"].between(960, 1140)
                    ].copy()
                    progress.progress(100, text="Prapemrosesan selesai")
                    proses.update(label="Prapemrosesan berhasil", state="complete", expanded=False)

                if clean.empty:
                    st.warning("Tidak ada data pada rentang jam sibuk. Periksa format kolom Waktu dan rentang waktu pengamatan.")

                st.session_state.processed = clean
                st.session_state.predictions = None
                st.session_state.rekomendasi_operasional = None

                st.success("Prapemrosesan selesai.")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Duplikat dihapus",
                    duplicates,
                )

                col2.metric(
                    "Baris tidak valid",
                    invalid,
                )

                col3.metric(
                    "Baris jam sibuk",
                    len(clean),
                )

                st.write(
                    "Tanggal unik:",
                    clean["Tanggal"].nunique(),
                )

                if clean.empty:
                    st.warning(
                        "Tidak ada baris yang memenuhi rentang "
                        "jam sibuk yang ditentukan."
                    )

            except Exception as exc:
                tampilkan_status_gagal(
                    "Prapemrosesan gagal",
                    exc,
                    [
                        "Pastikan lima kolom wajib tersedia: Tanggal, Waktu, Nomor Bus, Rute/Koridor, dan Jumlah Penumpang.",
                        "Pastikan tanggal dan waktu dapat dibaca serta jumlah penumpang berupa angka nonnegatif.",
                        "Pastikan data memiliki catatan pada jam sibuk pagi (07.00–09.30) atau sore (16.00–19.00).",
                    ],
                )

        if st.session_state.processed is not None:

            st.markdown("### Hasil prapemrosesan")

            st.dataframe(
                st.session_state.processed,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Unduh hasil prapemrosesan",
                data=st.session_state.processed.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="hasil_prapemrosesan.csv",
                mime="text/csv",
            )


# ============================================================
# 9. PREDIKSI
# ============================================================

elif page == "Prediksi":

    st.subheader("Prediksi Jumlah Penumpang")

    try:
        model = baca_model()

        st.success("Model XGBoost berhasil dimuat.")

        st.info(
            "Model menggunakan waktu pengamatan, hari dalam minggu, hari dalam bulan, "
            "bulan, dan rute sebagai fitur. Prediksi tetap bersifat estimasi karena "
            "dataset pengamatan saat ini hanya mencakup lima tanggal."
        )

        tanggal_mulai = st.date_input(
            "Tanggal awal prediksi",
            value=pd.Timestamp.today().date(),
        )

        semua_waktu = [
            "07:00",
            "07:10",
            "07:20",
            "07:30",
            "07:40",
            "07:50",
            "08:00",
            "08:10",
            "08:20",
            "08:30",
            "08:40",
            "08:50",
            "09:00",
            "09:10",
            "09:20",
            "09:30",
            "16:00",
            "16:10",
            "16:20",
            "16:30",
            "16:40",
            "16:50",
            "17:00",
            "17:10",
            "17:20",
            "17:30",
            "17:40",
            "17:50",
            "18:00",
            "18:10",
            "18:20",
            "18:30",
            "18:40",
            "18:50",
            "19:00",
        ]

        waktu_default = [
            "07:00",
            "08:00",
            "09:00",
            "09:30",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
        ]

        waktu_pilihan = st.multiselect(
            "Waktu pengamatan untuk diprediksi",
            options=semua_waktu,
            default=waktu_default,
        )

        if st.button("Jalankan Prediksi"):

            if not waktu_pilihan:
                st.warning("Pilih setidaknya satu waktu.")

            else:
                with st.status("Prediksi sedang diproses", expanded=True) as proses:
                    proses.write("Tahap 1/4: membentuk tanggal, waktu, dan rute prediksi.")
                    result = buat_titik_prediksi(
                        tanggal_mulai,
                        waktu_pilihan,
                    )
                    proses.write("Tahap 2/4: membaca metadata model.")
                    metadata = baca_metadata()
                    proses.write("Tahap 3/4: menyusun fitur sesuai metadata training.")
                    feature_data = bentuk_fitur_prediksi(result, metadata)
                    proses.write("Tahap 4/4: menjalankan prediksi XGBoost.")
                    predictions = model.predict(feature_data)
                    proses.update(label="Prediksi berhasil", state="complete", expanded=False)

                result["Prediksi Penumpang"] = np.maximum(
                    0,
                    predictions,
                )

                result["Prediksi Penumpang"] = (
                    result["Prediksi Penumpang"]
                    .round()
                    .astype(int)
                )

                st.session_state.predictions = result
                st.session_state.rekomendasi_operasional = None

                st.success(
                    "Prediksi berhasil dijalankan untuk tujuh hari."
                )

        if st.session_state.predictions is not None:

            result = st.session_state.predictions.copy()

            st.markdown("### Tabel hasil prediksi")

            st.dataframe(
                result[
                    [
                        "Tanggal",
                        "Waktu",
                        "Rute/Koridor",
                        "Periode",
                        "Prediksi Penumpang",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Jumlah titik prediksi",
                len(result),
            )

            col2.metric(
                "Jumlah tanggal",
                result["Tanggal"].nunique(),
            )

            st.markdown("### Grafik Prediksi Penumpang")

            chart_data = result.copy()

            chart_data["Waktu Gabungan"] = pd.to_datetime(
                chart_data["Tanggal"].dt.strftime("%Y-%m-%d")
                + " "
                + chart_data["Waktu"].astype(str),
                errors="coerce",
            )

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(14, 9),
            )

            for periode, ax in [
                ("Pagi", axes[0]),
                ("Sore", axes[1]),
            ]:

                subset = chart_data[
                    chart_data["Periode"] == periode
                ].copy()

                subset = subset.sort_values(
                    "Waktu Gabungan"
                )

                if subset.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"Tidak ada prediksi periode {periode}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                    ax.set_title(
                        f"Prediksi periode {periode}"
                    )

                    continue

                ax.plot(
                    subset["Waktu Gabungan"],
                    subset["Prediksi Penumpang"],
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                )

                jumlah_label = min(8, len(subset))

                posisi_label = np.linspace(
                    0,
                    len(subset) - 1,
                    num=jumlah_label,
                    dtype=int,
                )

                ax.set_xticks(
                    subset["Waktu Gabungan"].iloc[
                        posisi_label
                    ]
                )

                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter("%d-%m %H:%M")
                )

                ax.tick_params(
                    axis="x",
                    labelrotation=35,
                )

                for label in ax.get_xticklabels():
                    label.set_horizontalalignment("right")

                ax.set_title(
                    f"Prediksi Penumpang - {periode}"
                )

                ax.set_xlabel("Tanggal dan waktu")
                ax.set_ylabel("Prediksi jumlah penumpang")
                ax.grid(True, alpha=0.3)

            fig.suptitle(
                "Prediksi Jumlah Penumpang Rush Hour Selama 7 Hari"
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                "Unduh hasil prediksi",
                data=result.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )

    except FileNotFoundError as exc:
        tampilkan_status_gagal(
            "Komponen model belum tersedia",
            exc,
            [
                "Pastikan folder model berada satu direktori dengan app_terbaru_revisi(2).py.",
                "Pastikan xgboost_model.json dan metadata.joblib dihasilkan oleh skrip training yang sesuai.",
                "Jalankan training terlebih dahulu, lalu buka kembali halaman Prediksi.",
            ],
        )
    except Exception as exc:
        tampilkan_status_gagal(
            "Prediksi gagal dijalankan",
            exc,
            [
                "Pastikan model dan metadata berasal dari versi training yang sama.",
                "Periksa kesesuaian nama fitur pada metadata dengan fitur yang dibentuk aplikasi.",
                "Periksa kembali pilihan waktu dan tanggal prediksi, lalu coba ulang.",
            ],
        )


# ============================================================
# 10. REKOMENDASI LAYANAN
# ============================================================

elif page == "Rekomendasi Layanan":

    st.subheader(
        "Rekomendasi Operasional Penyediaan Bus"
    )

    result = st.session_state.predictions

    if result is None:

        st.warning(
            "Jalankan prediksi terlebih dahulu pada halaman Prediksi."
        )

    else:

        st.write(
            "Masukkan parameter operasional berdasarkan data aktual "
            "atau sumber resmi yang digunakan dalam penelitian."
        )

        st.warning(
            "Gunakan nilai operasional yang benar-benar diketahui. "
            "Aplikasi tidak mengisi parameter armada secara otomatis."
        )

        st.markdown("### 1. Parameter operasional")

        with st.form("form_rekomendasi_operasional"):

            rute = st.selectbox(
                "Rute/Koridor yang dianalisis",
                options=[
                    "Koridor 2",
                    "Rute 2A",
                ],
            )

            col1, col2 = st.columns(2)

            with col1:

                bus_tersedia = st.number_input(
                    "Jumlah bus tersedia",
                    min_value=0,
                    value=0,
                    step=1,
                )

                kapasitas_bus = st.number_input(
                    "Kapasitas penumpang per bus",
                    min_value=0,
                    value=0,
                    step=1,
                )

            with col2:

                headway_target = st.number_input(
                    "Headway target (menit)",
                    min_value=0,
                    value=0,
                    step=1,
                )

                waktu_putaran = st.number_input(
                    "Waktu putaran lengkap (menit)",
                    min_value=0,
                    value=0,
                    step=1,
                )

            submitted = st.form_submit_button(
                "Hitung Rekomendasi Operasional"
            )

        if submitted:

            if kapasitas_bus <= 0:
                st.error("Masukkan kapasitas bus lebih besar dari 0.")

            elif headway_target <= 0:
                st.error("Masukkan headway target lebih besar dari 0.")

            elif waktu_putaran <= 0:
                st.error("Masukkan waktu putaran lebih besar dari 0.")

            else:
                try:
                    progress = st.progress(0, text=f"Memulai perhitungan untuk {rute}...")

                    with st.status(
                        f"Menghitung rekomendasi operasional {rute}",
                        expanded=True,
                    ) as proses:
                        # Tahap 1: memastikan parameter yang dimasukkan valid.
                        proses.write("Tahap 1/5: memvalidasi parameter operasional.")
                        if bus_tersedia < 0:
                            raise ValueError("Jumlah bus tersedia tidak boleh negatif.")
                        progress.progress(15, text="Parameter operasional tervalidasi")

                        # Tahap 2: mengambil prediksi hanya untuk rute terpilih.
                        proses.write(f"Tahap 2/5: mengambil hasil prediksi untuk {rute}.")
                        prediksi_rute = result.copy()
                        prediksi_rute = prediksi_rute[
                            prediksi_rute["Rute/Koridor"].astype(str).str.strip() == rute
                        ]
                        if prediksi_rute.empty:
                            raise ValueError(
                                f"Tidak ditemukan hasil prediksi untuk {rute}. "
                                "Jalankan kembali prediksi dan pastikan rute tersebut tersedia."
                            )
                        prediksi_puncak = float(
                            prediksi_rute["Prediksi Penumpang"].max()
                        )
                        progress.progress(35, text=f"Data prediksi {rute} berhasil diambil")

                        # Tahap 3: menghitung kebutuhan armada berdasarkan headway.
                        proses.write("Tahap 3/5: menghitung kebutuhan armada berdasarkan headway dan waktu putaran.")
                        kebutuhan_armada_headway = math.ceil(
                            waktu_putaran / headway_target
                        )
                        progress.progress(55, text="Perhitungan kebutuhan berdasarkan headway selesai")

                        # Tahap 4: membandingkan prediksi puncak dengan kapasitas bus.
                        proses.write(f"Tahap 4/5: membandingkan prediksi puncak {rute} dengan kapasitas bus.")
                        rasio_kapasitas = prediksi_puncak / float(kapasitas_bus)
                        kebutuhan_armada_prediksi = max(
                            1, math.ceil(prediksi_puncak / float(kapasitas_bus))
                        )
                        kebutuhan_armada = max(
                            kebutuhan_armada_headway, kebutuhan_armada_prediksi
                        )
                        tambahan_bus = max(0, kebutuhan_armada - bus_tersedia)
                        progress.progress(80, text="Perbandingan kapasitas bus selesai")

                        # Tahap 5: menyusun status dan menyimpan hasil.
                        proses.write(f"Tahap 5/5: menyusun rekomendasi untuk {rute}.")
                        if prediksi_puncak >= kapasitas_bus:
                            status = (
                                "Prediksi puncak mencapai/melebihi kapasitas satu bus. "
                                "Perlu evaluasi penambahan kapasitas atau frekuensi layanan; "
                                "rekomendasi armada juga mempertimbangkan headway dan waktu putaran."
                            )
                        elif bus_tersedia < kebutuhan_armada:
                            status = (
                                "Armada tersedia di bawah estimasi kebutuhan gabungan "
                                "dari headway/waktu putaran dan indikator prediksi penumpang."
                            )
                        else:
                            status = (
                                "Armada tersedia memenuhi estimasi kebutuhan gabungan. "
                                "Tetap tinjau prediksi puncak dan kondisi operasional aktual."
                            )

                        st.session_state.rekomendasi_operasional = {
                            "Rute/Koridor": rute,
                            "Bus Tersedia": int(bus_tersedia),
                            "Kapasitas Bus": int(kapasitas_bus),
                            "Headway Target (menit)": int(headway_target),
                            "Waktu Putaran (menit)": int(waktu_putaran),
                            "Prediksi Puncak Penumpang": round(prediksi_puncak, 2),
                            "Rasio Prediksi terhadap Kapasitas": round(rasio_kapasitas, 4),
                            "Kebutuhan Armada berdasarkan Headway": int(kebutuhan_armada_headway),
                            "Kebutuhan Armada berdasarkan Prediksi": int(kebutuhan_armada_prediksi),
                            "Estimasi Kebutuhan Armada": int(kebutuhan_armada),
                            "Estimasi Tambahan Bus": int(tambahan_bus),
                            "Status": status,
                        }
                        progress.progress(100, text=f"Rekomendasi {rute} selesai")
                        proses.update(
                            label=f"Rekomendasi operasional {rute} berhasil dihitung",
                            state="complete",
                            expanded=False,
                        )

                except Exception as exc:
                    st.session_state.rekomendasi_operasional = None
                    tampilkan_status_gagal(
                        f"Perhitungan rekomendasi {rute} gagal",
                        exc,
                        [
                            "Pastikan hasil prediksi untuk rute yang dipilih sudah tersedia.",
                            "Pastikan kapasitas bus, headway, dan waktu putaran lebih besar dari 0.",
                            "Periksa kembali parameter operasional yang dimasukkan.",
                        ],
                    )

        rekomendasi = st.session_state.rekomendasi_operasional

        if rekomendasi is not None:

            st.markdown("### 2. Hasil perhitungan")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Prediksi puncak penumpang",
                f"{rekomendasi['Prediksi Puncak Penumpang']:.0f} orang",
            )
            col2.metric(
                "Rasio terhadap kapasitas",
                f"{rekomendasi['Rasio Prediksi terhadap Kapasitas'] * 100:.1f}%",
            )
            col3.metric(
                "Estimasi kebutuhan armada",
                f"{rekomendasi['Estimasi Kebutuhan Armada']} bus",
            )
            col4.metric(
                "Estimasi tambahan bus",
                f"{rekomendasi['Estimasi Tambahan Bus']} bus",
            )

            st.info(rekomendasi["Status"])

            tabel_rekomendasi = pd.DataFrame([{
                "Rute/Koridor": rekomendasi["Rute/Koridor"],
                "Bus Tersedia": rekomendasi["Bus Tersedia"],
                "Kapasitas Bus": rekomendasi["Kapasitas Bus"],
                "Headway Target (menit)": (
                    rekomendasi["Headway Target (menit)"]
                ),
                "Waktu Putaran (menit)": (
                    rekomendasi["Waktu Putaran (menit)"]
                ),
                "Prediksi Puncak Penumpang": (
                    rekomendasi["Prediksi Puncak Penumpang"]
                ),
                "Rasio Prediksi terhadap Kapasitas": (
                    rekomendasi["Rasio Prediksi terhadap Kapasitas"]
                ),
                "Kebutuhan Armada berdasarkan Headway": (
                    rekomendasi["Kebutuhan Armada berdasarkan Headway"]
                ),
                "Kebutuhan Armada berdasarkan Prediksi": (
                    rekomendasi["Kebutuhan Armada berdasarkan Prediksi"]
                ),
                "Estimasi Kebutuhan Armada": (
                    rekomendasi["Estimasi Kebutuhan Armada"]
                ),
                "Estimasi Tambahan Bus": (
                    rekomendasi["Estimasi Tambahan Bus"]
                ),
            }])

            st.dataframe(
                tabel_rekomendasi,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Unduh hasil rekomendasi operasional",
                data=tabel_rekomendasi.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="rekomendasi_operasional.csv",
                mime="text/csv",
            )

            st.caption(
                "Kebutuhan armada akhir menggunakan nilai terbesar dari: "
                "(1) pembulatan ke atas waktu putaran/headway, dan "
                "(2) pembulatan ke atas prediksi puncak penumpang/kapasitas bus. "
                "Estimasi ini adalah aturan keputusan rancangan, bukan pengganti "
                "perhitungan operasional resmi. Pastikan prediksi pada titik waktu "
                "dapat ditafsirkan sebagai permintaan yang relevan untuk kapasitas bus."
            )

        st.divider()

        st.markdown("### 3. Tinjauan prediksi penumpang")

        st.write(
            "Prediksi pada satu waktu tidak otomatis sama dengan "
            "jumlah penumpang yang datang selama satu interval headway."
        )

        st.dataframe(
            result[
                [
                    "Tanggal",
                    "Waktu",
                    "Rute/Koridor",
                    "Periode",
                    "Prediksi Penumpang",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Model menggunakan waktu, tanggal kalender, dan rute sebagai fitur. "
            "Hasil prediksi untuk Koridor 2 dan Rute 2A dihitung secara terpisah."
        )

        st.warning(
            "Prediksi tetap merupakan estimasi berdasarkan data pengamatan yang tersedia. "
            "Rasio prediksi terhadap kapasitas hanya indikator awal; validasi dengan data "
            "kapasitas resmi, pola penumpang per interval, armada cadangan, dan waktu tempuh "
            "diperlukan sebelum rekomendasi diterapkan secara operasional."
        )


# ============================================================
# 11. EVALUASI MODEL
# ============================================================

elif page == "Evaluasi Model":

    st.subheader("Evaluasi Model")

    if not METRICS_PATH.exists():

        st.warning(
            "File metrik belum ditemukan. Jalankan skrip training "
            "terlebih dahulu."
        )

    else:

        try:
            with open(
                METRICS_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                metrics = json.load(file)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "MAE",
                f"{metrics['MAE']:.3f}",
            )

            col2.metric(
                "MSE",
                f"{metrics['MSE']:.3f}",
            )

            col3.metric(
                "RMSE",
                f"{metrics['RMSE']:.3f}",
            )

            col4.metric(
                "R²",
                f"{metrics['R2']:.3f}",
            )

            if "jumlah_data" in metrics:
                st.write(
                    "Jumlah data:",
                    metrics["jumlah_data"],
                )

            if "jumlah_data_latih" in metrics:
                st.write(
                    "Data pelatihan:",
                    metrics["jumlah_data_latih"],
                )

            if "jumlah_data_uji" in metrics:
                st.write(
                    "Data pengujian:",
                    metrics["jumlah_data_uji"],
                )

            st.caption(
                "Metrik berasal dari evaluasi pada saat training. "
                "Nilai R² negatif menunjukkan model pada data uji "
                "lebih buruk dibandingkan baseline prediksi rata-rata "
                "target data uji."
            )

        except Exception as exc:
            st.error(
                f"Metrik evaluasi gagal dibaca: {exc}"
            )
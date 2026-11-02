from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

POSSIBLE_FILES = [
    DATA_DIR / "data_layanan_penumpang_bus_tj_halte_pulo_gadung.xlsx",
    BASE_DIR / "data_layanan_penumpang_bus_tj_halte_pulo_gadung.xlsx",
    DATA_DIR / "data_penumpang.csv",
    BASE_DIR / "data_penumpang.csv",
]

MODEL_PATH = MODEL_DIR / "xgboost_model.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.joblib"
STATUS_PATH = MODEL_DIR / "training_status.json"
CLEAN_DATA_PATH = MODEL_DIR / "data_rush_hour.csv"

REQUIRED_COLUMNS = [
    "Tanggal",
    "Waktu",
    "Nomor Bus",
    "Rute/Koridor",
    "Jumlah Penumpang",
]

NUMERIC_FEATURES = ["Waktu Menit", "Hari dalam minggu", "Hari dalam bulan", "Bulan"]
TARGET_COLUMN = "Jumlah Penumpang"
ROUTE_PREFIX = "Rute__"

# Periode mengikuti proposal skripsi.
MORNING_START = 7 * 60
MORNING_END = 9 * 60 + 30
EVENING_START = 16 * 60
EVENING_END = 19 * 60


# ============================================================
# FUNGSI BANTU
# ============================================================

def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4, ensure_ascii=False)


def find_dataset():
    for path in POSSIBLE_FILES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset tidak ditemukan.\n"
        "Letakkan file Excel di folder data dengan nama:\n"
        "data_layanan_penumpang_bus_tj_halte_pulo_gadung.xlsx"
    )


def read_dataset(path):
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Format dataset harus CSV atau Excel.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_date(series):
    text = series.astype("string").str.strip()

    try:
        result = pd.to_datetime(
            text,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    except (TypeError, ValueError):
        result = pd.to_datetime(
            text,
            dayfirst=True,
            errors="coerce",
        )

    return result.dt.normalize()


def parse_time(series):
    result = pd.Series(np.nan, index=series.index, dtype=float)

    for index, value in series.items():
        if pd.isna(value):
            continue

        # Objek waktu dari Excel.
        if hasattr(value, "hour") and hasattr(value, "minute"):
            result.loc[index] = value.hour * 60 + value.minute
            continue

        # Waktu Excel dalam pecahan satu hari.
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if 0 <= number < 1:
                result.loc[index] = round(number * 1440) % 1440
                continue

        text = str(value).strip()
        if not text:
            continue

        parsed = pd.to_datetime(
            text,
            format="%H:%M:%S",
            errors="coerce",
        )

        if pd.isna(parsed):
            parsed = pd.to_datetime(
                text,
                format="%H:%M",
                errors="coerce",
            )

        if pd.isna(parsed):
            parsed = pd.to_datetime(
                text,
                errors="coerce",
            )

        if not pd.isna(parsed):
            result.loc[index] = parsed.hour * 60 + parsed.minute

    return result


def classify_period(minutes):
    if pd.isna(minutes):
        return "Tidak diketahui"

    minutes = int(minutes)

    if MORNING_START <= minutes <= MORNING_END:
        return "Pagi"

    if EVENING_START <= minutes <= EVENING_END:
        return "Sore"

    return "Di luar rush hour"


def clean_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan: " + ", ".join(missing)
        )

    initial_rows = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df)

    df["Tanggal"] = parse_date(df["Tanggal"])
    df["Waktu Menit"] = parse_time(df["Waktu"])

    passenger_text = (
        df["Jumlah Penumpang"]
        .astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    df["Jumlah Penumpang"] = pd.to_numeric(
        passenger_text,
        errors="coerce",
    )

    valid = (
        df["Tanggal"].notna()
        & df["Waktu Menit"].notna()
        & df["Jumlah Penumpang"].notna()
        & (df["Jumlah Penumpang"] >= 0)
        & (
            df["Jumlah Penumpang"]
            == np.floor(df["Jumlah Penumpang"])
        )
    )

    invalid_rows = int((~valid).sum())
    df = df.loc[valid].copy()

    df["Waktu Menit"] = df["Waktu Menit"].astype(int)
    df["Jumlah Penumpang"] = df["Jumlah Penumpang"].astype(int)
    df["Periode"] = df["Waktu Menit"].apply(classify_period)

    df["Waktu"] = (
        (df["Waktu Menit"] // 60).astype(str).str.zfill(2)
        + ":"
        + (df["Waktu Menit"] % 60).astype(str).str.zfill(2)
    )

    rush = df[df["Periode"].isin(["Pagi", "Sore"])].copy()
    rush = rush.sort_values(["Tanggal", "Waktu Menit"]).reset_index(drop=True)

    return rush, duplicates_removed, invalid_rows, len(df)


def normalize_route(value):
    text = str(value).strip()
    # Menjaga nama kategori tetap konsisten antara training dan aplikasi.
    aliases = {
        "koridor 2": "Koridor 2",
        "rute 2a": "Rute 2A",
    }
    return aliases.get(text.casefold(), text)


def build_features(df, route_categories):
    """Membentuk fitur tanggal, waktu, dan rute dengan kolom konsisten."""
    work = df.copy()
    dates = pd.to_datetime(work["Tanggal"], errors="coerce")
    work["Waktu Menit"] = pd.to_numeric(work["Waktu Menit"], errors="coerce")
    work["Hari dalam minggu"] = dates.dt.dayofweek
    work["Hari dalam bulan"] = dates.dt.day
    work["Bulan"] = dates.dt.month
    route_values = work["Rute/Koridor"].map(normalize_route)
    for route in route_categories:
        work[f"{ROUTE_PREFIX}{route}"] = (route_values == route).astype(int)
    feature_columns = NUMERIC_FEATURES + [f"{ROUTE_PREFIX}{r}" for r in route_categories]
    return work.reindex(columns=feature_columns, fill_value=0).astype(float), feature_columns


def create_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )


def smote_regression(X_train, y_train, random_state=42):
    """
    Adaptasi SMOTE untuk target regresi kontinu.

    Nilai target dibagi ke dalam bin kuantil hanya untuk mendeteksi
    rentang target yang jarang. Sampel sintetis dibuat melalui interpolasi
    antara dua sampel pada rentang target yang jarang. Data uji tidak pernah
    diteruskan ke fungsi ini.

    Catatan: ini bukan SMOTE klasifikasi standar; metode harus dijelaskan
    sebagai adaptasi SMOTE untuk regresi dalam laporan metodologi.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)

    if len(y) < 10 or len(np.unique(y)) < 3:
        return X_train.copy(), y_train.copy(), 0

    # Bin dengan lebar rentang target yang sama dipakai untuk mendeteksi
    # rentang nilai penumpang yang jarang; qcut dihindari karena cenderung
    # membuat jumlah sampel tiap bin hampir sama.
    try:
        bins = pd.cut(y, bins=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    except ValueError:
        return X_train.copy(), y_train.copy(), 0

    bin_counts = pd.Series(bins).value_counts().sort_index()
    if len(bin_counts) < 2:
        return X_train.copy(), y_train.copy(), 0

    typical_count = int(bin_counts.median())
    rare_bins = set(bin_counts[bin_counts < typical_count * 0.75].index.tolist())
    rare_indices = np.array([i for i, b in enumerate(bins) if b in rare_bins], dtype=int)

    if len(rare_indices) < 2:
        return X_train.copy(), y_train.copy(), 0

    rng = np.random.default_rng(random_state)
    synthetic_X, synthetic_y = [], []
    # Tambah sampel secukupnya untuk mendekatkan jumlah sampel di rentang
    # target jarang ke jumlah tipikal, tanpa menyentuh data pengujian.
    for bin_id in sorted(rare_bins):
        members = np.array([i for i, b in enumerate(bins) if b == bin_id], dtype=int)
        if len(members) < 2:
            continue
        target_count = min(typical_count, len(members) * 2)
        needed = max(0, target_count - len(members))
        for _ in range(needed):
            i = int(rng.choice(members))
            candidates = members[members != i]
            if len(candidates) == 0:
                continue
            # Tetangga dipilih berdasarkan jarak pada fitur dan target yang
            # dinormalisasi sederhana agar target kontinu ikut diperhitungkan.
            x_scale = np.std(X[members], axis=0)
            x_scale[x_scale == 0] = 1.0
            y_scale = np.std(y[members]) or 1.0
            distances = np.sqrt(
                np.sum(((X[candidates] - X[i]) / x_scale) ** 2, axis=1)
                + ((y[candidates] - y[i]) / y_scale) ** 2
            )
            j = int(candidates[np.argmin(distances)])
            lam = float(rng.random())
            synthetic_X.append(X[i] + lam * (X[j] - X[i]))
            synthetic_y.append(y[i] + lam * (y[j] - y[i]))

    if not synthetic_y:
        return X_train.copy(), y_train.copy(), 0

    X_aug = np.vstack([X, np.asarray(synthetic_X)])
    y_aug = np.concatenate([y, np.asarray(synthetic_y)])
    return pd.DataFrame(X_aug, columns=X_train.columns), pd.Series(y_aug), len(synthetic_y)


# ============================================================
# TRAINING DAN EVALUASI
# ============================================================

def main():
    print("=" * 65)
    print("TRAINING XGBOOST - PENUMPANG RUSH HOUR")
    print("=" * 65)

    data_path = find_dataset()
    df = read_dataset(data_path)

    print(f"Dataset       : {data_path}")
    print(f"Baris awal    : {len(df)}")

    rush, duplicates_removed, invalid_rows, valid_rows = clean_dataset(df)

    print(f"Duplikat dihapus         : {duplicates_removed}")
    print(f"Baris tidak valid dihapus: {invalid_rows}")
    print(f"Baris valid              : {valid_rows}")
    print(f"Baris rush hour          : {len(rush)}")
    print(f"Tanggal unik             : {rush['Tanggal'].nunique()}")

    if rush.empty:
        raise ValueError(
            "Tidak ada data dalam periode pagi 07.00-09.30 "
            "atau sore 16.00-19.00."
        )

    rush.to_csv(
        CLEAN_DATA_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    route_categories = sorted(rush["Rute/Koridor"].dropna().map(normalize_route).unique().tolist())
    if not route_categories:
        raise ValueError("Kolom Rute/Koridor tidak memiliki kategori valid.")

    X, FEATURE_COLUMNS = build_features(rush, route_categories)
    y = rush[TARGET_COLUMN].astype(float)

    if len(rush) < 10:
        raise ValueError(
            "Jumlah data rush hour kurang dari 10 baris. "
            "Data belum cukup untuk pembagian latih dan uji."
        )

    # Evaluasi menggunakan data uji terpisah.
    # Data uji tidak ikut dipakai untuk fit model evaluasi.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # SMOTE adaptasi regresi hanya diterapkan pada data training.
    # X_test dan y_test tetap data asli untuk evaluasi yang adil.
    X_train_smote, y_train_smote, synthetic_count = smote_regression(
        X_train, y_train, random_state=42
    )
    print(f"Sampel sintetis SMOTE-regression: {synthetic_count}")
    print(f"Data latih sebelum SMOTE       : {len(X_train)}")
    print(f"Data latih setelah SMOTE       : {len(X_train_smote)}")

    evaluation_model = create_model()
    evaluation_model.fit(X_train_smote, y_train_smote)

    y_pred = np.maximum(
        evaluation_model.predict(X_test),
        0,
    )

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    # R2 dapat bernilai negatif; itu tidak otomatis berarti error kode.
    r2 = (
        float(r2_score(y_test, y_pred))
        if len(y_test) >= 2
        else None
    )

    metrics = {
        "dataset": data_path.name,
        "jumlah_baris_awal": int(len(df)),
        "duplikat_dihapus": int(duplicates_removed),
        "baris_tidak_valid_dihapus": int(invalid_rows),
        "baris_valid": int(valid_rows),
        "jumlah_data_rush_hour": int(len(rush)),
        "jumlah_tanggal_unik": int(rush["Tanggal"].nunique()),
        "jumlah_data_latih": int(len(X_train)),
        "jumlah_data_uji": int(len(X_test)),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": r2,
        "evaluasi_tersedia": True,
        "metode_evaluasi": "train_test_split 80:20, random_state=42",
        "metode_penyeimbangan": "Adaptasi SMOTE untuk regresi kontinu; hanya pada data latih",
        "sampel_sintetis_smote_regression": int(synthetic_count),
        "jumlah_data_latih_setelah_smote": int(len(X_train_smote)),
        "fitur": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "catatan": (
            "Model memakai waktu, hari dalam minggu, hari dalam bulan, bulan, "
            "dan rute sebagai fitur. Karena data hanya mencakup lima tanggal, "
            "hasil evaluasi dan prediksi untuk tanggal baru harus ditafsirkan "
            "secara hati-hati."
        ),
    }

    # Model final dilatih menggunakan seluruh data rush hour yang telah
    # diproses dengan adaptasi SMOTE-regression. Model evaluasi di atas
    # tetap diuji pada X_test/y_test asli.
    X_final_smote, y_final_smote, final_synthetic_count = smote_regression(
        X, y, random_state=42
    )
    final_model = create_model()
    final_model.fit(X_final_smote, y_final_smote)
    final_model.save_model(str(MODEL_PATH))

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "route_categories": route_categories,
        "target_column": TARGET_COLUMN,
        "morning_range": "07:00-09:30",
        "evening_range": "16:00-19:00",
        "location": "Halte Pulo Gadung",
        "services": ["Koridor 2", "Rute 2A"],
        "training_data": data_path.name,
        "training_rows": int(len(rush)),
        "final_synthetic_rows": int(final_synthetic_count),
        "smote_method": "Adaptasi SMOTE untuk regresi kontinu; interpolasi sampel pada rentang target jarang",
        "training_dates": sorted(
            rush["Tanggal"].dt.strftime("%Y-%m-%d").unique().tolist()
        ),
        "model_note": (
            "Fitur model meliputi waktu pengamatan, hari dalam minggu, "
            "hari dalam bulan, bulan, serta kategori Rute/Koridor. "
            "Tanggal dan rute kini ikut memengaruhi prediksi. Adaptasi "
            "SMOTE-regression digunakan hanya pada data latih; data uji tetap asli. "
            "Data pengamatan hanya mencakup lima tanggal sehingga kemampuan "
            "generalisasi ke tanggal baru masih terbatas."
        ),
    }

    joblib.dump(metadata, METADATA_PATH)
    save_json(METRICS_PATH, metrics)
    save_json(
        STATUS_PATH,
        {
            "trained": True,
            "evaluated": True,
            "message": "Training dan evaluasi selesai.",
        },
    )

    print("\nHASIL EVALUASI")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2 if r2 is not None else 'Tidak tersedia'}")

    print("\n" + "=" * 65)
    print("TRAINING SELESAI")
    print("=" * 65)
    print(f"Model    : {MODEL_PATH}")
    print(f"Metrik   : {METRICS_PATH}")
    print(f"Metadata : {METADATA_PATH}")
    print(f"Data olah: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR TRAINING: {error}")
        save_json(
            STATUS_PATH,
            {
                "trained": False,
                "evaluated": False,
                "message": str(error),
            },
        )
        raise SystemExit(1)
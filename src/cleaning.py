Fungsi-fungsi inti untuk tahap pembersihan data dan verifikasi bahasa
(Notebook 01: Cleaning Data & Language Detection).

Pipeline tahap ini:
    ingesti data mentah  ->  pembersihan  ->  filter label bahasa platform
    ->  verifikasi langdetect  ->  filter berdasarkan hasil deteksi

Bahasa target penelitian: Bahasa Inggris (en), Bahasa Indonesia (id),
dan Bahasa Spanyol (es).

Catatan:
    Kode ini merupakan ekstraksi fungsi inti dari notebook. Sel eksplorasi,
    pencetakan progres, dan visualisasi tidak disertakan. Untuk reproduksi
    penuh beserta luarannya, lihat notebook pada folder notebooks/.

import numpy as np
import pandas as pd

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Seed langdetect agar hasil deteksi konsisten (reproducible)
DetectorFactory.seed = 42

# Kode bahasa yang diterima dari label platform.
#   'en'        -> Bahasa Inggris
#   'id' / 'in' -> Bahasa Indonesia (platform memakai kedua variasi kode)
#   'es'        -> Bahasa Spanyol
BAHASA_PLATFORM = ["en", "id", "in", "es"]

# Bahasa target final (berdasarkan hasil deteksi)
TARGET_LANGS = ["en", "id", "es"]


def standardize_columns(df):
    """Menstandarkan nama kolom dataset mentah ke format penelitian dan
    memilih kolom yang dibutuhkan.

    Args:
        df (pd.DataFrame): dataframe mentah hasil pembacaan berkas.

    Returns:
        pd.DataFrame: dataframe dengan kolom terstandar.
    """
    df = df.rename(columns={
        "place_name" : "nama_tempat",
        "place_id"   : "id_tempat",
        "address"    : "alamat",
        "city"       : "kabupaten",
        "review_text": "review",
        "source"     : "sumber",
    })

    df["provinsi"] = "Bali"

    kolom_pilih = [
        "user_id", "nama_tempat", "id_tempat", "rating",
        "kabupaten", "provinsi", "review", "language", "sumber",
    ]
    kolom_ada = [c for c in kolom_pilih if c in df.columns]
    return df[kolom_ada].copy()


def clean_dataframe(df, min_length=10):
    """Membersihkan dataframe secara bertahap.

    Tahapan:
        1. Hapus review kosong (NaN atau hanya spasi).
        2. Hapus duplikat berdasarkan isi ulasan.
        3. Hapus ulasan yang terlalu pendek (< min_length karakter).

    Args:
        df (pd.DataFrame): dataframe hasil standardize_columns.
        min_length (int): panjang minimum karakter ulasan yang dipertahankan.

    Returns:
        pd.DataFrame: dataframe yang telah dibersihkan.
    """
    # Langkah 1: hapus review kosong / NaN
    df["review"] = df["review"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["review"]).copy()

    # Langkah 2: hapus duplikat berdasarkan isi ulasan
    df = df.drop_duplicates(subset=["review"]).copy()

    # Langkah 3: hapus ulasan terlalu pendek
    df = df[df["review"].astype(str).str.len() >= min_length].copy()

    return df.reset_index(drop=True)


def filter_by_platform_language(df):
    """Menyaring baris berdasarkan label bahasa platform.

    Tahap ini mendefinisikan cakupan populasi data: hanya ulasan yang
    label platformnya termasuk salah satu bahasa target yang dipertahankan.

    Args:
        df (pd.DataFrame): dataframe yang telah dibersihkan.

    Returns:
        pd.DataFrame: dataframe yang lolos filter label platform.
    """
    df = df[df["language"].isin(BAHASA_PLATFORM)].copy()
    return df.reset_index(drop=True)


def detect_language(text, min_length=10):
    """Mendeteksi bahasa dari teks menggunakan langdetect.

    Args:
        text (str): teks ulasan.
        min_length (int): panjang minimum agar deteksi dijalankan.

    Returns:
        str: kode bahasa (mis. 'en', 'id', 'es') atau 'unknown' jika gagal.
    """
    try:
        if not isinstance(text, str) or len(text.strip()) < min_length:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"


def map_to_target_lang(lang_code):
    """Memetakan kode bahasa hasil deteksi ke tiga bahasa target atau 'other'.

    Args:
        lang_code (str): kode bahasa dari langdetect.

    Returns:
        str: salah satu dari 'en', 'id', 'es', atau 'other'.
    """
    if lang_code == "en":
        return "en"
    elif lang_code in ["id", "in"]:
        return "id"
    elif lang_code == "es":
        return "es"
    else:
        return "other"


def run_language_detection(df, text_col="review", batch_size=500):
    """Menjalankan deteksi bahasa pada seluruh ulasan secara batch.

    Menambahkan dua kolom:
        - 'language_detected'        : kode mentah dari langdetect
        - 'language_detected_mapped' : hasil pemetaan ke bahasa target

    Args:
        df (pd.DataFrame): dataframe berisi kolom teks ulasan.
        text_col (str): nama kolom teks ulasan.
        batch_size (int): ukuran batch pemrosesan.

    Returns:
        pd.DataFrame: dataframe dengan kolom hasil deteksi bahasa.
    """
    total = len(df)
    detected_langs = []

    for i in range(0, total, batch_size):
        batch = df[text_col].iloc[i:i + batch_size]
        detected_langs.extend(batch.apply(detect_language).tolist())

    df["language_detected"] = detected_langs
    df["language_detected_mapped"] = df["language_detected"].apply(map_to_target_lang)
    return df


def analyze_mislabeling(df):
    """Menyusun crosstab antara label bahasa platform dan hasil deteksi,
    serta mengembalikan baris yang labelnya tidak sesuai (mislabeled).

    Args:
        df (pd.DataFrame): dataframe hasil run_language_detection.

    Returns:
        tuple: (crosstab (pd.DataFrame), mislabeled (pd.DataFrame))
    """
    # Simpan label asli platform sebelum kolom 'language' diperbarui
    df["language_original"] = df["language"].copy()

    crosstab = pd.crosstab(
        df["language_original"],
        df["language_detected_mapped"],
        margins=True,
        margins_name="Total",
    )

    mislabeled = df[df["language_original"] != df["language_detected_mapped"]]
    return crosstab, mislabeled


def filter_by_detected_language(df):
    """Menyaring baris berdasarkan hasil deteksi langdetect.

    Kolom 'language' diperbarui dengan hasil deteksi, lalu hanya baris
    yang terdeteksi sebagai bahasa target (en/id/es) yang dipertahankan.
    Kolom bantu deteksi dihapus setelahnya.

    Catatan: pastikan analyze_mislabeling() telah dipanggil sebelumnya
    agar kolom 'language_original' tersimpan.

    Args:
        df (pd.DataFrame): dataframe hasil run_language_detection.

    Returns:
        pd.DataFrame: dataframe final tahap cleaning.
    """
    df["language"] = df["language_detected_mapped"]

    df = df[df["language"].isin(TARGET_LANGS)].copy()
    df = df.drop(
        columns=["language_detected", "language_detected_mapped"],
        errors="ignore",
    )
    return df.reset_index(drop=True)
